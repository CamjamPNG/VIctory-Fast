//! Native D3D12 compute shader texture detiler (`VICTORY_GPU_DETILE=1`).
//!
//! Detiles RDNA2/GFX10.3 tiled and linear textures directly on the host GPU
//! using root constants and raw buffer root descriptors (`ByteAddressBuffer` /
//! `RWByteAddressBuffer`), requiring no descriptor heap so it can be recorded
//! onto any direct command list (including `HostList::prelude`).

use std::sync::{Mutex, OnceLock};

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{Host, host};

/// Whether GPU compute texture detiling is enabled (`VICTORY_GPU_DETILE=1`).
/// Opt-in, default off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_DETILE").is_some_and(|v| v == "1"))
}

/// Whether `layout` can be detiled by the compute shader.
pub(crate) fn supported(layout: &Layout) -> bool {
    layout.addressable()
}

/// Whether a tiled texture upload of `read` tiled bytes and `linear_total`
/// linear footprint bytes can use GPU compute detiling on a direct prelude list.
pub(crate) fn can_use(layout: &Layout, read: usize, linear_total: u64) -> bool {
    on()
        && supported(layout)
        && layout.desc.swizzle != Swizzle::LINEAR
        && read > 0
        && u32::try_from(read).is_ok()
        && u32::try_from(linear_total.next_multiple_of(4)).is_ok()
        && pass().is_some()
}

/// One subresource detile (or zero-fill) work item for [`record_detile_subresources`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubresourceJob {
    pub mip: u32,
    pub slice: u32,
    pub dst_offset: u64,
    pub row_pitch: u32,
    pub depth_pitch: u32,
    pub zero: bool,
}

/// Packs the per-axis address-bit scatter positions into three `u32` shift words
/// (4 bits per coordinate bit, up to 8 coordinate bits per axis) plus bit counts,
/// using `victory_formats::texture::Layout::swizzle_shifts` as the single source of truth.
fn swizzle_encoding(layout: &Layout) -> ([u32; 3], [u32; 3]) {
    layout.swizzle_shifts().unwrap_or(([0; 3], [0; 3]))
}

const FLAG_THICK: u32 = 1 << 0;
const FLAG_ZERO: u32 = 1 << 1;

const DETILE_PARAM_WORDS: u32 = 22;

#[repr(C)]
#[derive(Clone, Copy)]
struct DetileParams {
    width: u32,
    height: u32,
    depth: u32,
    element_bytes: u32,
    row_pitch: u32,
    depth_pitch: u32,
    dst_offset: u32,
    slice_size: u32,
    mip_offset: u32,
    first_slice: u32,
    mip_pitch: u32,
    block_w: u32,
    block_h: u32,
    block_d: u32,
    block_log2: u32,
    tail_x: u32,
    tail_y: u32,
    flags: u32,
    x_shifts: u32,
    y_shifts: u32,
    z_shifts: u32,
    nbits_packed: u32,
}

const _: () = assert!(std::mem::size_of::<DetileParams>() == DETILE_PARAM_WORDS as usize * 4);

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=22, b0), SRV(t0), UAV(u0)"

cbuffer DetileParams : register(b0) {
    uint width;
    uint height;
    uint depth;
    uint element_bytes;
    uint row_pitch;
    uint depth_pitch;
    uint dst_offset;
    uint slice_size;
    uint mip_offset;
    uint first_slice;
    uint mip_pitch;
    uint block_w;
    uint block_h;
    uint block_d;
    uint block_log2;
    uint tail_x;
    uint tail_y;
    uint flags;
    uint x_shifts;
    uint y_shifts;
    uint z_shifts;
    uint nbits_packed;
};

ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

uint scatter_bits(uint coord, uint shifts, uint nbits) {
    uint acc = 0u;
    [unroll]
    for (uint i = 0u; i < 8u; ++i) {
        if (i < nbits) {
            uint pos = (shifts >> (i * 4u)) & 0xFu;
            acc |= ((coord >> i) & 1u) << pos;
        }
    }
    return acc;
}

uint tiled_col_addr(uint x, uint log2_bw, uint bw_mask, uint x_nbits) {
    uint tx = x + tail_x;
    uint bx = tx >> log2_bw;
    uint lx = tx & bw_mask;
    return (bx << block_log2) + scatter_bits(lx, x_shifts, x_nbits);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void detile(uint3 tid : SV_DispatchThreadID) {
    uint y = tid.y;
    uint z = tid.z;
    if (y >= height || z >= depth) {
        return;
    }

    uint elems_per_thread = (element_bytes == 1u) ? 4u : ((element_bytes == 2u) ? 2u : 1u);
    uint x0 = tid.x * elems_per_thread;
    if (x0 >= width) {
        return;
    }

    uint dst_addr = dst_offset + z * depth_pitch + y * row_pitch + x0 * element_bytes;
    if ((flags & 2u) != 0u) {
        if (element_bytes <= 4u) {
            dst.Store(dst_addr, 0u);
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, uint2(0u, 0u));
        } else {
            dst.Store4(dst_addr, uint4(0u, 0u, 0u, 0u));
        }
        return;
    }

    uint s = first_slice + z;
    uint inner_z = 0u;
    if ((flags & 1u) != 0u) {
        uint log2_bd = firstbithigh(block_d);
        inner_z = s & (block_d - 1u);
        s = s >> log2_bd;
    }
    uint base = s * slice_size + mip_offset;

    if (block_log2 == 0u) {
        uint row_base = base + y * mip_pitch * element_bytes;
        uint src_addr = row_base + x0 * element_bytes;
        if (element_bytes == 1u) {
            uint w = src.Load(src_addr);
            uint rem = width - x0;
            if (rem < 4u) {
                w &= (1u << (rem * 8u)) - 1u;
            }
            dst.Store(dst_addr, w);
        } else if (element_bytes == 2u) {
            uint w = src.Load(src_addr);
            if (x0 + 1u >= width) {
                w &= 0xFFFFu;
            }
            dst.Store(dst_addr, w);
        } else if (element_bytes == 4u) {
            dst.Store(dst_addr, src.Load(src_addr));
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, src.Load2(src_addr));
        } else {
            dst.Store4(dst_addr, src.Load4(src_addr));
        }
        return;
    }

    uint x_nbits = nbits_packed & 0xFFu;
    uint y_nbits = (nbits_packed >> 8u) & 0xFFu;
    uint z_nbits = (nbits_packed >> 16u) & 0xFFu;

    uint log2_bw = firstbithigh(block_w);
    uint log2_bh = firstbithigh(block_h);
    uint bw_mask = block_w - 1u;

    uint ty = y + tail_y;
    uint by = ty >> log2_bh;
    uint ly = ty & (block_h - 1u);
    uint blocks_x = mip_pitch >> log2_bw;
    uint row_base = base
        + ((by * blocks_x) << block_log2)
        + (scatter_bits(ly, y_shifts, y_nbits) | scatter_bits(inner_z, z_shifts, z_nbits));

    if (element_bytes == 1u) {
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 4u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint sa = row_base + tiled_col_addr(x, log2_bw, bw_mask, x_nbits);
                uint w = src.Load(sa & ~3u);
                uint b = (w >> ((sa & 3u) * 8u)) & 0xFFu;
                packed |= b << (k * 8u);
            }
        }
        dst.Store(dst_addr, packed);
    } else if (element_bytes == 2u) {
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 2u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint sa = row_base + tiled_col_addr(x, log2_bw, bw_mask, x_nbits);
                uint w = src.Load(sa & ~3u);
                uint h = (w >> ((sa & 2u) * 8u)) & 0xFFFFu;
                packed |= h << (k * 16u);
            }
        }
        dst.Store(dst_addr, packed);
    } else {
        uint sa = row_base + tiled_col_addr(x0, log2_bw, bw_mask, x_nbits);
        if (element_bytes == 4u) {
            dst.Store(dst_addr, src.Load(sa));
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, src.Load2(sa));
        } else {
            dst.Store4(dst_addr, src.Load4(sa));
        }
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Dummy read-only buffer bound at `t0` when all jobs in a batch are `zero = true`
    /// and the caller passes `src_gpu_va == 0`.
    dummy_src: ID3D12Resource,
}

// SAFETY: D3D12 root signatures, pipelines and resources are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make_pass(host: &Host) -> windows::core::Result<Pass> {
    let code = crate::host::compile(SOURCE, s!("detile"), s!("cs_5_1"))?;
    let (dummy_src, _) = host
        .upload_buffer(256)
        .ok_or_else(windows::core::Error::empty)?;
    // SAFETY: root signature is embedded in the compiled shader blob; plain creation.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        let pso = host.device.CreateComputePipelineState(&desc)?;
        crate::host::name(&root, || "gpu detile root".into());
        crate::host::name(&pso, || "gpu detile pso".into());
        crate::host::name(&dummy_src, || "gpu detile dummy src".into());
        Ok(Pass {
            root,
            pso,
            dummy_src,
        })
    }
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make_pass(host)
            .inspect_err(|e| tracing::warn!(error = %e, "gpu detile compute pipeline unavailable"))
            .ok()
    })
    .as_ref()
}

/// Records compute detile dispatches for `jobs` onto `list`.
///
/// The caller owns resource state barriers on `dst_uav` (`COMMON -> UNORDERED_ACCESS`
/// before and `UNORDERED_ACCESS -> COPY_SOURCE` after). Returns `false` without
/// recording any commands if validation fails or any byte offset exceeds `u32::MAX`.
pub(crate) fn record_detile_subresources(
    list: &ID3D12GraphicsCommandList,
    layout: &Layout,
    src_gpu_va: u64,
    src_len: u64,
    dst_uav: &ID3D12Resource,
    dst_len: u64,
    jobs: &[SubresourceJob],
) -> bool {
    if !supported(layout) {
        return false;
    }
    if jobs.is_empty() {
        return true;
    }
    let Some(block_log2) = layout.desc.swizzle.block_log2() else {
        return false;
    };
    if block_log2 != 0 && !layout.block.iter().all(|&b| b.is_power_of_two()) {
        return false;
    }
    let Some(pass) = pass() else {
        return false;
    };
    // SAFETY: live D3D12 buffer resource.
    let (dst_uav_va, dst_uav_width) = unsafe { (dst_uav.GetGPUVirtualAddress(), dst_uav.GetDesc().Width) };
    if dst_uav_va == 0 || !dst_uav_va.is_multiple_of(4) {
        return false;
    }
    let needs_src = jobs.iter().any(|j| !j.zero);
    if needs_src && (src_gpu_va == 0 || !src_gpu_va.is_multiple_of(4)) {
        return false;
    }

    let (shifts, nbits) = swizzle_encoding(layout);
    let nbits_packed = nbits[0] | (nbits[1] << 8) | (nbits[2] << 16);
    let bpe = layout.desc.element_bytes;
    let elems_per_thread = if bpe < 4 { 4 / bpe } else { 1 };
    let base_flags = if layout.thick { FLAG_THICK } else { 0 };

    let mut dispatches = Vec::with_capacity(jobs.len());
    for job in jobs {
        let Some(m) = layout.mips.get(job.mip as usize) else {
            return false;
        };
        let first_slice = if layout.desc.dimension == Dimension::Tex3d {
            if job.slice != 0 {
                return false;
            }
            0
        } else {
            job.slice
        };
        if first_slice >= layout.slices {
            return false;
        }
        let row_bytes = u64::from(m.width) * u64::from(bpe);
        let row_bytes_aligned = row_bytes.next_multiple_of(4);
        if !job.dst_offset.is_multiple_of(4) || u64::from(job.row_pitch) < row_bytes {
            return false;
        }
        if m.height > 1
            && (!job.row_pitch.is_multiple_of(4) || u64::from(job.row_pitch) < row_bytes_aligned)
        {
            return false;
        }
        let Some(slice_exact) = (u64::from(m.height) - 1)
            .checked_mul(u64::from(job.row_pitch))
            .and_then(|o| o.checked_add(row_bytes))
        else {
            return false;
        };
        let Some(slice_aligned) = (u64::from(m.height) - 1)
            .checked_mul(u64::from(job.row_pitch))
            .and_then(|o| o.checked_add(row_bytes_aligned))
        else {
            return false;
        };
        if m.depth > 1
            && (!job.depth_pitch.is_multiple_of(4) || u64::from(job.depth_pitch) < slice_aligned)
        {
            return false;
        }
        let Some(depth_span) = (u64::from(m.depth) - 1).checked_mul(u64::from(job.depth_pitch))
        else {
            return false;
        };
        let Some(exact_end) = job
            .dst_offset
            .checked_add(depth_span)
            .and_then(|o| o.checked_add(slice_exact))
        else {
            return false;
        };
        let Some(aligned_end) = job
            .dst_offset
            .checked_add(depth_span)
            .and_then(|o| o.checked_add(slice_aligned))
        else {
            return false;
        };
        if exact_end > dst_len || aligned_end > dst_uav_width || u32::try_from(aligned_end).is_err() {
            return false;
        }
        let Ok(dst_offset) = u32::try_from(job.dst_offset) else {
            return false;
        };

        let (slice_size, mip_offset) = if job.zero {
            (0, 0)
        } else {
            let tiled_need = layout.bytes_from(job.mip);
            if src_len < tiled_need || u32::try_from(tiled_need).is_err() {
                return false;
            }
            let max_z = first_slice + m.depth - 1;
            let max_s = if layout.thick {
                max_z / layout.block[2]
            } else {
                max_z
            };
            let slice_size = if max_s > 0 {
                let Ok(v) = u32::try_from(layout.slice_size) else {
                    return false;
                };
                v
            } else {
                0
            };
            let Ok(mip_offset) = u32::try_from(m.offset) else {
                return false;
            };
            (slice_size, mip_offset)
        };

        let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
        let params = DetileParams {
            width: m.width,
            height: m.height,
            depth: m.depth,
            element_bytes: bpe,
            row_pitch: job.row_pitch,
            depth_pitch: job.depth_pitch,
            dst_offset,
            slice_size,
            mip_offset,
            first_slice,
            mip_pitch: m.pitch,
            block_w: layout.block[0],
            block_h: layout.block[1],
            block_d: layout.block[2],
            block_log2,
            tail_x,
            tail_y,
            flags: base_flags | if job.zero { FLAG_ZERO } else { 0 },
            x_shifts: shifts[0],
            y_shifts: shifts[1],
            z_shifts: shifts[2],
            nbits_packed,
        };
        let gx = m.width.div_ceil(elems_per_thread).div_ceil(8);
        let gy = m.height.div_ceil(8);
        let gz = m.depth;
        dispatches.push((params, gx, gy, gz));
    }

    let bound_src_va = if src_gpu_va != 0 {
        src_gpu_va
    } else {
        // SAFETY: live static upload buffer in `Pass`.
        unsafe { pass.dummy_src.GetGPUVirtualAddress() }
    };

    // SAFETY: open direct command list; root signature, PSO and buffer VAs are valid and
    // aligned; all bounds and 32-bit offsets have been validated above.
    unsafe {
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRootShaderResourceView(1, bound_src_va);
        list.SetComputeRootUnorderedAccessView(2, dst_uav_va);
        for &(params, gx, gy, gz) in &dispatches {
            list.SetComputeRoot32BitConstants(
                0,
                DETILE_PARAM_WORDS,
                (&raw const params).cast(),
                0,
            );
            list.Dispatch(gx, gy, gz);
        }
    }
    true
}

/// Single reusable device-local UAV scratch buffer (>= 64 KiB, growing on demand
/// and retiring superseded allocations; capped at 32 MiB retention so outlier
/// textures do not permanently pin VRAM). Safe to share across prelude uploads on
/// the single direct queue.
pub(crate) fn shared_scratch(host: &Host, size: u64) -> Option<ID3D12Resource> {
    struct Scratch(Option<(u64, ID3D12Resource)>);
    // SAFETY: D3D12 resources are free-threaded.
    unsafe impl Send for Scratch {}
    static SCRATCH: Mutex<Scratch> = Mutex::new(Scratch(None));
    const MAX_RETAINED: u64 = 32 << 20;

    let min_size = size.max(1 << 16);
    let class = min_size.checked_next_power_of_two().unwrap_or(min_size);
    let mut slot = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cap, buffer)) = slot.0.as_ref()
        && *cap >= class
    {
        return Some(buffer.clone());
    }
    let buffer = host.uav_buffer(class)?;
    crate::host::name(&buffer, || format!("gpu detile scratch {class:#x}"));
    if class <= MAX_RETAINED
        && let Some((_, old)) = slot.0.replace((class, buffer.clone()))
    {
        crate::host::retire(old);
    }
    Some(buffer)
}

fn buffer_barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed COM pointer; never released by the barrier struct.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: open command list and live resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Standalone synchronous GPU detile helper for testing and benchmarking.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn detile_to_vec(
    layout: &Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    row_pitch: usize,
    depth_pitch: usize,
    dst_len: usize,
) -> Option<Vec<u8>> {
    let host = host()?;
    let row_pitch = u32::try_from(row_pitch).ok()?;
    let depth_pitch = u32::try_from(depth_pitch).ok()?;

    let src_alloc = (tiled.len() as u64).next_multiple_of(4).max(4);
    let (src_buf, src_ptr) = host.upload_buffer(src_alloc)?;
    // SAFETY: `src_ptr` is mapped for `src_alloc >= tiled.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len());
        if (src_alloc as usize) > tiled.len() {
            std::ptr::write_bytes(src_ptr.add(tiled.len()), 0, src_alloc as usize - tiled.len());
        }
    }

    let uav_size = (dst_len as u64).next_multiple_of(4).max(4);
    let dst_uav = host.uav_buffer(uav_size)?;
    let readback = host.read_back_buffer(uav_size)?;
    let list = host.new_list().ok()?;

    // SAFETY: live upload buffer.
    let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
    buffer_barrier(
        &list.list,
        &dst_uav,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let job = SubresourceJob {
        mip,
        slice,
        dst_offset: 0,
        row_pitch,
        depth_pitch,
        zero: false,
    };
    if !record_detile_subresources(
        &list.list,
        layout,
        src_va,
        tiled.len() as u64,
        &dst_uav,
        uav_size,
        &[job],
    ) {
        return None;
    }
    buffer_barrier(
        &list.list,
        &dst_uav,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: open list; `dst_uav` is in `COPY_SOURCE` and `readback` is in `COPY_DEST`.
    unsafe {
        list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, uav_size);
    }
    list.close();
    host.execute_sync_internal(&list);

    let mut data = std::ptr::null_mut();
    let read_range = D3D12_RANGE {
        Begin: 0,
        End: dst_len,
    };
    let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: GPU execution has completed; `readback` is >= `dst_len` bytes.
    let out = unsafe {
        readback.Map(0, Some(&read_range), Some(&mut data)).ok()?;
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), dst_len).to_vec();
        readback.Unmap(0, Some(&empty_write));
        bytes
    };
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::Desc;
    use windows::core::Interface;

    fn fill_pattern(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0u8; len];
        for chunk in out.chunks_mut(8) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        out
    }

    fn max_mips(w: u32, h: u32, d: u32) -> u32 {
        32 - w.max(h).max(d).leading_zeros()
    }

    /// Runs a batched GPU detile of all specified `(mip, slice)` subresources into a single
    /// UAV scratch buffer and compares every subresource bit-for-bit against `Layout::detile`.
    fn verify_layout_subresources(layout: &Layout, seed: u64, subresources: &[(u32, u32)]) {
        let Some(host) = host() else {
            return;
        };
        let tiled = fill_pattern(layout.size as usize, seed);
        let bpe = layout.desc.element_bytes as usize;

        let mut jobs = Vec::with_capacity(subresources.len());
        let mut ranges = Vec::with_capacity(subresources.len());
        let mut cursor = 0u64;
        for &(mip, slice) in subresources {
            let m = &layout.mips[mip as usize];
            let row_pitch = (m.width as usize * bpe).next_multiple_of(256);
            let depth_pitch = row_pitch * m.height as usize;
            let bytes = if m.depth > 1 {
                depth_pitch * m.depth as usize
            } else {
                row_pitch * m.height as usize
            };
            cursor = cursor.next_multiple_of(512);
            jobs.push(SubresourceJob {
                mip,
                slice,
                dst_offset: cursor,
                row_pitch: row_pitch as u32,
                depth_pitch: depth_pitch as u32,
                zero: false,
            });
            ranges.push((cursor as usize, bytes, row_pitch, depth_pitch));
            cursor += bytes as u64;
        }
        let total = cursor.next_multiple_of(4).max(4);
        let (src_buf, src_ptr) = host.upload_buffer(layout.size.next_multiple_of(4).max(4)).unwrap();
        // SAFETY: mapped upload buffer covers `tiled.len()`.
        unsafe {
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len());
        }
        let dst_uav = shared_scratch(host, total).unwrap();
        let readback = host.read_back_buffer(total).unwrap();
        let list = host.new_list().unwrap();

        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        assert!(
            record_detile_subresources(
                &list.list,
                layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                total,
                &jobs,
            ),
            "record_detile_subresources failed for {:?}",
            layout.desc
        );
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: open list; `dst_uav` in `COPY_SOURCE`, `readback` in `COPY_DEST`.
        unsafe {
            list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, total);
        }
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        host.execute_sync_internal(&list);

        let mut data = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: total as usize,
        };
        let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: GPU execution finished.
        let gpu_all = unsafe {
            readback.Map(0, Some(&read_range), Some(&mut data)).unwrap();
            let v = std::slice::from_raw_parts(data.cast::<u8>(), total as usize).to_vec();
            readback.Unmap(0, Some(&empty_write));
            v
        };

        for (idx, &(mip, slice)) in subresources.iter().enumerate() {
            let (off, len, row_pitch, depth_pitch) = ranges[idx];
            let m = &layout.mips[mip as usize];
            let mut cpu = vec![0u8; len];
            layout
                .detile(&tiled, mip, slice, &mut cpu, row_pitch, depth_pitch)
                .unwrap();
            let gpu = &gpu_all[off..off + len];
            let row_bytes = m.width as usize * bpe;
            for z in 0..m.depth as usize {
                for y in 0..m.height as usize {
                    let row_start = z * depth_pitch + y * row_pitch;
                    assert_eq!(
                        &gpu[row_start..row_start + row_bytes],
                        &cpu[row_start..row_start + row_bytes],
                        "mismatch for desc={:?} mip={mip} slice={slice} z={z} y={y}",
                        layout.desc
                    );
                    // Verify trailing bytes within the final 4-byte dword are zeroed for bpe < 4.
                    let pad_end = row_bytes.next_multiple_of(4);
                    if pad_end > row_bytes {
                        assert!(
                            gpu[row_start + row_bytes..row_start + pad_end]
                                .iter()
                                .all(|&b| b == 0),
                            "non-zero trailing dword padding for desc={:?} mip={mip} slice={slice} z={z} y={y}",
                            layout.desc
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn swizzle_encoding_matches_block_dimensions() {
        let swizzles = [
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let (shifts, nbits) = swizzle_encoding(&layout);
                assert_eq!(1u32 << nbits[0], layout.block[0]);
                assert_eq!(1u32 << nbits[1], layout.block[1]);
                assert_eq!(1u32 << nbits[2], layout.block[2]);
                // Every scatter bit position must lie below block_log2 and be unique.
                let block_log2 = swizzle.block_log2().unwrap();
                let mut used = 0u32;
                for a in 0..3 {
                    for i in 0..nbits[a] {
                        let pos = (shifts[a] >> (i * 4)) & 0xF;
                        assert!(pos < block_log2);
                        assert_eq!(used & (1 << pos), 0);
                        used |= 1 << pos;
                    }
                }
            }
        }
        // Thick 3D modes
        for swizzle in [Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 64,
                    height: 64,
                    depth: 64,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(layout.thick);
                let (_, nbits) = swizzle_encoding(&layout);
                assert_eq!(1u32 << nbits[0], layout.block[0]);
                assert_eq!(1u32 << nbits[1], layout.block[1]);
                assert_eq!(1u32 << nbits[2], layout.block[2]);
            }
        }
    }

    #[test]
    fn gpu_detile_all_swizzles_and_element_sizes() {
        let Some(_) = host() else {
            return;
        };
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 1u64;
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 137,
                    height: 83,
                    depth: 1,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(supported(&layout));
                let tiled = fill_pattern(layout.size as usize, seed);
                seed += 1;
                let row_pitch = (layout.mips[0].width as usize * bpe as usize).next_multiple_of(4);
                let dst_len = row_pitch * layout.mips[0].height as usize;
                let mut cpu = vec![0u8; dst_len];
                layout
                    .detile(&tiled, 0, 0, &mut cpu, row_pitch, dst_len)
                    .unwrap();
                let gpu = detile_to_vec(&layout, &tiled, 0, 0, row_pitch, dst_len, dst_len)
                    .expect("detile_to_vec failed");
                assert_eq!(
                    gpu, cpu,
                    "bit-for-bit mismatch for swizzle={swizzle:?} bpe={bpe}"
                );
            }
            // Block-compressed formats (BC1/BC4 = 8B, BC3/BC5/BC6H/BC7 = 16B).
            for bpe in [8u32, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 260,
                    height: 140,
                    depth: 1,
                    mips: 4,
                    element_bytes: bpe,
                    block: 4,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_mip_chains_tails_and_npot() {
        let Some(_) = host() else {
            return;
        };
        let dims = [(73u32, 39u32), (109, 71), (255, 129), (5, 3), (1, 1)];
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 1000u64;
        for (w, h) in dims {
            let mips = max_mips(w, h, 1);
            for swizzle in swizzles {
                for bpe in [1u32, 2, 4, 8, 16] {
                    let layout = Layout::new(&Desc {
                        dimension: Dimension::Tex2d,
                        width: w,
                        height: h,
                        depth: 1,
                        mips,
                        element_bytes: bpe,
                        block: 1,
                        samples: 1,
                        swizzle,
                    })
                    .unwrap();
                    if matches!(
                        swizzle,
                        Swizzle::SW_4KB_S
                            | Swizzle::SW_4KB_D
                            | Swizzle::SW_64KB_S
                            | Swizzle::SW_64KB_D
                    ) && mips > 1
                    {
                        assert!(
                            layout.first_tail_mip < layout.desc.mips,
                            "expected mip tail for {w}x{h} {swizzle:?} bpe={bpe}"
                        );
                    }
                    let subs: Vec<(u32, u32)> = (0..mips).map(|m| (m, 0)).collect();
                    verify_layout_subresources(&layout, seed, &subs);
                    seed += 1;
                }
            }
        }

        // Also test 1D textures with odd widths and small bpe (partial final dwords).
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_256B_S, Swizzle::SW_4KB_S] {
            for bpe in [1u32, 2, 4, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex1d,
                    width: 131,
                    height: 1,
                    depth: 3,
                    mips: 6,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let mut subs = Vec::new();
                for s in 0..3 {
                    for m in 0..6 {
                        subs.push((m, s));
                    }
                }
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_2d_arrays() {
        let Some(_) = host() else {
            return;
        };
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 5000u64;
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let slices = 5u32;
                let mips = 7u32;
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 97,
                    height: 67,
                    depth: slices,
                    mips,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let mut subs = Vec::new();
                for s in 0..slices {
                    for m in 0..mips {
                        subs.push((m, s));
                    }
                }
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_3d_thin_and_thick_volumes() {
        let Some(_) = host() else {
            return;
        };
        let mut seed = 8000u64;

        // Thin linear 3D.
        for bpe in [1u32, 2, 4, 8, 16] {
            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex3d,
                width: 37,
                height: 23,
                depth: 11,
                mips: 6,
                element_bytes: bpe,
                block: 1,
                samples: 1,
                swizzle: Swizzle::LINEAR,
            })
            .unwrap();
            assert!(!layout.thick);
            let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
            verify_layout_subresources(&layout, seed, &subs);
            seed += 1;
        }

        // Thick 4KB_S and 64KB_S 3D volumes, including deep volumes with multiple depth slabs
        // in both standard mips and tail mips.
        for swizzle in [Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 41,
                    height: 29,
                    depth: 35,
                    mips: 6,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(layout.thick);
                assert!(layout.slices > 1, "expected multiple depth slabs");
                assert!(layout.first_tail_mip < layout.desc.mips, "expected 3D mip tail");
                let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;

                // Deep narrow volume where mip 0 itself starts in the tail and spans multiple slabs.
                let tail_deep = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 3,
                    height: 3,
                    depth: 37,
                    mips: 4,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(tail_deep.thick);
                assert_eq!(tail_deep.first_tail_mip, 0);
                assert!(tail_deep.slices > 1);
                let subs: Vec<(u32, u32)> = (0..tail_deep.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&tail_deep, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    #[ignore = "needs a D3D12 adapter; run with --ignored on a machine with a GPU"]
    fn gpu_detile_batched_prelude_layout_and_zero_subresources() {
        let Some(host) = host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 193,
            height: 115,
            depth: 2,
            mips: 8,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        assert!(layout.first_tail_mip < layout.desc.mips);

        // Verify shared_scratch reuses buffers within the same power-of-two class.
        let s1 = shared_scratch(host, 4096).unwrap();
        let s2 = shared_scratch(host, 65536).unwrap();
        let s3 = shared_scratch(host, 131072).unwrap();
        assert_eq!(s1.as_raw(), s2.as_raw());
        assert_ne!(s1.as_raw(), s3.as_raw());

        let tiled = fill_pattern(layout.size as usize, 0xCAFE_BABE);
        let bpe = layout.desc.element_bytes as usize;
        let mut jobs = Vec::new();
        let mut meta = Vec::new();
        let mut cursor = 0u64;
        for slice in 0..2u32 {
            for mip in 0..layout.desc.mips {
                let m = &layout.mips[mip as usize];
                let row_pitch = (m.width as usize * bpe).next_multiple_of(256);
                let bytes = row_pitch * m.height as usize;
                cursor = cursor.next_multiple_of(512);
                // Mark mip 0 of slice 0 and mip 6 (tail) of slice 1 as zero-filled (unheld on_demand).
                let zero = (slice == 0 && mip == 0) || (slice == 1 && mip == 6);
                jobs.push(SubresourceJob {
                    mip,
                    slice,
                    dst_offset: cursor,
                    row_pitch: row_pitch as u32,
                    depth_pitch: bytes as u32,
                    zero,
                });
                meta.push((cursor as usize, bytes, row_pitch, mip, slice, zero));
                cursor += bytes as u64;
            }
        }
        let total = cursor;

        // Pre-fill dst_uav with 0xA5 so zero-fill subresources are proven to actively write 0x00.
        let (poison_buf, poison_ptr) = host.upload_buffer(total).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::write_bytes(poison_ptr, 0xA5, total as usize) };
        let (src_buf, src_ptr) = host.upload_buffer(layout.size).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };

        let dst_uav = shared_scratch(host, total).unwrap();
        let readback = host.read_back_buffer(total).unwrap();
        let list = host.new_list().unwrap();

        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: open list; poison_buf is upload (`GENERIC_READ`), dst_uav is `COPY_DEST`.
        unsafe { list.list.CopyBufferRegion(&dst_uav, 0, &poison_buf, 0, total) };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
        assert!(record_detile_subresources(
            &list.list,
            &layout,
            src_va,
            tiled.len() as u64,
            &dst_uav,
            total,
            &jobs,
        ));
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: open list.
        unsafe { list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, total) };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        host.execute_sync_internal(&list);

        let mut data = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: total as usize,
        };
        let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: GPU execution finished.
        let gpu_all = unsafe {
            readback.Map(0, Some(&read_range), Some(&mut data)).unwrap();
            let v = std::slice::from_raw_parts(data.cast::<u8>(), total as usize).to_vec();
            readback.Unmap(0, Some(&empty_write));
            v
        };

        for (off, len, row_pitch, mip, slice, zero) in meta {
            let m = &layout.mips[mip as usize];
            let row_bytes = m.width as usize * bpe;
            let gpu = &gpu_all[off..off + len];
            if zero {
                for y in 0..m.height as usize {
                    let row = &gpu[y * row_pitch..y * row_pitch + row_bytes];
                    assert!(
                        row.iter().all(|&b| b == 0),
                        "expected zeroed row for mip={mip} slice={slice} y={y}"
                    );
                }
            } else {
                let mut cpu = vec![0u8; len];
                layout
                    .detile(&tiled, mip, slice, &mut cpu, row_pitch, len)
                    .unwrap();
                for y in 0..m.height as usize {
                    let s = y * row_pitch;
                    assert_eq!(
                        &gpu[s..s + row_bytes],
                        &cpu[s..s + row_bytes],
                        "mismatch for mip={mip} slice={slice} y={y}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_detile_validation_rejects_invalid_and_oversized() {
        let xor_layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        })
        .unwrap();
        assert!(!supported(&xor_layout));

        let Some(host) = host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let dst_uav = shared_scratch(host, 65536).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: live resource.
        let va = unsafe { dst_uav.GetGPUVirtualAddress() };

        // >4 GiB dst_offset rejected cleanly.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size,
            &dst_uav,
            u64::MAX,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: (u32::MAX as u64) + 1,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // Unaligned dst_offset rejected.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 2,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // Too-small src_len rejected.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size - 1,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 0,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // All-zero batch with src_gpu_va == 0 and src_len == 0 succeeds cleanly.
        assert!(record_detile_subresources(
            &list.list,
            &layout,
            0,
            0,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 0,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: true,
            }],
        ));
    }

    #[test]
    #[ignore = "benchmark"]
    fn gpu_detile_bench() {
        let Some(host) = host() else {
            eprintln!("no host D3D12 device available");
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 2048,
            height: 2048,
            depth: 1,
            mips: 12,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let tiled = fill_pattern(layout.size as usize, 0x1234_5678);

        let mut jobs = Vec::new();
        let mut cursor = 0u64;
        for mip in 0..layout.desc.mips {
            let m = &layout.mips[mip as usize];
            let row_pitch = (m.width as usize * 16).next_multiple_of(256);
            let bytes = row_pitch * m.height as usize;
            cursor = cursor.next_multiple_of(512);
            jobs.push(SubresourceJob {
                mip,
                slice: 0,
                dst_offset: cursor,
                row_pitch: row_pitch as u32,
                depth_pitch: bytes as u32,
                zero: false,
            });
            cursor += bytes as u64;
        }
        let total_linear = cursor as usize;

        // Warmup CPU
        let mut cpu_buf = vec![0u8; total_linear];
        for job in &jobs {
            let off = job.dst_offset as usize;
            let len = job.depth_pitch as usize;
            layout
                .detile(
                    &tiled,
                    job.mip,
                    0,
                    &mut cpu_buf[off..off + len],
                    job.row_pitch as usize,
                    len,
                )
                .unwrap();
        }

        const ITERS: u32 = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            for job in &jobs {
                let off = job.dst_offset as usize;
                let len = job.depth_pitch as usize;
                layout
                    .detile(
                        &tiled,
                        job.mip,
                        0,
                        &mut cpu_buf[off..off + len],
                        job.row_pitch as usize,
                        len,
                    )
                    .unwrap();
            }
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        // Warmup GPU (pre-uploaded source + shared UAV scratch, matching prelude path).
        let (src_buf, src_ptr) = host.upload_buffer(layout.size).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };
        let dst_uav = shared_scratch(host, cursor).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };

        let run_gpu_once = || {
            buffer_barrier(
                &list.list,
                &dst_uav,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            assert!(record_detile_subresources(
                &list.list,
                &layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                cursor,
                &jobs,
            ));
            buffer_barrier(
                &list.list,
                &dst_uav,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            list.close();
            host.execute_sync_internal(&list);
            list.reset();
        };
        for _ in 0..5 {
            run_gpu_once();
        }

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_gpu_once();
        }
        let gpu_sync_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        // Batched GPU (all ITERS recorded into one command list, matching prelude batching).
        let t2 = std::time::Instant::now();
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        for _ in 0..ITERS {
            assert!(record_detile_subresources(
                &list.list,
                &layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                cursor,
                &jobs,
            ));
        }
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        let record_us = t2.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
        host.execute_sync_internal(&list);
        list.reset();
        let gpu_batch_ms = t2.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        let mb = total_linear as f64 / (1024.0 * 1024.0);
        println!(
            "2048x2048 BC7 full 12-mip chain ({mb:.2} MiB linear, {ITERS} iters):\n  \
             CPU detile           = {cpu_ms:.3} ms ({:.2} GiB/s)\n  \
             GPU (1 submit/tex)   = {gpu_sync_ms:.3} ms ({:.2} GiB/s)\n  \
             GPU (batched submit) = {gpu_batch_ms:.3} ms ({:.2} GiB/s, CPU record = {record_us:.1} us/tex, speedup = {:.2}x)",
            (mb / 1024.0) / (cpu_ms / 1e3),
            (mb / 1024.0) / (gpu_sync_ms / 1e3),
            (mb / 1024.0) / (gpu_batch_ms / 1e3),
            cpu_ms / gpu_batch_ms,
        );
    }
}
