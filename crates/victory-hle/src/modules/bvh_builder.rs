//! `XboxBvhBuilder.dll`: HLE hook for console ray-tracing acceleration
//! structure building, copying, prebuild info, and static BVH4 rehydration.
//!
//! When `VICTORY_NATIVE_RT` is unset/0, all 7 C++ exports transparently forward
//! to the title's real `XboxBvhBuilder.dll` (loaded via `victory-loader` so
//! console imports bind cleanly even though HLE claims the DLL name).
//!
//! When `VICTORY_NATIVE_RT=1`, builds/copies/rehydrations are dual-fed into
//! `victory_gpu::dxr` while keeping the real builder running for fallback
//! software-RT shaders unless `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1` is set.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use victory_gpu::dxr::{
    self, GuestBatchedDesc2, GuestBuildAsDesc, GuestBuildAsInputs, GuestPrebuildInfo, GuestStreamDesc,
};
use victory_loader::{LoadedImage, UnimplementedPolicy};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress, LoadLibraryA};

use crate::{HleExport, HleModule, HleResolver};

const SYM_CTOR: &str = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
const SYM_DTOR: &str = "??1XboxBvhBuilder@@QEAA@XZ";
const SYM_BUILD_BATCHED_2X: &str = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
const SYM_COPY: &str = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";
const SYM_PREBUILD: &str = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
const SYM_REHYDRATE_BATCHED_2X: &str = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";
const SYM_SET_64BIT_MORTON: &str = "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z";

type FnCtor = unsafe extern "system" fn(*mut c_void, *mut c_void, u8) -> *mut c_void;
type FnDtor = unsafe extern "system" fn(*mut c_void);
type FnBuildBatched2X = unsafe extern "system" fn(*mut c_void, *const c_void, u32, *const GuestBatchedDesc2);
type FnCopy = unsafe extern "system" fn(*mut c_void, *const c_void, u64, u64, u32);
type FnPrebuild =
    unsafe extern "system" fn(*mut c_void, *const GuestBuildAsInputs, *const GuestStreamDesc, *mut GuestPrebuildInfo);
type FnRehydrateBatched2X =
    unsafe extern "system" fn(*mut c_void, *const c_void, u32, *const GuestBuildAsDesc, *const u64) -> i32;
type FnSet64BitMorton = unsafe extern "system" fn(*mut c_void, u8);

struct RealExports {
    /// Kept alive for the process lifetime when loaded directly via `victory-loader`.
    _image: Option<LoadedImage>,
    ctor: Option<FnCtor>,
    dtor: Option<FnDtor>,
    build_batched_2x: Option<FnBuildBatched2X>,
    copy: Option<FnCopy>,
    prebuild: Option<FnPrebuild>,
    rehydrate_batched_2x: Option<FnRehydrateBatched2X>,
    set_64bit_morton: Option<FnSet64BitMorton>,
}

// SAFETY: function pointers into a permanently mapped module image.
unsafe impl Send for RealExports {}
unsafe impl Sync for RealExports {}

fn real_exports() -> Option<&'static RealExports> {
    static REAL: OnceLock<Option<RealExports>> = OnceLock::new();
    REAL.get_or_init(resolve_real_dll).as_ref()
}

fn proc_addr(module: HMODULE, sym: &str) -> Option<usize> {
    let name = CString::new(sym).ok()?;
    // SAFETY: valid module handle and NUL-terminated export name.
    unsafe { GetProcAddress(module, name.as_ptr().cast()) }.map(|f| f as *const () as usize)
}

/// Locates the title's real `XboxBvhBuilder.dll`, loading and binding it through
/// `victory-loader` if it was not already mapped (since HLE claiming the DLL name
/// causes `load_title_dlls` to skip it).
fn resolve_real_dll() -> Option<RealExports> {
    // 1. Check if already mapped in-process (e.g. already loaded or attached).
    let existing = crate::runtime::image_named("XboxBvhBuilder.dll")
        .map(|base| base as HMODULE)
        .filter(|m| !m.is_null())
        .or_else(|| {
            // SAFETY: static NUL-terminated DLL name.
            let h = unsafe { GetModuleHandleA(c"XboxBvhBuilder.dll".as_ptr().cast()) };
            (!h.is_null()).then_some(h)
        });
    if let Some(module) = existing {
        tracing::info!(base = format_args!("{:#x}", module as usize), "bvh_builder: using already-mapped XboxBvhBuilder.dll");
        return Some(exports_from_module(module, None));
    }

    // 2. Find `XboxBvhBuilder.dll` beside the title executable or in `VICTORY_GAME_DIR`.
    let dll_path = find_title_bvh_dll()?;

    // 3. Prepare, map, bind console imports, and attach via `victory-loader`.
    if let Some(image) = load_console_dll(&dll_path) {
        let module = image.base() as HMODULE;
        tracing::info!(
            base = format_args!("{:#x}", image.base()),
            path = %dll_path.display(),
            "bvh_builder: mapped real XboxBvhBuilder.dll via victory-loader"
        );
        return Some(exports_from_module(module, Some(image)));
    }

    // 4. Fallback: plain Windows `LoadLibraryA` for non-console/host-linked copies.
    if let Some(cpath) = dll_path.to_str().and_then(|s| CString::new(s).ok()) {
        // SAFETY: NUL-terminated path.
        let module = unsafe { LoadLibraryA(cpath.as_ptr().cast()) };
        if !module.is_null() {
            tracing::info!(
                base = format_args!("{:#x}", module as usize),
                path = %dll_path.display(),
                "bvh_builder: loaded XboxBvhBuilder.dll via LoadLibraryA fallback"
            );
            return Some(exports_from_module(module, None));
        }
    }

    tracing::warn!(path = %dll_path.display(), "bvh_builder: could not load real XboxBvhBuilder.dll");
    None
}

fn find_title_bvh_dll() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(exe_path) = crate::runtime::main_image().and_then(crate::runtime::image_path)
        && let Some(dir) = exe_path.parent()
    {
        candidates.push(dir.join("XboxBvhBuilder.dll"));
    }
    if let Some(game_dir) = std::env::var_os("VICTORY_GAME_DIR") {
        candidates.push(PathBuf::from(game_dir).join("XboxBvhBuilder.dll"));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("XboxBvhBuilder.dll"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn load_console_dll(path: &Path) -> Option<LoadedImage> {
    let original = std::fs::read(path)
        .inspect_err(|e| tracing::warn!(error = %e, path = %path.display(), "bvh_builder: failed reading DLL"))
        .ok()?;
    let prepared = victory_loader::prepare(&original)
        .inspect_err(|e| tracing::warn!(error = %e, path = %path.display(), "bvh_builder: failed preparing DLL"))
        .ok()?;
    let title = crate::runtime::title();
    let cache_dir = if !title.storage.as_os_str().is_empty() {
        title.storage.with_file_name("bvh-builder-cache")
    } else {
        std::env::temp_dir().join("victory-bvh-builder-cache")
    };
    let image = victory_loader::load(&prepared, &cache_dir)
        .inspect_err(|e| tracing::warn!(error = %e, path = %path.display(), "bvh_builder: failed mapping DLL"))
        .ok()?;
    let resolver = HleResolver::default();
    match image.bind_imports(&resolver, UnimplementedPolicy::ReturnZero) {
        Ok(report) => {
            if !report.missing.is_empty() {
                tracing::warn!(
                    missing = report.missing.len(),
                    "bvh_builder: real XboxBvhBuilder.dll had unresolved imports"
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "bvh_builder: failed binding real XboxBvhBuilder.dll imports");
            return None;
        }
    }
    crate::runtime::set_image_path(image.base(), path);
    // SAFETY: imports are bound above; attaches the title DLL once before first call.
    if let Err(e) = unsafe { image.attach() } {
        tracing::warn!(error = %e, "bvh_builder: real XboxBvhBuilder.dll attach failed");
        return None;
    }
    Some(image)
}

fn exports_from_module(module: HMODULE, image: Option<LoadedImage>) -> RealExports {
    // SAFETY: transmuting resolved PE export addresses matching the exact MSVC x64 C++ signatures.
    unsafe {
        RealExports {
            _image: image,
            ctor: proc_addr(module, SYM_CTOR).map(|a| std::mem::transmute::<usize, FnCtor>(a)),
            dtor: proc_addr(module, SYM_DTOR).map(|a| std::mem::transmute::<usize, FnDtor>(a)),
            build_batched_2x: proc_addr(module, SYM_BUILD_BATCHED_2X)
                .map(|a| std::mem::transmute::<usize, FnBuildBatched2X>(a)),
            copy: proc_addr(module, SYM_COPY).map(|a| std::mem::transmute::<usize, FnCopy>(a)),
            prebuild: proc_addr(module, SYM_PREBUILD).map(|a| std::mem::transmute::<usize, FnPrebuild>(a)),
            rehydrate_batched_2x: proc_addr(module, SYM_REHYDRATE_BATCHED_2X)
                .map(|a| std::mem::transmute::<usize, FnRehydrateBatched2X>(a)),
            set_64bit_morton: proc_addr(module, SYM_SET_64BIT_MORTON)
                .map(|a| std::mem::transmute::<usize, FnSet64BitMorton>(a)),
        }
    }
}

extern "system" fn ctor(this: *mut c_void, device: *mut c_void, flag: u8) -> *mut c_void {
    if !this.is_null()
        && let Some(real) = real_exports().and_then(|r| r.ctor)
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::XboxBvhBuilder`.
        return unsafe { real(this, device, flag) };
    }
    this
}

extern "system" fn dtor(this: *mut c_void) {
    if !this.is_null()
        && let Some(real) = real_exports().and_then(|r| r.dtor)
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::~XboxBvhBuilder`.
        unsafe { real(this) };
    }
}

fn dispatch_build_batched_2x(
    suppress: bool,
    real: Option<FnBuildBatched2X>,
    this: *mut c_void,
    cmd_list: *const c_void,
    num_descs: u32,
    descs: *const GuestBatchedDesc2,
) {
    if !suppress
        && !this.is_null()
        && let Some(real) = real
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X`.
        unsafe { real(this, cmd_list, num_descs, descs) };
    }
}

extern "system" fn build_batched_2x(
    this: *mut c_void,
    cmd_list: *const c_void,
    num_descs: u32,
    descs: *const GuestBatchedDesc2,
) {
    dispatch_build_batched_2x(
        dxr::suppress_builder(),
        real_exports().and_then(|r| r.build_batched_2x),
        this,
        cmd_list,
        num_descs,
        descs,
    );
    if dxr::enabled() {
        dxr::enqueue_build_batched(cmd_list, num_descs, descs);
    }
}

fn dispatch_copy_as(
    suppress: bool,
    real: Option<FnCopy>,
    this: *mut c_void,
    cmd_list: *const c_void,
    dest_va: u64,
    src_va: u64,
    mode: u32,
) {
    if !suppress
        && !this.is_null()
        && let Some(real) = real
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::CopyRaytracingAccelerationStructure`.
        unsafe { real(this, cmd_list, dest_va, src_va, mode) };
    }
}

extern "system" fn copy_as(
    this: *mut c_void,
    cmd_list: *const c_void,
    dest_va: u64,
    src_va: u64,
    mode: u32,
) {
    dispatch_copy_as(
        dxr::suppress_builder(),
        real_exports().and_then(|r| r.copy),
        this,
        cmd_list,
        dest_va,
        src_va,
        mode,
    );
    if dxr::enabled() {
        dxr::enqueue_copy(cmd_list, dest_va, src_va, mode);
    }
}

fn dispatch_prebuild_info<F>(
    suppress: bool,
    real: Option<FnPrebuild>,
    host_query: F,
    this: *mut c_void,
    inputs: *const GuestBuildAsInputs,
    stream_desc: *const GuestStreamDesc,
    out_info: *mut GuestPrebuildInfo,
) where
    F: Fn(&GuestBuildAsInputs, Option<&GuestStreamDesc>) -> Option<GuestPrebuildInfo>,
{
    if out_info.is_null() {
        return;
    }
    // SAFETY: caller passes optional non-null stream descriptor.
    let stream_opt = (!stream_desc.is_null()).then(|| unsafe { *stream_desc });
    if suppress && !inputs.is_null() {
        // SAFETY: caller passes valid input pointer.
        let guest_inputs = unsafe { *inputs };
        if let Some(info) = host_query(&guest_inputs, stream_opt.as_ref()) {
            // SAFETY: non-null output pointer.
            unsafe { out_info.write(info) };
            return;
        }
        // When `host_query` returns `None` (e.g. non-empty `stream_desc` for
        // packed static BVH4 rehydration), fall through to real `XboxBvhBuilder.dll`
        // so unpacked console BVH4 buffers receive exact rehydration sizes.
    }
    if !this.is_null()
        && let Some(real) = real
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::GetRaytracingAccelerationStructurePrebuildInfo`.
        unsafe { real(this, inputs, stream_desc, out_info) };
    } else if !inputs.is_null() {
        // SAFETY: caller passes valid input pointer.
        let guest_inputs = unsafe { *inputs };
        let info = if suppress {
            dxr::fallback_prebuild_info(&guest_inputs, stream_opt.as_ref())
        } else {
            host_query(&guest_inputs, stream_opt.as_ref())
                .unwrap_or_else(|| dxr::fallback_prebuild_info(&guest_inputs, stream_opt.as_ref()))
        };
        // SAFETY: non-null output pointer.
        unsafe { out_info.write(info) };
    } else {
        // SAFETY: non-null output pointer; zero-initialize on null inputs.
        unsafe { out_info.write(GuestPrebuildInfo::default()) };
    }
}

extern "system" fn get_prebuild_info(
    this: *mut c_void,
    inputs: *const GuestBuildAsInputs,
    stream_desc: *const GuestStreamDesc,
    out_info: *mut GuestPrebuildInfo,
) {
    dispatch_prebuild_info(
        dxr::suppress_builder(),
        real_exports().and_then(|r| r.prebuild),
        dxr::prebuild_info_with_stream,
        this,
        inputs,
        stream_desc,
        out_info,
    );
}

extern "system" fn rehydrate_batched_2x(
    this: *mut c_void,
    cmd_list: *const c_void,
    num_descs: u32,
    descs: *const GuestBuildAsDesc,
    sizes: *const u64,
) -> i32 {
    // Always forward to the real console rehydrator when present: `memory.rs`
    // uploads packed BVH4 pages to GPU arenas, while `XboxBvhBuilder.dll` runs
    // the GPU compute passes that unpack them in place before `dxr.rs` extracts
    // triangles.
    let hr = if !this.is_null()
        && let Some(real) = real_exports().and_then(|r| r.rehydrate_batched_2x)
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X`.
        unsafe { real(this, cmd_list, num_descs, descs, sizes) }
    } else {
        0
    };
    if dxr::enabled() {
        dxr::enqueue_rehydrate_batched(cmd_list, num_descs, descs, sizes);
    }
    hr
}

extern "system" fn set_64bit_morton_codes(this: *mut c_void, enable: u8) {
    if !this.is_null()
        && let Some(real) = real_exports().and_then(|r| r.set_64bit_morton)
    {
        // SAFETY: forwarded to real `XboxBvhBuilder::Set64BitMortonCodes`.
        unsafe { real(this, enable) };
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "bvh_builder",
        dlls: &["xboxbvhbuilder.dll"],
        exports: vec![
            HleExport {
                name: SYM_CTOR,
                address: ctor as *const () as usize,
            },
            HleExport {
                name: SYM_DTOR,
                address: dtor as *const () as usize,
            },
            HleExport {
                name: SYM_BUILD_BATCHED_2X,
                address: build_batched_2x as *const () as usize,
            },
            HleExport {
                name: SYM_COPY,
                address: copy_as as *const () as usize,
            },
            HleExport {
                name: SYM_PREBUILD,
                address: get_prebuild_info as *const () as usize,
            },
            HleExport {
                name: SYM_REHYDRATE_BATCHED_2X,
                address: rehydrate_batched_2x as *const () as usize,
            },
            HleExport {
                name: SYM_SET_64BIT_MORTON,
                address: set_64bit_morton_codes as *const () as usize,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static MOCK_BUILD_CALLS: AtomicUsize = AtomicUsize::new(0);
    static MOCK_COPY_CALLS: AtomicUsize = AtomicUsize::new(0);
    static MOCK_PREBUILD_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn mock_build(
        _this: *mut c_void,
        _cmd: *const c_void,
        _num: u32,
        _descs: *const GuestBatchedDesc2,
    ) {
        MOCK_BUILD_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn mock_copy(
        _this: *mut c_void,
        _cmd: *const c_void,
        _dst: u64,
        _src: u64,
        _mode: u32,
    ) {
        MOCK_COPY_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn mock_prebuild(
        _this: *mut c_void,
        _inputs: *const GuestBuildAsInputs,
        _stream: *const GuestStreamDesc,
        out_info: *mut GuestPrebuildInfo,
    ) {
        MOCK_PREBUILD_CALLS.fetch_add(1, Ordering::Relaxed);
        if !out_info.is_null() {
            // SAFETY: non-null test output pointer.
            unsafe {
                out_info.write(GuestPrebuildInfo {
                    result_data_max_size_in_bytes: 0x2000,
                    scratch_data_size_in_bytes: 0x1000,
                    update_scratch_data_size_in_bytes: 0x400,
                });
            }
        }
    }

    #[test]
    fn module_exports_all_seven_bvh_builder_symbols() {
        let m = module();
        assert_eq!(m.name, "bvh_builder");
        assert_eq!(m.dlls, &["xboxbvhbuilder.dll"]);
        assert_eq!(m.exports.len(), 7);
        let expected = [
            SYM_CTOR,
            SYM_DTOR,
            SYM_BUILD_BATCHED_2X,
            SYM_COPY,
            SYM_PREBUILD,
            SYM_REHYDRATE_BATCHED_2X,
            SYM_SET_64BIT_MORTON,
        ];
        for (exp, sym) in m.exports.iter().zip(expected) {
            assert_eq!(exp.name, sym);
            assert_ne!(exp.address, 0);
        }
    }

    #[test]
    fn exports_behave_cleanly_without_real_dll() {
        let mut dummy = 0u64;
        let this = (&mut dummy as *mut u64).cast::<c_void>();

        assert_eq!(ctor(this, std::ptr::null_mut(), 1), this);
        assert_eq!(ctor(std::ptr::null_mut(), std::ptr::null_mut(), 0), std::ptr::null_mut());
        set_64bit_morton_codes(this, 1);
        set_64bit_morton_codes(std::ptr::null_mut(), 0);
        build_batched_2x(this, std::ptr::null(), 0, std::ptr::null());
        copy_as(this, std::ptr::null(), 0, 0, 0);
        assert_eq!(
            rehydrate_batched_2x(this, std::ptr::null(), 0, std::ptr::null(), std::ptr::null()),
            0
        );

        let inputs = GuestBuildAsInputs {
            kind: 1,
            flags: 0,
            num_descs: 0,
            descs_layout: 0,
            ptr_or_va: 0,
        };
        let mut info = GuestPrebuildInfo::default();
        get_prebuild_info(this, &inputs, std::ptr::null(), &mut info);
        assert!(info.result_data_max_size_in_bytes >= 256);
        assert_eq!(info.result_data_max_size_in_bytes % 256, 0);
        assert_eq!(info.scratch_data_size_in_bytes % 256, 0);

        let stream = GuestStreamDesc {
            size_in_bytes: 1500,
            data: std::ptr::null(),
        };
        let mut stream_info = GuestPrebuildInfo::default();
        get_prebuild_info(this, &inputs, &stream, &mut stream_info);
        assert!(stream_info.result_data_max_size_in_bytes >= 1536);
        assert_eq!(stream_info.result_data_max_size_in_bytes % 256, 0);

        // Null inputs / null out_info must not crash.
        let mut zero_info = GuestPrebuildInfo {
            result_data_max_size_in_bytes: 99,
            scratch_data_size_in_bytes: 99,
            update_scratch_data_size_in_bytes: 99,
        };
        get_prebuild_info(this, std::ptr::null(), std::ptr::null(), &mut zero_info);
        assert_eq!(zero_info.result_data_max_size_in_bytes, 0);
        get_prebuild_info(this, &inputs, std::ptr::null(), std::ptr::null_mut());

        dtor(this);
        dtor(std::ptr::null_mut());
    }

    #[test]
    fn suppression_gates_real_builder_and_falls_back_on_stream_desc() {
        let mut dummy = 0u64;
        let this = (&mut dummy as *mut u64).cast::<c_void>();

        let b0 = MOCK_BUILD_CALLS.load(Ordering::Relaxed);
        dispatch_build_batched_2x(true, Some(mock_build), this, std::ptr::null(), 0, std::ptr::null());
        assert_eq!(MOCK_BUILD_CALLS.load(Ordering::Relaxed), b0);
        dispatch_build_batched_2x(false, Some(mock_build), this, std::ptr::null(), 0, std::ptr::null());
        assert_eq!(MOCK_BUILD_CALLS.load(Ordering::Relaxed), b0 + 1);

        let c0 = MOCK_COPY_CALLS.load(Ordering::Relaxed);
        dispatch_copy_as(true, Some(mock_copy), this, std::ptr::null(), 0x100, 0x200, 0);
        assert_eq!(MOCK_COPY_CALLS.load(Ordering::Relaxed), c0);
        dispatch_copy_as(false, Some(mock_copy), this, std::ptr::null(), 0x100, 0x200, 0);
        assert_eq!(MOCK_COPY_CALLS.load(Ordering::Relaxed), c0 + 1);

        let inputs = GuestBuildAsInputs {
            kind: 1,
            flags: 0,
            num_descs: 0,
            descs_layout: 0,
            ptr_or_va: 0,
        };
        let host_info = GuestPrebuildInfo {
            result_data_max_size_in_bytes: 0x400,
            scratch_data_size_in_bytes: 0x200,
            update_scratch_data_size_in_bytes: 0x100,
        };

        // Suppressed + host query succeeds -> uses host result and does NOT invoke real builder.
        let p0 = MOCK_PREBUILD_CALLS.load(Ordering::Relaxed);
        let mut out = GuestPrebuildInfo::default();
        dispatch_prebuild_info(
            true,
            Some(mock_prebuild),
            |_, sd| sd.is_none_or(|s| s.size_in_bytes == 0).then_some(host_info),
            this,
            &inputs,
            std::ptr::null(),
            &mut out,
        );
        assert_eq!(MOCK_PREBUILD_CALLS.load(Ordering::Relaxed), p0);
        assert_eq!(out.result_data_max_size_in_bytes, 0x400);

        // Suppressed + non-empty stream_desc (host query returns None) -> falls back to real builder.
        let stream = GuestStreamDesc {
            size_in_bytes: 4096,
            data: std::ptr::null(),
        };
        dispatch_prebuild_info(
            true,
            Some(mock_prebuild),
            |_, sd| sd.is_none_or(|s| s.size_in_bytes == 0).then_some(host_info),
            this,
            &inputs,
            &stream,
            &mut out,
        );
        assert_eq!(MOCK_PREBUILD_CALLS.load(Ordering::Relaxed), p0 + 1);
        assert_eq!(out.result_data_max_size_in_bytes, 0x2000);

        // Unsuppressed -> invokes real builder directly.
        dispatch_prebuild_info(
            false,
            Some(mock_prebuild),
            |_, _| Some(host_info),
            this,
            &inputs,
            std::ptr::null(),
            &mut out,
        );
        assert_eq!(MOCK_PREBUILD_CALLS.load(Ordering::Relaxed), p0 + 2);
        assert_eq!(out.result_data_max_size_in_bytes, 0x2000);
    }
}
