//! Query heaps, query recording/resolution, and GPU predication on host D3D12.
//!
//! When [`enabled`] is true (`VICTORY_QUERIES` unset or `!= "0"`) and a host
//! D3D12 device is available, each guest query heap is backed by a real host
//! `ID3D12QueryHeap`. `COPY_QUEUE_TIMESTAMP` heaps are remapped to `TIMESTAMP`
//! because `victory-gpu` executes all command lists on a host DIRECT queue.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(windows)]
use crate::host::HostList;
use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// Whether native host D3D12 query heaps, query resolution, predication, and
/// host queue timestamp frequency are enabled (`VICTORY_QUERIES=1` enables;
/// default off until live-verified under the D3D12 debug layer).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUERIES").is_some_and(|v| v == "1"))
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    #[cfg(windows)]
    host: Option<ID3D12QueryHeap>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryInfo {
    result_bytes: u64,
    allows_begin: bool,
}

#[cfg(windows)]
const _: () = assert!(size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() == 88);
#[cfg(windows)]
const _: () = assert!(size_of::<D3D12_QUERY_DATA_SO_STATISTICS>() == 16);
#[cfg(windows)]
const _: () = assert!(size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>() == 112);

/// Validates a console `D3D12_QUERY_TYPE` value and returns its resolved byte
/// size and whether `BeginQuery` is permitted on it.
fn query_info(kind: u32) -> Option<QueryInfo> {
    match kind {
        0 | 1 => Some(QueryInfo {
            result_bytes: 8,
            allows_begin: true,
        }),
        // D3D12 forbids BeginQuery on TIMESTAMP queries.
        2 => Some(QueryInfo {
            result_bytes: 8,
            allows_begin: false,
        }),
        3 => Some(QueryInfo {
            result_bytes: 88,
            allows_begin: true,
        }),
        4..=7 => Some(QueryInfo {
            result_bytes: 16,
            allows_begin: true,
        }),
        10 => Some(QueryInfo {
            result_bytes: 112,
            allows_begin: true,
        }),
        _ => None,
    }
}

/// Whether a guest query heap of `heap_kind` accepts queries of `query_kind`.
fn heap_accepts_query(heap_kind: u32, query_kind: u32) -> bool {
    match query_kind {
        0 | 1 => heap_kind == 0,
        2 => heap_kind == 1 || heap_kind == 5,
        3 => heap_kind == 2,
        4..=7 => heap_kind == 3,
        10 => heap_kind == 8,
        _ => false,
    }
}

/// Validates a query index and heap/type pair for `BeginQuery` or `EndQuery`.
fn validate_mark(desc: Desc, kind: u32, index: u32, is_begin: bool) -> bool {
    let Some(info) = query_info(kind) else {
        return false;
    };
    if is_begin && !info.allows_begin {
        return false;
    }
    heap_accepts_query(desc.kind, kind) && index < desc.count
}

/// Validates a `ResolveQueryData` range and destination byte offset, returning
/// total resolved bytes on success.
fn validate_resolve(desc: Desc, kind: u32, first: u32, count: u32, offset: u64) -> Option<u64> {
    let info = query_info(kind)?;
    if !heap_accepts_query(desc.kind, kind) {
        return None;
    }
    let end_index = first.checked_add(count)?;
    if end_index > desc.count || !offset.is_multiple_of(8) {
        return None;
    }
    u64::from(count).checked_mul(info.result_bytes)
}

#[cfg(windows)]
fn map_heap_type(kind: u32) -> Option<D3D12_QUERY_HEAP_TYPE> {
    match kind {
        0 => Some(D3D12_QUERY_HEAP_TYPE_OCCLUSION),
        // COPY_QUEUE_TIMESTAMP (5) must map to TIMESTAMP because victory-gpu
        // executes all command lists on a host DIRECT queue.
        1 | 5 => Some(D3D12_QUERY_HEAP_TYPE_TIMESTAMP),
        2 => Some(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS),
        3 => Some(D3D12_QUERY_HEAP_TYPE_SO_STATISTICS),
        4 => Some(D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS),
        8 => Some(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1),
        _ => None,
    }
}

#[cfg(windows)]
fn map_query_type(kind: u32) -> Option<D3D12_QUERY_TYPE> {
    match kind {
        0 => Some(D3D12_QUERY_TYPE_OCCLUSION),
        1 => Some(D3D12_QUERY_TYPE_BINARY_OCCLUSION),
        2 => Some(D3D12_QUERY_TYPE_TIMESTAMP),
        3 => Some(D3D12_QUERY_TYPE_PIPELINE_STATISTICS),
        4 => Some(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0),
        5 => Some(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1),
        6 => Some(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM2),
        7 => Some(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3),
        10 => Some(D3D12_QUERY_TYPE_PIPELINE_STATISTICS1),
        _ => None,
    }
}

#[cfg(windows)]
fn map_predication_op(operation: u32) -> Option<D3D12_PREDICATION_OP> {
    match operation {
        0 => Some(D3D12_PREDICATION_OP_EQUAL_ZERO),
        1 => Some(D3D12_PREDICATION_OP_NOT_EQUAL_ZERO),
        _ => None,
    }
}

#[cfg(windows)]
fn create_host_heap_on(device: &ID3D12Device, desc: Desc) -> Option<ID3D12QueryHeap> {
    if desc.count == 0 {
        return None;
    }
    let host_type = map_heap_type(desc.kind)?;
    let host_desc = D3D12_QUERY_HEAP_DESC {
        Type: host_type,
        Count: desc.count,
        NodeMask: 0,
    };
    let mut heap: Option<ID3D12QueryHeap> = None;
    // SAFETY: valid device and query heap descriptor.
    match unsafe { device.CreateQueryHeap(&host_desc, &mut heap) } {
        Ok(()) => heap,
        Err(e) => {
            tracing::warn!(
                kind = desc.kind,
                count = desc.count,
                error = %e,
                "host CreateQueryHeap failed; keeping synthetic fallback"
            );
            None
        }
    }
}

#[cfg(windows)]
fn create_host_heap(desc: Desc) -> Option<ID3D12QueryHeap> {
    let host = crate::host::host()?;
    create_host_heap_on(&host.device, desc)
}

#[cfg(windows)]
fn heap_ref<'a>(heap: This) -> Option<&'a Heap> {
    if heap.is_null() {
        return None;
    }
    // SAFETY: guest COM objects begin with their vtable pointer; verify it is
    // our query heap table before reading `Heap` fields.
    let vtable = unsafe { *heap.cast::<*const *const ()>() };
    if vtable != FUNCTIONS.0.as_ptr() {
        return None;
    }
    // SAFETY: verified above to be a live `Heap`.
    Some(unsafe { &*heap.cast::<Heap>() })
}

#[cfg(windows)]
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// Records `BeginQuery` on the host list if `heap` has a real host query heap
/// and the arguments are valid. Returns `false` if skipped or rejected.
#[cfg(windows)]
pub(crate) fn begin_query(list: &HostList, heap: This, kind: u32, index: u32) -> bool {
    if !enabled() {
        return false;
    }
    let Some(heap) = heap_ref(heap) else {
        return false;
    };
    let Some(host_heap) = heap.host.as_ref() else {
        return false;
    };
    if !validate_mark(heap.desc, kind, index, true) {
        return false;
    }
    let Some(host_kind) = map_query_type(kind) else {
        return false;
    };
    list.keep(host_heap);
    // SAFETY: an open host command list and validated query heap/type/index.
    unsafe { list.list.BeginQuery(host_heap, host_kind, index) };
    true
}

/// Records `EndQuery` on the host list if `heap` has a real host query heap
/// and the arguments are valid. Returns `false` if skipped or rejected.
#[cfg(windows)]
pub(crate) fn end_query(list: &HostList, heap: This, kind: u32, index: u32) -> bool {
    if !enabled() {
        return false;
    }
    let Some(heap) = heap_ref(heap) else {
        return false;
    };
    let Some(host_heap) = heap.host.as_ref() else {
        return false;
    };
    if !validate_mark(heap.desc, kind, index, false) {
        return false;
    }
    let Some(host_kind) = map_query_type(kind) else {
        return false;
    };
    list.keep(host_heap);
    // SAFETY: an open host command list and validated query heap/type/index.
    unsafe { list.list.EndQuery(host_heap, host_kind, index) };
    true
}

/// Records `ResolveQueryData` on the host list into the destination buffer's
/// shadow copy and schedules shadow GPU write-back to guest RAM. Returns
/// `false` if native resolution cannot handle the call (caller may fall back).
#[cfg(windows)]
pub(crate) fn resolve_query_data(
    list: &HostList,
    heap: This,
    kind: u32,
    first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) -> bool {
    if !enabled() {
        return false;
    }
    let Some(heap) = heap_ref(heap) else {
        return false;
    };
    let Some(host_heap) = heap.host.as_ref() else {
        return false;
    };
    let Some(total_bytes) = validate_resolve(heap.desc, kind, first, count, offset) else {
        return false;
    };
    let Some((ptr, size)) = crate::resource::buffer_bytes(buffer) else {
        return false;
    };
    let Some(end) = offset.checked_add(total_bytes) else {
        return false;
    };
    if end > size {
        return false;
    }
    if count == 0 {
        return true;
    }
    let Some(address) = (ptr as u64).checked_add(offset) else {
        return false;
    };
    let Some(shadow) = crate::shadow::find(address, total_bytes) else {
        return false;
    };
    if shadow.left(address) < total_bytes {
        return false;
    }
    let shadow_offset = shadow.offset(address);
    if !shadow_offset.is_multiple_of(8) {
        return false;
    }
    let Some(host_kind) = map_query_type(kind) else {
        return false;
    };

    // Refresh current guest bytes first so the shadow's pre-submission snapshot
    // captures the prior contents (including non-zero values overwritten by 0).
    shadow.refresh(list, address, total_bytes);
    list.keep(host_heap);
    list.keep_shadow(shadow.clone());

    let before = [
        uav_barrier(),
        crate::host::transition(
            &shadow.buffer,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        ),
    ];
    let after = [crate::host::transition(
        &shadow.buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    )];
    // SAFETY: an open host command list; `host_heap` and `shadow` are kept
    // alive on `list`; range and 8-byte alignment are validated above.
    unsafe {
        list.list.ResourceBarrier(&before);
        list.list
            .ResolveQueryData(host_heap, host_kind, first, count, &shadow.buffer, shadow_offset);
        list.list.ResourceBarrier(&after);
    }

    shadow.gpu_writes(list, address, total_bytes);
    list.changed_something();
    true
}

/// Records `SetPredication` on the host list. Passing a null `buffer` clears
/// predication; otherwise validates alignment/bounds and binds the buffer's
/// shadow copy. Returns `false` if disabled or invalid.
#[cfg(windows)]
pub(crate) fn set_predication(
    list: &HostList,
    buffer: This,
    offset: u64,
    operation: u32,
) -> bool {
    if !enabled() {
        return false;
    }
    if buffer.is_null() {
        // SAFETY: an open host command list; `None` clears predication.
        unsafe {
            list.list
                .SetPredication(None, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
        return true;
    }
    let Some(host_op) = map_predication_op(operation) else {
        return false;
    };
    if !offset.is_multiple_of(8) {
        return false;
    }
    let Some((ptr, size)) = crate::resource::buffer_bytes(buffer) else {
        return false;
    };
    let Some(end) = offset.checked_add(8) else {
        return false;
    };
    if end > size {
        return false;
    }
    let Some(address) = (ptr as u64).checked_add(offset) else {
        return false;
    };
    let Some(shadow) = crate::shadow::find_any(address, 8) else {
        return false;
    };
    if shadow.left(address) < 8 {
        return false;
    }
    let shadow_offset = shadow.offset(address);
    if !shadow_offset.is_multiple_of(8) {
        return false;
    }

    shadow.refresh(list, address, 8);
    list.keep_shadow(shadow.clone());

    // Ensure prior GPU UAV writes complete before predication reads the buffer.
    // `resolve_query_data` already transitioned `COPY_DEST -> COMMON`, and
    // `COMMON` implicitly promotes to `PREDICATION` on a DIRECT list.
    let barrier = [uav_barrier()];
    // SAFETY: an open host command list; `shadow` is kept alive on `list`;
    // offset is 8-byte aligned and within bounds.
    unsafe {
        list.list.ResourceBarrier(&barrier);
        list.list
            .SetPredication(Some(&shadow.buffer), shadow_offset, host_op);
    }
    true
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    #[cfg(windows)]
    let host = if !out.is_null() && object::is(iid, &HEAP_IIDS) && enabled() && desc.count > 0 {
        create_host_heap(desc)
    } else {
        None
    };
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host,
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, E_POINTER, S_FALSE, S_OK, slot};

    #[test]
    fn null_descriptions_are_rejected() {
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), std::ptr::null(), &HEAP_IIDS[0], &mut heap),
            E_INVALIDARG
        );
        assert!(heap.is_null());
        // The description is checked first: a probe with no room for the
        // heap fails on it rather than answering S_FALSE.
        assert_eq!(
            create(
                std::ptr::null(),
                std::ptr::null(),
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            E_INVALIDARG
        );
    }

    #[test]
    fn creation_without_an_out_pointer_only_probes() {
        let desc = Desc {
            kind: 1, // timestamps
            count: 64,
            nodes: 0,
        };
        // A known interface with no out pointer asks whether creation would
        // succeed; the heap is dropped instead of handed out.
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );
        // A null interface id is tolerated: no interface, no heap.
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, std::ptr::null(), &mut heap),
            E_NOINTERFACE
        );
        assert!(heap.is_null());
    }

    #[test]
    fn heaps_answer_their_interfaces_and_count_references() {
        let desc = Desc {
            kind: 0, // occlusion
            count: 8,
            nodes: 0,
        };
        let null = std::ptr::null();
        let mut heap = null;
        assert_eq!(create(null, &desc, &HEAP_IIDS[0], &mut heap), S_OK);
        assert!(!heap.is_null());
        let query = slot::<extern "system" fn(This, *const Iid, *mut This) -> HRESULT>(heap, 0x0);
        // Every interface the heap answers to hands out the same object.
        for iid in &HEAP_IIDS {
            let mut out = null;
            assert_eq!(query(heap, iid, &mut out), S_OK);
            assert_eq!(out as usize, heap as usize);
        }
        // An unknown interface writes null behind and reports it missing.
        let unknown = [0u8; 16];
        let mut out = heap;
        assert_eq!(query(heap, &unknown, &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // A null interface id is tolerated rather than matched.
        let mut out = heap;
        assert_eq!(query(heap, std::ptr::null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // No room for the answer is a pointer error.
        assert_eq!(query(heap, &unknown, std::ptr::null_mut()), E_POINTER);
        // References: one from creation, four from the queries, one added;
        // each release hands back what is left, the last frees the heap.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(heap, 0x8)(heap), 6);
        let release = slot::<extern "system" fn(This) -> u32>(heap, 0x10);
        for left in [5, 4, 3, 2, 1, 0] {
            assert_eq!(release(heap), left);
        }
    }

    #[test]
    fn query_validation_rules() {
        // Heap type -> query type compatibility and result sizes.
        assert_eq!(
            query_info(0),
            Some(QueryInfo { result_bytes: 8, allows_begin: true })
        );
        assert_eq!(
            query_info(1),
            Some(QueryInfo { result_bytes: 8, allows_begin: true })
        );
        assert_eq!(
            query_info(2),
            Some(QueryInfo { result_bytes: 8, allows_begin: false })
        );
        assert_eq!(
            query_info(3),
            Some(QueryInfo { result_bytes: 88, allows_begin: true })
        );
        for so in 4..=7 {
            assert_eq!(
                query_info(so),
                Some(QueryInfo { result_bytes: 16, allows_begin: true })
            );
        }
        assert_eq!(
            query_info(10),
            Some(QueryInfo { result_bytes: 112, allows_begin: true })
        );
        assert_eq!(query_info(9), None);
        assert_eq!(query_info(99), None);

        // TIMESTAMP rejects BeginQuery, accepts EndQuery on both TIMESTAMP (1)
        // and COPY_QUEUE_TIMESTAMP (5) heaps.
        let ts_heap = Desc { kind: 1, count: 4, nodes: 0 };
        let copy_ts_heap = Desc { kind: 5, count: 4, nodes: 0 };
        assert!(!validate_mark(ts_heap, 2, 0, true));
        assert!(validate_mark(ts_heap, 2, 0, false));
        assert!(!validate_mark(copy_ts_heap, 2, 3, true));
        assert!(validate_mark(copy_ts_heap, 2, 3, false));
        // Out-of-bounds index rejected.
        assert!(!validate_mark(ts_heap, 2, 4, false));
        assert!(!validate_mark(copy_ts_heap, 2, 4, false));
        // Mismatched query type for heap rejected.
        assert!(!validate_mark(ts_heap, 0, 0, false));
        assert!(!validate_mark(ts_heap, 1, 0, true));

        // OCCLUSION heap accepts OCCLUSION (0) and BINARY_OCCLUSION (1) only.
        let occ_heap = Desc { kind: 0, count: 8, nodes: 0 };
        assert!(validate_mark(occ_heap, 0, 0, true));
        assert!(validate_mark(occ_heap, 1, 7, true));
        assert!(!validate_mark(occ_heap, 0, 8, true));
        assert!(!validate_mark(occ_heap, 2, 0, false));

        // PIPELINE_STATISTICS (2) and SO_STATISTICS (3) and PIPELINE_STATISTICS1 (8).
        let ps_heap = Desc { kind: 2, count: 2, nodes: 0 };
        assert!(validate_mark(ps_heap, 3, 1, true));
        assert!(!validate_mark(ps_heap, 10, 0, true));
        let so_heap = Desc { kind: 3, count: 2, nodes: 0 };
        for so in 4..=7 {
            assert!(validate_mark(so_heap, so, 0, true));
        }
        assert!(!validate_mark(so_heap, 0, 0, true));
        let ps1_heap = Desc { kind: 8, count: 2, nodes: 0 };
        assert!(validate_mark(ps1_heap, 10, 0, true));
        assert!(!validate_mark(ps1_heap, 3, 0, true));

        // Resolve validation: bounds, overflow, alignment.
        assert_eq!(validate_resolve(occ_heap, 0, 0, 8, 0), Some(64));
        assert_eq!(validate_resolve(occ_heap, 1, 6, 2, 16), Some(16));
        assert_eq!(validate_resolve(occ_heap, 0, 8, 0, 0), Some(0));
        assert_eq!(validate_resolve(ps_heap, 3, 0, 2, 8), Some(176));
        assert_eq!(validate_resolve(so_heap, 5, 1, 1, 24), Some(16));
        assert_eq!(validate_resolve(copy_ts_heap, 2, 0, 4, 0), Some(32));
        // OOB & overflow.
        assert_eq!(validate_resolve(occ_heap, 0, 7, 2, 0), None);
        assert_eq!(validate_resolve(occ_heap, 0, u32::MAX, 1, 0), None);
        // Unaligned offset (must be multiple of 8).
        assert_eq!(validate_resolve(occ_heap, 0, 0, 1, 4), None);
        assert_eq!(validate_resolve(occ_heap, 0, 0, 1, 1), None);
        // Mismatched heap/query.
        assert_eq!(validate_resolve(ts_heap, 0, 0, 1, 0), None);
    }

    #[cfg(windows)]
    #[test]
    fn heap_type_and_predication_mapping() {
        assert_eq!(map_heap_type(0), Some(D3D12_QUERY_HEAP_TYPE_OCCLUSION));
        assert_eq!(map_heap_type(1), Some(D3D12_QUERY_HEAP_TYPE_TIMESTAMP));
        assert_eq!(map_heap_type(2), Some(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS));
        assert_eq!(map_heap_type(3), Some(D3D12_QUERY_HEAP_TYPE_SO_STATISTICS));
        assert_eq!(map_heap_type(4), Some(D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS));
        // CRITICAL: COPY_QUEUE_TIMESTAMP (5) maps to TIMESTAMP for DIRECT lists.
        assert_eq!(map_heap_type(5), Some(D3D12_QUERY_HEAP_TYPE_TIMESTAMP));
        assert_eq!(map_heap_type(8), Some(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1));
        assert_eq!(map_heap_type(6), None);
        assert_eq!(map_heap_type(99), None);

        assert_eq!(map_predication_op(0), Some(D3D12_PREDICATION_OP_EQUAL_ZERO));
        assert_eq!(map_predication_op(1), Some(D3D12_PREDICATION_OP_NOT_EQUAL_ZERO));
        assert_eq!(map_predication_op(2), None);
    }

    #[cfg(windows)]
    mod gpu_tests {
        use super::*;
        use windows::Win32::Foundation::{HANDLE, RECT};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{
            D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
        };
        use windows::Win32::Graphics::Dxgi::*;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::{Interface, s};

        struct WarpDevice {
            device: ID3D12Device,
            queue: ID3D12CommandQueue,
            allocator: ID3D12CommandAllocator,
            list: ID3D12GraphicsCommandList,
            fence: ID3D12Fence,
            submitted: u64,
            event: HANDLE,
            /// Held last so WARP is released only after all D3D12 objects drop.
            _turn: std::sync::MutexGuard<'static, ()>,
        }

        impl WarpDevice {
            fn new() -> Option<Self> {
                let turn = crate::indirect::tests::warp_turn();
                // SAFETY: plain DXGI and D3D12 test setup on WARP.
                unsafe {
                    let factory: IDXGIFactory4 =
                        CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                    let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                    let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                        return None;
                    }
                    let mut device: Option<ID3D12Device> = None;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                    let device = device?;
                    let queue: ID3D12CommandQueue = device
                        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                            ..Default::default()
                        })
                        .ok()?;
                    let allocator: ID3D12CommandAllocator = device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .ok()?;
                    let list: ID3D12GraphicsCommandList = device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                        .ok()?;
                    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                    let event = CreateEventW(None, false, false, None).ok()?;
                    Some(Self {
                        device,
                        queue,
                        allocator,
                        list,
                        fence,
                        submitted: 0,
                        event,
                        _turn: turn,
                    })
                }
            }

            fn buffer(
                &self,
                heap: D3D12_HEAP_TYPE,
                size: u64,
                state: D3D12_RESOURCE_STATES,
            ) -> ID3D12Resource {
                let mut res: Option<ID3D12Resource> = None;
                // SAFETY: valid device and buffer description.
                unsafe {
                    self.device
                        .CreateCommittedResource(
                            &D3D12_HEAP_PROPERTIES {
                                Type: heap,
                                ..Default::default()
                            },
                            D3D12_HEAP_FLAG_NONE,
                            &D3D12_RESOURCE_DESC {
                                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                                Width: size,
                                Height: 1,
                                DepthOrArraySize: 1,
                                MipLevels: 1,
                                Format: DXGI_FORMAT_UNKNOWN,
                                SampleDesc: DXGI_SAMPLE_DESC {
                                    Count: 1,
                                    Quality: 0,
                                },
                                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                ..Default::default()
                            },
                            state,
                            None,
                            &mut res,
                        )
                        .expect("CreateCommittedResource buffer");
                }
                res.expect("buffer created")
            }

            fn render_target(&self, width: u32, height: u32) -> (ID3D12Resource, ID3D12DescriptorHeap) {
                let mut rt: Option<ID3D12Resource> = None;
                // SAFETY: valid device and 2D render target description.
                unsafe {
                    self.device
                        .CreateCommittedResource(
                            &D3D12_HEAP_PROPERTIES {
                                Type: D3D12_HEAP_TYPE_DEFAULT,
                                ..Default::default()
                            },
                            D3D12_HEAP_FLAG_NONE,
                            &D3D12_RESOURCE_DESC {
                                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                                Width: u64::from(width),
                                Height: height,
                                DepthOrArraySize: 1,
                                MipLevels: 1,
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                SampleDesc: DXGI_SAMPLE_DESC {
                                    Count: 1,
                                    Quality: 0,
                                },
                                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                ..Default::default()
                            },
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            None,
                            &mut rt,
                        )
                        .expect("CreateCommittedResource RTV");
                    let rt = rt.expect("RTV texture");
                    let rtv_heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                            NodeMask: 0,
                        })
                        .expect("CreateDescriptorHeap RTV");
                    self.device
                        .CreateRenderTargetView(&rt, None, rtv_heap.GetCPUDescriptorHandleForHeapStart());
                    (rt, rtv_heap)
                }
            }

            fn fullscreen_pso(&self) -> (ID3D12RootSignature, ID3D12PipelineState) {
                const HLSL: &str = r#"
float4 vs_main(uint id : SV_VertexID) : SV_Position {
    float2 pos = float2((id == 1) ? 3.0 : -1.0, (id == 2) ? -3.0 : 1.0);
    return float4(pos, 0.0, 1.0);
}
float4 ps_main() : SV_Target {
    return float4(1.0, 0.5, 0.25, 1.0);
}
"#;
                fn compile(entry: windows::core::PCSTR, target: windows::core::PCSTR) -> Vec<u8> {
                    let mut blob: Option<ID3DBlob> = None;
                    let mut errs: Option<ID3DBlob> = None;
                    // SAFETY: valid HLSL source and out pointers.
                    let hr = unsafe {
                        D3DCompile(
                            HLSL.as_ptr().cast(),
                            HLSL.len(),
                            s!("query_test"),
                            None,
                            None,
                            entry,
                            target,
                            0,
                            0,
                            &mut blob,
                            Some(&mut errs),
                        )
                    };
                    if let Err(e) = hr {
                        let msg = errs
                            .map(|b| unsafe {
                                String::from_utf8_lossy(std::slice::from_raw_parts(
                                    b.GetBufferPointer().cast::<u8>(),
                                    b.GetBufferSize(),
                                ))
                                .into_owned()
                            })
                            .unwrap_or_default();
                        panic!("D3DCompile failed ({e}): {msg}");
                    }
                    let blob = blob.expect("shader blob");
                    // SAFETY: valid shader blob buffer.
                    unsafe {
                        std::slice::from_raw_parts(
                            blob.GetBufferPointer().cast::<u8>(),
                            blob.GetBufferSize(),
                        )
                        .to_vec()
                    }
                }

                let vs = compile(s!("vs_main"), s!("vs_5_0"));
                let ps = compile(s!("ps_main"), s!("ps_5_0"));

                let mut rs_blob: Option<ID3DBlob> = None;
                // SAFETY: empty root signature serialized and created on device.
                unsafe {
                    D3D12SerializeRootSignature(
                        &D3D12_ROOT_SIGNATURE_DESC {
                            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                            ..Default::default()
                        },
                        D3D_ROOT_SIGNATURE_VERSION_1,
                        &mut rs_blob,
                        None,
                    )
                    .expect("D3D12SerializeRootSignature");
                    let rs_blob = rs_blob.expect("root sig blob");
                    let rs_bytes = std::slice::from_raw_parts(
                        rs_blob.GetBufferPointer().cast::<u8>(),
                        rs_blob.GetBufferSize(),
                    );
                    let root: ID3D12RootSignature = self
                        .device
                        .CreateRootSignature(0, rs_bytes)
                        .expect("CreateRootSignature");
                    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        VS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: vs.as_ptr().cast(),
                            BytecodeLength: vs.len(),
                        },
                        PS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: ps.as_ptr().cast(),
                            BytecodeLength: ps.len(),
                        },
                        BlendState: D3D12_BLEND_DESC {
                            RenderTarget: [
                                D3D12_RENDER_TARGET_BLEND_DESC {
                                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                                    ..Default::default()
                                },
                                Default::default(),
                                Default::default(),
                                Default::default(),
                                Default::default(),
                                Default::default(),
                                Default::default(),
                                Default::default(),
                            ],
                            ..Default::default()
                        },
                        SampleMask: u32::MAX,
                        RasterizerState: D3D12_RASTERIZER_DESC {
                            FillMode: D3D12_FILL_MODE_SOLID,
                            CullMode: D3D12_CULL_MODE_NONE,
                            DepthClipEnable: true.into(),
                            ..Default::default()
                        },
                        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                        NumRenderTargets: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        ..Default::default()
                    };
                    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                    let pso: ID3D12PipelineState = self
                        .device
                        .CreateGraphicsPipelineState(&desc)
                        .expect("CreateGraphicsPipelineState");
                    (root, pso)
                }
            }

            fn execute_and_wait(&mut self) {
                // SAFETY: open list closed, submitted to direct queue, waited on fence.
                unsafe {
                    self.list.Close().expect("Close");
                    let cmd: ID3D12CommandList = self.list.cast().expect("cast ID3D12CommandList");
                    self.queue.ExecuteCommandLists(&[Some(cmd)]);
                    self.submitted += 1;
                    self.queue
                        .Signal(&self.fence, self.submitted)
                        .expect("Signal");
                    if self.fence.GetCompletedValue() < self.submitted {
                        self.fence
                            .SetEventOnCompletion(self.submitted, self.event)
                            .expect("SetEventOnCompletion");
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    self.allocator.Reset().expect("allocator Reset");
                    self.list.Reset(&self.allocator, None).expect("list Reset");
                }
            }

            fn read_u64s(&self, readback: &ID3D12Resource, count: usize) -> Vec<u64> {
                let mut ptr = std::ptr::null_mut();
                let range = D3D12_RANGE {
                    Begin: 0,
                    End: count * 8,
                };
                // SAFETY: readback buffer large enough for `count` u64s and GPU idle.
                unsafe {
                    readback.Map(0, Some(&range), Some(&mut ptr)).expect("Map");
                    let vals = std::slice::from_raw_parts(ptr.cast::<u64>(), count).to_vec();
                    readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
                    vals
                }
            }
        }

        impl Drop for WarpDevice {
            fn drop(&mut self) {
                // SAFETY: owned event handle.
                unsafe {
                    let _ = windows::Win32::Foundation::CloseHandle(self.event);
                }
            }
        }

        #[test]
        fn warp_timestamp_queries_and_copy_queue_remap() {
            let Some(mut gpu) = WarpDevice::new() else {
                return;
            };
            // SAFETY: valid direct command queue.
            let freq = unsafe { gpu.queue.GetTimestampFrequency() }
                .expect("GetTimestampFrequency");
            assert!(freq > 0, "timestamp frequency must be positive");

            // Standard TIMESTAMP heap (kind 1) and remapped COPY_QUEUE_TIMESTAMP heap (kind 5).
            let ts_heap = create_host_heap_on(
                &gpu.device,
                Desc {
                    kind: 1,
                    count: 4,
                    nodes: 0,
                },
            )
            .expect("TIMESTAMP query heap");
            let copy_ts_heap = create_host_heap_on(
                &gpu.device,
                Desc {
                    kind: 5,
                    count: 4,
                    nodes: 0,
                },
            )
            .expect("COPY_QUEUE_TIMESTAMP remapped to TIMESTAMP query heap");

            let src = gpu.buffer(D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
            let dst = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 4096, D3D12_RESOURCE_STATE_COMMON);
            let resolved = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COMMON);
            let readback = gpu.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST);

            // SAFETY: valid open list, live query heaps and buffers.
            unsafe {
                gpu.list.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                gpu.list.CopyBufferRegion(&dst, 0, &src, 0, 4096);
                gpu.list.EndQuery(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

                gpu.list.EndQuery(&copy_ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                gpu.list.CopyBufferRegion(&dst, 0, &src, 0, 4096);
                gpu.list.EndQuery(&copy_ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &resolved,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                gpu.list.ResolveQueryData(&ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &resolved, 0);
                gpu.list.ResolveQueryData(
                    &copy_ts_heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    0,
                    2,
                    &resolved,
                    16,
                );
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &resolved,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                gpu.list.CopyBufferRegion(&readback, 0, &resolved, 0, 32);
            }

            gpu.execute_and_wait();
            let vals = gpu.read_u64s(&readback, 4);
            assert!(
                vals[1] >= vals[0],
                "TIMESTAMP t1 ({}) must be >= t0 ({})",
                vals[1],
                vals[0]
            );
            assert!(
                vals[3] >= vals[2],
                "remapped COPY_QUEUE_TIMESTAMP t1 ({}) must be >= t0 ({})",
                vals[3],
                vals[2]
            );
        }

        #[test]
        fn warp_occlusion_and_binary_occlusion_queries() {
            let Some(mut gpu) = WarpDevice::new() else {
                return;
            };
            let heap = create_host_heap_on(
                &gpu.device,
                Desc {
                    kind: 0,
                    count: 4,
                    nodes: 0,
                },
            )
            .expect("OCCLUSION query heap");
            let (_rt, rtv_heap) = gpu.render_target(64, 64);
            let (root, pso) = gpu.fullscreen_pso();
            let resolved = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COMMON);
            let readback = gpu.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST);

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 64.0,
                Height: 64.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: 64,
                bottom: 64,
            };

            // SAFETY: valid open list, pipeline, RTV, query heap, and buffers.
            unsafe {
                let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                gpu.list.OMSetRenderTargets(1, Some(&rtv), false, None);
                gpu.list.RSSetViewports(&[vp]);
                gpu.list.RSSetScissorRects(&[scissor]);
                gpu.list.SetGraphicsRootSignature(&root);
                gpu.list.SetPipelineState(&pso);
                gpu.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                // Query 0: OCCLUSION over fullscreen 64x64 triangle -> 4096 samples.
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0);

                // Query 1: BINARY_OCCLUSION over fullscreen triangle -> 1.
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);

                // Query 2: OCCLUSION over empty draw -> 0 samples.
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2);
                gpu.list.DrawInstanced(0, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2);

                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &resolved,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                gpu.list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, &resolved, 0);
                gpu.list
                    .ResolveQueryData(&heap, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1, 1, &resolved, 8);
                gpu.list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2, 1, &resolved, 16);
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &resolved,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                gpu.list.CopyBufferRegion(&readback, 0, &resolved, 0, 24);
            }

            gpu.execute_and_wait();
            let vals = gpu.read_u64s(&readback, 3);
            assert_eq!(vals[0], 64 * 64, "fullscreen 64x64 triangle occlusion count");
            assert_eq!(vals[1], 1, "binary occlusion visible");
            assert_eq!(vals[2], 0, "empty draw occlusion count");
        }

        #[test]
        fn warp_set_predication_suppresses_and_permits_draws() {
            let Some(mut gpu) = WarpDevice::new() else {
                return;
            };
            let heap = create_host_heap_on(
                &gpu.device,
                Desc {
                    kind: 0,
                    count: 8,
                    nodes: 0,
                },
            )
            .expect("OCCLUSION query heap");
            let (_rt, rtv_heap) = gpu.render_target(64, 64);
            let (root, pso) = gpu.fullscreen_pso();
            let pred_buf = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COMMON);
            let out_buf = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COMMON);
            let readback = gpu.buffer(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST);

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 64.0,
                Height: 64.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: 64,
                bottom: 64,
            };

            // SAFETY: valid open list, pipeline, RTV, query heap, and buffers.
            unsafe {
                let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                gpu.list.OMSetRenderTargets(1, Some(&rtv), false, None);
                gpu.list.RSSetViewports(&[vp]);
                gpu.list.RSSetScissorRects(&[scissor]);
                gpu.list.SetGraphicsRootSignature(&root);
                gpu.list.SetPipelineState(&pso);
                gpu.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                // Slot 0 = 0 (empty draw), Slot 1 = 4096 (fullscreen draw).
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0);
                gpu.list.DrawInstanced(0, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0);

                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 1);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 1);

                // Resolve slots 0..2 into `pred_buf` using our COMMON -> COPY_DEST -> COMMON pattern.
                gpu.list.ResourceBarrier(&[
                    uav_barrier(),
                    crate::host::transition(
                        &pred_buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ),
                ]);
                gpu.list
                    .ResolveQueryData(&heap, D3D12_QUERY_TYPE_OCCLUSION, 0, 2, &pred_buf, 0);
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &pred_buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);

                // Slot 2: EQUAL_ZERO on 0 (offset 0) -> draw skipped (0 samples).
                gpu.list.ResourceBarrier(&[uav_barrier()]);
                gpu.list
                    .SetPredication(Some(&pred_buf), 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2);

                // Slot 3: EQUAL_ZERO on 4096 (offset 8) -> draw runs (4096 samples).
                gpu.list
                    .SetPredication(Some(&pred_buf), 8, D3D12_PREDICATION_OP_EQUAL_ZERO);
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 3);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 3);

                // Slot 4: NOT_EQUAL_ZERO on 0 (offset 0) -> draw runs (4096 samples).
                gpu.list
                    .SetPredication(Some(&pred_buf), 0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 4);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 4);

                // Slot 5: NOT_EQUAL_ZERO on 4096 (offset 8) -> draw skipped (0 samples).
                gpu.list
                    .SetPredication(Some(&pred_buf), 8, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 5);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 5);

                // Slot 6: Clear predication -> draw runs (4096 samples).
                gpu.list
                    .SetPredication(None, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
                gpu.list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 6);
                gpu.list.DrawInstanced(3, 1, 0, 0);
                gpu.list.EndQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, 6);

                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &out_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                gpu.list
                    .ResolveQueryData(&heap, D3D12_QUERY_TYPE_OCCLUSION, 2, 5, &out_buf, 0);
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &out_buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                gpu.list.CopyBufferRegion(&readback, 0, &out_buf, 0, 40);
            }

            gpu.execute_and_wait();
            let vals = gpu.read_u64s(&readback, 5);
            assert_eq!(vals[0], 0, "EQUAL_ZERO with 0 must skip draw");
            assert_eq!(vals[1], 64 * 64, "EQUAL_ZERO with non-zero must allow draw");
            assert_eq!(vals[2], 64 * 64, "NOT_EQUAL_ZERO with 0 must allow draw");
            assert_eq!(vals[3], 0, "NOT_EQUAL_ZERO with non-zero must skip draw");
            assert_eq!(vals[4], 64 * 64, "cleared predication must allow draw");
        }
    }
}
