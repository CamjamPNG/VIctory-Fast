//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`) and selects an FSR profile.
//! Other profiles keep the title's own console DLLs.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Upscaling runs by default;
//! PC Frame Generation is opt-in via `VICTORY_FFX_FG=1`.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_DESC, ID3D12Device, ID3D12Resource,
};
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
pub const DLL: &str = "amd_fidelityfx_upscaler_dx12.dll";
/// The optional PC build's frame generator (`VICTORY_FFX_FG=1`).
pub const FG_DLL: &str = "amd_fidelityfx_framegeneration_dx12.dll";

// Return codes.
const OK: u32 = 0;
const UNKNOWN_TYPE: u32 = 2;
const NO_PROVIDER: u32 = 4;
const PARAMETER: u32 = 6;

// Common description types.
const CONSOLE_BACKEND: u64 = 0x100_0002;
const HOST_BACKEND: u64 = 2;
const GET_VERSIONS: u64 = 4;
const OVERRIDE_VERSION: u64 = 5;
const GET_PROVIDER_VERSION: u64 = 6;

// Upscaler description types (effect id 0x1_0000).
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
const GET_RENDER_RESOLUTION: u64 = 0x1_0003;
const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;
const DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;

// Frame Generation description types (effect id 0x2_0000).
const EFFECT_MASK: u64 = 0xffff_0000;
const EFFECT_FRAMEGEN: u64 = 0x2_0000;
const CREATE_FRAMEGEN: u64 = 0x2_0001;
const CONFIGURE_FRAMEGEN: u64 = 0x2_0002;
const DISPATCH_FRAMEGEN: u64 = 0x2_0003;
const DISPATCH_FRAMEGEN_PREPARE: u64 = 0x2_0004;
const QUERY_FRAMEGEN_MEMORY_USAGE_V2: u64 = 0x2_000b;
const DISPATCH_FRAMEGEN_PREPARE_V2: u64 = 0x2_000c;

// FfxApiResourceDescription field indices (`ffx_api_types.h`):
// [0]=type, [1]=format, [2]=width, [3]=height, [4]=depth, [5]=mipCount, [6]=flags, [7]=usage.
const DESC_WIDTH: usize = 2;
const DESC_HEIGHT: usize = 3;
const DESC_DEPTH: usize = 4;
const DESC_MIP_COUNT: usize = 5;

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// The backend description: a device.
#[repr(C)]
struct Backend {
    header: Header,
    device: *mut c_void,
}

/// A resource, its description (type, format, width, height, depth, mips,
/// flags, usage) and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    settings: [u32; 10],
}

/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
#[derive(Clone, Copy)]
struct GetVersions {
    header: Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryUsageWithDevice {
    header: Header,
    device: *mut c_void,
    max_render: [u32; 2],
    max_output: [u32; 2],
    flags: u32,
    usage: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GetRenderResolution {
    header: Header,
    display_width: u32,
    display_height: u32,
    quality_mode: u32,
    out_render_width: *mut u32,
    out_render_height: *mut u32,
}

#[repr(C)]
struct ProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

#[repr(C)]
struct OverrideVersion {
    header: Header,
    id: u64,
}

/// `ffxCreateContextDescFrameGeneration` (`0x2_0001`).
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateFrameGen {
    header: Header,
    flags: u32,
    display_size: [u32; 2],
    max_render_size: [u32; 2],
    back_buffer_format: u32,
}

/// `ffxConfigureDescFrameGeneration` (`0x2_0002`).
#[repr(C)]
#[derive(Clone, Copy)]
struct ConfigureFrameGen {
    header: Header,
    swap_chain: *mut c_void,
    present_callback: usize,
    present_callback_user_context: *mut c_void,
    frame_generation_callback: usize,
    frame_generation_callback_user_context: *mut c_void,
    frame_generation_enabled: bool,
    allow_async_workloads: bool,
    hudless_color: Resource,
    flags: u32,
    only_present_generated: bool,
    generation_rect: [i32; 4],
    frame_id: u64,
}

/// `ffxDispatchDescFrameGeneration` (`0x2_0003`).
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchFrameGen {
    header: Header,
    command_list: *mut c_void,
    present_color: Resource,
    outputs: [Resource; 4],
    num_generated_frames: u32,
    reset: bool,
    back_buffer_transfer_function: u32,
    min_max_luminance: [f32; 2],
    generation_rect: [i32; 4],
    frame_id: u64,
}

/// `ffxDispatchDescFrameGenerationPrepare` (`0x2_0004`).
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchFrameGenPrepare {
    header: Header,
    frame_id: u64,
    flags: u32,
    command_list: *mut c_void,
    render_size: [u32; 2],
    jitter_offset: [f32; 2],
    motion_vector_scale: [f32; 2],
    frame_time_delta: f32,
    unused_reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_vertical: f32,
    view_space_to_meters_factor: f32,
    depth: Resource,
    motion_vectors: Resource,
}

/// `ffxDispatchDescFrameGenerationPrepareV2` (`0x2_000c`).
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchFrameGenPrepareV2 {
    base: DispatchFrameGenPrepare,
    camera_position: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_forward: [f32; 3],
}

/// `ffxQueryDescFrameGenerationGetGPUMemoryUsageV2` (`0x2_000b`).
#[repr(C)]
#[derive(Clone, Copy)]
struct QueryFrameGenMemoryUsageV2 {
    header: Header,
    device: *mut c_void,
    max_render: [u32; 2],
    display_size: [u32; 2],
    create_flags: u32,
    dispatch_flags: u32,
    back_buffer_format: u32,
    hudless_format: u32,
    usage: *mut c_void,
}

// Compile-time ABI layout checks against the FidelityFX SDK 2.x headers.
const _: () = assert!(size_of::<Header>() == 0x10);
const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Backend>() == 0x18);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);
const _: () = assert!(size_of::<GetVersions>() == 0x38);
const _: () = assert!(size_of::<MemoryUsageWithDevice>() == 0x38);
const _: () = assert!(size_of::<GetRenderResolution>() == 0x30);
const _: () = assert!(size_of::<ProviderVersion>() == 0x20);
const _: () = assert!(size_of::<OverrideVersion>() == 0x18);
const _: () = assert!(size_of::<CreateFrameGen>() == 0x28);
const _: () = assert!(size_of::<ConfigureFrameGen>() == 0x90);
const _: () = assert!(size_of::<DispatchFrameGen>() == 0x138);
const _: () = assert!(size_of::<DispatchFrameGenPrepare>() == 0xb8);
const _: () = assert!(size_of::<DispatchFrameGenPrepareV2>() == 0xe8);
const _: () = assert!(size_of::<QueryFrameGenMemoryUsageV2>() == 0x40);

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

struct Host {
    _library: Library,
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
    provider_id: u64,
    provider_version: String,
}

struct Library(windows_sys::Win32::Foundation::HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this owns the reference acquired by LoadLibraryExW.
        unsafe {
            windows_sys::Win32::Foundation::FreeLibrary(self.0);
        }
    }
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

static HOST: OnceLock<Host> = OnceLock::new();
static FG_HOST: OnceLock<Host> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextKind {
    Upscale,
    FrameGen,
}

enum CreationDesc {
    Upscale(Box<CreateUpscale>),
    // Retained for the FFX context lifetime contract (pointers stay valid until destroy).
    FrameGen { _desc: Box<CreateFrameGen> },
}

struct Context {
    kind: ContextKind,
    // Serializes recording and destruction for this particular temporal history.
    live: Mutex<bool>,
    first_dispatch: std::sync::Once,
    // The FFX API requires pointers in the creation descriptor to remain
    // live through destroy, including our replacement backend.
    creation: CreateChain,
}

struct CreateChain {
    description: CreationDesc,
    _backend: Box<Backend>,
    _version: Box<OverrideVersion>,
}

// SAFETY: after Create, the boxed descriptors are retained immutably.
// The device is held by Host; the remaining extension pointers belong
// to the title under the FFX context lifetime contract.
unsafe impl Send for CreateChain {}
// SAFETY: as above, no thread mutates the retained descriptors.
unsafe impl Sync for CreateChain {}

// Typical active context count is 1-2; a tiny Vec under a shared read lock
// avoids exclusive mutex contention and tree traversal on the dispatch hot path.
static CONTEXTS: RwLock<Vec<(usize, Arc<Context>)>> = RwLock::new(Vec::new());

fn lookup_context(handle: usize) -> Option<Arc<Context>> {
    CONTEXTS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(h, _)| *h == handle)
        .map(|(_, c)| Arc::clone(c))
}

fn insert_context(handle: usize, ctx: Arc<Context>) {
    let mut contexts = CONTEXTS.write().unwrap_or_else(|e| e.into_inner());
    if let Some((_, slot)) = contexts.iter_mut().find(|(h, _)| *h == handle) {
        *slot = ctx;
    } else {
        contexts.push((handle, ctx));
    }
}

fn remove_context(handle: usize, ctx: &Arc<Context>) {
    let mut contexts = CONTEXTS.write().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = contexts
        .iter()
        .position(|(h, current)| *h == handle && Arc::ptr_eq(current, ctx))
    {
        contexts.swap_remove(pos);
    }
}

fn is_framegen_kind(kind: u64) -> bool {
    (kind & EFFECT_MASK) == EFFECT_FRAMEGEN
}

fn host_for_kind(kind: ContextKind) -> Option<&'static Host> {
    match kind {
        ContextKind::Upscale => HOST.get(),
        ContextKind::FrameGen => FG_HOST.get(),
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

fn strict_validate() -> bool {
    static VALIDATE: OnceLock<bool> = OnceLock::new();
    *VALIDATE.get_or_init(|| env_flag("VICTORY_FFX_VALIDATE"))
}

fn fits(extent: [u32; 2], maximum: [u32; 2]) -> bool {
    extent[0] != 0 && extent[1] != 0 && extent[0] <= maximum[0] && extent[1] <= maximum[1]
}

fn dispatch_output(
    render: [u32; 2],
    output: [u32; 2],
    max_render: [u32; 2],
    max_output: [u32; 2],
) -> Option<[u32; 2]> {
    let output = if output == [0, 0] { max_output } else { output };
    (fits(render, max_render) && fits(output, max_output) && fits(render, output)).then_some(output)
}

fn minimum_extent(index: usize, flags: u32, render: [u32; 2], output: [u32; 2]) -> [u32; 2] {
    match index {
        6 => output,
        2 if flags & DISPLAY_RESOLUTION_MOTION_VECTORS != 0 => output,
        3 => [1, 1], // Optional exposure texture.
        // Color (0), depth (1), render-res MVs (2), reactive (4), transparencyAndComposition (5).
        _ => render,
    }
}

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

fn sanitize_resource_desc(
    index: usize,
    r: &mut Resource,
    host_desc: &D3D12_RESOURCE_DESC,
    strict: bool,
) -> Result<(), &'static str> {
    let host_w = u32::try_from(host_desc.Width).unwrap_or(u32::MAX);
    let host_h = host_desc.Height;
    if r.description[DESC_WIDTH] > host_w || r.description[DESC_HEIGHT] > host_h {
        if strict {
            return Err("resource description extent exceeds host backing texture");
        }
        once!(
            warn,
            index,
            desc_size = ?[r.description[DESC_WIDTH], r.description[DESC_HEIGHT]],
            host_size = ?[host_w, host_h],
            "FidelityFX: clamping resource description extent to host backing texture size"
        );
        r.description[DESC_WIDTH] = r.description[DESC_WIDTH].min(host_w);
        r.description[DESC_HEIGHT] = r.description[DESC_HEIGHT].min(host_h);
    }

    let host_depth = u32::from(host_desc.DepthOrArraySize);
    if host_depth > 0 && r.description[DESC_DEPTH] > host_depth {
        if strict {
            return Err("resource description depth exceeds host texture depth");
        }
        once!(
            warn,
            index,
            desc_depth = r.description[DESC_DEPTH],
            host_depth,
            "FidelityFX: clamping resource description depth to host texture"
        );
        r.description[DESC_DEPTH] = host_depth;
    }

    let host_mips = u32::from(host_desc.MipLevels);
    if host_mips > 0 && r.description[DESC_MIP_COUNT] > host_mips {
        if strict {
            return Err("resource description mip count exceeds host texture mip levels");
        }
        once!(
            warn,
            index,
            desc_mips = r.description[DESC_MIP_COUNT],
            host_mips,
            "FidelityFX: clamping resource description mip count to host texture mip levels"
        );
        r.description[DESC_MIP_COUNT] = host_mips;
    }

    Ok(())
}

/// Validates host texture dimensions against the active dispatch rectangle and
/// clamps (or rejects in strict mode) oversized console resource descriptions
/// (extents, depth, mip count) so the PC SDK does not hit validation errors.
fn validate_resource_desc(
    index: usize,
    r: &mut Resource,
    host_desc: &D3D12_RESOURCE_DESC,
    flags: u32,
    render: [u32; 2],
    output: [u32; 2],
    strict: bool,
) -> Result<(), &'static str> {
    let needed = minimum_extent(index, flags, render, output);
    if host_desc.Width < u64::from(needed[0]) || host_desc.Height < needed[1] {
        return Err(if matches!(index, 4 | 5) {
            "optional mask host texture is smaller than render size"
        } else {
            "host texture cannot cover the active dispatch rectangle"
        });
    }
    sanitize_resource_desc(index, r, host_desc, strict)
}

/// Select the PC provider before the title's import registry is initialized.
pub fn use_host_dir(dir: &Path) -> Result<String, String> {
    let (host, version) = load(dir)?;
    HOST.set(host)
        .map_err(|_| String::from("FidelityFX provider already initialized"))?;
    tracing::info!(
        folder = %dir.display(),
        dll = %dir.join(DLL).display(),
        version = %version,
        "FidelityFX: loaded PC FSR upscaler provider"
    );

    if env_flag("VICTORY_FFX_FG") {
        let fg_path = std::env::var_os("VICTORY_FFX_FG_DLL")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| dir.join(FG_DLL));
        match load_dll(&fg_path, FG_DLL, CREATE_FRAMEGEN, None, false) {
            Ok((fg_host, fg_version)) => {
                if FG_HOST.set(fg_host).is_ok() {
                    tracing::info!(
                        dll = %fg_path.display(),
                        version = %fg_version,
                        "FidelityFX: loaded opt-in PC Frame Generation provider"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    dll = %fg_path.display(),
                    error = %err,
                    "FidelityFX: VICTORY_FFX_FG=1 requested but frame generation DLL could not be loaded; continuing with upscaler only"
                );
            }
        }
    }

    Ok(version)
}

/// Attempt to load the PC Frame Generation DLL from `dir` into `FG_HOST`.
pub fn load_frame_generation_from(dir: &Path) -> Result<String, String> {
    let (fg_host, fg_version) = load_dll(&dir.join(FG_DLL), FG_DLL, CREATE_FRAMEGEN, None, false)?;
    FG_HOST
        .set(fg_host)
        .map_err(|_| String::from("FidelityFX frame generation provider already initialized"))?;
    tracing::info!(
        folder = %dir.display(),
        dll = %dir.join(FG_DLL).display(),
        version = %fg_version,
        "FidelityFX: loaded PC Frame Generation provider"
    );
    Ok(fg_version)
}

/// Whether the host's FidelityFX upscaler serves the title's.
pub fn serving() -> bool {
    HOST.get().is_some()
}

/// Whether the host's opt-in FidelityFX Frame Generation provider is active.
pub fn frame_generation_serving() -> bool {
    FG_HOST.get().is_some()
}

fn load(dir: &Path) -> Result<(Host, String), String> {
    let requested = std::env::var("VICTORY_FFX_VERSION").ok();
    let strict = env_flag("VICTORY_FFX_STRICT_VERSION");
    load_dll(
        &dir.join(DLL),
        DLL,
        CREATE_UPSCALE,
        requested.as_deref(),
        strict,
    )
}

fn load_dll(
    dll_path: &Path,
    display_name: &str,
    create_kind: u64,
    requested: Option<&str>,
    strict: bool,
) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(dll_path).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!("no {display_name} there"));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{display_name} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let library = Library(module);
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{display_name} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the SDK's ffxQuery export, with its declared signature.
    let query = unsafe { std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?) };
    let (provider_id, version) = select_version(query, &device, create_kind, requested, strict)?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query,
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            _library: library,
            device,
            provider_id,
            provider_version: version.clone(),
        }
    };
    Ok((host, version))
}

fn parse_semver(name: &str) -> Option<(u32, u32, u32, u32)> {
    let start = name.find(|c: char| c.is_ascii_digit())?;
    let mut cursor = &name[start..];
    let mut parts = [0u32; 4];
    let mut count = 0usize;
    while count < 4 {
        let len = cursor.bytes().take_while(u8::is_ascii_digit).count();
        if len == 0 {
            break;
        }
        parts[count] = cursor[..len].parse().ok()?;
        count += 1;
        cursor = &cursor[len..];
        let Some(after_dot) = cursor.strip_prefix('.') else {
            break;
        };
        cursor = after_dot;
    }
    (count >= 2).then_some((parts[0], parts[1], parts[2], parts[3]))
}

fn version_matches_prefix(name: &str, prefix: &str) -> bool {
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        return false;
    }
    let candidate = name.trim();
    let rest = if let Some(r) = candidate.strip_prefix(trimmed) {
        r
    } else if let Some(start) = candidate.find(|c: char| c.is_ascii_digit()) {
        let Some(r) = candidate[start..].strip_prefix(trimmed) else {
            return false;
        };
        r
    } else {
        return false;
    };
    rest.is_empty() || !rest.starts_with(|c: char| c.is_ascii_digit())
}

fn is_fsr_4_1_2(name: &str) -> bool {
    version_matches_prefix(name, "4.1.2")
}

/// Pure provider version selection logic.
///
/// - Explicit `requested` (non-empty and not `"auto"`): selects the highest
///   semantic version matching that prefix at a version boundary.
/// - Unset or `"auto"`: prefers `4.1.2` if present; otherwise errors when
///   `strict` is true, or falls back to the highest available semantic version.
fn pick_provider_version(
    candidates: &[(u64, String)],
    requested: Option<&str>,
    strict: bool,
) -> Result<(u64, String), String> {
    if candidates.is_empty() {
        return Err(String::from("no FidelityFX provider versions available"));
    }
    let available = || {
        candidates
            .iter()
            .map(|(_, n)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let explicit = requested
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"));

    if let Some(req) = explicit {
        return candidates
            .iter()
            .filter(|(_, name)| version_matches_prefix(name, req))
            .max_by_key(|(_, name)| parse_semver(name))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "no FidelityFX provider matching '{req}' (available: {})",
                    available()
                )
            });
    }

    if let Some(preferred) = candidates
        .iter()
        .filter(|(_, name)| is_fsr_4_1_2(name))
        .max_by_key(|(_, name)| parse_semver(name))
    {
        return Ok(preferred.clone());
    }

    if strict {
        return Err(format!(
            "no FSR 4.1.2 provider for this adapter (available: {})",
            available()
        ));
    }

    candidates
        .iter()
        .max_by_key(|(_, name)| parse_semver(name))
        .cloned()
        .ok_or_else(|| format!("no FidelityFX provider available (available: {})", available()))
}

fn query_versions(
    query_fn: Call,
    device: &ID3D12Device,
    create_kind: u64,
) -> Result<Vec<(u64, String)>, String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind,
        device: device.as_raw(),
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { query_fn(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no provider for kind {create_kind:#x} on the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    let mut ids = vec![0u64; names.len()];
    query.names = names.as_mut_ptr();
    query.ids = ids.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    let code = unsafe { query_fn(null_mut(), &mut query.header) };
    if code != OK {
        return Err(format!(
            "provider version names could not be queried (code {code})"
        ));
    }
    let mut candidates = Vec::with_capacity(count as usize);
    for (id, name) in ids.into_iter().zip(names).take(count as usize) {
        if !name.is_null() {
            // SAFETY: NUL-terminated name owned by the SDK.
            let name = unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
            candidates.push((id, name));
        }
    }
    Ok(candidates)
}

/// The SDK supplies the version id to put in the context creation chain.
fn select_version(
    query_fn: Call,
    device: &ID3D12Device,
    create_kind: u64,
    requested: Option<&str>,
    strict: bool,
) -> Result<(u64, String), String> {
    let candidates = query_versions(query_fn, device, create_kind)?;
    let selected = pick_provider_version(&candidates, requested, strict)?;
    let explicit = requested
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"));
    if create_kind == CREATE_UPSCALE && explicit.is_none() && !is_fsr_4_1_2(&selected.1) {
        let available = candidates
            .iter()
            .map(|(_, n)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::warn!(
            selected = %selected.1,
            available = %available,
            "FidelityFX: FSR 4.1.2 unavailable for this adapter; auto-selected highest available provider"
        );
    } else if let Some(req) = explicit {
        tracing::info!(
            requested = %req,
            selected = %selected.1,
            "FidelityFX: selected provider version via VICTORY_FFX_VERSION"
        );
    }
    Ok(selected)
}

/// Puts the host backend after the title's creation description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
fn link_header(first: &mut Header, backend: &mut Backend) {
    let after = first.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    first.next = &raw mut backend.header;
}

fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    link_header(&mut upscale.header, backend);
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

fn verify_created_provider(
    host: &Host,
    context: *mut *mut c_void,
    memory: *const c_void,
) -> Result<(u64, String), u32> {
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    let query_code = unsafe { (host.query)(context, &mut version.header) };
    let name = if !version.name.is_null() {
        // SAFETY: a NUL-terminated name the DLL keeps.
        unsafe { CStr::from_ptr(version.name) }
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("?")
    };
    if query_code != OK || version.id != host.provider_id {
        tracing::warn!(
            query_code,
            expected = %host.provider_version,
            expected_id = format_args!("{:#x}", host.provider_id),
            actual = %name,
            actual_id = format_args!("{:#x}", version.id),
            "FidelityFX created a different provider than requested"
        );
        // SAFETY: the PC provider just created the context with this allocator.
        unsafe { (host.destroy)(context, memory) };
        return Err(NO_PROVIDER);
    }
    Ok((version.id, name))
}

extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    match kind {
        CREATE_UPSCALE => {
            let Some(host) = HOST.get() else {
                return NO_PROVIDER;
            };
            // SAFETY: an upscaler description, by its type.
            let mut upscale = Box::new(unsafe { desc.cast::<CreateUpscale>().read() });
            let mut backend = Box::new(Backend {
                header: Header {
                    kind: HOST_BACKEND,
                    next: null_mut(),
                },
                device: host.device.as_raw(),
            });
            link(&mut upscale, &mut backend);
            let mut requested = Box::new(OverrideVersion {
                header: Header {
                    kind: OVERRIDE_VERSION,
                    next: backend.header.next,
                },
                id: host.provider_id,
            });
            backend.header.next = &raw mut requested.header;
            // SAFETY: a PC chain; the title's later descriptions live for the call.
            let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
            if code != OK {
                tracing::error!(
                    code,
                    render = ?upscale.max_render,
                    output = ?upscale.max_output,
                    flags = format_args!("{:#x}", upscale.flags),
                    "FidelityFX: the host refused the title's upscaler"
                );
                return code;
            }
            let (provider_id, name) = match verify_created_provider(host, context, memory) {
                Ok(v) => v,
                Err(err) => return err,
            };
            tracing::info!(
                provider = %name,
                id = format_args!("{provider_id:#x}"),
                dimensions = format_args!(
                    "{}x{} -> {}x{}",
                    upscale.max_render[0],
                    upscale.max_render[1],
                    upscale.max_output[0],
                    upscale.max_output[1]
                ),
                render = ?upscale.max_render,
                output = ?upscale.max_output,
                flags = format_args!("{:#x}", upscale.flags),
                "FidelityFX: created PC FSR upscaler context"
            );
            // SAFETY: a successful SDK creation wrote an opaque context into this slot.
            insert_context(
                unsafe { *context } as usize,
                Arc::new(Context {
                    kind: ContextKind::Upscale,
                    live: Mutex::new(true),
                    first_dispatch: std::sync::Once::new(),
                    creation: CreateChain {
                        description: CreationDesc::Upscale(upscale),
                        _backend: backend,
                        _version: requested,
                    },
                }),
            );
            OK
        }
        CREATE_FRAMEGEN => {
            let Some(fg_host) = FG_HOST.get() else {
                once!(
                    warn,
                    "FidelityFX: frame generation context requested but PC Frame Generation is not active (set VICTORY_FFX_FG=1 with amd_fidelityfx_framegeneration_dx12.dll)"
                );
                return NO_PROVIDER;
            };
            // SAFETY: a frame generation creation description, by its type.
            let mut fg = Box::new(unsafe { desc.cast::<CreateFrameGen>().read() });
            let mut backend = Box::new(Backend {
                header: Header {
                    kind: HOST_BACKEND,
                    next: null_mut(),
                },
                device: fg_host.device.as_raw(),
            });
            link_header(&mut fg.header, &mut backend);
            let mut requested = Box::new(OverrideVersion {
                header: Header {
                    kind: OVERRIDE_VERSION,
                    next: backend.header.next,
                },
                id: fg_host.provider_id,
            });
            backend.header.next = &raw mut requested.header;
            // SAFETY: a PC chain; the title's later descriptions live for the call.
            let code = unsafe { (fg_host.create)(context, &mut fg.header, memory) };
            if code != OK {
                tracing::error!(
                    code,
                    display = ?fg.display_size,
                    max_render = ?fg.max_render_size,
                    flags = format_args!("{:#x}", fg.flags),
                    "FidelityFX: the host refused the title's frame generation context"
                );
                return code;
            }
            let (provider_id, name) = match verify_created_provider(fg_host, context, memory) {
                Ok(v) => v,
                Err(err) => return err,
            };
            tracing::info!(
                provider = %name,
                id = format_args!("{provider_id:#x}"),
                display = ?fg.display_size,
                max_render = ?fg.max_render_size,
                flags = format_args!("{:#x}", fg.flags),
                "FidelityFX: created PC Frame Generation context"
            );
            // SAFETY: a successful SDK creation wrote an opaque context into this slot.
            insert_context(
                unsafe { *context } as usize,
                Arc::new(Context {
                    kind: ContextKind::FrameGen,
                    live: Mutex::new(true),
                    first_dispatch: std::sync::Once::new(),
                    creation: CreateChain {
                        description: CreationDesc::FrameGen { _desc: fg },
                        _backend: backend,
                        _version: requested,
                    },
                }),
            );
            OK
        }
        _ => {
            once!(
                warn,
                kind = format_args!("{kind:#x}"),
                "FidelityFX: unsupported effect creation was refused"
            );
            NO_PROVIDER
        }
    }
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes its context. Capture its value before destroy clears it.
    let handle = unsafe { *context } as usize;
    let Some(ctx) = lookup_context(handle) else {
        return PARAMETER;
    };
    let Some(host) = host_for_kind(ctx.kind) else {
        return NO_PROVIDER;
    };
    let mut live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    // The guest releases a context after submitting its frames. The host
    // may still be executing their lists, even after guest Execute returns.
    victory_gpu::external::finish_submitted();
    // SAFETY: the host created this context; live lock prevents concurrent dispatch.
    let code = unsafe { (host.destroy)(context, memory) };
    if code == OK {
        *live = false;
        remove_context(handle, &ctx);
    }
    code
}

/// Settings (key-value pairs, debug messages) pass through; Frame Generation
/// configuration translates `HUDLessColor` when present.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a configuration description header.
    let kind = unsafe { (*desc).kind };
    let host = if !context.is_null() {
        // SAFETY: non-null context slot.
        lookup_context(unsafe { *context } as usize)
            .and_then(|ctx| host_for_kind(ctx.kind))
            .or_else(|| {
                if is_framegen_kind(kind) {
                    FG_HOST.get()
                } else {
                    HOST.get()
                }
            })
    } else if is_framegen_kind(kind) {
        FG_HOST.get()
    } else {
        HOST.get()
    };
    let Some(host) = host else {
        return NO_PROVIDER;
    };

    if kind == CONFIGURE_FRAMEGEN {
        // SAFETY: a frame generation configuration description, by its type.
        let mut cfg = unsafe { desc.cast::<ConfigureFrameGen>().read() };
        let mut _hudless_keepalive: Option<ID3D12Resource> = None;
        if !cfg.hudless_color.resource.is_null() {
            let guest = cfg.hudless_color.resource.cast::<Object>().cast_const();
            if let Some(host_tex) = victory_gpu::external::host_texture(guest) {
                // SAFETY: live D3D12 resource.
                let host_desc = unsafe { host_tex.GetDesc() };
                if let Err(reason) = sanitize_resource_desc(
                    0,
                    &mut cfg.hudless_color,
                    &host_desc,
                    strict_validate(),
                ) {
                    once!(warn, reason, "FidelityFX: rejected FG HUDLessColor description");
                    return PARAMETER;
                }
                cfg.hudless_color.resource = host_tex.as_raw();
                _hudless_keepalive = Some(host_tex);
            } else {
                once!(
                    warn,
                    "FidelityFX: FG HUDLessColor had no host copy; clearing HUDLessColor for this configure call"
                );
                cfg.hudless_color.resource = null_mut();
            }
        }
        // SAFETY: translated FG configure struct; keepalive holds host HUDLessColor.
        return unsafe { (host.configure)(context, &raw mut cfg.header) };
    }

    // SAFETY: the title passes its context and descriptions.
    unsafe { (host.configure)(context, desc) }
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    // Substitute the native device; the SDK returns values through the
    // title's original output pointers in these request structures.
    if kind == GET_VERSIONS {
        // SAFETY: GetVersions by its type.
        let mut query = unsafe { *desc.cast::<GetVersions>() };
        let host = if query.create_kind == CREATE_FRAMEGEN {
            FG_HOST.get()
        } else {
            HOST.get()
        };
        let Some(host) = host else {
            return NO_PROVIDER;
        };
        query.device = host.device.as_raw();
        // SAFETY: query with host device substituted.
        return unsafe { (host.query)(context, &mut query.header) };
    }
    if kind == MEMORY_USAGE_WITH_DEVICE {
        let Some(host) = HOST.get() else {
            return NO_PROVIDER;
        };
        // SAFETY: MemoryUsageWithDevice by its type.
        let mut query = unsafe { *desc.cast::<MemoryUsageWithDevice>() };
        query.device = host.device.as_raw();
        // SAFETY: query with host device substituted.
        return unsafe { (host.query)(context, &mut query.header) };
    }
    if kind == QUERY_FRAMEGEN_MEMORY_USAGE_V2 {
        let Some(fg_host) = FG_HOST.get() else {
            return NO_PROVIDER;
        };
        // SAFETY: QueryFrameGenMemoryUsageV2 by its type.
        let mut query = unsafe { *desc.cast::<QueryFrameGenMemoryUsageV2>() };
        query.device = fg_host.device.as_raw();
        // SAFETY: query with host device substituted.
        return unsafe { (fg_host.query)(context, &mut query.header) };
    }

    let host = if !context.is_null() {
        // SAFETY: non-null context slot.
        lookup_context(unsafe { *context } as usize)
            .and_then(|ctx| host_for_kind(ctx.kind))
            .or_else(|| {
                if is_framegen_kind(kind) {
                    FG_HOST.get()
                } else {
                    HOST.get()
                }
            })
    } else if is_framegen_kind(kind) {
        FG_HOST.get()
    } else {
        HOST.get()
    };
    let Some(host) = host else {
        return NO_PROVIDER;
    };

    // SAFETY: as above.
    let code = unsafe { (host.query)(context, desc) };
    if kind == GET_RENDER_RESOLUTION && code != OK {
        tracing::warn!(
            code,
            "FidelityFX provider refused the render-resolution query"
        );
    }
    code
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    match kind {
        DISPATCH_UPSCALE => dispatch_upscale(context, desc),
        DISPATCH_FRAMEGEN => dispatch_framegen(context, desc),
        DISPATCH_FRAMEGEN_PREPARE => dispatch_framegen_prepare(context, desc),
        DISPATCH_FRAMEGEN_PREPARE_V2 => dispatch_framegen_prepare_v2(context, desc),
        _ => {
            once!(
                warn,
                kind = format_args!("{kind:#x}"),
                "FidelityFX: unsupported dispatch type was refused"
            );
            UNKNOWN_TYPE
        }
    }
}

fn dispatch_upscale(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    // SAFETY: an upscaling, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    if context.is_null() {
        once!(warn, "FidelityFX dispatch arrived without a context");
        return PARAMETER;
    }
    if let Some(index) = [0, 1, 2, 6]
        .into_iter()
        .find(|&i| dispatch.resources[i].resource.is_null())
    {
        once!(
            warn,
            index,
            "FidelityFX dispatch is missing a required texture"
        );
        return PARAMETER;
    }
    // SAFETY: the title passes the same opaque context slot as for creation.
    let handle = unsafe { *context } as usize;
    let Some(ctx) = lookup_context(handle) else {
        once!(
            warn,
            "FidelityFX dispatch referenced an unregistered context"
        );
        return PARAMETER;
    };
    let CreationDesc::Upscale(create) = &ctx.creation.description else {
        once!(
            warn,
            "FidelityFX upscale dispatch referenced a non-upscale context"
        );
        return PARAMETER;
    };
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let Some(output_size) = dispatch_output(
        dispatch.render_size,
        dispatch.upscale_size,
        create.max_render,
        create.max_output,
    ) else {
        once!(
            warn,
            render = ?dispatch.render_size,
            output = ?dispatch.upscale_size,
            max_render = ?create.max_render,
            max_output = ?create.max_output,
            "FidelityFX rejected dispatch dimensions outside the context's allocations"
        );
        return PARAMETER;
    };
    let list = dispatch.list.cast::<Object>().cast_const();
    victory_gpu::external::record(list, |list| {
        record_dispatch(list, host, context, &ctx, create, &mut dispatch, output_size)
    })
    .unwrap_or_else(|| {
        once!(
            warn,
            "FidelityFX could not record to the title command list"
        );
        PARAMETER
    })
}

fn record_dispatch(
    list: &victory_gpu::external::Recording<'_>,
    host: &Host,
    context: *mut *mut c_void,
    ctx: &Context,
    create: &CreateUpscale,
    dispatch: &mut DispatchUpscale,
    output_size: [u32; 2],
) -> u32 {
    let output_guest = dispatch.resources[6].resource.cast::<Object>();
    // Alive for the call (the list holds them too); stack-allocated, zero heap allocs.
    let mut textures: [Option<ID3D12Resource>; 7] = [const { None }; 7];
    let mut backing_extents = [None; 7];
    let mut host_formats = [None; 7];
    let strict = strict_validate();

    for (index, r) in dispatch.resources.iter_mut().enumerate() {
        if r.resource.is_null() {
            continue;
        }
        let Some(wanted) = host_state(r.state) else {
            once!(
                warn,
                index,
                state = r.state,
                "FidelityFX: a resource has an unsupported host state; the upscaling did not run"
            );
            return PARAMETER;
        };
        let guest_obj = r.resource.cast::<Object>();
        let Some(prepared) = list.prepare_texture(guest_obj, wanted, index < 6) else {
            once!(
                warn,
                index,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return PARAMETER;
        };
        backing_extents[index] = Some((prepared.desc.Width, prepared.desc.Height));
        host_formats[index] = Some(prepared.desc.Format.0);
        if let Err(reason) = validate_resource_desc(
            index,
            r,
            &prepared.desc,
            create.flags,
            dispatch.render_size,
            output_size,
            strict,
        ) {
            once!(
                warn,
                index,
                reason,
                width = prepared.desc.Width,
                height = prepared.desc.Height,
                "FidelityFX resource failed validation"
            );
            return PARAMETER;
        }
        r.resource = prepared.resource.as_raw();
        textures[index] = Some(prepared.resource);
    }

    once!(
        info,
        formats = ?host_formats,
        "FidelityFX: first upscaling, host formats of its textures"
    );
    // Flushes all queued state transitions in one batched ResourceBarrier call.
    dispatch.list = list.commands().as_raw();
    // SAFETY: the PC dispatch on host objects. The context is live and locked.
    let code = unsafe { (host.dispatch)(context, &raw mut dispatch.header) };
    if code == OK {
        victory_gpu::external::output_written(output_guest);
        ctx.first_dispatch.call_once(|| {
            tracing::info!(
                provider = %host.provider_version,
                dimensions = format_args!(
                    "{}x{} -> {}x{}",
                    dispatch.render_size[0],
                    dispatch.render_size[1],
                    output_size[0],
                    output_size[1]
                ),
                render = ?dispatch.render_size,
                output = ?output_size,
                ?backing_extents,
                jitter = ?dispatch.jitter,
                motion_vector_scale = ?dispatch.motion_scale,
                "FidelityFX recorded the title's temporal dispatch"
            );
        });
    } else {
        once!(
            warn,
            code,
            provider = %host.provider_version,
            "FidelityFX provider rejected the title dispatch"
        );
    }
    code
}

fn prepare_fg_resource(
    list: &victory_gpu::external::Recording<'_>,
    index: usize,
    r: &mut Resource,
    is_input: bool,
    strict: bool,
) -> Result<Option<ID3D12Resource>, u32> {
    if r.resource.is_null() {
        return Ok(None);
    }
    let Some(wanted) = host_state(r.state) else {
        once!(
            warn,
            index,
            state = r.state,
            "FidelityFX FG: resource has an unsupported host state"
        );
        return Err(PARAMETER);
    };
    let guest_obj = r.resource.cast::<Object>();
    let Some(prepared) = list.prepare_texture(guest_obj, wanted, is_input) else {
        once!(
            warn,
            index,
            state = r.state,
            "FidelityFX FG: resource has no host texture in its state"
        );
        return Err(PARAMETER);
    };
    if let Err(reason) = sanitize_resource_desc(index, r, &prepared.desc, strict) {
        once!(warn, index, reason, "FidelityFX FG: resource failed validation");
        return Err(PARAMETER);
    }
    r.resource = prepared.resource.as_raw();
    Ok(Some(prepared.resource))
}

fn dispatch_framegen(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(fg_host) = FG_HOST.get() else {
        return NO_PROVIDER;
    };
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: a frame generation dispatch, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchFrameGen>().read() };
    if dispatch.present_color.resource.is_null()
        || dispatch.num_generated_frames == 0
        || dispatch.num_generated_frames > 4
    {
        return PARAMETER;
    }
    let out_count = dispatch.num_generated_frames as usize;
    if dispatch.outputs[..out_count]
        .iter()
        .any(|o| o.resource.is_null())
    {
        return PARAMETER;
    }
    // SAFETY: non-null context slot.
    let handle = unsafe { *context } as usize;
    let Some(ctx) = lookup_context(handle) else {
        return PARAMETER;
    };
    if ctx.kind != ContextKind::FrameGen {
        return PARAMETER;
    }
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let list = dispatch.command_list.cast::<Object>().cast_const();
    victory_gpu::external::record(list, |list| {
        let strict = strict_validate();
        let mut textures: [Option<ID3D12Resource>; 5] = [const { None }; 5];
        let mut guest_outputs: [*const Object; 4] = [null(); 4];
        match prepare_fg_resource(list, 0, &mut dispatch.present_color, true, strict) {
            Ok(tex) => textures[0] = tex,
            Err(code) => return code,
        }
        for i in 0..out_count {
            guest_outputs[i] = dispatch.outputs[i].resource.cast::<Object>().cast_const();
            match prepare_fg_resource(list, i + 1, &mut dispatch.outputs[i], false, strict) {
                Ok(tex) => textures[i + 1] = tex,
                Err(code) => return code,
            }
        }
        dispatch.command_list = list.commands().as_raw();
        // SAFETY: PC FG dispatch on host objects; context is live and locked.
        let code = unsafe { (fg_host.dispatch)(context, &raw mut dispatch.header) };
        if code == OK {
            for &out in &guest_outputs[..out_count] {
                if !out.is_null() {
                    victory_gpu::external::output_written(out);
                }
            }
            ctx.first_dispatch.call_once(|| {
                tracing::info!(
                    provider = %fg_host.provider_version,
                    generated = dispatch.num_generated_frames,
                    frame_id = dispatch.frame_id,
                    "FidelityFX Frame Generation recorded its first dispatch"
                );
            });
        }
        code
    })
    .unwrap_or(PARAMETER)
}

fn record_framegen_prepare_resources(
    list: &victory_gpu::external::Recording<'_>,
    prepare: &mut DispatchFrameGenPrepare,
    textures: &mut [Option<ID3D12Resource>; 2],
) -> Result<(), u32> {
    let strict = strict_validate();
    textures[0] = prepare_fg_resource(list, 0, &mut prepare.depth, true, strict)?;
    textures[1] = prepare_fg_resource(list, 1, &mut prepare.motion_vectors, true, strict)?;
    prepare.command_list = list.commands().as_raw();
    Ok(())
}

fn dispatch_framegen_prepare(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(fg_host) = FG_HOST.get() else {
        return NO_PROVIDER;
    };
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: DispatchFrameGenPrepare by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchFrameGenPrepare>().read() };
    // SAFETY: non-null context slot.
    let handle = unsafe { *context } as usize;
    let Some(ctx) = lookup_context(handle) else {
        return PARAMETER;
    };
    if ctx.kind != ContextKind::FrameGen {
        return PARAMETER;
    }
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let list = dispatch.command_list.cast::<Object>().cast_const();
    victory_gpu::external::record(list, |list| {
        let mut textures: [Option<ID3D12Resource>; 2] = [const { None }; 2];
        if let Err(code) = record_framegen_prepare_resources(list, &mut dispatch, &mut textures) {
            return code;
        }
        // SAFETY: PC FG prepare dispatch on host objects; context is live and locked.
        unsafe { (fg_host.dispatch)(context, &raw mut dispatch.header) }
    })
    .unwrap_or(PARAMETER)
}

fn dispatch_framegen_prepare_v2(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(fg_host) = FG_HOST.get() else {
        return NO_PROVIDER;
    };
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: DispatchFrameGenPrepareV2 by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchFrameGenPrepareV2>().read() };
    // SAFETY: non-null context slot.
    let handle = unsafe { *context } as usize;
    let Some(ctx) = lookup_context(handle) else {
        return PARAMETER;
    };
    if ctx.kind != ContextKind::FrameGen {
        return PARAMETER;
    }
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let list = dispatch.base.command_list.cast::<Object>().cast_const();
    victory_gpu::external::record(list, |list| {
        let mut textures: [Option<ID3D12Resource>; 2] = [const { None }; 2];
        if let Err(code) =
            record_framegen_prepare_resources(list, &mut dispatch.base, &mut textures)
        {
            return code;
        }
        // SAFETY: PC FG prepare V2 dispatch on host objects; context is live and locked.
        unsafe { (fg_host.dispatch)(context, &raw mut dispatch.base.header) }
    })
    .unwrap_or(PARAMETER)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_host_desc(width: u64, height: u32, depth: u16, mips: u16) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Width: width,
            Height: height,
            DepthOrArraySize: depth,
            MipLevels: mips,
            ..Default::default()
        }
    }

    fn make_resource(width: u32, height: u32, depth: u32, mips: u32) -> Resource {
        Resource {
            resource: 0x1000 as *mut c_void,
            description: [2, 10, width, height, depth, mips, 0, 0],
            state: 0x4,
        }
    }

    #[test]
    fn validates_logical_rectangles_against_the_current_context() {
        for (render, display) in [
            ([1920, 1080], [2560, 1440]),
            ([1706, 960], [2560, 1440]),
            ([2560, 1440], [3840, 2160]),
            ([1920, 1080], [3840, 2160]),
        ] {
            assert_eq!(
                dispatch_output(render, display, render, display),
                Some(display)
            );
            assert_eq!(
                dispatch_output(render, [0, 0], render, display),
                Some(display)
            );
            assert_eq!(minimum_extent(2, 0x29, render, display), render);
            assert_eq!(minimum_extent(6, 0x29, render, display), display);
        }
        for display in [[1920, 1080], [2560, 1440], [3840, 2160]] {
            for divisor in [1.0_f32, 1.5, 1.7, 2.0, 3.0] {
                let render = display.map(|v| (v as f32 / divisor) as u32);
                assert_eq!(
                    dispatch_output(render, display, render, display),
                    Some(display)
                );
                assert_eq!(
                    dispatch_output(render, [0, 0], render, display),
                    Some(display)
                );
                assert_eq!(minimum_extent(2, 0x29, render, display), render);
                assert_eq!(minimum_extent(2, 0x2b, render, display), display);
                assert_eq!(minimum_extent(6, 0x29, render, display), display);
                assert_eq!(minimum_extent(3, 0x29, render, display), [1, 1]);
                assert_eq!(minimum_extent(4, 0x29, render, display), render);
                assert_eq!(minimum_extent(5, 0x29, render, display), render);
            }
        }
        assert_eq!(
            dispatch_output([0, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1281, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [0, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [3000, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [960, 540], [1280, 720], [2560, 1440]),
            None
        );
    }

    #[test]
    fn validates_and_clamps_resource_descriptions_and_optional_masks() {
        let render = [1280, 720];
        let output = [1920, 1080];

        // Valid color, exposure, reactive mask, transparency mask, output.
        let mut color = make_resource(1280, 720, 1, 1);
        assert!(validate_resource_desc(
            0,
            &mut color,
            &make_host_desc(1280, 720, 1, 1),
            0,
            render,
            output,
            true,
        )
        .is_ok());

        let mut exposure = make_resource(1, 1, 1, 0);
        assert!(validate_resource_desc(
            3,
            &mut exposure,
            &make_host_desc(1, 1, 1, 1),
            0,
            render,
            output,
            true,
        )
        .is_ok());
        assert_eq!(exposure.description[DESC_MIP_COUNT], 0);

        // Undersized reactive mask (index 4) and transparency mask (index 5) rejected.
        let mut small_reactive = make_resource(640, 360, 1, 1);
        assert!(validate_resource_desc(
            4,
            &mut small_reactive,
            &make_host_desc(640, 360, 1, 1),
            0,
            render,
            output,
            false,
        )
        .is_err());
        let mut small_trans = make_resource(1280, 719, 1, 1);
        assert!(validate_resource_desc(
            5,
            &mut small_trans,
            &make_host_desc(1280, 719, 1, 1),
            0,
            render,
            output,
            false,
        )
        .is_err());

        // Oversized reactive mask descriptor extent: clamped when non-strict, rejected when strict.
        let host_render = make_host_desc(1280, 720, 1, 2);
        let mut oversized_reactive = make_resource(1920, 1080, 1, 1);
        assert!(validate_resource_desc(
            4,
            &mut oversized_reactive,
            &host_render,
            0,
            render,
            output,
            true,
        )
        .is_err());
        assert!(validate_resource_desc(
            4,
            &mut oversized_reactive,
            &host_render,
            0,
            render,
            output,
            false,
        )
        .is_ok());
        assert_eq!(oversized_reactive.description[DESC_WIDTH], 1280);
        assert_eq!(oversized_reactive.description[DESC_HEIGHT], 720);

        // Oversized mip count & depth: clamped when non-strict, rejected when strict.
        let mut over_mips = make_resource(1280, 720, 4, 9);
        assert!(validate_resource_desc(
            0,
            &mut over_mips,
            &host_render,
            0,
            render,
            output,
            true,
        )
        .is_err());
        assert!(validate_resource_desc(
            0,
            &mut over_mips,
            &host_render,
            0,
            render,
            output,
            false,
        )
        .is_ok());
        assert_eq!(over_mips.description[DESC_DEPTH], 1);
        assert_eq!(over_mips.description[DESC_MIP_COUNT], 2);
    }

    #[test]
    fn picks_provider_version_by_override_preference_and_fallback() {
        let mixed = vec![
            (1, String::from("3.1.4")),
            (2, String::from("3.1.5")),
            (3, String::from("4.1.1")),
            (4, String::from("4.1.2")),
            (5, String::from("4.10.0")),
        ];

        // Exact match.
        assert_eq!(
            pick_provider_version(&mixed, Some("3.1.4"), false).unwrap(),
            (1, String::from("3.1.4"))
        );
        // Prefix match respects version boundaries ("4.1" picks "4.1.2", NOT "4.10.0").
        assert_eq!(
            pick_provider_version(&mixed, Some("4.1"), false).unwrap(),
            (4, String::from("4.1.2"))
        );
        // Prefix match picks highest semantic match ("3.1" picks "3.1.5").
        assert_eq!(
            pick_provider_version(&mixed, Some("3.1"), false).unwrap(),
            (2, String::from("3.1.5"))
        );
        // Auto-selection prefers 4.1.2 when present.
        assert_eq!(
            pick_provider_version(&mixed, None, false).unwrap(),
            (4, String::from("4.1.2"))
        );
        assert_eq!(
            pick_provider_version(&mixed, Some("auto"), true).unwrap(),
            (4, String::from("4.1.2"))
        );
        assert_eq!(
            pick_provider_version(&mixed, Some("AUTO"), true).unwrap(),
            (4, String::from("4.1.2"))
        );

        // When 4.1.2 is absent: non-strict falls back to highest semver; strict fails.
        let fsr3_only = vec![
            (10, String::from("2.3.2")),
            (11, String::from("3.1.4")),
            (12, String::from("3.1.5")),
        ];
        assert_eq!(
            pick_provider_version(&fsr3_only, None, false).unwrap(),
            (12, String::from("3.1.5"))
        );
        let strict_err = pick_provider_version(&fsr3_only, None, true).unwrap_err();
        assert!(strict_err.contains("4.1.2"));
        assert!(strict_err.contains("3.1.5"));

        // Invalid override returns error listing available versions.
        let bad_err = pick_provider_version(&fsr3_only, Some("4.1.2"), false).unwrap_err();
        assert!(bad_err.contains("4.1.2"));
        assert!(bad_err.contains("3.1.4"));

        // Empty list errors cleanly.
        assert!(pick_provider_version(&[], None, false).is_err());

        // 4-part semver sorts higher than 3-part base.
        let patch4 = vec![
            (20, String::from("4.1.2")),
            (21, String::from("4.1.2.1")),
        ];
        assert_eq!(
            pick_provider_version(&patch4, None, false).unwrap(),
            (21, String::from("4.1.2.1"))
        );
    }

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    #[test]
    fn framegen_abi_linking_and_missing_dll_graceful() {
        // Verify FrameGen creation chain linking replaces CONSOLE_BACKEND.
        let mut ext = header(0x2_000e, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut ext),
            device: 0x1111 as *mut c_void,
        };
        let mut fg = CreateFrameGen {
            header: header(CREATE_FRAMEGEN, &raw mut console.header),
            flags: 0,
            display_size: [1920, 1080],
            max_render_size: [1280, 720],
            back_buffer_format: 10,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x2222 as *mut c_void,
        };
        link_header(&mut fg.header, &mut backend);
        assert_eq!(fg.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut ext);

        assert!(is_framegen_kind(CREATE_FRAMEGEN));
        assert!(is_framegen_kind(CONFIGURE_FRAMEGEN));
        assert!(is_framegen_kind(DISPATCH_FRAMEGEN));
        assert!(is_framegen_kind(DISPATCH_FRAMEGEN_PREPARE));
        assert!(is_framegen_kind(DISPATCH_FRAMEGEN_PREPARE_V2));
        assert!(!is_framegen_kind(CREATE_UPSCALE));
        assert!(!is_framegen_kind(DISPATCH_UPSCALE));

        // Missing FG DLL fails cleanly without mutating FG_HOST.
        let dir = tempfile::tempdir().unwrap();
        let err = load_frame_generation_from(dir.path()).unwrap_err();
        assert!(err.contains(FG_DLL), "unexpected error: {err}");
        assert!(!frame_generation_serving());
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).err().unwrap().contains(DLL));
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }

    #[test]
    fn real_fidelityfx_dll_queries_versions_and_creates_context() {
        let dir = std::env::var_os("VICTORY_FIDELITYFX_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\Victory\fidelityfx-4.1.2"));
        let dll_path = dir.join(DLL);
        if !dll_path.is_file() {
            eprintln!(
                "skipping real_fidelityfx_dll_queries_versions_and_creates_context: {} not present",
                dll_path.display()
            );
            return;
        }

        // Load locally without mutating global HOST so other unit tests stay isolated.
        let (host, selected_version) = load(&dir).expect("failed to load real FidelityFX DLL");
        let all_versions = query_versions(host.query, &host.device, CREATE_UPSCALE)
            .expect("failed to query upscaler versions from real DLL");
        eprintln!(
            "FidelityFX real DLL selected: {selected_version} (id={:#x}); offered versions: {:?}",
            host.provider_id, all_versions
        );
        assert!(!all_versions.is_empty());

        // Create a real 1280x720 -> 1920x1080 upscaler context.
        let mut override_ver = OverrideVersion {
            header: header(OVERRIDE_VERSION, null_mut()),
            id: host.provider_id,
        };
        let mut backend = Backend {
            header: header(HOST_BACKEND, &raw mut override_ver.header),
            device: host.device.as_raw(),
        };
        let mut create = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut backend.header),
            flags: 0,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut ctx: *mut c_void = null_mut();
        // SAFETY: valid PC creation chain targeting host D3D12 device.
        let create_code = unsafe { (host.create)(&mut ctx, &mut create.header, null()) };
        assert_eq!(create_code, OK, "ffxCreateContext failed with {create_code}");
        assert!(!ctx.is_null());

        // Query GET_PROVIDER_VERSION.
        let mut provider = ProviderVersion {
            header: header(GET_PROVIDER_VERSION, null_mut()),
            id: 0,
            name: null(),
        };
        // SAFETY: valid query on created context.
        let ver_code = unsafe { (host.query)(&mut ctx, &mut provider.header) };
        assert_eq!(ver_code, OK, "GET_PROVIDER_VERSION failed with {ver_code}");
        assert_eq!(provider.id, host.provider_id);
        assert!(!provider.name.is_null());
        // SAFETY: SDK-owned NUL-terminated string.
        let active_name = unsafe { CStr::from_ptr(provider.name) }.to_string_lossy();
        eprintln!("FidelityFX active context provider: {active_name}");

        // Query GET_RENDER_RESOLUTION (Quality mode = 1 -> 1.5x => 1920x1080 -> 1280x720).
        let mut out_w = 0u32;
        let mut out_h = 0u32;
        let mut res_query = GetRenderResolution {
            header: header(GET_RENDER_RESOLUTION, null_mut()),
            display_width: 1920,
            display_height: 1080,
            quality_mode: 1,
            out_render_width: &mut out_w,
            out_render_height: &mut out_h,
        };
        // SAFETY: valid query on created context.
        let res_code = unsafe { (host.query)(&mut ctx, &mut res_query.header) };
        assert_eq!(res_code, OK, "GET_RENDER_RESOLUTION failed with {res_code}");
        eprintln!("FidelityFX GET_RENDER_RESOLUTION(1920x1080, Quality) -> {out_w}x{out_h}");
        assert!(out_w > 0 && out_w <= 1920);
        assert!(out_h > 0 && out_h <= 1080);

        // Cleanly destroy the context.
        // SAFETY: matching destroy call for the created context.
        let destroy_code = unsafe { (host.destroy)(&mut ctx, null()) };
        assert_eq!(destroy_code, OK, "ffxDestroyContext failed with {destroy_code}");
    }
}
