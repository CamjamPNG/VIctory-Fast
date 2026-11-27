//! Overrides of shared Windows APIs whose console meaning differs.
//!
//! Only the differences live here; everything else passes through to the
//! host (see `HostPolicy`).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};

use crate::{HleModule, HostPolicy, hle_export, registry, runtime};

/// `GetModuleHandleW(NULL)` must return the title's image, not our exe;
/// title images and our console modules are found by name.
pub extern "system" fn GetModuleHandleW(name: *const u16) -> *mut c_void {
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    find_module(&wide(name), false).unwrap_or_else(|| host::get_module_handle_w(name))
}

/// See [`GetModuleHandleW`].
pub extern "system" fn GetModuleHandleA(name: *const u8) -> *mut c_void {
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    find_module(&narrow(name), false).unwrap_or_else(|| host::get_module_handle_a(name))
}

/// A NUL-terminated string from the title.
fn narrow(text: *const u8) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe { CStr::from_ptr(text.cast()) }
        .to_string_lossy()
        .into_owned()
}

/// See [`narrow`].
fn wide(text: *const u16) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    }
}

/// A title image's path is its original file, not our cache copy;
/// `NULL` means the title's exe.
pub extern "system" fn GetModuleFileNameW(module: *mut c_void, out: *mut u16, size: u32) -> u32 {
    let Some(path) = title_path(module) else {
        return host::get_module_file_name_w(module, out, size);
    };
    let wide: Vec<u16> = path.encode_utf16().collect();
    copy_truncated(&wide, title_buffer(out, size))
}

/// See [`GetModuleFileNameW`].
pub extern "system" fn GetModuleFileNameA(module: *mut c_void, out: *mut u8, size: u32) -> u32 {
    let Some(path) = title_path(module) else {
        return host::get_module_file_name_a(module, out, size);
    };
    copy_truncated(path.as_bytes(), title_buffer(out, size))
}

/// The title's buffer of `size` elements; none for a null pointer (a slice
/// may not start at null, even an empty one, so `(NULL, 0)` would be
/// undefined behaviour).
fn title_buffer<'a, T>(out: *mut T, size: u32) -> &'a mut [T] {
    if out.is_null() {
        return &mut [];
    }
    // SAFETY: the title passes a buffer of `size` elements.
    unsafe { std::slice::from_raw_parts_mut(out, size as usize) }
}

fn title_path(module: *mut c_void) -> Option<String> {
    let base = if module.is_null() {
        runtime::main_image()?
    } else {
        module as usize
    };
    Some(runtime::image_path(base)?.to_string_lossy().into_owned())
}

/// `GetModuleFileName`'s contract: copy with a NUL; if it does not fit,
/// truncate to the buffer, report `ERROR_INSUFFICIENT_BUFFER` and return
/// the buffer size.
fn copy_truncated<T: Copy + Default>(text: &[T], out: &mut [T]) -> u32 {
    let Some(room) = out.len().checked_sub(1) else {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    };
    let n = text.len().min(room);
    out[..n].copy_from_slice(&text[..n]);
    out[n] = T::default();
    if n < text.len() {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return out.len() as u32;
    }
    n as u32
}

const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

/// Console-only DLLs we implement load as a pseudo handle (titles load
/// `xgameruntime.dll` this way), the title's DLLs as the images we map
/// (on first use if nothing imports them); anything else loads on the host.
pub extern "system" fn LoadLibraryExA(
    name: *const u8,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    if !name.is_null()
        && let Some(handle) = find_module(&narrow(name), runs_code(flags))
    {
        return handle;
    }
    host::load_library_ex_a(name, file, flags)
}

/// See [`LoadLibraryExA`].
pub extern "system" fn LoadLibraryExW(
    name: *const u16,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    if !name.is_null()
        && let Some(handle) = find_module(&wide(name), runs_code(flags))
    {
        return handle;
    }
    host::load_library_ex_w(name, file, flags)
}

/// False for loads as data or resources (`DONT_RESOLVE_DLL_REFERENCES`,
/// `LOAD_LIBRARY_AS_DATAFILE(_EXCLUSIVE)`, `AS_IMAGE_RESOURCE`): those run
/// no code, so a title DLL we have not mapped can go to the host.
fn runs_code(flags: u32) -> bool {
    flags & (0x1 | 0x2 | 0x20 | 0x40) == 0
}

pub extern "system" fn LoadLibraryA(name: *const u8) -> *mut c_void {
    LoadLibraryExA(name, std::ptr::null_mut(), 0)
}

pub extern "system" fn LoadLibraryW(name: *const u16) -> *mut c_void {
    LoadLibraryExW(name, std::ptr::null_mut(), 0)
}

/// Looks up our implementations behind a pseudo handle; real modules go to
/// the host.
pub extern "system" fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void {
    // `NULL` is the process's exe, which for the title is its own image.
    let module = if module.is_null() {
        runtime::main_image().unwrap_or(0) as *mut c_void
    } else {
        module
    };
    let Some(m) = registry().iter().find(|m| handle(m) == module) else {
        return host_override(module, name).unwrap_or_else(|| host::get_proc_address(module, name));
    };
    // Values below 0x10000 are ordinals; we export by name only.
    if (name as usize) < 0x1_0000 {
        tracing::warn!(
            module = m.name,
            ordinal = name as usize,
            "GetProcAddress by ordinal"
        );
        return std::ptr::null_mut();
    }
    // SAFETY: not an ordinal, so the title passes a NUL-terminated name.
    let name = unsafe { CStr::from_ptr(name.cast()) }.to_string_lossy();
    match m.exports.iter().find(|e| e.name == name) {
        Some(e) => e.address as *mut c_void,
        None => {
            tracing::warn!(module = m.name, %name, "GetProcAddress: not implemented");
            std::ptr::null_mut()
        }
    }
}

/// What a host module's `name` (or ordinal) is bound to at import binding
/// when we override it: a title that looks a function up by name must not
/// get past our overrides (the network guard, path rewriting).
fn host_override(module: *mut c_void, name: *const u8) -> Option<*mut c_void> {
    if module.is_null() || runtime::image_path(module as usize).is_some() {
        return None;
    }
    let symbol = if (name as usize) < 0x1_0000 {
        format!("#{}", name as usize)
    } else {
        narrow(name)
    };
    let mut path = [0u16; 260];
    let len = host::get_module_file_name_w(module, path.as_mut_ptr(), path.len() as u32) as usize;
    let path = String::from_utf16_lossy(&path[..len.min(path.len())]);
    let file = path.rsplit('\\').next()?;
    crate::registry::lookup(file, &symbol).map(|address| address as *mut c_void)
}

/// Our pseudo modules and the title's images stay loaded: `LoadLibrary`
/// took no reference on them, so the host's count must not drop.
pub extern "system" fn FreeLibrary(module: *mut c_void) -> i32 {
    if registry().iter().any(|m| handle(m) == module)
        || runtime::image_path(module as usize).is_some()
    {
        return 1;
    }
    host::free_library(module)
}

/// Title images hear of thread starts and ends from us, not from the
/// Windows loader (`victory_loader::disable_thread_calls`).
pub extern "system" fn DisableThreadLibraryCalls(module: *mut c_void) -> i32 {
    host::disable_thread_library_calls(module)
}

/// Title debug output goes to our log (no debugger needs to be attached).
pub extern "system" fn OutputDebugStringA(text: *const u8) {
    if !text.is_null() {
        let text = narrow(text);
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// See [`OutputDebugStringA`].
pub extern "system" fn OutputDebugStringW(text: *const u16) {
    if !text.is_null() {
        let text = wide(text);
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// The CRT's `exit`, through which the title's entry point returns: logs
/// the code and the caller, then exits on the host.
pub extern "C" fn exit(code: i32) -> ! {
    tracing::warn!(code, from = %host::callers(), "title exits");
    host::exit(code)
}

/// The CRT's `abort` (also reached from `terminate`): logged like [`exit`].
pub extern "C" fn abort() -> ! {
    tracing::error!(from = %host::callers(), "title aborts");
    host::abort()
}

/// The CRT's `terminate` and the STL's invalid-argument exit: logged, then
/// the host's `abort`.
pub extern "C" fn terminate() -> ! {
    tracing::error!(from = %host::callers(), "title calls terminate");
    host::abort()
}

/// See [`terminate`].
pub extern "C" fn _invalid_parameter_noinfo_noreturn() -> ! {
    tracing::error!(from = %host::callers(), "title passed an invalid parameter");
    host::abort()
}

/// The title's unhandled-exception filter. Ours stands in front of it and
/// reports the crash first; the title still gets its own back.
static TITLE_FILTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub extern "system" fn SetUnhandledExceptionFilter(filter: usize) -> usize {
    host::set_unhandled_exception_filter(report_unhandled);
    TITLE_FILTER.swap(filter, std::sync::atomic::Ordering::AcqRel)
}

extern "system" fn report_unhandled(info: *const c_void) -> i32 {
    host::report_unhandled(info);
    match TITLE_FILTER.load(std::sync::atomic::Ordering::Acquire) {
        0 => 0, // EXCEPTION_CONTINUE_SEARCH
        filter => {
            type Filter = extern "system" fn(*const c_void) -> i32;
            // SAFETY: the title registered this filter.
            let filter: Filter = unsafe { std::mem::transmute(filter) };
            filter(info)
        }
    }
}

/// See [`exit`]; only the current process is logged.
pub extern "system" fn TerminateProcess(process: *mut c_void, code: u32) -> i32 {
    if process as isize == -1 {
        tracing::warn!(code, from = %host::callers(), "title terminates the process");
    }
    host::terminate_process(process, code)
}

/// The pseudo handle of one of our modules: its address, never a real image.
fn handle(m: &HleModule) -> *mut c_void {
    m as *const HleModule as *mut c_void
}

/// The title's images and our console modules by name; `None` leaves the
/// name to the host, which only DLLs `HostPolicy` passes through get (as at
/// import binding); others fail. With `load`, a title DLL not mapped yet is
/// mapped now.
fn find_module(path: &str, load: bool) -> Option<*mut c_void> {
    let file = path.rsplit(['\\', '/']).next().unwrap_or(path);
    // As the Windows loader does, a name without extension means a DLL.
    let file = if file.contains('.') {
        file.to_owned()
    } else {
        format!("{file}.dll")
    };
    // Loads go through the title DLL loader first: it waits for a DLL that
    // another thread is still initialising, as the Windows loader lock does.
    if load {
        match runtime::load_title_dll(&file) {
            Ok(Some(base)) => return Some(base as *mut c_void),
            Ok(None) => {}
            // Not the host: it would load the unpatched DLL from the title's folder.
            Err(error) => {
                tracing::error!(dll = file, %error, "mapping a title DLL failed");
                host::set_last_error(ERROR_MOD_NOT_FOUND);
                return Some(std::ptr::null_mut());
            }
        }
    }
    if let Some(base) = runtime::image_named(&file) {
        return Some(base as *mut c_void);
    }
    // DLLs the host has too are real host modules, whatever `--strict` says.
    let host_dll = HostPolicy::default().allows(&file);
    if !host_dll && let Some(m) = registry().iter().find(|m| m.serves(&file)) {
        if load {
            tracing::info!(dll = file, "loaded as a pseudo module");
        }
        return Some(handle(m));
    }
    if host_dll {
        return None;
    }
    // Any other host DLL (wininet, urlmon, ...) is beyond what we guard, and
    // the console does not have it.
    refuse(&file);
    host::set_last_error(ERROR_MOD_NOT_FOUND);
    Some(std::ptr::null_mut())
}

/// Logs a DLL name the title is refused, once.
fn refuse(file: &str) {
    static REFUSED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let file = file.to_ascii_lowercase();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if !refused.contains(&file) {
        tracing::warn!(dll = file, "a host DLL outside the host policy: refused");
        refused.push(file);
    }
}

const ERROR_MOD_NOT_FOUND: u32 = 126;

/// The title's argument list: the program, then what was given after `--`.
/// The host's own would be ours (`victory run <exe> ...`), and the title
/// looks at fixed places in it (`-skipToIIS` only as the first argument).
struct TitleArgv {
    count: std::cell::UnsafeCell<i32>,
    list: std::cell::UnsafeCell<*mut *mut c_char>,
}

// SAFETY: never written here after it was made; the title's C runtime start
// reads it once, on one thread.
unsafe impl Send for TitleArgv {}
// SAFETY: as above.
unsafe impl Sync for TitleArgv {}

/// Kept for the whole run: the title holds on to the pointers.
fn title_argv(program: &str, args: &[String]) -> TitleArgv {
    let list: Vec<*mut c_char> = std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .filter_map(|a| std::ffi::CString::new(a).ok())
        .map(std::ffi::CString::into_raw)
        .collect();
    let count = list.len() as i32;
    // `argv[argc]` is null.
    let list: Box<[*mut c_char]> = list.into_iter().chain(Some(std::ptr::null_mut())).collect();
    TitleArgv {
        count: count.into(),
        list: Box::leak(list).as_mut_ptr().into(),
    }
}

fn argv() -> &'static TitleArgv {
    static ARGV: std::sync::OnceLock<TitleArgv> = std::sync::OnceLock::new();
    ARGV.get_or_init(|| {
        // The program's name stays what the host says (as it was before).
        let program = std::env::args_os().next().unwrap_or_default();
        let args = runtime::title_args();
        tracing::info!(?args, "the title's arguments");
        title_argv(&program.to_string_lossy(), args)
    })
}

/// See [`TitleArgv`].
pub extern "C" fn __p___argc() -> *mut i32 {
    argv().count.get()
}

/// See [`TitleArgv`].
pub extern "C" fn __p___argv() -> *mut *mut *mut c_char {
    argv().list.get()
}

/// The title's thread priorities, but for the lowest two, which become
/// "below normal". Six of the title's threads run at "idle", the streaming
/// pager's among them; on the console they have what the rest of the title
/// leaves of its cores. A PC's processors are also wanted by other programs
/// and by our own workers (sixteen when pipelines are made): with every
/// one of them busy such a thread ran once in four seconds or so, when the
/// host's scheduler took pity, and the loading that waits for it with it
/// (run stream-3: loads seen to be done 1.1 s late on average, 4.8 s at
/// worst). `VICTORY_TITLE_PRIORITIES=0`: as the title sets them.
fn title_priority(asked: i32) -> i32 {
    const IDLE: i32 = -15;
    const LOWEST: i32 = -2;
    const BELOW_NORMAL: i32 = -1;
    match asked {
        IDLE..=LOWEST => BELOW_NORMAL,
        other => other,
    }
}

/// [`title_priority`].
pub extern "system" fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32 {
    static AS_SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let as_set = *AS_SET.get_or_init(|| std::env::var_os("VICTORY_TITLE_PRIORITIES").is_some_and(|v| v == "0"));
    let priority = if as_set { priority } else { title_priority(priority) };
    // SAFETY: the title passes a thread handle.
    unsafe { windows_sys::Win32::System::Threading::SetThreadPriority(thread, priority) }
}

/// Win32 `GROUP_AFFINITY` (`winnt.h`, 16 bytes on x64).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GroupAffinity {
    pub mask: usize,
    pub group: u16,
    pub reserved: [u16; 3],
}

const _: () = assert!(size_of::<GroupAffinity>() == 16);
const _: () = assert!(std::mem::offset_of!(GroupAffinity, mask) == 0);
const _: () = assert!(std::mem::offset_of!(GroupAffinity, group) == 8);
const _: () = assert!(std::mem::offset_of!(GroupAffinity, reserved) == 10);

const ERROR_INVALID_HANDLE: u32 = 6;
const ERROR_INVALID_PARAMETER: u32 = 87;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TitleAffinityMode {
    Ignore,
    Passthrough,
}

fn parse_title_affinity(raw: Option<&str>) -> TitleAffinityMode {
    match raw.map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("passthrough") || v == "0" => {
            TitleAffinityMode::Passthrough
        }
        _ => TitleAffinityMode::Ignore,
    }
}

/// Title thread affinity policy (`VICTORY_TITLE_AFFINITY=ignore|passthrough`,
/// default `ignore`): console titles hard-pin threads to cores `0..6`, which
/// collides with Victory's worker threads on multi-core PCs. In `ignore` mode,
/// threads keep full-process affinity while the requested masks are shadowed
/// for `GetThreadGroupAffinity` / `PreviousGroupAffinity`.
fn title_affinity_mode() -> TitleAffinityMode {
    static MODE: std::sync::OnceLock<TitleAffinityMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = parse_title_affinity(std::env::var("VICTORY_TITLE_AFFINITY").ok().as_deref());
        tracing::info!(?mode, "title thread affinity policy (VICTORY_TITLE_AFFINITY)");
        mode
    })
}

fn shadow_affinity() -> &'static std::sync::Mutex<std::collections::HashMap<u64, GroupAffinity>> {
    static SHADOW: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<u64, GroupAffinity>>,
    > = std::sync::OnceLock::new();
    SHADOW.get_or_init(Default::default)
}

fn current_or_real_affinity(
    map: &std::collections::HashMap<u64, GroupAffinity>,
    key: u64,
    thread: *mut c_void,
) -> GroupAffinity {
    if let Some(&saved) = map.get(&key) {
        return saved;
    }
    let mut real = GroupAffinity::default();
    if host::get_thread_group_affinity(thread, &mut real) != 0 && real.mask != 0 {
        return real;
    }
    GroupAffinity {
        mask: host::process_affinity_mask(),
        group: 0,
        reserved: [0; 3],
    }
}

pub extern "system" fn SetThreadGroupAffinity(
    thread: *mut c_void,
    group_affinity: *const GroupAffinity,
    previous_group_affinity: *mut GroupAffinity,
) -> i32 {
    if title_affinity_mode() == TitleAffinityMode::Passthrough {
        return host::set_thread_group_affinity(thread, group_affinity, previous_group_affinity);
    }
    if group_affinity.is_null() {
        host::set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    // SAFETY: checked non-null; caller passes a valid `GROUP_AFFINITY`.
    let asked = unsafe { *group_affinity };
    if asked.mask == 0 || asked.reserved != [0; 3] {
        host::set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    let Some(key) = host::thread_key(thread) else {
        host::set_last_error(ERROR_INVALID_HANDLE);
        return 0;
    };
    let prev = {
        let mut map = shadow_affinity().lock().unwrap_or_else(|e| e.into_inner());
        let prev = current_or_real_affinity(&map, key, thread);
        map.insert(key, asked);
        prev
    };
    if !previous_group_affinity.is_null() {
        // SAFETY: optional output parameter checked non-null.
        unsafe { previous_group_affinity.write(prev) };
    }
    1
}

pub extern "system" fn GetThreadGroupAffinity(
    thread: *mut c_void,
    group_affinity: *mut GroupAffinity,
) -> i32 {
    if group_affinity.is_null() {
        host::set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    if title_affinity_mode() == TitleAffinityMode::Passthrough {
        return host::get_thread_group_affinity(thread, group_affinity);
    }
    if let Some(key) = host::thread_key(thread)
        && let Some(saved) = shadow_affinity()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .copied()
    {
        // SAFETY: checked non-null.
        unsafe { group_affinity.write(saved) };
        return 1;
    }
    host::get_thread_group_affinity(thread, group_affinity)
}

pub extern "system" fn SetThreadAffinityMask(
    thread: *mut c_void,
    thread_affinity_mask: usize,
) -> usize {
    if title_affinity_mode() == TitleAffinityMode::Passthrough {
        return host::set_thread_affinity_mask(thread, thread_affinity_mask);
    }
    if thread_affinity_mask == 0 {
        host::set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    let Some(key) = host::thread_key(thread) else {
        host::set_last_error(ERROR_INVALID_HANDLE);
        return 0;
    };
    let mut map = shadow_affinity().lock().unwrap_or_else(|e| e.into_inner());
    let prev = current_or_real_affinity(&map, key, thread);
    let prev_mask = if prev.mask != 0 {
        prev.mask
    } else {
        host::process_affinity_mask()
    };
    map.insert(
        key,
        GroupAffinity {
            mask: thread_affinity_mask,
            group: prev.group,
            reserved: [0; 3],
        },
    );
    prev_mask
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernel",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-libraryloader-*",
            "api-ms-win-core-processthreads-*",
            "api-ms-win-core-processtopology-*",
            "api-ms-win-core-debug-*",
            "api-ms-win-core-errorhandling-*",
            "api-ms-win-crt-runtime-*",
        ],
        exports: vec![
            hle_export!(__p___argc),
            hle_export!(__p___argv),
            hle_export!(GetModuleHandleW),
            hle_export!(GetModuleHandleA),
            hle_export!(GetModuleFileNameW),
            hle_export!(GetModuleFileNameA),
            hle_export!(LoadLibraryExA),
            hle_export!(LoadLibraryExW),
            hle_export!(LoadLibraryA),
            hle_export!(LoadLibraryW),
            hle_export!(GetProcAddress),
            hle_export!(FreeLibrary),
            hle_export!(DisableThreadLibraryCalls),
            hle_export!(OutputDebugStringA),
            hle_export!(OutputDebugStringW),
            hle_export!(exit),
            hle_export!(abort),
            hle_export!(terminate),
            hle_export!(_invalid_parameter_noinfo_noreturn),
            hle_export!(TerminateProcess),
            hle_export!(SetUnhandledExceptionFilter),
            hle_export!(SetThreadPriority),
            hle_export!(SetThreadGroupAffinity),
            hle_export!(GetThreadGroupAffinity),
            hle_export!(SetThreadAffinityMask),
        ],
    }
}

/// The return addresses above the caller, as `module+rva`, for logs.
#[cfg(feature = "diagnostics")]
pub(crate) fn callers() -> String {
    host::callers()
}

#[cfg(windows)]
mod host {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{FreeLibrary, SetLastError};
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::LibraryLoader::{
        DisableThreadLibraryCalls, GetModuleFileNameA, GetModuleFileNameW, GetModuleHandleA,
        GetModuleHandleW, GetProcAddress, LoadLibraryExA, LoadLibraryExW,
    };
    use windows_sys::Win32::System::Threading::TerminateProcess;

    unsafe extern "C" {
        #[link_name = "exit"]
        fn crt_exit(code: i32) -> !;
        #[link_name = "abort"]
        fn crt_abort() -> !;
    }

    // SAFETY (every function): forwarding the title's own arguments to the
    // host API of the same name.

    pub(super) fn get_module_handle_w(name: *const u16) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetModuleHandleW(name) }
    }

    pub(super) fn get_module_handle_a(name: *const u8) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetModuleHandleA(name) }
    }

    pub(super) fn get_module_file_name_w(module: *mut c_void, out: *mut u16, size: u32) -> u32 {
        // SAFETY: see above.
        unsafe { GetModuleFileNameW(module, out, size) }
    }

    pub(super) fn get_module_file_name_a(module: *mut c_void, out: *mut u8, size: u32) -> u32 {
        // SAFETY: see above.
        unsafe { GetModuleFileNameA(module, out, size) }
    }

    pub(super) fn set_last_error(code: u32) {
        // SAFETY: see above.
        unsafe { SetLastError(code) }
    }

    pub(super) fn load_library_ex_a(name: *const u8, file: *mut c_void, flags: u32) -> *mut c_void {
        // SAFETY: see above.
        unsafe { LoadLibraryExA(name, file, flags) }
    }

    pub(super) fn load_library_ex_w(
        name: *const u16,
        file: *mut c_void,
        flags: u32,
    ) -> *mut c_void {
        // SAFETY: see above.
        unsafe { LoadLibraryExW(name, file, flags) }
    }

    pub(super) fn get_proc_address(module: *mut c_void, name: *const u8) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetProcAddress(module, name) }.map_or(std::ptr::null_mut(), |f| f as *mut c_void)
    }

    pub(super) fn free_library(module: *mut c_void) -> i32 {
        // SAFETY: see above.
        unsafe { FreeLibrary(module) }
    }

    pub(super) fn disable_thread_library_calls(module: *mut c_void) -> i32 {
        match victory_loader::disable_thread_calls(module as usize) {
            Some(done) => i32::from(done),
            // SAFETY: see above.
            None => unsafe { DisableThreadLibraryCalls(module) },
        }
    }

    pub(super) fn exit(code: i32) -> ! {
        // SAFETY: see above.
        unsafe { crt_exit(code) }
    }

    pub(super) fn abort() -> ! {
        // SAFETY: see above.
        unsafe { crt_abort() }
    }

    pub(super) fn terminate_process(process: *mut c_void, code: u32) -> i32 {
        // SAFETY: see above.
        unsafe { TerminateProcess(process, code) }
    }

    pub(super) fn set_unhandled_exception_filter(filter: extern "system" fn(*const c_void) -> i32) {
        // SAFETY: our filter takes the exception pointers as an opaque pointer.
        unsafe {
            windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(
                std::mem::transmute::<
                    extern "system" fn(*const c_void) -> i32,
                    windows_sys::Win32::System::Diagnostics::Debug::LPTOP_LEVEL_EXCEPTION_FILTER,
                >(filter),
            )
        };
    }

    pub(super) fn report_unhandled(info: *const c_void) {
        // SAFETY: the system's exception pointers, passed to the filter.
        unsafe { victory_loader::report_unhandled(info.cast()) };
    }

    pub(super) fn note_title_output(line: &str) {
        victory_loader::note_title_output(line);
    }

    pub(super) fn thread_key(thread: *mut c_void) -> Option<u64> {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, GetCurrentThreadId, GetThreadId,
        };
        if thread.is_null() {
            return None;
        }
        // SAFETY: `GetThreadId` accepts real and pseudo thread handles.
        let tid = unsafe { GetThreadId(thread) };
        if tid != 0 {
            return Some(u64::from(tid));
        }
        // SAFETY: plain current-thread queries.
        if thread == unsafe { GetCurrentThread() } {
            return Some(u64::from(unsafe { GetCurrentThreadId() }));
        }
        Some((1u64 << 63) | (thread as usize as u64))
    }

    pub(super) fn process_affinity_mask() -> usize {
        type FnGetProcessAffinityMask =
            unsafe extern "system" fn(*mut c_void, *mut usize, *mut usize) -> i32;
        static FUNC: std::sync::OnceLock<Option<FnGetProcessAffinityMask>> =
            std::sync::OnceLock::new();
        let f = *FUNC.get_or_init(|| {
            crate::host::function("kernel32.dll", "GetProcessAffinityMask").map(|a| {
                // SAFETY: exact Win32 `GetProcessAffinityMask` signature from System32 `kernel32.dll`.
                unsafe { std::mem::transmute::<usize, FnGetProcessAffinityMask>(a) }
            })
        });
        let (mut proc_mask, mut sys_mask) = (0usize, 0usize);
        if let Some(f) = f {
            // SAFETY: valid process pseudo-handle and output pointers.
            let ok = unsafe {
                f(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    &mut proc_mask,
                    &mut sys_mask,
                )
            };
            if ok != 0 && proc_mask != 0 {
                return proc_mask;
            }
        }
        1
    }

    pub(super) fn set_thread_group_affinity(
        thread: *mut c_void,
        group_affinity: *const super::GroupAffinity,
        previous_group_affinity: *mut super::GroupAffinity,
    ) -> i32 {
        type FnSet = unsafe extern "system" fn(
            *mut c_void,
            *const super::GroupAffinity,
            *mut super::GroupAffinity,
        ) -> i32;
        static FUNC: std::sync::OnceLock<Option<FnSet>> = std::sync::OnceLock::new();
        let Some(f) = *FUNC.get_or_init(|| {
            crate::host::function("kernel32.dll", "SetThreadGroupAffinity").map(|a| {
                // SAFETY: exact Win32 `SetThreadGroupAffinity` signature from System32 `kernel32.dll`.
                unsafe { std::mem::transmute::<usize, FnSet>(a) }
            })
        }) else {
            return 0;
        };
        // SAFETY: forwarding caller arguments to host `SetThreadGroupAffinity`.
        unsafe { f(thread, group_affinity, previous_group_affinity) }
    }

    pub(super) fn get_thread_group_affinity(
        thread: *mut c_void,
        group_affinity: *mut super::GroupAffinity,
    ) -> i32 {
        type FnGet = unsafe extern "system" fn(*mut c_void, *mut super::GroupAffinity) -> i32;
        static FUNC: std::sync::OnceLock<Option<FnGet>> = std::sync::OnceLock::new();
        let Some(f) = *FUNC.get_or_init(|| {
            crate::host::function("kernel32.dll", "GetThreadGroupAffinity").map(|a| {
                // SAFETY: exact Win32 `GetThreadGroupAffinity` signature from System32 `kernel32.dll`.
                unsafe { std::mem::transmute::<usize, FnGet>(a) }
            })
        }) else {
            return 0;
        };
        // SAFETY: forwarding caller arguments to host `GetThreadGroupAffinity`.
        unsafe { f(thread, group_affinity) }
    }

    pub(super) fn set_thread_affinity_mask(thread: *mut c_void, mask: usize) -> usize {
        type FnSetMask = unsafe extern "system" fn(*mut c_void, usize) -> usize;
        static FUNC: std::sync::OnceLock<Option<FnSetMask>> = std::sync::OnceLock::new();
        let Some(f) = *FUNC.get_or_init(|| {
            crate::host::function("kernel32.dll", "SetThreadAffinityMask").map(|a| {
                // SAFETY: exact Win32 `SetThreadAffinityMask` signature from System32 `kernel32.dll`.
                unsafe { std::mem::transmute::<usize, FnSetMask>(a) }
            })
        }) else {
            return 0;
        };
        // SAFETY: forwarding caller arguments to host `SetThreadAffinityMask`.
        unsafe { f(thread, mask) }
    }

    /// Our caller's callers, as `module+rva`.
    pub(super) fn callers() -> String {
        let mut frames = [std::ptr::null_mut(); 12];
        // SAFETY: fills at most `frames.len()` return addresses.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                2,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        let names: Vec<_> = frames[..usize::from(n)]
            .iter()
            .map(|&f| victory_loader::locate(f as usize))
            .collect();
        names.join(" < ")
    }
}

#[cfg(not(windows))]
mod host {
    use std::ffi::c_void;

    pub(super) fn get_module_handle_w(_: *const u16) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_module_handle_a(_: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn load_library_ex_a(_: *const u8, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_module_file_name_w(_: *mut c_void, _: *mut u16, _: u32) -> u32 {
        0
    }

    pub(super) fn get_module_file_name_a(_: *mut c_void, _: *mut u8, _: u32) -> u32 {
        0
    }

    pub(super) fn set_last_error(_: u32) {}

    pub(super) fn load_library_ex_w(_: *const u16, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_proc_address(_: *mut c_void, _: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn free_library(_: *mut c_void) -> i32 {
        0
    }

    pub(super) fn disable_thread_library_calls(_: *mut c_void) -> i32 {
        0
    }

    pub(super) fn exit(code: i32) -> ! {
        std::process::exit(code)
    }

    pub(super) fn abort() -> ! {
        std::process::abort()
    }

    pub(super) fn terminate_process(_: *mut c_void, code: u32) -> i32 {
        std::process::exit(code as i32)
    }

    pub(super) fn set_unhandled_exception_filter(_: extern "system" fn(*const c_void) -> i32) {}

    pub(super) fn report_unhandled(_: *const c_void) {}

    pub(super) fn note_title_output(_: &str) {}

    pub(super) fn thread_key(thread: *mut c_void) -> Option<u64> {
        (!thread.is_null()).then_some(thread as usize as u64)
    }

    pub(super) fn process_affinity_mask() -> usize {
        1
    }

    pub(super) fn set_thread_group_affinity(
        _: *mut c_void,
        _: *const super::GroupAffinity,
        _: *mut super::GroupAffinity,
    ) -> i32 {
        0
    }

    pub(super) fn get_thread_group_affinity(
        _: *mut c_void,
        _: *mut super::GroupAffinity,
    ) -> i32 {
        0
    }

    pub(super) fn set_thread_affinity_mask(_: *mut c_void, _: usize) -> usize {
        0
    }

    pub(super) fn callers() -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The title's idle and lowest threads run below normal; the others as
    /// it sets them.
    #[test]
    fn no_title_thread_waits_for_idle_processors() {
        let given: Vec<i32> = [-15, -2, -1, 0, 1, 2, 15].into_iter().map(title_priority).collect();
        assert_eq!(given, [-1, -1, -1, 0, 1, 2, 15]);
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority};
        std::thread::spawn(|| {
            // SAFETY: plain calls on this thread.
            let thread = unsafe { GetCurrentThread() };
            assert_eq!(SetThreadPriority(thread, -15), 1);
            // SAFETY: as above.
            assert_eq!(unsafe { GetThreadPriority(thread) }, -1);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn title_images_report_their_original_path() {
        let base = 0x7000_0000_0000;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\game.exe"));
        let module = base as *mut c_void;
        let mut buf = [0xFFFFu16; 32];
        let n = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(
            String::from_utf16_lossy(&buf[..n as usize]),
            r"C:\title\game.exe"
        );
        assert_eq!(buf[n as usize], 0);

        // Too small: truncated, NUL-terminated, the buffer size returned.
        let mut small = [0u8; 4];
        assert_eq!(GetModuleFileNameA(module, small.as_mut_ptr(), 4), 4);
        assert_eq!(&small, &[b'C', b':', b'\\', 0]);
    }

    /// The title's arguments come right after the program, and the list ends
    /// in a null pointer; without any there is only the program.
    #[test]
    fn the_title_gets_its_own_arguments() {
        let read = |made: &TitleArgv| {
            // SAFETY: made above, kept for the whole run.
            unsafe {
                let (count, list) = (*made.count.get() as usize, *made.list.get());
                assert!((*list.add(count)).is_null());
                (0..count)
                    .map(|i| CStr::from_ptr(*list.add(i)).to_str().unwrap().to_owned())
                    .collect::<Vec<_>>()
            }
        };
        let some = title_argv("victory.exe", &["-skipToIIS".into(), "-x=1".into()]);
        assert_eq!(read(&some), ["victory.exe", "-skipToIIS", "-x=1"]);
        assert_eq!(read(&title_argv("victory.exe", &[])), ["victory.exe"]);
        // What the title is given: the same list every time.
        assert_eq!(__p___argc(), __p___argc());
        // SAFETY: as above.
        unsafe { assert_eq!((*__p___argc(), (**__p___argv()).is_null()), (1, false)) };
    }

    fn w(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    #[cfg(windows)]
    #[test]
    fn host_lookups_by_name_get_our_overrides() {
        let kernel32 = GetModuleHandleW(w("kernel32.dll").as_ptr());
        assert!(!kernel32.is_null());
        let ours = crate::registry::lookup("kernel32.dll", "CreateFileW").unwrap();
        assert_eq!(
            GetProcAddress(kernel32, c"CreateFileW".as_ptr().cast()) as usize,
            ours
        );
        // A plain name stays the host's.
        let host = host::get_proc_address(kernel32, c"GetTickCount".as_ptr().cast());
        assert!(!host.is_null());
        assert_eq!(
            GetProcAddress(kernel32, c"GetTickCount".as_ptr().cast()),
            host
        );
        // Host DLLs outside the host policy do not exist for the title.
        assert!(LoadLibraryW(w("wininet.dll").as_ptr()).is_null());
        assert!(LoadLibraryA(c"C:\\Windows\\System32\\urlmon".as_ptr().cast()).is_null());
        assert!(GetModuleHandleW(w("WININET.DLL").as_ptr()).is_null());
        assert_eq!(
            // SAFETY: reads this thread's last error.
            unsafe { windows_sys::Win32::Foundation::GetLastError() },
            ERROR_MOD_NOT_FOUND
        );
        assert!(!LoadLibraryW(w("kernel32").as_ptr()).is_null());
        // By ordinal too (ws2_32 #4 is `connect`).
        let ws2 = LoadLibraryW(w("ws2_32.dll").as_ptr());
        assert_eq!(
            GetProcAddress(ws2, 4 as *const u8) as usize,
            crate::registry::lookup("ws2_32.dll", "connect").unwrap()
        );
    }

    #[test]
    fn title_dlls_and_pseudo_modules_are_found_by_name() {
        let base = 0x7100_0000_0000usize;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\Mapped.dll"));
        runtime::set_title_dll_loader(Box::new(|file| match file {
            "late.dll" => Ok(Some(0x7200_0000_0000)),
            "broken.dll" => Err("bad image".into()),
            _ => Ok(None),
        }));
        let mapped = base as *mut c_void;
        assert_eq!(GetModuleHandleW(w("MAPPED").as_ptr()), mapped);
        assert_eq!(GetModuleHandleA(c"sub\\mapped.dll".as_ptr().cast()), mapped);
        assert_eq!(LoadLibraryW(w("mapped.dll").as_ptr()), mapped);
        assert_eq!(FreeLibrary(mapped), 1, "title images stay loaded");

        // Not mapped yet: `LoadLibrary` maps it, `GetModuleHandle` does not.
        assert!(GetModuleHandleW(w("late.dll").as_ptr()).is_null());
        assert_eq!(
            LoadLibraryA(c"late".as_ptr().cast()) as usize,
            0x7200_0000_0000
        );
        assert!(LoadLibraryA(c"broken.dll".as_ptr().cast()).is_null());
        // As data it runs no code, so the host loads it (and finds nothing).
        let data = LoadLibraryExW(w("late.dll").as_ptr(), std::ptr::null_mut(), 0x2);
        assert_ne!(data as usize, 0x7200_0000_0000);

        let runtime = LoadLibraryA(c"xgameruntime.dll".as_ptr().cast());
        assert!(!runtime.is_null());
        assert_eq!(GetModuleHandleW(w("XGameRuntime.dll").as_ptr()), runtime);
    }

    /// No buffer: nothing written, the error says it was too small.
    #[test]
    fn a_title_image_name_into_no_buffer() {
        let base = 0x7300_0000_0000;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\game.exe"));
        let module = base as *mut c_void;
        assert_eq!(GetModuleFileNameW(module, std::ptr::null_mut(), 0), 0);
        assert_eq!(GetModuleFileNameA(module, std::ptr::null_mut(), 0), 0);
        #[cfg(windows)]
        {
            // SAFETY: reads this thread's last error.
            let last = unsafe { windows_sys::Win32::Foundation::GetLastError() };
            assert_eq!(last, ERROR_INSUFFICIENT_BUFFER);
        }
    }

    #[test]
    fn title_affinity_hook_shadows_masks_and_preserves_host_affinity() {
        assert_eq!(parse_title_affinity(None), TitleAffinityMode::Ignore);
        assert_eq!(parse_title_affinity(Some("ignore")), TitleAffinityMode::Ignore);
        assert_eq!(parse_title_affinity(Some("passthrough")), TitleAffinityMode::Passthrough);
        assert_eq!(parse_title_affinity(Some("PASSTHROUGH")), TitleAffinityMode::Passthrough);
        assert_eq!(parse_title_affinity(Some("0")), TitleAffinityMode::Passthrough);

        // Verify API-set resolution catches processtopology and processthreads imports.
        for dll in [
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-processtopology-l1-1-0.dll",
            "api-ms-win-core-processthreads-l1-1-0.dll",
        ] {
            assert_eq!(
                registry::lookup(dll, "SetThreadGroupAffinity"),
                Some(SetThreadGroupAffinity as *const () as usize)
            );
            assert_eq!(
                registry::lookup(dll, "GetThreadGroupAffinity"),
                Some(GetThreadGroupAffinity as *const () as usize)
            );
            assert_eq!(
                registry::lookup(dll, "SetThreadAffinityMask"),
                Some(SetThreadAffinityMask as *const () as usize)
            );
        }

        #[cfg(windows)]
        std::thread::spawn(|| {
            use windows_sys::Win32::System::Threading::GetCurrentThread;
            // SAFETY: plain pseudo-handle query.
            let thread = unsafe { GetCurrentThread() };
            let mut real_before = GroupAffinity::default();
            assert_ne!(host::get_thread_group_affinity(thread, &mut real_before), 0);

            // Reject invalid mask (0) and non-zero reserved words.
            let bad_zero = GroupAffinity { mask: 0, group: 0, reserved: [0; 3] };
            assert_eq!(SetThreadGroupAffinity(thread, &bad_zero, std::ptr::null_mut()), 0);
            let bad_res = GroupAffinity { mask: 1, group: 0, reserved: [1, 0, 0] };
            assert_eq!(SetThreadGroupAffinity(thread, &bad_res, std::ptr::null_mut()), 0);
            assert_eq!(SetThreadAffinityMask(thread, 0), 0);

            // Set group affinity to core 0 only; previous returns real initial affinity,
            // GetThreadGroupAffinity returns shadowed mask, while real OS affinity stays untouched.
            let req = GroupAffinity { mask: 1, group: 0, reserved: [0; 3] };
            let mut prev = GroupAffinity::default();
            assert_eq!(SetThreadGroupAffinity(thread, &req, &mut prev), 1);
            assert_eq!(prev, real_before);

            let mut got = GroupAffinity::default();
            assert_eq!(GetThreadGroupAffinity(thread, &mut got), 1);
            assert_eq!(got, req);

            let mut real_after = GroupAffinity::default();
            assert_ne!(host::get_thread_group_affinity(thread, &mut real_after), 0);
            assert_eq!(real_after, real_before);

            // SetThreadAffinityMask returns previous shadowed mask and updates shadow.
            assert_eq!(SetThreadAffinityMask(thread, 0b1010), 1);
            assert_eq!(GetThreadGroupAffinity(thread, &mut got), 1);
            assert_eq!(got.mask, 0b1010);
            assert_eq!(got.group, 0);
        })
        .join()
        .unwrap();
    }
}
