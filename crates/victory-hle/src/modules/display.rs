//! The display and what the system shell does for a title: HDR mode, screen
//! timeout deferrals, launching URIs, screenshots.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, NOT_SUPPORTED, Object, S_OK, add_ref, missing, put, text,
};

/// HDR mode; asked for with this interface id since 2020.
pub const HDR_PROVIDER: Guid = Guid::new(
    0x03f0_fe74,
    0xfdd9,
    0x4e5c,
    [0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5],
);
pub const HDR_IID: Guid = Guid::new(
    0x35f0_7670,
    0x706e,
    0x4bfb,
    [0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3],
);
/// Launching URIs and screen timeout deferrals; the interface id is the
/// provider's.
pub const SHELL: Guid = Guid::new(
    0x1b33_9674,
    0x328d,
    0x4283,
    [0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39],
);
/// Screenshots and clips.
pub const CAPTURE_PROVIDER: Guid = Guid::new(
    0xa4f1_aee2,
    0x4bf1,
    0x4485,
    [0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27],
);
/// As the title (2026) asks for it; it calls 0x28 (take a
/// screenshot) with the public API's arguments.
pub const CAPTURE_IID_2026: Guid = Guid::new(
    0x22e6_72d7,
    0xb4e3,
    0x406c,
    [0xbd, 0x50, 0x8f, 0x0d, 0x25, 0x23, 0x6f, 0x9e],
);

pub static HDR_OBJECT: Object = Object::new("hdr", &HDR_VTABLE);
pub static SHELL_OBJECT: Object = Object::new("shell", &SHELL_VTABLE);
pub static CAPTURE_OBJECT: Object = Object::new("capture", &CAPTURE_VTABLE);

pub const INTERFACES: &[Interface] = &[
    (HDR_PROVIDER, HDR_IID, &HDR_OBJECT),
    (SHELL, SHELL, &SHELL_OBJECT),
    (CAPTURE_PROVIDER, CAPTURE_IID_2026, &CAPTURE_OBJECT),
];

type This = *const Object;

/// `XDisplayHdrModeResult` values from Xbox GDK `XDisplay.h`.
const HDR_ENABLED: u32 = 1;
const HDR_DISABLED: u32 = 2;

/// `XDisplayHdrModeInfo` from Xbox GDK `XDisplay.h` (nits).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XDisplayHdrModeInfo {
    pub min_tone_map_luminance: f32,
    pub max_tone_map_luminance: f32,
    pub max_full_frame_tone_map_luminance: f32,
}
const _: () = assert!(size_of::<XDisplayHdrModeInfo>() == 12);

/// `APPCAPTURE_MAX_LOCALID_LENGTH` from Xbox GDK `XAppCapture.h`.
pub const APPCAPTURE_MAX_LOCALID_LENGTH: usize = 1024;
/// `XAppCaptureScreenshotFormatFlag::SDR`.
pub const SCREENSHOT_FORMAT_SDR: u16 = 1;
/// `XAppCaptureScreenshotFormatFlag::HDR`.
pub const SCREENSHOT_FORMAT_HDR: u16 = 2;

/// `XAppCaptureTakeScreenshotResult` from Xbox GDK `XAppCapture.h`.
#[repr(C)]
pub struct XAppCaptureTakeScreenshotResult {
    pub local_id: [u8; APPCAPTURE_MAX_LOCALID_LENGTH],
    pub available_screenshot_formats: u16,
}
const _: () = assert!(size_of::<XAppCaptureTakeScreenshotResult>() == 1026);

/// Populates `XDisplayHdrModeInfo` and returns `HDR_ENABLED` (1) when
/// `VICTORY_HDR=1` is active and the host display reports HDR10, otherwise
/// `HDR_DISABLED` (2).
fn apply_hdr_result(
    preference: u32,
    caps: Option<(f32, f32, f32)>,
    info: *mut c_void,
) -> u32 {
    let Some((min_nits, max_nits, max_full_nits)) = caps else {
        tracing::info!(preference, "HDR mode: disabled (SDR presentation)");
        return HDR_DISABLED;
    };
    if !info.is_null() {
        let mode_info = XDisplayHdrModeInfo {
            min_tone_map_luminance: min_nits,
            max_tone_map_luminance: max_nits,
            max_full_frame_tone_map_luminance: max_full_nits,
        };
        // SAFETY: checked non-null; the title passes an `XDisplayHdrModeInfo` out pointer.
        unsafe { info.cast::<XDisplayHdrModeInfo>().write(mode_info) };
    }
    tracing::info!(
        preference,
        min_nits,
        max_nits,
        max_full_nits,
        "HDR mode: enabled (HDR10 presentation)"
    );
    HDR_ENABLED
}

extern "system" fn try_enable_hdr(_: This, preference: u32, info: *mut c_void) -> u32 {
    #[cfg(windows)]
    let caps = victory_gpu::present::hdr_for_title().map(|c| {
        (
            c.min_luminance,
            c.max_luminance,
            c.max_full_frame_luminance,
        )
    });
    #[cfg(not(windows))]
    let caps = None;
    apply_hdr_result(preference, caps, info)
}

/// Nothing launches URIs for the title: no store, browser or console app
/// exists here, and the console's are online services.
extern "system" fn launch_uri(_: This, _user: *const c_void, uri: *const c_char) -> HRESULT {
    tracing::info!(uri = text(uri).unwrap_or_default(), "URI not launched");
    NOT_SUPPORTED
}

/// A deferral keeps the host display on, as on the console, through a
/// power request; the handle is that request.
#[cfg(windows)]
extern "system" fn acquire_deferral(_: This, out: *mut *mut c_void) -> HRESULT {
    use windows_sys::Win32::System::Power::{
        PowerCreateRequest, PowerRequestDisplayRequired, PowerSetRequest,
    };
    use windows_sys::Win32::System::Threading::{
        POWER_REQUEST_CONTEXT_SIMPLE_STRING, REASON_CONTEXT,
    };
    if out.is_null() {
        return super::xasync::E_INVALIDARG;
    }
    let mut reason: Vec<u16> = "Title display deferral\0".encode_utf16().collect();
    let mut context = REASON_CONTEXT {
        Flags: POWER_REQUEST_CONTEXT_SIMPLE_STRING,
        ..Default::default()
    };
    context.Reason.SimpleReasonString = reason.as_mut_ptr();
    // SAFETY: a valid context; the string is copied by the call.
    let request = unsafe { PowerCreateRequest(&context) };
    if request as isize == -1 {
        return last_error();
    }
    // SAFETY: the request we just created.
    if unsafe { PowerSetRequest(request, PowerRequestDisplayRequired) } == 0 {
        let hr = last_error();
        // SAFETY: as above; not handed out.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(request) };
        return hr;
    }
    put(out, request)
}

#[cfg(windows)]
extern "system" fn close_deferral(_: This, handle: *mut c_void) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Power::{PowerClearRequest, PowerRequestDisplayRequired};
    if !handle.is_null() {
        // SAFETY: a request from `acquire_deferral`.
        unsafe {
            PowerClearRequest(handle, PowerRequestDisplayRequired);
            CloseHandle(handle);
        }
    }
}

#[cfg(windows)]
fn last_error() -> HRESULT {
    // SAFETY: no preconditions.
    let code = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    (0x8007_0000 | (code & 0xffff)) as i32
}

/// Encodes BGRA8 pixel rows (`row_pitch` bytes apart) into an 8-bit RGB PNG.
fn encode_bgra8_png(width: u32, height: u32, row_pitch: usize, bgra8: &[u8]) -> Option<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || row_pitch < w * 4 || bgra8.len() < (h - 1) * row_pitch + w * 4 {
        return None;
    }
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for y in 0..h {
        raw.push(0); // PNG filter type None
        let row = &bgra8[y * row_pitch..y * row_pitch + w * 4];
        for px in row.chunks_exact(4) {
            raw.extend_from_slice(&[px[2], px[1], px[0]]);
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    write_png_chunk(&mut png, b"IHDR", &ihdr);
    let mut compressed = vec![0u8; zlib_rs::compress_bound(raw.len())];
    let (idat, rc) = zlib_rs::compress_slice(&mut compressed, &raw, zlib_rs::DeflateConfig::new(3));
    if rc != zlib_rs::ReturnCode::Ok {
        return None;
    }
    write_png_chunk(&mut png, b"IDAT", idat);
    write_png_chunk(&mut png, b"IEND", &[]);
    Some(png)
}

fn write_png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = png_crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn screenshot_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("VICTORY_SCREENSHOT_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("screenshots")
}

fn write_screenshot_result(
    out: *mut XAppCaptureTakeScreenshotResult,
    local_id: &str,
    hdr: bool,
) {
    let mut res = XAppCaptureTakeScreenshotResult {
        local_id: [0u8; APPCAPTURE_MAX_LOCALID_LENGTH],
        available_screenshot_formats: if hdr {
            SCREENSHOT_FORMAT_SDR | SCREENSHOT_FORMAT_HDR
        } else {
            SCREENSHOT_FORMAT_SDR
        },
    };
    let bytes = local_id.as_bytes();
    let n = bytes.len().min(APPCAPTURE_MAX_LOCALID_LENGTH - 1);
    res.local_id[..n].copy_from_slice(&bytes[..n]);
    // SAFETY: caller validated `out` is non-null.
    unsafe { out.write(res) };
}

fn save_screenshot_png(dir: &Path, width: u32, height: u32, row_pitch: usize, bgra8: &[u8]) -> Option<(String, PathBuf)> {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let png = encode_bgra8_png(width, height, row_pitch, bgra8)?;
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let local_id = format!("victory-shot-{seq}");
    let path = dir.join(format!("screenshot-{millis}-{seq}.png"));
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &png)) {
        tracing::warn!(error = %e, path = %path.display(), "screenshot PNG could not be written");
        return None;
    }
    Some((local_id, path))
}

/// Captures the next presented frame on demand (zero per-frame overhead when
/// idle), encodes it as PNG into `VICTORY_SCREENSHOT_DIR` (default
/// `./screenshots`), and fills `XAppCaptureTakeScreenshotResult`.
extern "system" fn take_screenshot(_: This, _user: *const c_void, result: *mut c_void) -> HRESULT {
    if result.is_null() {
        return super::xasync::E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let Some(frame) = victory_gpu::last_frame() else {
            tracing::info!("screenshot not taken: no presented frame available");
            return NOT_SUPPORTED;
        };
        let dir = screenshot_dir();
        let Some((local_id, path)) =
            save_screenshot_png(&dir, frame.width, frame.height, frame.row_pitch, &frame.bgra8)
        else {
            return NOT_SUPPORTED;
        };
        write_screenshot_result(
            result.cast::<XAppCaptureTakeScreenshotResult>(),
            &local_id,
            frame.hdr,
        );
        tracing::info!(
            local_id,
            width = frame.width,
            height = frame.height,
            hdr = frame.hdr,
            path = %path.display(),
            "screenshot saved"
        );
        S_OK
    }
    #[cfg(not(windows))]
    {
        tracing::info!("screenshot not taken: unsupported platform");
        NOT_SUPPORTED
    }
}

static HDR_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),
    try_enable_hdr as *const (), // 0x20
]);

#[cfg(windows)]
static SHELL_VTABLE: Methods<6> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    launch_uri as *const (),       // 0x18
    acquire_deferral as *const (), // 0x20
    close_deferral as *const (),   // 0x28
]);
#[cfg(not(windows))]
static SHELL_VTABLE: Methods<4> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    launch_uri as *const (),
]);

static CAPTURE_VTABLE: Methods<6> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),    // 0x18 diagnostic screenshot
    missing::<4> as *const (),    // 0x20 diagnostic clip
    take_screenshot as *const (), // 0x28
]);

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::{S_OK, method};

    #[test]
    fn sdr_and_display_deferrals() {
        type Hdr = extern "system" fn(This, u32, *mut c_void) -> u32;
        assert_eq!(
            method::<Hdr>(&HDR_OBJECT, 0x20)(&HDR_OBJECT, 0, std::ptr::null_mut()),
            HDR_DISABLED
        );
        type Acquire = extern "system" fn(This, *mut *mut c_void) -> HRESULT;
        type Close = extern "system" fn(This, *mut c_void);
        let mut handle = std::ptr::null_mut();
        match method::<Acquire>(&SHELL_OBJECT, 0x20)(&SHELL_OBJECT, &mut handle) {
            S_OK => {
                assert!(!handle.is_null());
                method::<Close>(&SHELL_OBJECT, 0x28)(&SHELL_OBJECT, handle);
            }
            // Windows can reject display power requests on some hosts.
            NOT_SUPPORTED => assert!(handle.is_null()),
            hr => panic!("unexpected display deferral HRESULT {:#010x}", hr as u32),
        }
    }

    #[test]
    fn hdr_info_populates_luminance_when_enabled() {
        let mut info = XDisplayHdrModeInfo {
            min_tone_map_luminance: -1.0,
            max_tone_map_luminance: -1.0,
            max_full_frame_tone_map_luminance: -1.0,
        };
        let status = apply_hdr_result(
            0,
            Some((0.005, 1000.0, 450.0)),
            (&raw mut info).cast::<c_void>(),
        );
        assert_eq!(status, HDR_ENABLED);
        assert_eq!(info.min_tone_map_luminance, 0.005);
        assert_eq!(info.max_tone_map_luminance, 1000.0);
        assert_eq!(info.max_full_frame_tone_map_luminance, 450.0);

        // Disabled leaves caller struct untouched.
        let disabled = apply_hdr_result(0, None, (&raw mut info).cast::<c_void>());
        assert_eq!(disabled, HDR_DISABLED);
        assert_eq!(info.max_tone_map_luminance, 1000.0);
    }

    #[test]
    fn screenshot_png_encoding_and_capture_result() {
        // 2x1 BGRA8 pixels: pure red (B=0,G=0,R=255,A=255) and pure blue (B=255,G=0,R=0,A=255).
        let bgra = [0u8, 0, 255, 255, 255, 0, 0, 255];
        let dir = tempfile::tempdir().unwrap();
        let (local_id, path) = save_screenshot_png(dir.path(), 2, 1, 8, &bgra).unwrap();
        assert!(local_id.starts_with("victory-shot-"));
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&bytes[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(bytes[16..20].try_into().unwrap()), 2);
        assert_eq!(u32::from_be_bytes(bytes[20..24].try_into().unwrap()), 1);

        let mut out = std::mem::MaybeUninit::<XAppCaptureTakeScreenshotResult>::zeroed();
        write_screenshot_result(out.as_mut_ptr(), &local_id, true);
        // SAFETY: written above.
        let out = unsafe { out.assume_init() };
        assert_eq!(
            out.available_screenshot_formats,
            SCREENSHOT_FORMAT_SDR | SCREENSHOT_FORMAT_HDR
        );
        let id_str = std::ffi::CStr::from_bytes_until_nul(&out.local_id)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(id_str, local_id);

        // Null result pointer returns E_INVALIDARG; without presented frames returns NOT_SUPPORTED.
        type Shot = extern "system" fn(This, *const c_void, *mut c_void) -> HRESULT;
        let shot = method::<Shot>(&CAPTURE_OBJECT, 0x28);
        assert_eq!(
            shot(&CAPTURE_OBJECT, std::ptr::null(), std::ptr::null_mut()),
            crate::modules::xasync::E_INVALIDARG
        );
        let mut res = std::mem::MaybeUninit::<XAppCaptureTakeScreenshotResult>::zeroed();
        assert_eq!(
            shot(
                &CAPTURE_OBJECT,
                std::ptr::null(),
                res.as_mut_ptr().cast::<c_void>()
            ),
            NOT_SUPPORTED
        );
    }
}
