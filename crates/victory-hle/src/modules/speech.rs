//! Accessibility: speech synthesis for the title's narration.
//!
//! Uses Windows SAPI 5 via COM on a dedicated STA worker thread to synthesize
//! plain text (slot `0x68`) and SSML (slot `0xa8`) into 16 kHz mono 16-bit
//! RIFF/WAVE buffers, and enumerates installed SAPI voices (slot `0x40`).
//! Set `VICTORY_SPEECH=0` to opt out into silent 44-byte WAVE streams; COM or
//! missing-voice failures also fall back safely to silence.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::ffi::{CString, OsStr, c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, mpsc};

use super::xasync::{E_INVALIDARG, E_NOT_SUFFICIENT_BUFFER};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

/// Closed captions, high contrast, speech to text and synthesis.
pub const PROVIDER: Guid = Guid::new(
    0x3e24_1977,
    0x6237,
    0x41e9,
    [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1],
);
/// Unchanged since 2020.
pub const IID: Guid = Guid::new(
    0xd722_b373,
    0x8c4d,
    0x4692,
    [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d],
);

pub static OBJECT: Object = Object::new("accessibility", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT)];

/// A WAVE file of 16 kHz mono 16-bit PCM with no samples: RIFF size 36,
/// a 16-byte format chunk (PCM, 1 channel, 16,000 Hz, 32,000 bytes/s,
/// block 2, 16 bits), an empty data chunk.
const SILENCE: [u8; 44] =
    *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";

type This = *const Object;

/// `XSpeechSynthesizerVoiceInformation` passed to the title's enumeration callback.
#[repr(C)]
struct VoiceInfo {
    description: *const c_char,
    display_name: *const c_char,
    /// `XSpeechSynthesizerVoiceGender`: `0` = Female, `1` = Male.
    gender: u32,
    voice_id: *const c_char,
    language: *const c_char,
}

type VoicesCallback = extern "system" fn(*const VoiceInfo, *mut c_void) -> bool;

struct VoiceRecord {
    description: CString,
    display_name: CString,
    gender: u32,
    voice_id: CString,
    language: CString,
}

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(1);
static SYNTHS: OnceLock<Mutex<HashMap<usize, Option<String>>>> = OnceLock::new();
static STREAMS: OnceLock<Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();

fn synths() -> MutexGuard<'static, HashMap<usize, Option<String>>> {
    SYNTHS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn streams() -> MutexGuard<'static, HashMap<usize, Vec<u8>>> {
    STREAMS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn alloc_handle() -> usize {
    loop {
        let h = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        if h != 0 {
            return h;
        }
    }
}

fn speech_enabled() -> bool {
    std::env::var_os("VICTORY_SPEECH").as_deref() != Some(OsStr::new("0"))
}

fn synthesize_wav(voice_id: Option<String>, input: &str, is_ssml: bool, enabled: bool) -> Vec<u8> {
    if !enabled || input.is_empty() {
        return SILENCE.to_vec();
    }
    #[cfg(windows)]
    {
        if let Some(wav) = sapi::synthesize(voice_id, input.to_owned(), is_ssml) {
            return wav;
        }
    }
    #[cfg(not(windows))]
    let _ = (voice_id, is_ssml);
    SILENCE.to_vec()
}

extern "system" fn enumerate_voices(
    _: This,
    context: *mut c_void,
    callback: *const c_void,
) -> HRESULT {
    if callback.is_null() {
        return E_INVALIDARG;
    }
    if !speech_enabled() {
        return S_OK;
    }
    #[cfg(windows)]
    {
        // SAFETY: title supplies `XSpeechSynthesizerInstalledVoicesCallback*`.
        let cb: VoicesCallback = unsafe { std::mem::transmute(callback) };
        for voice in sapi::enumerate_voices() {
            let info = VoiceInfo {
                description: voice.description.as_ptr(),
                display_name: voice.display_name.as_ptr(),
                gender: voice.gender,
                voice_id: voice.voice_id.as_ptr(),
                language: voice.language.as_ptr(),
            };
            if !cb(&info, context) {
                break;
            }
        }
    }
    #[cfg(not(windows))]
    let _ = context;
    S_OK
}

extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let handle = alloc_handle();
    synths().insert(handle, None);
    put(out, handle as *const u8)
}

extern "system" fn close_synthesizer(_: This, handle: *const u8) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    if synths().remove(&(handle as usize)).is_some() {
        S_OK
    } else {
        E_INVALIDARG
    }
}

extern "system" fn set_default_voice(_: This, handle: *const u8) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    match synths().get_mut(&(handle as usize)) {
        Some(voice) => {
            *voice = None;
            S_OK
        }
        None => E_INVALIDARG,
    }
}

extern "system" fn set_custom_voice(
    _: This,
    handle: *const u8,
    voice: *const c_char,
) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    let Some(id) = text(voice) else {
        return E_INVALIDARG;
    };
    match synths().get_mut(&(handle as usize)) {
        Some(slot) => {
            *slot = Some(id.to_owned());
            S_OK
        }
        None => E_INVALIDARG,
    }
}

fn create_stream_inner(
    synth: *const u8,
    words: *const c_char,
    out: *mut *const u8,
    is_ssml: bool,
) -> HRESULT {
    if synth.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let voice_id = match synths().get(&(synth as usize)) {
        Some(v) => v.clone(),
        None => return E_INVALIDARG,
    };
    let input = text(words).unwrap_or_default();
    let enabled = speech_enabled();
    let wav = synthesize_wav(voice_id, &input, is_ssml, enabled);
    tracing::debug!(
        text = %input,
        ssml = is_ssml,
        bytes = wav.len(),
        enabled,
        "speech stream created"
    );
    let handle = alloc_handle();
    streams().insert(handle, wav);
    put(out, handle as *const u8)
}

extern "system" fn create_stream_text(
    _: This,
    synth: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    create_stream_inner(synth, words, out, false)
}

extern "system" fn create_stream_ssml(
    _: This,
    synth: *const u8,
    ssml: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    create_stream_inner(synth, ssml, out, true)
}

extern "system" fn close_stream(_: This, handle: *const u8) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    if streams().remove(&(handle as usize)).is_some() {
        S_OK
    } else {
        E_INVALIDARG
    }
}

extern "system" fn stream_size(_: This, handle: *const u8, out: *mut usize) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    match streams().get(&(handle as usize)) {
        Some(wav) => put(out, wav.len()),
        None => E_INVALIDARG,
    }
}

extern "system" fn stream_data(
    _: This,
    handle: *const u8,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    let map = streams();
    let Some(wav) = map.get(&(handle as usize)) else {
        return E_INVALIDARG;
    };
    if buffer.is_null() || size < wav.len() {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size >= wav.len()` bytes, checked above.
    unsafe { buffer.copy_from_nonoverlapping(wav.as_ptr(), wav.len()) };
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(wav.len()) };
    }
    S_OK
}

static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),          // 0x18 closed captions
    missing::<4> as *const (),          // 0x20
    missing::<5> as *const (),          // 0x28 high contrast
    missing::<6> as *const (),          // 0x30 speech to text
    missing::<7> as *const (),          // 0x38
    enumerate_voices as *const (),      // 0x40
    create as *const (),                // 0x48
    close_synthesizer as *const (),     // 0x50
    set_default_voice as *const (),     // 0x58
    set_custom_voice as *const (),      // 0x60
    create_stream_text as *const (),    // 0x68 from text
    close_stream as *const (),          // 0x70 a stream
    stream_size as *const (),           // 0x78
    stream_data as *const (),           // 0x80
    missing::<17> as *const (),         // 0x88 speech to text
    missing::<18> as *const (),         // 0x90
    missing::<19> as *const (),         // 0x98
    missing::<20> as *const (),         // 0xa0
    create_stream_ssml as *const (),    // 0xa8 from SSML
]);

#[cfg(windows)]
mod sapi {
    use super::*;
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

    const COINIT_APARTMENTTHREADED: u32 = 0x2;
    const CLSCTX_ALL: u32 = 0x17;
    const STATFLAG_NONAME: u32 = 1;
    const SPF_IS_XML: u32 = 1 << 3;
    const SPF_IS_NOT_XML: u32 = 1 << 4;

    const CLSID_SP_VOICE: Guid = Guid::new(
        0x9674_9377,
        0x3391,
        0x11d2,
        [0x9e, 0xe3, 0x00, 0xc0, 0x4f, 0x79, 0x73, 0x96],
    );
    const IID_ISP_VOICE: Guid = Guid::new(
        0x6c44_df74,
        0x72b9,
        0x4992,
        [0xa1, 0xec, 0xef, 0x99, 0x6e, 0x04, 0x22, 0xd4],
    );
    const CLSID_SP_STREAM: Guid = Guid::new(
        0x715d_9c59,
        0x4442,
        0x11d2,
        [0x96, 0x05, 0x00, 0xc0, 0x4f, 0x8e, 0xe6, 0x28],
    );
    const IID_ISP_STREAM: Guid = Guid::new(
        0x12e3_cca9,
        0x7518,
        0x44c5,
        [0xa5, 0xe7, 0xba, 0x5a, 0x79, 0xcb, 0x92, 0x9e],
    );
    const SPDFID_WAVE_FORMAT_EX: Guid = Guid::new(
        0xc31a_dbae,
        0x527f,
        0x4ff5,
        [0xa2, 0x30, 0xf6, 0x2b, 0xb6, 0x1f, 0xf7, 0x0c],
    );
    const CLSID_SP_OBJECT_TOKEN_CATEGORY: Guid = Guid::new(
        0xa910_187f,
        0x0c7a,
        0x45ac,
        [0x92, 0xcc, 0x59, 0xed, 0xaf, 0xb7, 0x7b, 0x53],
    );
    const IID_ISP_OBJECT_TOKEN_CATEGORY: Guid = Guid::new(
        0x2d3d_3845,
        0x39af,
        0x4850,
        [0xbb, 0xf9, 0x40, 0xb4, 0x97, 0x80, 0x01, 0x1d],
    );

    #[repr(C, packed)]
    #[derive(Clone, Copy)]
    struct WaveFormatEx {
        format_tag: u16,
        channels: u16,
        samples_per_sec: u32,
        avg_bytes_per_sec: u32,
        block_align: u16,
        bits_per_sample: u16,
        cb_size: u16,
    }

    impl WaveFormatEx {
        const PCM_16KHZ_MONO: Self = WaveFormatEx {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 16_000,
            avg_bytes_per_sec: 32_000,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
    }

    #[repr(C)]
    struct StatStg {
        pwcs_name: *mut u16,
        kind: u32,
        cb_size: u64,
        mtime: u64,
        ctime: u64,
        atime: u64,
        grf_mode: u32,
        grf_locks_supported: u32,
        clsid: Guid,
        grf_state_bits: u32,
        reserved: u32,
    }

    type FnCoInitializeEx = unsafe extern "system" fn(*const c_void, u32) -> HRESULT;
    type FnCoUninitialize = unsafe extern "system" fn();
    type FnCoCreateInstance = unsafe extern "system" fn(
        *const Guid,
        *mut c_void,
        u32,
        *const Guid,
        *mut *mut c_void,
    ) -> HRESULT;
    type FnCoTaskMemFree = unsafe extern "system" fn(*mut c_void);
    type FnCreateStreamOnHGlobal =
        unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> HRESULT;
    type FnGetHGlobalFromStream =
        unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
    type FnGlobalLock = unsafe extern "system" fn(*mut c_void) -> *mut c_void;
    type FnGlobalUnlock = unsafe extern "system" fn(*mut c_void) -> i32;
    type FnLcidToLocaleName = unsafe extern "system" fn(u32, *mut u16, i32, u32) -> i32;

    struct WinApi {
        co_initialize_ex: FnCoInitializeEx,
        co_uninitialize: FnCoUninitialize,
        co_create_instance: FnCoCreateInstance,
        co_task_mem_free: FnCoTaskMemFree,
        create_stream_on_hglobal: FnCreateStreamOnHGlobal,
        get_hglobal_from_stream: FnGetHGlobalFromStream,
        global_lock: FnGlobalLock,
        global_unlock: FnGlobalUnlock,
        lcid_to_locale_name: Option<FnLcidToLocaleName>,
    }

    impl WinApi {
        fn load() -> Option<Self> {
            // SAFETY: loading standard system DLLs and resolving documented exports.
            unsafe {
                let ole32 = LoadLibraryA(c"ole32.dll".as_ptr().cast());
                let kernel32 = LoadLibraryA(c"kernel32.dll".as_ptr().cast());
                if ole32.is_null() || kernel32.is_null() {
                    return None;
                }
                macro_rules! sym {
                    ($dll:expr, $name:expr, $ty:ty) => {{
                        let p = GetProcAddress($dll, $name.as_ptr().cast())?;
                        std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(p)
                    }};
                }
                let lcid_to_locale_name =
                    GetProcAddress(kernel32, c"LCIDToLocaleName".as_ptr().cast()).map(|p| {
                        std::mem::transmute::<
                            unsafe extern "system" fn() -> isize,
                            FnLcidToLocaleName,
                        >(p)
                    });
                Some(WinApi {
                    co_initialize_ex: sym!(ole32, c"CoInitializeEx", FnCoInitializeEx),
                    co_uninitialize: sym!(ole32, c"CoUninitialize", FnCoUninitialize),
                    co_create_instance: sym!(ole32, c"CoCreateInstance", FnCoCreateInstance),
                    co_task_mem_free: sym!(ole32, c"CoTaskMemFree", FnCoTaskMemFree),
                    create_stream_on_hglobal: sym!(
                        ole32,
                        c"CreateStreamOnHGlobal",
                        FnCreateStreamOnHGlobal
                    ),
                    get_hglobal_from_stream: sym!(
                        ole32,
                        c"GetHGlobalFromStream",
                        FnGetHGlobalFromStream
                    ),
                    global_lock: sym!(kernel32, c"GlobalLock", FnGlobalLock),
                    global_unlock: sym!(kernel32, c"GlobalUnlock", FnGlobalUnlock),
                    lcid_to_locale_name,
                })
            }
        }
    }

    struct ComPtr(*mut c_void);

    impl ComPtr {
        fn as_ptr(&self) -> *mut c_void {
            self.0
        }

        /// # Safety
        /// `slot` must be a valid vtable index on this COM interface.
        unsafe fn slot(&self, slot: usize) -> *const () {
            unsafe {
                let vtbl = *(self.0 as *const *const *const ());
                *vtbl.add(slot)
            }
        }
    }

    impl Drop for ComPtr {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: slot 2 is `IUnknown::Release`.
                unsafe {
                    let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                        std::mem::transmute(self.slot(2));
                    release(self.0);
                }
            }
        }
    }

    fn utf16_nul(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    unsafe fn wide_to_string(ptr: *const u16) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0;
        // SAFETY: NUL-terminated UTF-16 string from COM.
        while unsafe { *ptr.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `len` u16 code units before NUL.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        String::from_utf16_lossy(slice)
    }

    fn to_cstring(s: &str) -> CString {
        CString::new(s.replace('\0', "")).unwrap_or_default()
    }

    fn build_wav(wfx: WaveFormatEx, pcm: &[u8]) -> Vec<u8> {
        let data_len = pcm.len() as u32;
        let riff_len = 36u32.saturating_add(data_len);
        let mut wav = Vec::with_capacity(44 + pcm.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&riff_len.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        let tag = wfx.format_tag;
        let ch = wfx.channels;
        let rate = wfx.samples_per_sec;
        let avg = wfx.avg_bytes_per_sec;
        let align = wfx.block_align;
        let bits = wfx.bits_per_sample;
        wav.extend_from_slice(&tag.to_le_bytes());
        wav.extend_from_slice(&ch.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&avg.to_le_bytes());
        wav.extend_from_slice(&align.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(pcm);
        wav
    }

    fn lcid_hex_to_bcp47(api: &WinApi, raw: &str) -> String {
        let first = raw.split(';').next().unwrap_or(raw).trim();
        let Ok(lcid) = u32::from_str_radix(first, 16) else {
            return "en-US".to_owned();
        };
        if let Some(f) = api.lcid_to_locale_name {
            let mut buf = [0u16; 85];
            // SAFETY: buffer holds 85 UTF-16 code units (LOCALE_NAME_MAX_LENGTH).
            let n = unsafe { f(lcid, buf.as_mut_ptr(), buf.len() as i32, 0) };
            if n > 1 {
                return String::from_utf16_lossy(&buf[..(n as usize - 1)]);
            }
        }
        "en-US".to_owned()
    }

    fn datakey_get_string(api: &WinApi, key: &ComPtr, name: Option<&str>) -> Option<String> {
        let wide_name = name.map(utf16_nul);
        let name_ptr = wide_name.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut out: *mut u16 = std::ptr::null_mut();
        // SAFETY: `ISpDataKey::GetStringValue` is vtable slot 6.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const u16, *mut *mut u16) -> HRESULT =
                std::mem::transmute(key.slot(6));
            f(key.as_ptr(), name_ptr, &mut out)
        };
        if hr < 0 || out.is_null() {
            return None;
        }
        // SAFETY: COM-allocated NUL-terminated UTF-16 string.
        let value = unsafe { wide_to_string(out) };
        unsafe { (api.co_task_mem_free)(out.cast()) };
        Some(value)
    }

    fn token_get_id(api: &WinApi, token: &ComPtr) -> Option<String> {
        let mut out: *mut u16 = std::ptr::null_mut();
        // SAFETY: `ISpObjectToken::GetId` is vtable slot 16.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut *mut u16) -> HRESULT =
                std::mem::transmute(token.slot(16));
            f(token.as_ptr(), &mut out)
        };
        if hr < 0 || out.is_null() {
            return None;
        }
        // SAFETY: COM-allocated NUL-terminated UTF-16 string.
        let value = unsafe { wide_to_string(out) };
        unsafe { (api.co_task_mem_free)(out.cast()) };
        Some(value)
    }

    fn enum_voice_tokens(api: &WinApi) -> Option<ComPtr> {
        let mut cat_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: creating SAPI's `SpObjectTokenCategory`.
        let hr = unsafe {
            (api.co_create_instance)(
                &CLSID_SP_OBJECT_TOKEN_CATEGORY,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_ISP_OBJECT_TOKEN_CATEGORY,
                &mut cat_raw,
            )
        };
        if hr < 0 || cat_raw.is_null() {
            return None;
        }
        let cat = ComPtr(cat_raw);
        let voices_path = utf16_nul(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices");
        // SAFETY: `ISpObjectTokenCategory::SetId` is vtable slot 15.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const u16, i32) -> HRESULT =
                std::mem::transmute(cat.slot(15));
            f(cat.as_ptr(), voices_path.as_ptr(), 0)
        };
        if hr < 0 {
            return None;
        }
        let mut enum_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ISpObjectTokenCategory::EnumTokens` is vtable slot 18.
        let hr = unsafe {
            let f: unsafe extern "system" fn(
                *mut c_void,
                *const u16,
                *const u16,
                *mut *mut c_void,
            ) -> HRESULT = std::mem::transmute(cat.slot(18));
            f(cat.as_ptr(), std::ptr::null(), std::ptr::null(), &mut enum_raw)
        };
        if hr < 0 || enum_raw.is_null() {
            return None;
        }
        Some(ComPtr(enum_raw))
    }

    fn collect_voices(api: &WinApi) -> Vec<VoiceRecord> {
        let Some(enum_tokens) = enum_voice_tokens(api) else {
            return Vec::new();
        };
        let mut count = 0u32;
        // SAFETY: `IEnumSpObjectTokens::GetCount` is vtable slot 8.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT =
                std::mem::transmute(enum_tokens.slot(8));
            f(enum_tokens.as_ptr(), &mut count)
        };
        if hr < 0 {
            return Vec::new();
        }
        let mut voices = Vec::with_capacity(count as usize);
        let attrs_name = utf16_nul("Attributes");
        for i in 0..count {
            let mut tok_raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `IEnumSpObjectTokens::Item` is vtable slot 7.
            let hr = unsafe {
                let f: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT =
                    std::mem::transmute(enum_tokens.slot(7));
                f(enum_tokens.as_ptr(), i, &mut tok_raw)
            };
            if hr < 0 || tok_raw.is_null() {
                continue;
            }
            let token = ComPtr(tok_raw);
            let Some(voice_id) = token_get_id(api, &token) else {
                continue;
            };
            let mut display_name = datakey_get_string(api, &token, None).unwrap_or_default();
            let mut gender = 0u32; // Female default
            let mut language = "en-US".to_owned();

            let mut attrs_raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ISpDataKey::OpenKey` is vtable slot 9.
            let hr = unsafe {
                let f: unsafe extern "system" fn(
                    *mut c_void,
                    *const u16,
                    *mut *mut c_void,
                ) -> HRESULT = std::mem::transmute(token.slot(9));
                f(token.as_ptr(), attrs_name.as_ptr(), &mut attrs_raw)
            };
            if hr >= 0 && !attrs_raw.is_null() {
                let attrs = ComPtr(attrs_raw);
                if let Some(name) = datakey_get_string(api, &attrs, Some("Name"))
                    && !name.is_empty()
                {
                    display_name = name;
                }
                if let Some(g) = datakey_get_string(api, &attrs, Some("Gender"))
                    && g.eq_ignore_ascii_case("male")
                {
                    gender = 1;
                }
                if let Some(lang) = datakey_get_string(api, &attrs, Some("Language")) {
                    language = lcid_hex_to_bcp47(api, &lang);
                }
            }
            if display_name.is_empty() {
                display_name = voice_id.clone();
            }
            let description = format!("{display_name} ({language})");
            voices.push(VoiceRecord {
                description: to_cstring(&description),
                display_name: to_cstring(&display_name),
                gender,
                voice_id: to_cstring(&voice_id),
                language: to_cstring(&language),
            });
        }
        voices
    }

    fn find_voice_token(api: &WinApi, wanted: &str) -> Option<ComPtr> {
        let enum_tokens = enum_voice_tokens(api)?;
        let mut count = 0u32;
        // SAFETY: `IEnumSpObjectTokens::GetCount` is vtable slot 8.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT =
                std::mem::transmute(enum_tokens.slot(8));
            f(enum_tokens.as_ptr(), &mut count)
        };
        if hr < 0 {
            return None;
        }
        for i in 0..count {
            let mut tok_raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: `IEnumSpObjectTokens::Item` is vtable slot 7.
            let hr = unsafe {
                let f: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> HRESULT =
                    std::mem::transmute(enum_tokens.slot(7));
                f(enum_tokens.as_ptr(), i, &mut tok_raw)
            };
            if hr < 0 || tok_raw.is_null() {
                continue;
            }
            let token = ComPtr(tok_raw);
            if token_get_id(api, &token).is_some_and(|id| id.eq_ignore_ascii_case(wanted)) {
                return Some(token);
            }
            if datakey_get_string(api, &token, None)
                .is_some_and(|n| n.eq_ignore_ascii_case(wanted))
            {
                return Some(token);
            }
        }
        None
    }

    fn run_synthesize(
        api: &WinApi,
        voice_id: Option<&str>,
        text: &str,
        is_ssml: bool,
    ) -> Option<Vec<u8>> {
        let mut mem_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `CreateStreamOnHGlobal` allocates a growable in-memory `IStream`.
        let hr = unsafe { (api.create_stream_on_hglobal)(std::ptr::null_mut(), 1, &mut mem_raw) };
        if hr < 0 || mem_raw.is_null() {
            return None;
        }
        let mem_stream = ComPtr(mem_raw);

        let mut sp_stream_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: creating SAPI's `SpStream`.
        let hr = unsafe {
            (api.co_create_instance)(
                &CLSID_SP_STREAM,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_ISP_STREAM,
                &mut sp_stream_raw,
            )
        };
        if hr < 0 || sp_stream_raw.is_null() {
            return None;
        }
        let sp_stream = ComPtr(sp_stream_raw);

        let mut wfx = WaveFormatEx::PCM_16KHZ_MONO;
        // SAFETY: `ISpStream::SetBaseStream` is vtable slot 15.
        let hr = unsafe {
            let f: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const Guid,
                *const WaveFormatEx,
            ) -> HRESULT = std::mem::transmute(sp_stream.slot(15));
            f(
                sp_stream.as_ptr(),
                mem_stream.as_ptr(),
                &SPDFID_WAVE_FORMAT_EX,
                &wfx,
            )
        };
        if hr < 0 {
            return None;
        }

        let mut voice_raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: creating SAPI's `SpVoice`.
        let hr = unsafe {
            (api.co_create_instance)(
                &CLSID_SP_VOICE,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_ISP_VOICE,
                &mut voice_raw,
            )
        };
        if hr < 0 || voice_raw.is_null() {
            return None;
        }
        let sp_voice = ComPtr(voice_raw);

        if let Some(wanted) = voice_id
            && let Some(token) = find_voice_token(api, wanted)
        {
            // SAFETY: `ISpVoice::SetVoice` is vtable slot 18.
            unsafe {
                let f: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT =
                    std::mem::transmute(sp_voice.slot(18));
                let _ = f(sp_voice.as_ptr(), token.as_ptr());
            }
        }

        // SAFETY: `ISpVoice::SetOutput` is vtable slot 13; `fAllowFormatChanges = FALSE`
        // asks SAPI to convert into our 16 kHz mono 16-bit PCM stream.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> HRESULT =
                std::mem::transmute(sp_voice.slot(13));
            f(sp_voice.as_ptr(), sp_stream.as_ptr(), 0)
        };
        if hr < 0 {
            return None;
        }

        let wide = utf16_nul(text);
        let flags = if is_ssml { SPF_IS_XML } else { SPF_IS_NOT_XML };
        // SAFETY: `ISpVoice::Speak` is vtable slot 20 (synchronous without `SPF_ASYNC`).
        let hr = unsafe {
            let f: unsafe extern "system" fn(
                *mut c_void,
                *const u16,
                u32,
                *mut u32,
            ) -> HRESULT = std::mem::transmute(sp_voice.slot(20));
            f(sp_voice.as_ptr(), wide.as_ptr(), flags, std::ptr::null_mut())
        };

        // Detach output before reading so SAPI flushes everything into `mem_stream`.
        // SAFETY: `ISpVoice::SetOutput(NULL, FALSE)` detaches output cleanly.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> HRESULT =
                std::mem::transmute(sp_voice.slot(13));
            let _ = f(sp_voice.as_ptr(), std::ptr::null_mut(), 0);
        }
        if hr < 0 {
            return None;
        }

        // Confirm the stream's actual WAVEFORMATEX before closing.
        let mut fmt_id = Guid::new(0, 0, 0, [0; 8]);
        let mut out_wfx: *mut WaveFormatEx = std::ptr::null_mut();
        // SAFETY: `ISpStreamFormat::GetFormat` is vtable slot 14.
        let fmt_hr = unsafe {
            let f: unsafe extern "system" fn(
                *mut c_void,
                *mut Guid,
                *mut *mut WaveFormatEx,
            ) -> HRESULT = std::mem::transmute(sp_stream.slot(14));
            f(sp_stream.as_ptr(), &mut fmt_id, &mut out_wfx)
        };
        if fmt_hr >= 0 && !out_wfx.is_null() {
            if fmt_id == SPDFID_WAVE_FORMAT_EX {
                // SAFETY: `CoTaskMem`-allocated `WAVEFORMATEX` returned by `GetFormat`.
                wfx = unsafe { out_wfx.read_unaligned() };
            }
            unsafe { (api.co_task_mem_free)(out_wfx.cast()) };
        }

        // SAFETY: `ISpStream::Close` is vtable slot 18.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void) -> HRESULT =
                std::mem::transmute(sp_stream.slot(18));
            let _ = f(sp_stream.as_ptr());
        }

        let mut stat: StatStg = unsafe { std::mem::zeroed() };
        // SAFETY: `IStream::Stat` is vtable slot 12.
        let hr = unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *mut StatStg, u32) -> HRESULT =
                std::mem::transmute(mem_stream.slot(12));
            f(mem_stream.as_ptr(), &mut stat, STATFLAG_NONAME)
        };
        if hr < 0 || stat.cb_size == 0 || stat.cb_size > (64 << 20) {
            return None;
        }
        let pcm_len = stat.cb_size as usize;

        let mut hglobal: *mut c_void = std::ptr::null_mut();
        // SAFETY: `mem_stream` was created by `CreateStreamOnHGlobal`.
        let hr = unsafe { (api.get_hglobal_from_stream)(mem_stream.as_ptr(), &mut hglobal) };
        if hr < 0 || hglobal.is_null() {
            return None;
        }
        // SAFETY: locking the backing `HGLOBAL` to read `pcm_len` bytes.
        let locked = unsafe { (api.global_lock)(hglobal) };
        if locked.is_null() {
            return None;
        }
        // SAFETY: `IStream::Stat` reported `pcm_len` valid bytes written to the stream.
        let pcm = unsafe { std::slice::from_raw_parts(locked.cast::<u8>(), pcm_len) };
        let wav = build_wav(wfx, pcm);
        unsafe { (api.global_unlock)(hglobal) };
        Some(wav)
    }

    enum Job {
        EnumerateVoices(mpsc::Sender<Vec<VoiceRecord>>),
        Synthesize {
            voice_id: Option<String>,
            text: String,
            is_ssml: bool,
            reply: mpsc::Sender<Option<Vec<u8>>>,
        },
    }

    /// Runs all SAPI COM work on a dedicated STA thread so we never disturb
    /// the caller's COM apartment.
    fn worker() -> Option<&'static mpsc::Sender<Job>> {
        static TX: OnceLock<Option<mpsc::Sender<Job>>> = OnceLock::new();
        TX.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Job>();
            let (ready_tx, ready_rx) = mpsc::channel::<bool>();
            std::thread::Builder::new()
                .name("victory SAPI speech".into())
                .spawn(move || {
                    let Some(api) = WinApi::load() else {
                        let _ = ready_tx.send(false);
                        return;
                    };
                    // SAFETY: initializing COM STA once for this dedicated worker thread.
                    let hr = unsafe {
                        (api.co_initialize_ex)(std::ptr::null(), COINIT_APARTMENTTHREADED)
                    };
                    let initialized = hr >= 0;
                    if !initialized {
                        let _ = ready_tx.send(false);
                        return;
                    }
                    let _ = ready_tx.send(true);
                    while let Ok(job) = rx.recv() {
                        match job {
                            Job::EnumerateVoices(reply) => {
                                let _ = reply.send(collect_voices(&api));
                            }
                            Job::Synthesize {
                                voice_id,
                                text,
                                is_ssml,
                                reply,
                            } => {
                                let wav = run_synthesize(&api, voice_id.as_deref(), &text, is_ssml);
                                let _ = reply.send(wav);
                            }
                        }
                    }
                    // SAFETY: balanced with successful `CoInitializeEx` above.
                    unsafe { (api.co_uninitialize)() };
                })
                .ok()?;
            ready_rx.recv().ok().filter(|&ok| ok).map(|_| tx)
        })
        .as_ref()
    }

    pub(super) fn enumerate_voices() -> Vec<VoiceRecord> {
        let Some(tx) = worker() else {
            return Vec::new();
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send(Job::EnumerateVoices(reply_tx)).is_err() {
            return Vec::new();
        }
        reply_rx.recv().unwrap_or_default()
    }

    pub(super) fn synthesize(
        voice_id: Option<String>,
        text: String,
        is_ssml: bool,
    ) -> Option<Vec<u8>> {
        let tx = worker()?;
        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(Job::Synthesize {
            voice_id,
            text,
            is_ssml,
            reply: reply_tx,
        })
        .ok()?;
        reply_rx.recv().ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn silent_mode_produces_empty_wave_file() {
        let wav = synthesize_wav(None, "Hello", false, false);
        assert_eq!(wav.len(), 44);
        assert_eq!(
            (&wav[..4], &wav[8..16], &wav[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 0);
    }

    #[test]
    fn sapi_synthesizes_valid_non_silent_wave_and_handles_buffer_checks() {
        let this: This = &OBJECT;
        type Create = extern "system" fn(This, *mut *const u8) -> HRESULT;
        type Close = extern "system" fn(This, *const u8) -> HRESULT;
        type Stream = extern "system" fn(This, *const u8, *const c_char, *mut *const u8) -> HRESULT;
        type Size = extern "system" fn(This, *const u8, *mut usize) -> HRESULT;
        type Data = extern "system" fn(This, *const u8, usize, *mut u8, *mut usize) -> HRESULT;
        type EnumVoices = extern "system" fn(This, *mut c_void, *const c_void) -> HRESULT;

        // Enumerate voices through the vtable (slot 0x40).
        extern "system" fn collect_cb(info: *const VoiceInfo, ctx: *mut c_void) -> bool {
            // SAFETY: test passes `&mut Vec<String>` as context and valid `info`.
            let list = unsafe { &mut *ctx.cast::<Vec<String>>() };
            let info = unsafe { &*info };
            let id = unsafe { CStr::from_ptr(info.voice_id) }
                .to_string_lossy()
                .into_owned();
            let name = unsafe { CStr::from_ptr(info.display_name) }
                .to_string_lossy()
                .into_owned();
            assert!(!id.is_empty() && !name.is_empty());
            list.push(id);
            true
        }
        let mut voices: Vec<String> = Vec::new();
        assert_eq!(
            method::<EnumVoices>(&OBJECT, 0x40)(
                this,
                (&raw mut voices).cast(),
                collect_cb as *const c_void
            ),
            S_OK
        );

        let (mut synth, mut stream, mut size) = (std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let say = method::<Stream>(&OBJECT, 0x68);
        assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
        assert_eq!(method::<Size>(&OBJECT, 0x78)(this, stream, &mut size), S_OK);
        assert!(size >= 44);

        // Too-small buffer must fail with E_NOT_SUFFICIENT_BUFFER.
        let mut tiny = [0u8; 8];
        let mut used = 0;
        let data = method::<Data>(&OBJECT, 0x80);
        assert_eq!(
            data(this, stream, tiny.len(), tiny.as_mut_ptr(), &mut used),
            E_NOT_SUFFICIENT_BUFFER
        );

        let mut wav = vec![0u8; size];
        assert_eq!(data(this, stream, size, wav.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(used, size);
        assert_eq!(
            (&wav[..4], &wav[8..16], &wav[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        let data_bytes = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize;
        assert_eq!(44 + data_bytes, size);

        // When Windows SAPI has installed voices and speech isn't disabled, expect real audio.
        if speech_enabled() && !voices.is_empty() {
            assert!(size > 44, "expected non-empty SAPI WAV output");
            let samples: Vec<i16> = wav[44..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c))
                .collect();
            let energy: f64 =
                samples.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
                    / samples.len().max(1) as f64;
            assert!(energy > 10.0, "expected non-zero speech energy, got {energy}");

            // Also verify SSML synthesis (slot 0xa8).
            let mut ssml_stream = std::ptr::null();
            let mut ssml_size = 0;
            let say_ssml = method::<Stream>(&OBJECT, 0xa8);
            assert_eq!(
                say_ssml(
                    this,
                    synth,
                    c"<speak version='1.0' xml:lang='en-US'>Hello</speak>".as_ptr(),
                    &mut ssml_stream
                ),
                S_OK
            );
            assert_eq!(
                method::<Size>(&OBJECT, 0x78)(this, ssml_stream, &mut ssml_size),
                S_OK
            );
            assert!(ssml_size > 44, "expected non-empty SSML WAV output");
            assert_eq!(method::<Close>(&OBJECT, 0x70)(this, ssml_stream), S_OK);
        }

        // Closing frees handles; subsequent calls reject closed handles.
        assert_eq!(method::<Close>(&OBJECT, 0x70)(this, stream), S_OK);
        assert_eq!(
            method::<Size>(&OBJECT, 0x78)(this, stream, &mut size),
            E_INVALIDARG
        );
        assert_eq!(method::<Close>(&OBJECT, 0x50)(this, synth), S_OK);
        assert_eq!(method::<Close>(&OBJECT, 0x50)(this, synth), E_INVALIDARG);
    }
}
