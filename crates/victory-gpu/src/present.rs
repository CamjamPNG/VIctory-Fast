//! Native PC DXGI presentation, HDR10 output, host display mode enumeration,
//! and zero-cost-when-idle on-demand screenshot capture.
//!
//! Flip-Model (`DXGI_SWAP_EFFECT_FLIP_DISCARD`) windowed/borderless swap chains
//! created with `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING` + `DXGI_MWA_NO_ALT_ENTER`
//! and presented with `Present(0, DXGI_PRESENT_ALLOW_TEARING)` are safe on
//! Windows 10/11: they enable VRR (FreeSync / G-Sync) and eliminate DWM vsync
//! stalls while keeping Alt+Enter from transitioning into exclusive fullscreen
//! (where `DXGI_PRESENT_ALLOW_TEARING` is disallowed by DXGI).
//!
//! Environment switches:
//! - `VICTORY_TEARING`: opt-out (`0` disables; unset or non-`0` enables when
//!   hardware supports `DXGI_FEATURE_PRESENT_ALLOW_TEARING`).
//! - `VICTORY_VSYNC`: opt-in (`1` presents with `SyncInterval = 1` and
//!   suppresses `DXGI_PRESENT_ALLOW_TEARING`).
//! - `VICTORY_WAITABLE_SWAPCHAIN`: opt-in (`1` creates the swap chain with
//!   `DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT`).
//! - `VICTORY_MAX_FRAME_LATENCY`: optional `u32` clamped to `1..=16` (default
//!   `1` when waitable swapchain is enabled).
//! - `VICTORY_HDR`: opt-in (`1` enables HDR10 output when the host monitor is
//!   actively in HDR10 mode).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{HMONITOR, MONITOR_DEFAULTTONEAREST, MonitorFromWindow};
use windows::Win32::System::Threading::WaitForSingleObjectEx;
use windows::core::Interface;

/// Parsed presentation environment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PresentConfig {
    tearing: bool,
    vsync: bool,
    waitable: bool,
    max_frame_latency: u32,
    hdr: bool,
}

impl PresentConfig {
    fn from_vars(
        tearing: Option<&str>,
        vsync: Option<&str>,
        waitable: Option<&str>,
        max_latency: Option<&str>,
        hdr: Option<&str>,
    ) -> Self {
        Self {
            tearing: parse_tearing(tearing),
            vsync: parse_opt_in(vsync),
            waitable: parse_opt_in(waitable),
            max_frame_latency: parse_max_frame_latency(max_latency),
            hdr: parse_opt_in(hdr),
        }
    }
}

fn parse_tearing(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

fn parse_opt_in(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

fn parse_max_frame_latency(value: Option<&str>) -> u32 {
    value
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(1, |n| n.clamp(1, 16))
}

fn config() -> PresentConfig {
    static CONFIG: OnceLock<PresentConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let tearing = std::env::var("VICTORY_TEARING").ok();
        let vsync = std::env::var("VICTORY_VSYNC").ok();
        let waitable = std::env::var("VICTORY_WAITABLE_SWAPCHAIN").ok();
        let max_latency = std::env::var("VICTORY_MAX_FRAME_LATENCY").ok();
        let hdr = std::env::var("VICTORY_HDR").ok();
        PresentConfig::from_vars(
            tearing.as_deref(),
            vsync.as_deref(),
            waitable.as_deref(),
            max_latency.as_deref(),
            hdr.as_deref(),
        )
    })
}

/// Queries whether the host DXGI runtime and driver support tearing
/// (`DXGI_FEATURE_PRESENT_ALLOW_TEARING`) on flip-model swap chains.
pub(crate) fn query_tearing_support(factory: &IDXGIFactory2) -> bool {
    let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allow: u32 = 0;
    // SAFETY: `allow` is a 4-byte `BOOL`-sized output buffer matching `DXGI_FEATURE_PRESENT_ALLOW_TEARING`.
    let ok = unsafe {
        factory5
            .CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                (&raw mut allow).cast(),
                std::mem::size_of::<u32>() as u32,
            )
            .is_ok()
    };
    ok && allow != 0
}

/// Host monitor HDR10 capability and luminance metadata from `IDXGIOutput6::GetDesc1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrCapabilities {
    pub supported: bool,
    pub min_luminance: f32,
    pub max_luminance: f32,
    pub max_full_frame_luminance: f32,
}

impl Default for HdrCapabilities {
    fn default() -> Self {
        Self {
            supported: false,
            min_luminance: 0.0,
            max_luminance: 0.0,
            max_full_frame_luminance: 0.0,
        }
    }
}

fn hdr_caps_from_desc1(desc: &DXGI_OUTPUT_DESC1) -> HdrCapabilities {
    let supported = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    let min_luminance = if desc.MinLuminance.is_finite() && desc.MinLuminance >= 0.0 {
        desc.MinLuminance
    } else {
        0.0
    };
    let max_luminance = if desc.MaxLuminance.is_finite() && desc.MaxLuminance > 0.0 {
        desc.MaxLuminance
    } else if supported {
        1000.0
    } else {
        0.0
    };
    let max_full_frame_luminance =
        if desc.MaxFullFrameLuminance.is_finite() && desc.MaxFullFrameLuminance > 0.0 {
            desc.MaxFullFrameLuminance
        } else if supported {
            400.0
        } else {
            0.0
        };
    HdrCapabilities {
        supported,
        min_luminance,
        max_luminance,
        max_full_frame_luminance,
    }
}

fn query_hdr_for_hwnd(hwnd: Option<HWND>) -> HdrCapabilities {
    // SAFETY: plain DXGI factory and monitor queries.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return HdrCapabilities::default();
        };
        let target_monitor: Option<HMONITOR> =
            hwnd.map(|h| MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST));
        let mut fallback: Option<HdrCapabilities> = None;
        let mut adapter_idx = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(adapter_idx) {
            adapter_idx += 1;
            let mut output_idx = 0u32;
            while let Ok(output) = adapter.EnumOutputs(output_idx) {
                output_idx += 1;
                let Ok(output6) = output.cast::<IDXGIOutput6>() else {
                    continue;
                };
                let Ok(desc) = output6.GetDesc1() else {
                    continue;
                };
                let caps = hdr_caps_from_desc1(&desc);
                if target_monitor.is_some_and(|m| !m.is_invalid() && m == desc.Monitor) {
                    return caps;
                }
                if fallback.is_none() {
                    fallback = Some(caps);
                    if target_monitor.is_none() {
                        return caps;
                    }
                }
            }
        }
        fallback.unwrap_or_default()
    }
}

/// Queries the real host monitor's HDR10 status and luminance via `IDXGIOutput6::GetDesc1`.
pub fn query_hdr_capabilities() -> HdrCapabilities {
    query_hdr_for_hwnd(None)
}

/// Returns `Some(caps)` iff `VICTORY_HDR=1` is set AND the host display reports HDR10
/// (`DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`).
pub fn hdr_for_title() -> Option<HdrCapabilities> {
    if !config().hdr {
        return None;
    }
    let caps = query_hdr_capabilities();
    caps.supported.then_some(caps)
}

/// True iff `hdr_for_title().is_some()`.
pub fn hdr_enabled() -> bool {
    hdr_for_title().is_some()
}

/// A progressive display mode enumerated from the primary host monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostDisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_num: u32,
    pub refresh_den: u32,
    pub format: u32,
}

fn normalize_display_modes(raw: &[DXGI_MODE_DESC]) -> Vec<HostDisplayMode> {
    let mut seen = HashSet::new();
    let mut modes = Vec::new();
    for m in raw {
        let progressive = matches!(
            m.ScanlineOrdering,
            DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE | DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED
        );
        if !progressive
            || m.Width == 0
            || m.Height == 0
            || m.RefreshRate.Numerator == 0
            || m.RefreshRate.Denominator == 0
        {
            continue;
        }
        let key = (
            m.Width,
            m.Height,
            m.RefreshRate.Numerator,
            m.RefreshRate.Denominator,
        );
        if seen.insert(key) {
            modes.push(HostDisplayMode {
                width: m.Width,
                height: m.Height,
                refresh_num: m.RefreshRate.Numerator,
                refresh_den: m.RefreshRate.Denominator,
                format: m.Format.0 as u32,
            });
        }
    }
    modes.sort_by(|a, b| {
        let a_pixels = u64::from(a.width) * u64::from(a.height);
        let b_pixels = u64::from(b.width) * u64::from(b.height);
        let a_hz = u128::from(a.refresh_num) * u128::from(b.refresh_den);
        let b_hz = u128::from(b.refresh_num) * u128::from(a.refresh_den);
        (b_pixels, b.width, b.height, b_hz).cmp(&(a_pixels, a.width, a.height, a_hz))
    });
    modes
}

/// Enumerates progressive display modes from the primary host `IDXGIOutput::GetDisplayModeList`.
pub fn host_display_modes() -> Vec<HostDisplayMode> {
    // SAFETY: plain DXGI adapter/output enumeration.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return Vec::new();
        };
        let Ok(adapter) = factory.EnumAdapters1(0) else {
            return Vec::new();
        };
        let Ok(output) = adapter.EnumOutputs(0) else {
            return Vec::new();
        };
        let mut count = 0u32;
        if output
            .GetDisplayModeList(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_ENUM_MODES(0),
                &mut count,
                None,
            )
            .is_err()
            || count == 0
        {
            return Vec::new();
        }
        let mut raw = vec![DXGI_MODE_DESC::default(); count as usize];
        if output
            .GetDisplayModeList(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_ENUM_MODES(0),
                &mut count,
                Some(raw.as_mut_ptr()),
            )
            .is_err()
        {
            return Vec::new();
        }
        raw.truncate(count as usize);
        normalize_display_modes(&raw)
    }
}

fn mode_refresh_interval(mode: HostDisplayMode) -> Option<Duration> {
    if mode.refresh_num == 0 || mode.refresh_den == 0 {
        return None;
    }
    let num = u128::from(mode.refresh_num);
    let den = u128::from(mode.refresh_den);
    let nanos = (den * 1_000_000_000 + num / 2) / num;
    u64::try_from(nanos).ok().map(Duration::from_nanos)
}

/// Returns the refresh interval of the highest refresh rate on the primary
/// display mode (e.g. 144 Hz -> 6.944 ms, 120 Hz -> 8.333 ms).
pub fn primary_refresh_interval() -> Option<Duration> {
    let modes = host_display_modes();
    if modes.is_empty() {
        return None;
    }
    // Prefer the highest refresh rate at the primary output's current desktop
    // resolution when available, falling back to the highest-resolution mode.
    let desktop_size: Option<(u32, u32)> = (|| {
        // SAFETY: plain DXGI primary output description query.
        unsafe {
            let factory = CreateDXGIFactory2::<IDXGIFactory2>(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter = factory.EnumAdapters1(0).ok()?;
            let output = adapter.EnumOutputs(0).ok()?;
            let desc = output.GetDesc().ok()?;
            let w = desc
                .DesktopCoordinates
                .right
                .saturating_sub(desc.DesktopCoordinates.left);
            let h = desc
                .DesktopCoordinates
                .bottom
                .saturating_sub(desc.DesktopCoordinates.top);
            (w > 0 && h > 0).then_some((w as u32, h as u32))
        }
    })();
    let chosen = desktop_size
        .and_then(|(w, h)| modes.iter().find(|m| m.width == w && m.height == h))
        .or_else(|| modes.first())?;
    mode_refresh_interval(*chosen)
}

/// Active presentation state owned by `Host`'s `Presenter`.
pub(crate) struct PresentState {
    tearing_enabled: bool,
    vsync: bool,
    waitable: Option<HANDLE>,
    max_frame_latency: u32,
    swap_chain_flags: DXGI_SWAP_CHAIN_FLAG,
    buffer_count: u32,
    hdr_active: bool,
    color_space: DXGI_COLOR_SPACE_TYPE,
}

// SAFETY: `waitable` is a kernel event handle created by DXGI for this swap
// chain; waiting on it is thread-safe and it is closed once in `Drop`.
unsafe impl Send for PresentState {}
// SAFETY: `PresentState` is accessed under `Host::presenter`'s mutex.
unsafe impl Sync for PresentState {}

impl Drop for PresentState {
    fn drop(&mut self) {
        if let Some(handle) = self.waitable.take()
            && !handle.is_invalid()
        {
            // SAFETY: handle returned by `GetFrameLatencyWaitableObject`, owned by `PresentState`.
            let _ = unsafe { CloseHandle(handle) };
        }
    }
}

fn sdr_swap_format(src_format: DXGI_FORMAT, two_plane_compose: bool) -> Option<DXGI_FORMAT> {
    if two_plane_compose {
        return Some(DXGI_FORMAT_R8G8B8A8_UNORM);
    }
    Some(match src_format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

fn select_swap_format_for(
    src_format: DXGI_FORMAT,
    two_plane_compose: bool,
    hdr: bool,
) -> Option<DXGI_FORMAT> {
    if two_plane_compose {
        return Some(if hdr {
            DXGI_FORMAT_R10G10B10A2_UNORM
        } else {
            DXGI_FORMAT_R8G8B8A8_UNORM
        });
    }
    // Single-plane `CopyResource` requires matching DXGI format families:
    // 10-bit/16-bit HDR surfaces already map to R10G10B10A2_UNORM /
    // R16G16B16A16_FLOAT, while 8-bit SDR surfaces (e.g. splash screens) stay
    // 8-bit until the title switches to its HDR target.
    sdr_swap_format(src_format, false)
}

/// Selects the host swap-chain format for a presented title surface.
pub(crate) fn select_swap_format(
    src_format: DXGI_FORMAT,
    two_plane_compose: bool,
) -> Option<DXGI_FORMAT> {
    let hdr = two_plane_compose && hdr_enabled();
    select_swap_format_for(src_format, two_plane_compose, hdr)
}

fn desired_color_space(format: DXGI_FORMAT, want_hdr: bool) -> DXGI_COLOR_SPACE_TYPE {
    if !want_hdr {
        return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    }
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        DXGI_FORMAT_R10G10B10A2_UNORM => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        _ => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
    }
}

fn apply_color_space(
    swap_chain: &IDXGISwapChain3,
    format: DXGI_FORMAT,
    want_hdr: bool,
) -> (bool, DXGI_COLOR_SPACE_TYPE) {
    let target = desired_color_space(format, want_hdr);
    if target != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 {
        // SAFETY: `swap_chain` is live.
        let support = unsafe { swap_chain.CheckColorSpaceSupport(target) }.unwrap_or(0);
        if (support & (DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32)) != 0
            && unsafe { swap_chain.SetColorSpace1(target) }.is_ok()
        {
            return (true, target);
        }
    }
    // SAFETY: fallback to SDR sRGB color space.
    let _ = unsafe { swap_chain.SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) };
    (false, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)
}

/// Creates a flip-model DXGI swap chain configured for tearing, optional
/// waitable frame latency, and optional HDR10 output.
pub(crate) fn create_swap_chain(
    queue: &ID3D12CommandQueue,
    hwnd: HWND,
    width: u32,
    height: u32,
    requested_format: DXGI_FORMAT,
) -> windows::core::Result<(IDXGISwapChain3, DXGI_FORMAT, PresentState)> {
    let cfg = config();
    // SAFETY: plain DXGI factory and swap-chain creation for a live queue and HWND.
    unsafe {
        let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let tearing_enabled = cfg.tearing && query_tearing_support(&factory);
        if tearing_enabled {
            let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        }

        let mut flag_bits = 0i32;
        if tearing_enabled {
            flag_bits |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0;
        }
        if cfg.waitable {
            flag_bits |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0;
        }
        let swap_chain_flags = DXGI_SWAP_CHAIN_FLAG(flag_bits);
        let buffer_count = if cfg.waitable {
            (cfg.max_frame_latency + 1).clamp(2, 16)
        } else {
            2
        };

        let hdr_supported = cfg.hdr && query_hdr_for_hwnd(Some(hwnd)).supported;
        let actual_format = requested_format;

        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: actual_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: buffer_count,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            Flags: swap_chain_flags.0 as u32,
            ..Default::default()
        };

        let swap_chain: IDXGISwapChain3 = factory
            .CreateSwapChainForHwnd(queue, hwnd, &desc, None, None)?
            .cast()?;

        let mut waitable = None;
        if cfg.waitable {
            let _ = swap_chain.SetMaximumFrameLatency(cfg.max_frame_latency);
            let handle = swap_chain.GetFrameLatencyWaitableObject();
            if !handle.is_invalid() {
                waitable = Some(handle);
            }
        }

        let (hdr_active, color_space) = apply_color_space(&swap_chain, actual_format, hdr_supported);

        tracing::info!(
            width,
            height,
            format = actual_format.0,
            buffers = buffer_count,
            tearing = tearing_enabled,
            vsync = cfg.vsync,
            waitable = waitable.is_some(),
            max_latency = cfg.max_frame_latency,
            hdr = hdr_active,
            color_space = color_space.0,
            "presenting to a window"
        );

        let state = PresentState {
            tearing_enabled,
            vsync: cfg.vsync,
            waitable,
            max_frame_latency: cfg.max_frame_latency,
            swap_chain_flags,
            buffer_count,
            hdr_active,
            color_space,
        };
        Ok((swap_chain, actual_format, state))
    }
}

impl PresentState {
    pub(crate) fn swap_chain_flags(&self) -> DXGI_SWAP_CHAIN_FLAG {
        self.swap_chain_flags
    }

    pub(crate) fn buffer_count(&self) -> u32 {
        self.buffer_count
    }

    pub(crate) fn hdr_active(&self) -> bool {
        self.hdr_active
    }

    pub(crate) fn wait_for_latency(&self) {
        if let Some(handle) = self.waitable
            && !handle.is_invalid()
        {
            // SAFETY: valid DXGI waitable object handle; 1 s timeout prevents deadlock.
            let _ = unsafe { WaitForSingleObjectEx(handle, 1000, false) };
        }
    }

    pub(crate) fn on_resized(&mut self, swap_chain: &IDXGISwapChain3, format: DXGI_FORMAT) {
        let want_hdr = self.hdr_active || (config().hdr && query_hdr_capabilities().supported);
        let (hdr_active, color_space) = apply_color_space(swap_chain, format, want_hdr);
        self.hdr_active = hdr_active;
        self.color_space = color_space;
        tracing::debug!(
            format = format.0,
            hdr = self.hdr_active,
            color_space = self.color_space.0,
            max_latency = self.max_frame_latency,
            "re-applied swap chain color space after resize"
        );
    }

    pub(crate) fn present(&mut self, swap_chain: &IDXGISwapChain3) {
        HAS_PRESENTED.store(true, Ordering::Release);
        // SAFETY: `swap_chain` is live.
        unsafe {
            if self.vsync {
                let _ = swap_chain.Present(1, DXGI_PRESENT(0));
            } else if self.tearing_enabled {
                let hr = swap_chain.Present(0, DXGI_PRESENT_ALLOW_TEARING);
                if hr.is_err() {
                    let _ = swap_chain.Present(0, DXGI_PRESENT(0));
                }
            } else {
                let _ = swap_chain.Present(0, DXGI_PRESENT(0));
            }
        }
    }
}

/// Captured presented frame in tightly-packed 8-bit BGRA format.
#[derive(Debug, Clone)]
pub struct ScreenshotFrame {
    pub width: u32,
    pub height: u32,
    pub row_pitch: usize,
    /// 8-bit BGRA pixels (`height * row_pitch` bytes, tightly packed `row_pitch = width * 4`).
    pub bgra8: Vec<u8>,
    pub hdr: bool,
}

static HAS_PRESENTED: AtomicBool = AtomicBool::new(false);
static SCREENSHOT_PENDING: AtomicBool = AtomicBool::new(false);
static NEXT_SHOT_ID: AtomicU64 = AtomicU64::new(1);
static SCREENSHOT_WAITERS: Mutex<Vec<(u64, SyncSender<ScreenshotFrame>)>> = Mutex::new(Vec::new());

/// Hot-path check: one relaxed atomic load, zero locks or allocations when idle.
#[inline]
pub(crate) fn screenshot_pending() -> bool {
    SCREENSHOT_PENDING.load(Ordering::Relaxed)
}

/// Requests the next presented frame as BGRA8 with zero overhead when idle.
/// Returns `None` immediately if no frame has ever been presented yet, or on timeout.
pub fn take_screenshot(timeout: Duration) -> Option<ScreenshotFrame> {
    if !HAS_PRESENTED.load(Ordering::Acquire) {
        return None;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let id = NEXT_SHOT_ID.fetch_add(1, Ordering::Relaxed);
    {
        let mut waiters = SCREENSHOT_WAITERS.lock().unwrap_or_else(|e| e.into_inner());
        waiters.push((id, tx));
        SCREENSHOT_PENDING.store(true, Ordering::Release);
    }
    match rx.recv_timeout(timeout) {
        Ok(frame) => Some(frame),
        Err(_) => {
            let mut waiters = SCREENSHOT_WAITERS.lock().unwrap_or_else(|e| e.into_inner());
            waiters.retain(|(wid, _)| *wid != id);
            if waiters.is_empty() {
                SCREENSHOT_PENDING.store(false, Ordering::Release);
            }
            None
        }
    }
}

/// Convenience wrapper matching `display.rs`'s API (`500ms` timeout).
pub fn last_frame() -> Option<ScreenshotFrame> {
    take_screenshot(Duration::from_millis(500))
}

fn record_copy_to_readback(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    // SAFETY: `texture` is live; query layout for subresource 0.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    if total == 0 {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain readback buffer creation and copy into an open command list.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Records an on-demand readback copy of the swap-chain back buffer (expected in
/// `COPY_DEST` and left back in `COPY_DEST`).
pub(crate) fn record_screenshot_readback(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    back: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    // SAFETY: `list` is open and `back` is in `COPY_DEST`.
    unsafe {
        list.ResourceBarrier(&[crate::host::transition(
            back,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        let res = record_copy_to_readback(device, list, back);
        list.ResourceBarrier(&[crate::host::transition(
            back,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        res
    }
}

/// Maps a completed screenshot readback buffer, converts it to BGRA8, and
/// delivers the frame to all pending `take_screenshot` callers.
pub(crate) fn deliver_screenshot(
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    hdr: bool,
) {
    let frame = map_footprint_to_bgra8(buffer, footprint, hdr);
    let waiters = {
        let mut guard = SCREENSHOT_WAITERS.lock().unwrap_or_else(|e| e.into_inner());
        SCREENSHOT_PENDING.store(false, Ordering::Release);
        std::mem::take(&mut *guard)
    };
    let Some(frame) = frame else {
        return;
    };
    let mut iter = waiters.into_iter().peekable();
    while let Some((_, tx)) = iter.next() {
        if iter.peek().is_some() {
            let _ = tx.try_send(frame.clone());
        } else {
            let _ = tx.try_send(frame);
            break;
        }
    }
}

/// Executes a synchronous D3D12 GPU readback of `texture` into a BGRA8 `ScreenshotFrame`.
#[allow(dead_code)]
pub(crate) fn read_back_texture_bgra8(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: D3D12_RESOURCE_STATES,
    hdr: bool,
) -> Option<ScreenshotFrame> {
    let copy = host.new_list().ok()?;
    let list = &copy.list;
    // SAFETY: `copy` is open and `texture` is live in `state`.
    let (buffer, footprint) = unsafe {
        if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
            list.ResourceBarrier(&[crate::host::transition(
                texture,
                state,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        let readback = record_copy_to_readback(&host.device, list, texture);
        if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
            list.ResourceBarrier(&[crate::host::transition(
                texture,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                state,
            )]);
        }
        copy.close();
        readback?
    };
    host.execute(&[&copy]);
    map_footprint_to_bgra8(&buffer, &footprint, hdr)
}

fn map_footprint_to_bgra8(
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    hdr: bool,
) -> Option<ScreenshotFrame> {
    let f = footprint.Footprint;
    let src_pitch = f.RowPitch as usize;
    let total_bytes = src_pitch.checked_mul(f.Height as usize)?;
    let mut mapped = std::ptr::null_mut();
    // SAFETY: GPU copy has completed before mapping; read slice stays inside footprint bounds.
    unsafe {
        buffer.Map(0, None, Some(&mut mapped)).ok()?;
        let base = mapped.cast::<u8>().add(footprint.Offset as usize);
        let rows = std::slice::from_raw_parts(base, total_bytes);
        let converted = convert_rows_to_bgra8(f.Format, f.Width, f.Height, src_pitch, rows, hdr);
        buffer.Unmap(0, None);
        converted
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            let mut f = frac;
            let mut e: i32 = -14;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            f32::from_bits(sign | (((e + 127) as u32) << 23) | (f << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xff << 23) | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

fn pq_to_linear_nits(e: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 32.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 128.0;
    const C3: f32 = 2392.0 / 128.0;
    let ep = e.clamp(0.0, 1.0).powf(1.0 / M2);
    let num = (ep - C1).max(0.0);
    let den = (C2 - C3 * ep).max(1e-6);
    (num / den).powf(1.0 / M1) * 10_000.0
}

fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 };
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn tonemap_reinhard_srgb(linear_sdr_scaled: f32) -> u8 {
    let x = if linear_sdr_scaled.is_finite() {
        linear_sdr_scaled.max(0.0)
    } else {
        0.0
    };
    linear_to_srgb_u8(x / (1.0 + x))
}

fn convert_rows_to_bgra8(
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
    src_pitch: usize,
    rows: &[u8],
    hdr: bool,
) -> Option<ScreenshotFrame> {
    let w = width as usize;
    let h = height as usize;
    let dst_pitch = w.checked_mul(4)?;
    let mut bgra8 = vec![0u8; dst_pitch.checked_mul(h)?];

    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => {
            for y in 0..h {
                let src = &rows[y * src_pitch..y * src_pitch + dst_pitch];
                let dst = &mut bgra8[y * dst_pitch..(y + 1) * dst_pitch];
                dst.copy_from_slice(src);
            }
        }
        DXGI_FORMAT_B8G8R8X8_UNORM | DXGI_FORMAT_B8G8R8X8_UNORM_SRGB => {
            for y in 0..h {
                let src = &rows[y * src_pitch..y * src_pitch + dst_pitch];
                let dst = &mut bgra8[y * dst_pitch..(y + 1) * dst_pitch];
                for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 255;
                }
            }
        }
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => {
            for y in 0..h {
                let src = &rows[y * src_pitch..y * src_pitch + dst_pitch];
                let dst = &mut bgra8[y * dst_pitch..(y + 1) * dst_pitch];
                for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = s[3];
                }
            }
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            for y in 0..h {
                let src = &rows[y * src_pitch..y * src_pitch + dst_pitch];
                let dst = &mut bgra8[y * dst_pitch..(y + 1) * dst_pitch];
                for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                    let word = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                    let r10 = word & 0x3ff;
                    let g10 = (word >> 10) & 0x3ff;
                    let b10 = (word >> 20) & 0x3ff;
                    let a2 = (word >> 30) & 0x3;
                    let a8 = ((a2 * 255 + 1) / 3) as u8;
                    if hdr {
                        // ST.2084 PQ (BT.2020) -> linear nits -> BT.709 -> Reinhard -> sRGB.
                        let r_nits = pq_to_linear_nits(r10 as f32 / 1023.0);
                        let g_nits = pq_to_linear_nits(g10 as f32 / 1023.0);
                        let b_nits = pq_to_linear_nits(b10 as f32 / 1023.0);
                        let r709 = (1.6605 * r_nits - 0.5876 * g_nits - 0.0728 * b_nits).max(0.0);
                        let g709 = (-0.1246 * r_nits + 1.1329 * g_nits - 0.0083 * b_nits).max(0.0);
                        let b709 = (-0.0182 * r_nits - 0.1006 * g_nits + 1.1187 * b_nits).max(0.0);
                        const SDR_WHITE_NITS: f32 = 203.0;
                        d[0] = tonemap_reinhard_srgb(b709 / SDR_WHITE_NITS);
                        d[1] = tonemap_reinhard_srgb(g709 / SDR_WHITE_NITS);
                        d[2] = tonemap_reinhard_srgb(r709 / SDR_WHITE_NITS);
                        d[3] = a8;
                    } else {
                        d[0] = ((b10 * 255 + 511) / 1023) as u8;
                        d[1] = ((g10 * 255 + 511) / 1023) as u8;
                        d[2] = ((r10 * 255 + 511) / 1023) as u8;
                        d[3] = a8;
                    }
                }
            }
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            let src_row_bytes = w.checked_mul(8)?;
            for y in 0..h {
                let src = &rows[y * src_pitch..y * src_pitch + src_row_bytes];
                let dst = &mut bgra8[y * dst_pitch..(y + 1) * dst_pitch];
                for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(4)) {
                    let r = f16_to_f32(u16::from_le_bytes([s[0], s[1]]));
                    let g = f16_to_f32(u16::from_le_bytes([s[2], s[3]]));
                    let b = f16_to_f32(u16::from_le_bytes([s[4], s[5]]));
                    let a = f16_to_f32(u16::from_le_bytes([s[6], s[7]]));
                    if hdr {
                        // scRGB: 1.0 == 80 nits; normalize to 203 nits SDR paper white.
                        const SCRGB_SCALE: f32 = 80.0 / 203.0;
                        d[0] = tonemap_reinhard_srgb(b * SCRGB_SCALE);
                        d[1] = tonemap_reinhard_srgb(g * SCRGB_SCALE);
                        d[2] = tonemap_reinhard_srgb(r * SCRGB_SCALE);
                    } else {
                        d[0] = linear_to_srgb_u8(b);
                        d[1] = linear_to_srgb_u8(g);
                        d[2] = linear_to_srgb_u8(r);
                    }
                    let a_clamped = if a.is_finite() { a.clamp(0.0, 1.0) } else { 1.0 };
                    d[3] = (a_clamped * 255.0).round() as u8;
                }
            }
        }
        other => {
            tracing::warn!(format = other.0, "screenshot: unsupported back-buffer format");
            return None;
        }
    }

    Some(ScreenshotFrame {
        width,
        height,
        row_pitch: dst_pitch,
        bgra8,
        hdr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parsing_defaults_and_overrides() {
        let def = PresentConfig::from_vars(None, None, None, None, None);
        assert!(def.tearing);
        assert!(!def.vsync);
        assert!(!def.waitable);
        assert_eq!(def.max_frame_latency, 1);
        assert!(!def.hdr);

        let disabled =
            PresentConfig::from_vars(Some("0"), Some("1"), Some("1"), Some("3"), Some("1"));
        assert!(!disabled.tearing);
        assert!(disabled.vsync);
        assert!(disabled.waitable);
        assert_eq!(disabled.max_frame_latency, 3);
        assert!(disabled.hdr);

        // Non-"0" tearing stays enabled; latency clamps to 1..=16; invalid defaults to 1.
        assert!(parse_tearing(Some("1")));
        assert!(parse_tearing(Some("yes")));
        assert!(!parse_tearing(Some(" 0 ")));
        assert_eq!(parse_max_frame_latency(Some("0")), 1);
        assert_eq!(parse_max_frame_latency(Some("99")), 16);
        assert_eq!(parse_max_frame_latency(Some("bad")), 1);
        assert!(!parse_opt_in(Some("0")));
        assert!(!parse_opt_in(Some("true")));
        assert!(parse_opt_in(Some(" 1 ")));
    }

    #[test]
    fn swap_format_selection_sdr_and_hdr() {
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R8G8B8A8_UNORM, false, false),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_B8G8R8A8_UNORM, false, false),
            Some(DXGI_FORMAT_B8G8R8A8_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R10G10B10A2_UNORM, false, false),
            Some(DXGI_FORMAT_R10G10B10A2_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R16G16B16A16_FLOAT, false, false),
            Some(DXGI_FORMAT_R16G16B16A16_FLOAT)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_B8G8R8A8_UNORM, true, false),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );

        // HDR10 promotes two-plane composition to R10G10B10A2_UNORM, while
        // single-plane CopyResource preserves each source surface's format family.
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R8G8B8A8_UNORM, false, true),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R10G10B10A2_UNORM, false, true),
            Some(DXGI_FORMAT_R10G10B10A2_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_B8G8R8A8_UNORM, true, true),
            Some(DXGI_FORMAT_R10G10B10A2_UNORM)
        );
        assert_eq!(
            select_swap_format_for(DXGI_FORMAT_R16G16B16A16_FLOAT, false, true),
            Some(DXGI_FORMAT_R16G16B16A16_FLOAT)
        );
    }

    #[test]
    fn real_dxgi_tearing_support_query() {
        // SAFETY: plain DXGI factory creation.
        let factory: IDXGIFactory2 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.expect("DXGI factory");
        let tearing = query_tearing_support(&factory);
        eprintln!("Host DXGI tearing support (DXGI_FEATURE_PRESENT_ALLOW_TEARING): {tearing}");
    }

    #[test]
    fn real_dxgi_hdr_capabilities_query() {
        let caps = query_hdr_capabilities();
        eprintln!(
            "Host IDXGIOutput6 HDR10 capabilities: supported={}, min_nits={}, max_nits={}, max_full_frame_nits={}",
            caps.supported, caps.min_luminance, caps.max_luminance, caps.max_full_frame_luminance
        );
        assert!(caps.min_luminance.is_finite() && caps.min_luminance >= 0.0);
        assert!(caps.max_luminance.is_finite() && caps.max_luminance >= 0.0);
        assert!(caps.max_full_frame_luminance.is_finite() && caps.max_full_frame_luminance >= 0.0);
    }

    #[test]
    fn real_dxgi_host_display_modes_and_refresh_interval() {
        let modes = host_display_modes();
        eprintln!(
            "Host DXGI progressive display modes ({} total): {:?}",
            modes.len(),
            &modes[..modes.len().min(8)]
        );
        assert!(!modes.is_empty(), "expected at least one progressive display mode");
        for m in &modes {
            assert!(m.width > 0 && m.height > 0);
            assert!(m.refresh_num > 0 && m.refresh_den > 0);
        }
        let interval = primary_refresh_interval();
        eprintln!("Host primary display refresh interval: {interval:?}");
        assert!(interval.is_some_and(|d| d > Duration::ZERO && d <= Duration::from_millis(50)));
    }

    #[test]
    fn pixel_format_conversions_to_bgra8() {
        // RGBA8 -> BGRA8 (with padded pitch).
        let rgba = [10u8, 20, 30, 255, 0, 0, 0, 0];
        let frame = convert_rows_to_bgra8(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 8, &rgba, false)
            .expect("rgba8");
        assert_eq!(frame.row_pitch, 4);
        assert_eq!(frame.bgra8, vec![30, 20, 10, 255]);

        // BGRA8 -> BGRA8 passthrough.
        let bgra = [30u8, 20, 10, 200];
        let frame = convert_rows_to_bgra8(DXGI_FORMAT_B8G8R8A8_UNORM, 1, 1, 4, &bgra, false)
            .expect("bgra8");
        assert_eq!(frame.bgra8, vec![30, 20, 10, 200]);

        // R10G10B10A2 SDR: R=1023, G=512, B=0, A=3.
        let packed = 1023u32 | (512u32 << 10) | (3u32 << 30);
        let frame = convert_rows_to_bgra8(
            DXGI_FORMAT_R10G10B10A2_UNORM,
            1,
            1,
            4,
            &packed.to_le_bytes(),
            false,
        )
        .expect("r10g10b10a2 sdr");
        assert_eq!(frame.bgra8[0], 0);
        assert!((frame.bgra8[1] as i16 - 128).abs() <= 1);
        assert_eq!(frame.bgra8[2], 255);
        assert_eq!(frame.bgra8[3], 255);

        // R10G10B10A2 HDR10 ST.2084: black stays black, PQ white (~0.58 => ~203 nits) maps mid-high grey.
        let black = 3u32 << 30;
        let f_black = convert_rows_to_bgra8(
            DXGI_FORMAT_R10G10B10A2_UNORM,
            1,
            1,
            4,
            &black.to_le_bytes(),
            true,
        )
        .expect("hdr10 black");
        assert_eq!(f_black.bgra8, vec![0, 0, 0, 255]);
        assert!(f_black.hdr);

        let pq_mid = 594u32; // ~0.5806 => ~203 nits => Reinhard 0.5 => sRGB ~188.
        let white_pq = pq_mid | (pq_mid << 10) | (pq_mid << 20) | (3u32 << 30);
        let f_white = convert_rows_to_bgra8(
            DXGI_FORMAT_R10G10B10A2_UNORM,
            1,
            1,
            4,
            &white_pq.to_le_bytes(),
            true,
        )
        .expect("hdr10 white");
        assert!((170..=205).contains(&f_white.bgra8[0]), "{:?}", f_white.bgra8);
        assert!((170..=205).contains(&f_white.bgra8[1]), "{:?}", f_white.bgra8);
        assert!((170..=205).contains(&f_white.bgra8[2]), "{:?}", f_white.bgra8);
        assert_eq!(f_white.bgra8[3], 255);

        // R16G16B16A16_FLOAT: R=1.0 (0x3c00), G=0.0 (0x0000), B=0.25 (0x3400), A=1.0 (0x3c00).
        let mut f16_px = Vec::new();
        for bits in [0x3c00u16, 0x0000, 0x3400, 0x3c00] {
            f16_px.extend_from_slice(&bits.to_le_bytes());
        }
        let f_f16 = convert_rows_to_bgra8(DXGI_FORMAT_R16G16B16A16_FLOAT, 1, 1, 8, &f16_px, false)
            .expect("rgba16f");
        assert_eq!(f_f16.bgra8[2], 255); // R = 1.0 -> 255
        assert_eq!(f_f16.bgra8[1], 0); // G = 0.0 -> 0
        assert!((135..=138).contains(&f_f16.bgra8[0]), "{:?}", f_f16.bgra8); // linear 0.25 -> sRGB ~137
        assert_eq!(f_f16.bgra8[3], 255);
    }

    #[test]
    fn idle_screenshot_is_zero_cost_and_returns_none_before_present() {
        assert!(!screenshot_pending());
        if !HAS_PRESENTED.load(Ordering::Acquire) {
            let start = std::time::Instant::now();
            assert!(last_frame().is_none());
            assert!(start.elapsed() < Duration::from_millis(50));
            assert!(!screenshot_pending());
        }
    }

    #[test]
    fn real_d3d12_gpu_readback_texture_bgra8() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping D3D12 screenshot readback test: no host device");
            return;
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 2,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain committed render target creation on live host device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut resource,
                )
                .expect("create RT");
        }
        let tex = resource.expect("RT resource");
        let clear = host.new_list().expect("clear list");
        // SAFETY: clear to (R=0.25, G=0.5, B=1.0, A=1.0).
        unsafe {
            clear.list.ClearRenderTargetView(
                host.rtv(Some(&tex), None),
                &[0.25, 0.5, 1.0, 1.0],
                None,
            );
            clear.close();
        }
        host.execute(&[&clear]);

        let shot =
            read_back_texture_bgra8(host, &tex, D3D12_RESOURCE_STATE_RENDER_TARGET, false)
                .expect("readback frame");
        assert_eq!((shot.width, shot.height, shot.row_pitch), (4, 2, 16));
        assert_eq!(shot.bgra8.len(), 32);
        assert!(!shot.hdr);
        // Expected BGRA8: B=255, G=128, R=64, A=255.
        for px in shot.bgra8.chunks_exact(4) {
            assert_eq!(px[0], 255, "B mismatch: {px:?}");
            assert!(px[1].abs_diff(128) <= 1, "G mismatch: {px:?}");
            assert!(px[2].abs_diff(64) <= 1, "R mismatch: {px:?}");
            assert_eq!(px[3], 255, "A mismatch: {px:?}");
        }
        eprintln!(
            "Real D3D12 GPU readback BGRA8 pixel[0]: {:?}",
            &shot.bgra8[..4]
        );
    }
}
