//! The console's audio decoder (`xapu.dll`): Opus decode, Opus decode +
//! sample rate conversion, and float PCM sample rate conversion.
//!
//! A title connects once for one kind of processing, gets a block of memory
//! for its packets and output, and then queues per-stream commands
//! (activate, process, reset, deactivate) and dequeues one result per
//! command. We process each command when it is queued, so its result is ready
//! at once; the order of results is the order of commands.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Mutex;

use opus_pure::OpusDecoder;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const QUEUE_FULL: HRESULT = 0x8AC8_1000_u32 as i32;
/// No result yet, which titles poll for.
pub(crate) const PENDING_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;
const INVALID_COMMAND: HRESULT = 0x8AC8_1002_u32 as i32;
const INVALID_STREAM: HRESULT = 0x8AC8_1003_u32 as i32;
const ALREADY_ACTIVE: HRESULT = 0x8AC8_1005_u32 as i32;
const BAD_PACKET: HRESULT = 0x8AC8_1008_u32 as i32;

/// Processing kinds (`XApuProcessType`).
const DECODE_OPUS: u32 = 1;
const DECODE_CONVERT_OPUS: u32 = 2;
const CONVERT_PCM: u32 = 3;

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;
const NOTIFY: u32 = 4;
const RESET: u32 = 5;

/// Command options (`XApuCommandOptions`).
const SIGNAL: u32 = 1;
const FLUSH: u32 = 2;
const RESET_DECODER_ONLY: u32 = 4;
/// Connect options (`XApuConnectOptions`).
const DEINTERLEAVED_INPUT: u32 = 2;
const DEINTERLEAVED_OUTPUT: u32 = 4;
const RESULT_EVENT: u32 = 8;

/// Result state flags (`XApuStateOptions`).
const STATE_OUTPUT_READY: u32 = 1;
const STATE_NEED_MORE_INPUT: u32 = 2;
const STATE_MORE_OUTPUT_AVAILABLE: u32 = 4;

/// Pitch and ramp limits (`XApuDecodeConvertCommand`).
const MIN_PITCH: f32 = 0.125;
const MAX_PITCH: f32 = 8.0;
const MIN_PITCH_RAMP: f32 = 1.0 / (1u32 << 18) as f32;

/// The longest Opus packet duration at 48 kHz (120 ms).
const MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct ConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
pub struct ConnectOutput {
    memory: *mut c_void,
    result_event: HANDLE,
}

/// What every command starts with, and what its result echoes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct CommandId {
    kind: u32,
    stream: u32,
    sequence: u32,
    options: u32,
    context: u64,
}

#[repr(C, packed)]
struct Activate {
    id: CommandId,
    channels: u32,
    init_length: u32,
    init: *const c_void,
    start_pitch: f32,
    processing_length: u32,
    processing: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Process {
    id: CommandId,
    pitch_ramp: f32,
    target_pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ApuResult {
    id: CommandId,
    state: u32,
    state_data: u64,
    usec: u64,
    error: HRESULT,
    frames: u32,
    output_length: u32,
    output: *mut u8,
    pitch: f32,
    input_length: u32,
    input: *const u8,
}

// SAFETY: the raw pointers are title addresses carried through, never
// dereferenced after the result is made.
unsafe impl Send for ApuResult {}

pub struct Connection {
    memory: *mut u8,
    layout: Layout,
    event: HANDLE,
    process: u32,
    frames_per_output: usize,
    options: u32,
    per_stream: u32,
    state: Mutex<State>,
}

struct Stream {
    decoder: Option<OpusDecoder>,
    channels: usize,
    pitch: f32,
    /// Interleaved input PCM awaiting conversion; frame 0 is always the
    /// previous history frame for 4-tap cubic Hermite interpolation.
    in_buf: Vec<f32>,
    /// Whole input frames to drop before resuming (when pitch > 1 steps past
    /// the end of the previous input buffer).
    skip_in: usize,
    /// Fractional input frame phase in `[0.0, 1.0)`.
    phase: f64,
    /// Converted interleaved output frames waiting for a full output block.
    out_buf: Vec<f32>,
}

impl Stream {
    fn new(decoder: Option<OpusDecoder>, channels: usize, pitch: f32) -> Self {
        Stream {
            decoder,
            channels,
            pitch,
            in_buf: vec![0.0; channels],
            skip_in: 0,
            phase: 0.0,
            out_buf: Vec::new(),
        }
    }

    fn reset_resampler(&mut self) {
        self.in_buf.clear();
        self.in_buf.resize(self.channels, 0.0);
        self.skip_in = 0;
        self.phase = 0.0;
        self.out_buf.clear();
    }

    /// Appends interleaved input frames, honoring any integer carry `skip_in`.
    fn push_input(&mut self, frames: &[f32]) {
        let ch = self.channels;
        let count = frames.len() / ch;
        if count == 0 {
            return;
        }
        if self.skip_in >= count {
            let last = (count - 1) * ch;
            self.in_buf[..ch].copy_from_slice(&frames[last..last + ch]);
            self.skip_in -= count;
            return;
        }
        if self.skip_in > 0 {
            let last = (self.skip_in - 1) * ch;
            self.in_buf[..ch].copy_from_slice(&frames[last..last + ch]);
            let start = self.skip_in * ch;
            self.skip_in = 0;
            self.in_buf.extend_from_slice(&frames[start..]);
        } else {
            self.in_buf.extend_from_slice(frames);
        }
    }

    /// Runs the 4-point Catmull-Rom resampler until `out_buf` has `want_frames`
    /// frames or input runs out; returns the number of input frames consumed.
    fn resample_into(
        &mut self,
        want_frames: usize,
        pitch_ramp: f32,
        target_pitch: f32,
        flush: bool,
    ) -> usize {
        let ch = self.channels;
        let total = self.in_buf.len() / ch;
        if total <= 1 {
            return 0;
        }
        let target = if target_pitch.is_finite() && target_pitch > 0.0 {
            target_pitch.clamp(MIN_PITCH, MAX_PITCH)
        } else {
            self.pitch
        };
        let ramp = if pitch_ramp.is_finite() && pitch_ramp >= MIN_PITCH_RAMP {
            pitch_ramp.min(MAX_PITCH)
        } else {
            0.0
        };
        let mut idx: usize = 1;
        while self.out_buf.len() / ch < want_frames {
            if idx >= total {
                break;
            }
            if !flush && self.phase > 0.0 && idx + 2 >= total {
                break;
            }
            if ramp > 0.0 {
                if self.pitch < target {
                    self.pitch = (self.pitch + ramp).min(target);
                } else if self.pitch > target {
                    self.pitch = (self.pitch - ramp).max(target);
                }
            }
            let t = self.phase as f32;
            if t == 0.0 {
                let base = idx * ch;
                for c in 0..ch {
                    self.out_buf.push(self.in_buf[base + c]);
                }
            } else {
                let i0 = (idx - 1) * ch;
                let i1 = idx * ch;
                let i2 = (idx + 1).min(total - 1) * ch;
                let i3 = (idx + 2).min(total - 1) * ch;
                for c in 0..ch {
                    self.out_buf.push(catmull_rom(
                        self.in_buf[i0 + c],
                        self.in_buf[i1 + c],
                        self.in_buf[i2 + c],
                        self.in_buf[i3 + c],
                        t,
                    ));
                }
            }
            let pos = self.phase + self.pitch as f64;
            let step = pos.floor() as usize;
            idx += step;
            self.phase = pos - step as f64;
        }
        let consumed = (idx - 1).min(total - 1);
        if idx > total {
            let last = (total - 1) * ch;
            self.in_buf.copy_within(last..last + ch, 0);
            self.in_buf.truncate(ch);
            self.skip_in += idx - total;
        } else if idx > 1 {
            self.in_buf.drain(..(idx - 1) * ch);
        }
        consumed
    }
}

#[inline]
fn catmull_rom(ym1: f32, y0: f32, y1: f32, y2: f32, t: f32) -> f32 {
    let c0 = y0;
    let c1 = 0.5 * (y1 - ym1);
    let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
    let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
    ((c3 * t + c2) * t + c1) * t + c0
}

struct State {
    streams: Vec<Option<Stream>>,
    /// Reusable PCM scratch buffer so `process` does no heap allocation per command.
    scratch: Vec<f32>,
    /// Results not yet dequeued, per stream.
    queued: Vec<u32>,
    results: VecDeque<ApuResult>,
    commands: u32,
    dequeued: u32,
}

fn lock(c: &Connection) -> std::sync::MutexGuard<'_, State> {
    c.state.lock().unwrap_or_else(|e| e.into_inner())
}

pub extern "system" fn XApuConnect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
) -> HRESULT {
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    let input = unsafe { &*input };
    if !(1..=500).contains(&input.streams)
        || !(1..=16).contains(&input.commands_per_stream)
        || input.memory == 0
    {
        return E_INVALIDARG;
    }
    match input.process {
        DECODE_OPUS => {
            if input.frames_per_output != 0 {
                return E_INVALIDARG;
            }
        }
        DECODE_CONVERT_OPUS | CONVERT_PCM => {
            if !matches!(input.frames_per_output, 256 | 480 | 512 | 960 | 1024) {
                return E_INVALIDARG;
            }
        }
        0 => return E_INVALIDARG,
        _ => {
            tracing::warn!(
                process = input.process,
                "audio decoder: processing kind not implemented"
            );
            return E_NOTIMPL;
        }
    }
    let Ok(layout) = Layout::from_size_align(input.memory as usize, 16) else {
        return E_INVALIDARG;
    };
    // SAFETY: non-zero size.
    let memory = unsafe { std::alloc::alloc_zeroed(layout) };
    if memory.is_null() {
        return E_OUTOFMEMORY;
    }
    let event = if input.options & RESULT_EVENT != 0 {
        // SAFETY: plain auto-reset event, no name.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if event.is_null() {
            // SAFETY: allocated above with `layout`.
            unsafe { std::alloc::dealloc(memory, layout) };
            return E_OUTOFMEMORY;
        }
        event
    } else {
        std::ptr::null_mut()
    };
    let streams = input.streams as usize;
    tracing::info!(
        process = input.process,
        streams,
        frames_per_output = input.frames_per_output,
        memory = input.memory,
        options = input.options,
        "audio decoder connected"
    );
    let connection = Box::new(Connection {
        memory,
        layout,
        event,
        process: input.process,
        frames_per_output: input.frames_per_output as usize,
        options: input.options,
        per_stream: input.commands_per_stream,
        state: Mutex::new(State {
            streams: (0..streams).map(|_| None).collect(),
            scratch: Vec::new(),
            queued: vec![0; streams],
            results: VecDeque::new(),
            commands: 0,
            dequeued: 0,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        output.write(ConnectOutput {
            memory: memory.cast(),
            result_event: event,
        });
        handle.write(Box::into_raw(connection));
    }
    S_OK
}

pub extern "system" fn XApuDisconnect(handle: *mut Connection) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XApuConnect`; disconnected once.
    let connection = unsafe { Box::from_raw(handle) };
    if !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { CloseHandle(connection.event) };
    }
    // SAFETY: allocated in `XApuConnect` with this layout.
    unsafe { std::alloc::dealloc(connection.memory, connection.layout) };
    S_OK
}

pub extern "system" fn XApuEnqueueCommand(
    handle: *mut Connection,
    command: *const CommandId,
    total: *mut u32,
) -> HRESULT {
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`; every command starts with an id.
    let (connection, id) = unsafe { (&*handle, command.read_unaligned()) };
    let mut state = lock(connection);
    let stream = id.stream as usize;
    if stream >= state.streams.len() {
        return INVALID_STREAM;
    }
    if state.queued[stream] >= connection.per_stream {
        return QUEUE_FULL;
    }
    let mut result = ApuResult {
        id,
        state: 0,
        state_data: 0,
        usec: 0,
        error: S_OK,
        frames: 0,
        output_length: 0,
        output: std::ptr::null_mut(),
        pitch: 1.0,
        input_length: 0,
        input: std::ptr::null(),
    };
    let state_ref = &mut *state;
    match id.kind {
        ACTIVATE => {
            // SAFETY: an activate command is this struct.
            let act = unsafe { command.cast::<Activate>().read_unaligned() };
            let channels = act.channels;
            let start_pitch = act.start_pitch;
            tracing::debug!(
                stream,
                process = connection.process,
                channels,
                start_pitch,
                "APU stream activated"
            );
            let slot = &mut state_ref.streams[stream];
            result.error = if slot.is_some() {
                ALREADY_ACTIVE
            } else if !(1..=2).contains(&channels)
                || (connection.process != DECODE_OPUS
                    && (!start_pitch.is_finite() || start_pitch < 0.0))
            {
                E_INVALIDARG
            } else {
                let pitch = if connection.process == DECODE_OPUS || start_pitch == 0.0 {
                    1.0
                } else {
                    start_pitch.clamp(MIN_PITCH, MAX_PITCH)
                };
                match connection.process {
                    DECODE_OPUS | DECODE_CONVERT_OPUS => {
                        match OpusDecoder::new(48_000, channels as usize) {
                            Ok(d) => {
                                *slot = Some(Stream::new(Some(d), channels as usize, pitch));
                                result.pitch = pitch;
                                S_OK
                            }
                            Err(_) => E_INVALIDARG,
                        }
                    }
                    CONVERT_PCM => {
                        *slot = Some(Stream::new(None, channels as usize, pitch));
                        result.pitch = pitch;
                        S_OK
                    }
                    _ => E_NOTIMPL,
                }
            };
        }
        DEACTIVATE => state_ref.streams[stream] = None,
        RESET => {
            if let Some(s) = &mut state_ref.streams[stream] {
                if let Some(d) = &mut s.decoder {
                    let _ = d.reset_state();
                }
                if !(connection.process == DECODE_CONVERT_OPUS
                    && id.options & RESET_DECODER_ONLY != 0)
                {
                    s.reset_resampler();
                }
                result.pitch = s.pitch;
            }
        }
        NOTIFY => {}
        PROCESS => {
            // SAFETY: a process command is this struct.
            let command = unsafe { command.cast::<Process>().read_unaligned() };
            match &mut state_ref.streams[stream] {
                Some(s) => match connection.process {
                    DECODE_OPUS => process_decode_opus(
                        s,
                        &mut state_ref.scratch,
                        connection.options,
                        &command,
                        &mut result,
                    ),
                    DECODE_CONVERT_OPUS | CONVERT_PCM => process_convert(
                        s,
                        &mut state_ref.scratch,
                        connection.process,
                        connection.frames_per_output,
                        connection.options,
                        &command,
                        &mut result,
                    ),
                    _ => result.error = E_NOTIMPL,
                },
                None => result.error = INVALID_STREAM,
            }
        }
        _ => return INVALID_COMMAND,
    }
    state_ref.queued[stream] += 1;
    state_ref.commands = state_ref.commands.wrapping_add(1);
    state_ref.results.push_back(result);
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state_ref.commands) };
    }
    if id.options & SIGNAL != 0 && !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { SetEvent(connection.event) };
    }
    S_OK
}

/// Writes `frames` of interleaved f32 samples into `output`, supporting
/// unaligned pointers and deinterleaved stereo output with `deinterleave_stride`
/// frames between the left and right channel bases.
fn write_pcm_output(
    samples: &[f32],
    frames: usize,
    channels: usize,
    options: u32,
    deinterleave_stride: usize,
    output: *mut u8,
    output_length: usize,
) -> usize {
    let bytes = frames * channels * 4;
    if bytes == 0 {
        return 0;
    }
    if options & DEINTERLEAVED_OUTPUT != 0 && channels == 2 {
        let right_offset = if (deinterleave_stride + frames) * 4 <= output_length {
            deinterleave_stride * 4
        } else {
            frames * 4
        };
        let span = (right_offset + frames * 4).min(output_length);
        // SAFETY: `output` is non-null and holds at least `span <= output_length` bytes;
        // writing raw bytes avoids requiring 4-byte alignment on the title's `*mut u8`.
        let out = unsafe { std::slice::from_raw_parts_mut(output, span) };
        let (left_buf, rest) = out.split_at_mut(right_offset);
        let (left, _) = left_buf[..frames * 4].as_chunks_mut::<4>();
        let (right, _) = rest[..frames * 4].as_chunks_mut::<4>();
        for (i, frame) in samples[..frames * 2].as_chunks::<2>().0.iter().enumerate() {
            left[i] = frame[0].to_le_bytes();
            right[i] = frame[1].to_le_bytes();
        }
    } else {
        // SAFETY: `output` is non-null and holds at least `bytes <= output_length` bytes.
        let out = unsafe { std::slice::from_raw_parts_mut(output, bytes) };
        for (dst, &sample) in out
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&samples[..frames * channels])
        {
            *dst = sample.to_le_bytes();
        }
    }
    bytes
}

/// Decodes one Opus packet into the command's output buffer, keeping the frames
/// from `first_frame` on, at most `frame_count` of them.
fn process_decode_opus(
    stream: &mut Stream,
    scratch: &mut Vec<f32>,
    options: u32,
    command: &Process,
    result: &mut ApuResult,
) {
    let Some(decoder) = &mut stream.decoder else {
        result.error = INVALID_STREAM;
        return;
    };
    let channels = stream.channels;
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    // SAFETY: the title's packet in its connection memory.
    let packet = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
    let needed = MAX_FRAMES * channels;
    if scratch.len() < needed {
        scratch.resize(needed, 0.0);
    }
    let decoded = match decoder.decode(packet, MAX_FRAMES, &mut scratch[..needed]) {
        Ok(n) => n,
        Err(_) => {
            result.error = BAD_PACKET;
            return;
        }
    };
    let first = (command.first_frame as usize).min(decoded);
    let frames = (decoded - first)
        .min(command.frame_count as usize)
        .min(output_length as usize / (4 * channels));
    let kept = &scratch[first * channels..(first + frames) * channels];
    let bytes = write_pcm_output(
        kept,
        frames,
        channels,
        options,
        frames,
        output,
        output_length as usize,
    );
    result.frames = frames as u32;
    result.output_length = bytes as u32;
}

/// Decodes (when `DECODE_CONVERT_OPUS`) or reads float32 PCM (`CONVERT_PCM`)
/// and converts sample rate with dynamic pitch ramping and persistent phase.
fn process_convert(
    stream: &mut Stream,
    scratch: &mut Vec<f32>,
    kind: u32,
    frames_per_output: usize,
    options: u32,
    command: &Process,
    result: &mut ApuResult,
) {
    let channels = stream.channels;
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    result.pitch = stream.pitch;
    if output.is_null() {
        result.error = E_INVALIDARG;
        return;
    }
    if input_length > 0 {
        if input.is_null() {
            result.error = E_INVALIDARG;
            return;
        }
        match kind {
            DECODE_CONVERT_OPUS => {
                let Some(decoder) = &mut stream.decoder else {
                    result.error = INVALID_STREAM;
                    return;
                };
                // SAFETY: the title's Opus packet in its connection memory.
                let packet = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
                let needed = MAX_FRAMES * channels;
                if scratch.len() < needed {
                    scratch.resize(needed, 0.0);
                }
                let decoded = match decoder.decode(packet, MAX_FRAMES, &mut scratch[..needed]) {
                    Ok(n) => n,
                    Err(_) => {
                        result.error = BAD_PACKET;
                        return;
                    }
                };
                let first = (command.first_frame as usize).min(decoded);
                let count = (decoded - first).min(command.frame_count as usize);
                stream.push_input(&scratch[first * channels..(first + count) * channels]);
            }
            CONVERT_PCM => {
                // SAFETY: the title's float32 PCM buffer in its connection memory.
                let raw = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
                let avail_frames = raw.len() / (4 * channels);
                let frame_limit = if command.frame_count == u32::MAX {
                    avail_frames
                } else {
                    (command.frame_count as usize).min(avail_frames)
                };
                let first = (command.first_frame as usize).min(frame_limit);
                let count = frame_limit - first;
                scratch.clear();
                scratch.reserve(count * channels);
                if options & DEINTERLEAVED_INPUT != 0 && channels == 2 {
                    let half = raw.len() / 2;
                    let (left_bytes, right_bytes) = raw.split_at(half);
                    let (left_chunks, _) = left_bytes.as_chunks::<4>();
                    let (right_chunks, _) = right_bytes.as_chunks::<4>();
                    let max_pair = left_chunks.len().min(right_chunks.len());
                    let end = (first + count).min(max_pair);
                    for i in first..end {
                        scratch.push(f32::from_le_bytes(left_chunks[i]));
                        scratch.push(f32::from_le_bytes(right_chunks[i]));
                    }
                } else {
                    let (chunks, _) = raw.as_chunks::<4>();
                    let start = first * channels;
                    let end = (first + count) * channels;
                    for &c in &chunks[start..end] {
                        scratch.push(f32::from_le_bytes(c));
                    }
                }
                stream.push_input(scratch);
            }
            _ => {
                result.error = E_NOTIMPL;
                return;
            }
        }
    }

    let flush = command.id.options & FLUSH != 0;
    let cap_frames = (output_length as usize / (4 * channels)).min(frames_per_output);
    let consumed = stream.resample_into(cap_frames, command.pitch_ramp, command.target_pitch, flush);
    let avail_out = stream.out_buf.len() / channels;
    let emit = if flush {
        avail_out.min(cap_frames)
    } else if avail_out >= frames_per_output {
        frames_per_output.min(cap_frames)
    } else {
        0
    };
    let bytes = if emit > 0 {
        let written = write_pcm_output(
            &stream.out_buf,
            emit,
            channels,
            options,
            frames_per_output,
            output,
            output_length as usize,
        );
        stream.out_buf.drain(..emit * channels);
        written
    } else {
        0
    };

    let stored_out = stream.out_buf.len() / channels;
    let stored_in = (stream.in_buf.len() / channels).saturating_sub(1);
    let mut state_flags = 0u32;
    if emit > 0 {
        state_flags |= STATE_OUTPUT_READY;
    }
    let est_remaining_out =
        stored_out + ((stored_in as f64) / (stream.pitch as f64).max(MIN_PITCH as f64)).floor() as usize;
    if est_remaining_out >= frames_per_output || (flush && (stored_out > 0 || stored_in > 0)) {
        state_flags |= STATE_MORE_OUTPUT_AVAILABLE;
    } else {
        state_flags |= STATE_NEED_MORE_INPUT;
    }
    let state_words = [
        stored_out.min(u16::MAX as usize) as u16,
        consumed.min(u16::MAX as usize) as u16,
        0u16,
        stored_in.min(u16::MAX as usize) as u16,
    ];
    result.state = state_flags;
    result.state_data = u64::from_le_bytes([
        state_words[0].to_le_bytes()[0],
        state_words[0].to_le_bytes()[1],
        state_words[1].to_le_bytes()[0],
        state_words[1].to_le_bytes()[1],
        state_words[2].to_le_bytes()[0],
        state_words[2].to_le_bytes()[1],
        state_words[3].to_le_bytes()[0],
        state_words[3].to_le_bytes()[1],
    ]);
    result.frames = emit as u32;
    result.output_length = bytes as u32;
    result.pitch = stream.pitch;
}

pub extern "system" fn XApuDequeueResult(handle: *mut Connection, out: *mut ApuResult) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`.
    let mut state = lock(unsafe { &*handle });
    let Some(result) = state.results.pop_front() else {
        return PENDING_RESULTS;
    };
    let stream = result.id.stream as usize;
    state.queued[stream] -= 1;
    state.dequeued = state.dequeued.wrapping_add(1);
    // SAFETY: the title passes room for one result.
    unsafe { out.write_unaligned(result) };
    S_OK
}

pub extern "system" fn XApuGetResultCount(handle: *mut Connection, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: a connection from `XApuConnect`.
    let state = lock(unsafe { &*handle });
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.dequeued) };
    }
    state.results.len() as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "apu",
        dlls: &["xapu.dll"],
        exports: vec![
            hle_export!(XApuConnect),
            hle_export!(XApuDisconnect),
            hle_export!(XApuEnqueueCommand),
            hle_export!(XApuDequeueResult),
            hle_export!(XApuGetResultCount),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus_pure::{Application, OpusEncoder};

    fn id(kind: u32, stream: u32) -> CommandId {
        CommandId {
            kind,
            stream,
            sequence: 7,
            options: 0,
            context: 0x1234,
        }
    }

    #[test]
    fn decodes_opus_as_the_title_drives_it() {
        assert_eq!(size_of::<ApuResult>(), 0x50);
        // 256 mono/stereo streams, two commands each, as the title connects.
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 256,
            commands_per_stream: 2,
            memory: 0x10_4000,
            frames_per_output: 0,
            options: 1,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();

        let activate = Activate {
            id: id(ACTIVATE, 3),
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
            S_OK
        );

        // A 20 ms stereo packet of a 440 Hz tone.
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let process = Process {
            id: id(PROCESS, 3),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 0xf00 * 2,
            output: base.wrapping_add(0x14000),
        };
        let mut total = 0;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            S_OK
        );
        assert_eq!(total, 2);
        // Two results queued for stream 3: its queue is full.
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            QUEUE_FULL
        );
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 2);

        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let (kind, error, context) = (first.id.kind, first.error, first.id.context);
        assert_eq!((kind, error, context), (ACTIVATE, S_OK, 0x1234));
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let decoded = unsafe { result.assume_init() };
        let (kind, error, frames, length) = (
            decoded.id.kind,
            decoded.error,
            decoded.frames,
            decoded.output_length,
        );
        assert_eq!((kind, error, frames, length), (PROCESS, S_OK, 960, 960 * 8));
        // SAFETY: the decoder wrote 960 stereo frames there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x14000).cast::<f32>(), 1920) };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / 1920.0;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        assert_eq!(
            XApuDequeueResult(handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut dequeued = 0;
        assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
        assert_eq!(dequeued, 2);

        // A process on a stream that was never activated fails in its result.
        let mut stray = process;
        stray.id.stream = 4;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const stray).cast(), &mut total),
            S_OK
        );
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let error = unsafe { result.assume_init() }.error;
        assert_eq!(error, INVALID_STREAM);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn decodes_into_unaligned_output_interleaved_and_deinterleaved() {
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| {
                let phase = (i / 2) as f32 * std::f32::consts::TAU / 48_000.0;
                if i % 2 == 0 {
                    (phase * 440.0).sin() * 0.5
                } else {
                    (phase * 880.0).sin() * 0.25
                }
            })
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();

        for (options, deinterleaved) in [(0u32, false), (DEINTERLEAVED_OUTPUT, true)] {
            let input = ConnectInput {
                process: DECODE_OPUS,
                streams: 4,
                commands_per_stream: 4,
                memory: 0x4_0000,
                frames_per_output: 0,
                options,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            let base = output.memory.cast::<u8>();

            // Invalid channel counts fail cleanly without activating the stream.
            for bad_channels in [0u32, 3, u32::MAX] {
                let bad = Activate {
                    id: id(ACTIVATE, 0),
                    channels: bad_channels,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const bad).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, E_INVALIDARG, "channels={bad_channels}");
            }

            // Activate two identical stereo decoders so aligned and unaligned outputs match bit-for-bit.
            for stream in [0u32, 1] {
                let activate = Activate {
                    id: id(ACTIVATE, stream),
                    channels: 2,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, S_OK);
            }

            // SAFETY: inside the connection memory.
            unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
            let aligned_out = base.wrapping_add(0x10000);
            let unaligned_out = base.wrapping_add(0x20001); // 1 mod 4: misaligned for f32

            for (stream, out_ptr, frame_count) in [
                (0u32, aligned_out, 960u32),
                (1u32, unaligned_out, 960u32),
                (1u32, base.wrapping_add(0x30003), 0u32), // zero frames at misaligned ptr
            ] {
                let proc_cmd = Process {
                    id: id(PROCESS, stream),
                    pitch_ramp: 0.0,
                    target_pitch: 1.0,
                    first_frame: 0,
                    frame_count,
                    input_length: n as u32,
                    input: base,
                    output_length: 960 * 8,
                    output: out_ptr,
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const proc_cmd).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let decoded = unsafe { res.assume_init() };
                let (err, frames, len) = (decoded.error, decoded.frames, decoded.output_length);
                assert_eq!((err, frames, len), (S_OK, frame_count, frame_count * 8));
            }

            // SAFETY: both output regions are inside connection memory and hold 960 * 8 bytes.
            let (aligned_bytes, unaligned_bytes) = unsafe {
                (
                    std::slice::from_raw_parts(aligned_out, 960 * 8),
                    std::slice::from_raw_parts(unaligned_out, 960 * 8),
                )
            };
            assert_eq!(
                unaligned_bytes, aligned_bytes,
                "unaligned output must match aligned output (deinterleaved={deinterleaved})"
            );
            let samples: Vec<f32> = unaligned_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
            assert!(
                energy > 0.01,
                "expected decoded tone (deinterleaved={deinterleaved}), energy={energy}"
            );
            assert_eq!(XApuDisconnect(handle), S_OK);
        }
    }

    fn dequeue_one(handle: *mut Connection) -> ApuResult {
        let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
        // SAFETY: written by `XApuDequeueResult`.
        unsafe { res.assume_init() }
    }

    #[test]
    fn decode_convert_opus_supports_pitch_scaling_and_ramping() {
        // Encode two 20 ms mono Opus packets of a 440 Hz tone.
        let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
        let make_pkt = |enc: &mut OpusEncoder, start_sample: usize| {
            let pcm: Vec<f32> = (0..960)
                .map(|i| {
                    ((start_sample + i) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin()
                        * 0.5
                })
                .collect();
            let mut pkt = vec![0u8; 1500];
            let n = enc.encode(&pcm, 960, &mut pkt).unwrap();
            pkt.truncate(n);
            pkt
        };
        let pkt0 = make_pkt(&mut encoder, 0);
        let pkt1 = make_pkt(&mut encoder, 960);

        for pitch in [0.5f32, 1.0, 2.0] {
            let input = ConnectInput {
                process: DECODE_CONVERT_OPUS,
                streams: 2,
                commands_per_stream: 4,
                memory: 0x4_0000,
                frames_per_output: 480,
                options: 0,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            let base = output.memory.cast::<u8>();

            let act = Activate {
                id: id(ACTIVATE, 0),
                channels: 1,
                init_length: 0,
                init: std::ptr::null(),
                start_pitch: pitch,
                processing_length: 0,
                processing: std::ptr::null_mut(),
            };
            assert_eq!(
                XApuEnqueueCommand(handle, (&raw const act).cast(), std::ptr::null_mut()),
                S_OK
            );
            assert_eq!({ dequeue_one(handle).error }, S_OK);

            // Feed both packets and drain all output chunks until flushed.
            let mut collected = Vec::new();
            for (i, pkt) in [&pkt0, &pkt1].iter().enumerate() {
                // SAFETY: inside connection memory.
                unsafe { std::ptr::copy_nonoverlapping(pkt.as_ptr(), base, pkt.len()) };
                let out_ptr = base.wrapping_add(0x4000);
                let proc_cmd = Process {
                    id: id(PROCESS, 0),
                    pitch_ramp: 0.0,
                    target_pitch: pitch,
                    first_frame: 0,
                    frame_count: u32::MAX,
                    input_length: pkt.len() as u32,
                    input: base,
                    output_length: 480 * 4,
                    output: out_ptr,
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const proc_cmd).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = dequeue_one(handle);
                let (err, mut frames, mut st, p) =
                    (res.error, res.frames, res.state, res.pitch);
                assert_eq!(err, S_OK, "pitch={pitch} pkt={i}");
                assert!((p - pitch).abs() < 1e-6);
                if frames > 0 {
                    assert_ne!(st & STATE_OUTPUT_READY, 0);
                    // SAFETY: `frames` floats written.
                    let slice =
                        unsafe { std::slice::from_raw_parts(out_ptr.cast::<f32>(), frames as usize) };
                    collected.extend_from_slice(slice);
                }
                while st & STATE_MORE_OUTPUT_AVAILABLE != 0 {
                    let drain = Process {
                        id: id(PROCESS, 0),
                        pitch_ramp: 0.0,
                        target_pitch: pitch,
                        first_frame: 0,
                        frame_count: 0,
                        input_length: 0,
                        input: std::ptr::null(),
                        output_length: 480 * 4,
                        output: out_ptr,
                    };
                    assert_eq!(
                        XApuEnqueueCommand(handle, (&raw const drain).cast(), std::ptr::null_mut()),
                        S_OK
                    );
                    res = dequeue_one(handle);
                    frames = res.frames;
                    st = res.state;
                    if frames > 0 {
                        let slice = unsafe {
                            std::slice::from_raw_parts(out_ptr.cast::<f32>(), frames as usize)
                        };
                        collected.extend_from_slice(slice);
                    } else {
                        break;
                    }
                }
            }
            // Flush any remaining tail frames.
            let out_ptr = base.wrapping_add(0x4000);
            let mut flush_id = id(PROCESS, 0);
            flush_id.options = FLUSH;
            let flush_cmd = Process {
                id: flush_id,
                pitch_ramp: 0.0,
                target_pitch: pitch,
                first_frame: 0,
                frame_count: 0,
                input_length: 0,
                input: std::ptr::null(),
                output_length: 480 * 4,
                output: out_ptr,
            };
            assert_eq!(
                XApuEnqueueCommand(handle, (&raw const flush_cmd).cast(), std::ptr::null_mut()),
                S_OK
            );
            let res = dequeue_one(handle);
            let frames = res.frames;
            if frames > 0 {
                // SAFETY: `frames` floats written.
                let slice =
                    unsafe { std::slice::from_raw_parts(out_ptr.cast::<f32>(), frames as usize) };
                collected.extend_from_slice(slice);
            }

            let expected = (1920.0 / pitch).round() as usize;
            assert!(
                collected.len().abs_diff(expected) <= 4,
                "pitch={pitch}: got {} frames, expected ~{expected}",
                collected.len()
            );
            let energy: f32 =
                collected.iter().map(|s| s * s).sum::<f32>() / collected.len().max(1) as f32;
            assert!(energy > 0.01, "pitch={pitch}: expected non-silent tone, energy={energy}");
            assert_eq!(XApuDisconnect(handle), S_OK);
        }

        // Dynamic pitch ramp: start at 1.0, ramp at 0.002/frame toward 2.0 over 256 frames.
        let input = ConnectInput {
            process: DECODE_CONVERT_OPUS,
            streams: 1,
            commands_per_stream: 4,
            memory: 0x2_0000,
            frames_per_output: 256,
            options: 0,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();
        let act = Activate {
            id: id(ACTIVATE, 0),
            channels: 1,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const act).cast(), std::ptr::null_mut()),
            S_OK
        );
        assert_eq!({ dequeue_one(handle).error }, S_OK);
        // SAFETY: inside connection memory.
        unsafe { std::ptr::copy_nonoverlapping(pkt0.as_ptr(), base, pkt0.len()) };
        let proc_cmd = Process {
            id: id(PROCESS, 0),
            pitch_ramp: 0.002,
            target_pitch: 2.0,
            first_frame: 0,
            frame_count: u32::MAX,
            input_length: pkt0.len() as u32,
            input: base,
            output_length: 256 * 4,
            output: base.wrapping_add(0x4000),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const proc_cmd).cast(), std::ptr::null_mut()),
            S_OK
        );
        let res = dequeue_one(handle);
        let (err, frames, pitch_after) = (res.error, res.frames, res.pitch);
        assert_eq!((err, frames), (S_OK, 256));
        assert!(
            (pitch_after - (1.0 + 256.0 * 0.002)).abs() < 1e-4,
            "expected ramped pitch ~1.512, got {pitch_after}"
        );
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn convert_pcm_preserves_phase_continuity_and_supports_deinterleaved_io() {
        // Build a 1024-frame stereo linear-phase sine wave.
        let total_frames = 1024usize;
        let left: Vec<f32> = (0..total_frames)
            .map(|i| (i as f32 * 300.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.6)
            .collect();
        let right: Vec<f32> = (0..total_frames)
            .map(|i| (i as f32 * 600.0 * std::f32::consts::TAU / 48_000.0).cos() * 0.4)
            .collect();

        // Compare converting all 1024 frames in one shot vs two 512-frame chunks at non-integer pitch 1.37.
        let run_convert = |chunk_size: usize, options: u32| -> (Vec<f32>, Vec<f32>) {
            let fpo = 256usize;
            let input = ConnectInput {
                process: CONVERT_PCM,
                streams: 1,
                commands_per_stream: 4,
                memory: 0x4_0000,
                frames_per_output: fpo as u32,
                options,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            let base = output.memory.cast::<u8>();
            let act = Activate {
                id: id(ACTIVATE, 0),
                channels: 2,
                init_length: 0,
                init: std::ptr::null(),
                start_pitch: 1.37,
                processing_length: 0,
                processing: std::ptr::null_mut(),
            };
            assert_eq!(
                XApuEnqueueCommand(handle, (&raw const act).cast(), std::ptr::null_mut()),
                S_OK
            );
            assert_eq!({ dequeue_one(handle).error }, S_OK);

            let (mut out_l, mut out_r) = (Vec::new(), Vec::new());
            let out_ptr = base.wrapping_add(0x1_0000);
            let deint_in = options & DEINTERLEAVED_INPUT != 0;
            let deint_out = options & DEINTERLEAVED_OUTPUT != 0;

            let mut collect = |frames: usize| {
                if frames == 0 {
                    return;
                }
                if deint_out {
                    // Right channel starts at `fpo` frames even when `frames < fpo`.
                    let l_slice = unsafe {
                        std::slice::from_raw_parts(out_ptr.cast::<f32>(), frames)
                    };
                    let r_slice = unsafe {
                        std::slice::from_raw_parts(out_ptr.wrapping_add(fpo * 4).cast::<f32>(), frames)
                    };
                    out_l.extend_from_slice(l_slice);
                    out_r.extend_from_slice(r_slice);
                } else {
                    let slice = unsafe {
                        std::slice::from_raw_parts(out_ptr.cast::<f32>(), frames * 2)
                    };
                    for pair in slice.chunks_exact(2) {
                        out_l.push(pair[0]);
                        out_r.push(pair[1]);
                    }
                }
            };

            let mut off = 0;
            while off < total_frames {
                let n = (total_frames - off).min(chunk_size);
                let in_bytes = n * 8;
                let dst = unsafe { std::slice::from_raw_parts_mut(base, in_bytes) };
                if deint_in {
                    let (dl, dr) = dst.split_at_mut(n * 4);
                    for i in 0..n {
                        dl[i * 4..(i + 1) * 4].copy_from_slice(&left[off + i].to_le_bytes());
                        dr[i * 4..(i + 1) * 4].copy_from_slice(&right[off + i].to_le_bytes());
                    }
                } else {
                    for i in 0..n {
                        dst[i * 8..i * 8 + 4].copy_from_slice(&left[off + i].to_le_bytes());
                        dst[i * 8 + 4..i * 8 + 8].copy_from_slice(&right[off + i].to_le_bytes());
                    }
                }
                let cmd = Process {
                    id: id(PROCESS, 0),
                    pitch_ramp: 0.0,
                    target_pitch: 1.37,
                    first_frame: 0,
                    frame_count: n as u32,
                    input_length: in_bytes as u32,
                    input: base,
                    output_length: (fpo * 8) as u32,
                    output: out_ptr,
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const cmd).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = dequeue_one(handle);
                assert_eq!({ res.error }, S_OK);
                collect(res.frames as usize);
                while res.state & STATE_MORE_OUTPUT_AVAILABLE != 0 {
                    let drain = Process {
                        id: id(PROCESS, 0),
                        pitch_ramp: 0.0,
                        target_pitch: 1.37,
                        first_frame: 0,
                        frame_count: 0,
                        input_length: 0,
                        input: std::ptr::null(),
                        output_length: (fpo * 8) as u32,
                        output: out_ptr,
                    };
                    assert_eq!(
                        XApuEnqueueCommand(handle, (&raw const drain).cast(), std::ptr::null_mut()),
                        S_OK
                    );
                    res = dequeue_one(handle);
                    assert_eq!({ res.error }, S_OK);
                    collect(res.frames as usize);
                    if res.frames == 0 {
                        break;
                    }
                }
                off += n;
            }
            // Final flush.
            let mut fid = id(PROCESS, 0);
            fid.options = FLUSH;
            let flush = Process {
                id: fid,
                pitch_ramp: 0.0,
                target_pitch: 1.37,
                first_frame: 0,
                frame_count: 0,
                input_length: 0,
                input: std::ptr::null(),
                output_length: (fpo * 8) as u32,
                output: out_ptr,
            };
            assert_eq!(
                XApuEnqueueCommand(handle, (&raw const flush).cast(), std::ptr::null_mut()),
                S_OK
            );
            let res = dequeue_one(handle);
            assert_eq!({ res.error }, S_OK);
            collect(res.frames as usize);

            assert_eq!(XApuDisconnect(handle), S_OK);
            (out_l, out_r)
        };

        let single = run_convert(1024, 0);
        let chunked = run_convert(512, 0);
        let deint = run_convert(512, DEINTERLEAVED_INPUT | DEINTERLEAVED_OUTPUT);

        assert_eq!(single.0.len(), chunked.0.len());
        assert_eq!(single, chunked, "chunked conversion must match single-shot bit-for-bit");
        assert_eq!(single, deint, "deinterleaved I/O must match interleaved I/O bit-for-bit");
        assert!((single.0.len() as isize - (1024.0 / 1.37f64).round() as isize).abs() <= 2);
    }
}
