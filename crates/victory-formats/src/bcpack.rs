//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format,
//! kept in `bcpack-tables.bin` beside this file ([`Tables::builtin`]).

use std::cell::RefCell;
use std::fmt;

/// Raw binary tables shipped with Victory, exposed for GPU BCPack consumers.
pub const BUILTIN_TABLES: &[u8] = include_bytes!("bcpack-tables.bin");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const HISTORY: usize = 256;
const DICTIONARY: usize = 255;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;

/// Bytes of the table file (see [`Tables::parse`]).
pub const TABLES_SIZE: usize = MODELS * BUCKETS * 2
    + 256
    + 64
    + 256
    + 64
    + 64
    + 512 * 2
    + DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);

/// Endpoints and indices of BC1, BC3, BC4 and BC5: small unsigned values,
/// summed per element modulo the element's width.
#[derive(Clone, Copy, Default)]
struct Plain {
    ep: [u8; 8],
    ix: [u8; 32],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ep {
    /// Endpoint 0 R G B, then endpoints 1-3, left-aligned and biased.
    e: [[u16; 3]; 4],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ix {
    mode: u8,
    ix: [i16; 16],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ep {
    ctrl: u16,
    /// Per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned.
    rows: [[u16; 8]; 3],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ix {
    bits: [u8; 2],
    ix: [[i16; 16]; 2],
}

/// Per-bucket rANS decode parameters packed into 16 bytes for single-line access.
#[derive(Clone, Copy)]
#[repr(C)]
struct BucketInfo {
    recip: u64,
    start: u16,
    freq: u16,
    base: u8,
    _pad: [u8; 3],
}

/// Per-model rANS lookup tables packed together for cache locality.
#[derive(Clone, Copy)]
struct Model {
    /// Bucket boundaries (`start[used..=BUCKETS] == 65536` sentinel).
    start: [u32; BUCKETS + 1],
    /// Per-bucket decode parameters (`info[0]` adjusted for slot 0 unused).
    info: [BucketInfo; BUCKETS],
    /// Bucket of the first slot of each 256-slot window.
    bucket: [u8; 256],
}

/// The format's constant tables.
pub struct Tables {
    models: Box<[Model; MODELS]>,
    rank1: [u8; 256],
    rank3a: [u8; 64],
    rank3c: [u8; 256],
    rank4: [u8; 64],
    rank5: [u8; 64],
    ctrl7: [u16; 512],
    bc1: Vec<Plain>,
    bc3: Vec<Plain>,
    bc4: Vec<Plain>,
    bc5: Vec<Plain>,
    bc6: Vec<(Bc6Ep, Bc6Ix)>,
    bc7: Vec<(Bc7Ep, Bc7Ix)>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        head.try_into().unwrap_or([0; N])
    }
    fn u8(&mut self) -> u8 {
        self.bytes::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes())
    }
}

impl Tables {
    /// The tables VIctory ships with.
    pub fn builtin() -> Result<Tables, Error> {
        Tables::parse(BUILTIN_TABLES)
    }

    /// Reads the table file: 128 x 9 u16 model frequencies (per bucket, the
    /// frequency of each of its symbols), the rank tables (BC1 256 B, BC3
    /// alpha 64 B and colour 256 B, BC4 64 B, BC5 64 B), 512 u16 BC7 control
    /// words, then 255 dictionary entries per format in format order: BC1
    /// 6 + 16 B, BC3 8 + 32 B, BC4 2 + 16 B, BC5 4 + 32 B, BC6H (u8 mode,
    /// 12 u16, u8 mode, 16 i16), BC7 (u16 control, 24 u16, 2 u8, 2 B,
    /// 16 + 15 i16). Little-endian, no padding. Every model must tile 65536.
    pub fn parse(bytes: &[u8]) -> Result<Tables, Error> {
        if bytes.len() != TABLES_SIZE {
            return Err(Error("table file has the wrong size"));
        }
        let mut r = Reader(bytes);
        let mut models = Box::new(
            [Model {
                start: [0; BUCKETS + 1],
                info: [BucketInfo {
                    recip: 0,
                    start: 0,
                    freq: 0,
                    base: 0,
                    _pad: [0; 3],
                }; BUCKETS],
                bucket: [0; 256],
            }; MODELS],
        );
        for (m, model) in models.iter_mut().enumerate() {
            let mut raw_freq = [0u32; BUCKETS];
            for f in &mut raw_freq {
                *f = u32::from(r.u16());
            }
            // Model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols.
            let used = m / 16 + 2;
            let mut at = 0u32;
            for b in 0..used {
                if raw_freq[b] == 0 {
                    return Err(Error("a model has an empty bucket"));
                }
                model.start[b] = at;
                model.info[b] = BucketInfo {
                    recip: if b < 2 {
                        0
                    } else {
                        (1u64 << 32).div_ceil(u64::from(raw_freq[b]))
                    },
                    start: if b == 0 { 1 } else { at as u16 },
                    freq: if b == 0 {
                        (raw_freq[0] - 1) as u16
                    } else {
                        raw_freq[b] as u16
                    },
                    base: if b == 0 { 0 } else { 1u8 << (b - 1) },
                    _pad: [0; 3],
                };
                at += raw_freq[b] << b.saturating_sub(1);
            }
            if at != 1 << 16 {
                return Err(Error("a model does not tile 65536"));
            }
            model.start[used..].fill(1 << 16);
            for (window, b) in model.bucket.iter_mut().enumerate() {
                let first = (window as u32) << 8;
                let idx = model.start[1..].iter().filter(|&&s| s <= first).count();
                // Flag bit 7 when another bucket starts inside this 256-slot window.
                let crosses = model.start[idx + 1] < first + 256;
                *b = (idx as u8) | if crosses { 0x80 } else { 0 };
            }
        }
        let rank1 = r.bytes();
        let rank3a = r.bytes();
        let rank3c = r.bytes();
        let rank4 = r.bytes();
        let rank5 = r.bytes();
        let mut ctrl7 = [0u16; 512];
        for c in &mut ctrl7 {
            *c = r.u16();
        }
        let mut plain = |ep: usize, ix: usize| {
            (0..DICTIONARY)
                .map(|_| {
                    let mut p = Plain::default();
                    for v in &mut p.ep[..ep] {
                        *v = r.u8();
                    }
                    for v in &mut p.ix[..ix] {
                        *v = r.u8();
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        let bc1 = plain(6, 16);
        let bc3 = plain(8, 32);
        let bc4 = plain(2, 16);
        let bc5 = plain(4, 32);
        let bc6 = (0..DICTIONARY)
            .map(|_| {
                // The endpoint entry's mode byte plays no part in decoding.
                r.u8();
                let mut ep = Bc6Ep::default();
                for v in ep.e.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc6Ix {
                    mode: r.u8(),
                    ..Default::default()
                };
                for v in &mut ix.ix {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        let bc7 = (0..DICTIONARY)
            .map(|_| {
                let mut ep = Bc7Ep {
                    ctrl: r.u16(),
                    ..Default::default()
                };
                for v in ep.rows.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc7Ix {
                    bits: [r.u8(), r.u8()],
                    ..Default::default()
                };
                // Unused: the widths again in all entries but the first.
                r.u16();
                // Only 15 slots for set 1, which no entry uses (0 bits).
                for v in &mut ix.ix.as_flattened_mut()[..31] {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        Ok(Tables {
            models,
            rank1,
            rank3a,
            rank3c,
            rank4,
            rank5,
            ctrl7,
            bc1,
            bc3,
            bc4,
            bc5,
            bc6,
            bc7,
        })
    }
}

const MAX_TLS_SCRATCH: usize = 16 * 1024 * 1024;

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

const BC1_WIDTHS: [u8; 12] = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7];
const BC3_WIDTHS: [u8; 12] = [3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7];
const BC4_WIDTHS: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const BC5_WIDTHS: [u8; 12] = [3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const BC6_WIDTHS: [u8; 16] = [3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4];
const BC7_WIDTHS: [u8; 16] = [3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4];

/// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let res = decode_with_scratch(bc, src, dst, tables, &mut scratch);
        if scratch.capacity() > MAX_TLS_SCRATCH {
            scratch.clear();
            scratch.shrink_to(MAX_TLS_SCRATCH);
        }
        res
    })
}

/// Like [`decode`], but reuses caller-provided `scratch` for intermediate
/// symbol records instead of a thread-local buffer.
pub fn decode_with_scratch(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables: &Tables,
    scratch: &mut Vec<u8>,
) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let blocks = dst.len() / block;
    let stride = match bc {
        1 | 4 => 12,
        3 | 5 => 24,
        _ => 32,
    };
    let records_len = blocks * stride;
    if scratch.len() < records_len {
        scratch.resize(records_len, 0);
    }
    let records_buf = &mut scratch[..records_len];
    match bc {
        1 => lanes_slice::<12, 1>(tables, &BC1_WIDTHS, blocks, src, records_buf)?,
        3 => lanes_slice::<12, 2>(tables, &BC3_WIDTHS, blocks, src, records_buf)?,
        4 => lanes_slice::<12, 1>(tables, &BC4_WIDTHS, blocks, src, records_buf)?,
        5 => lanes_slice::<12, 2>(tables, &BC5_WIDTHS, blocks, src, records_buf)?,
        6 => lanes_slice::<16, 2>(tables, &BC6_WIDTHS, blocks, src, records_buf)?,
        _ => lanes_slice::<16, 2>(tables, &BC7_WIDTHS, blocks, src, records_buf)?,
    }
    let records = records_buf.chunks_exact(stride);
    let out = dst.chunks_exact_mut(block);
    let t = tables;
    match bc {
        1 => plain::<false, 6, 16>(records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
        3 => plain::<true, 8, 32>(records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
        4 => plain::<false, 2, 16>(records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
        5 => plain::<true, 4, 32>(records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
        6 => bc6(records, out, &t.bc6),
        _ => bc7(records, out, t),
    }
}

// Lanes.

#[derive(Clone, Copy)]
struct Lane {
    x: u32,
    left: u32,
    /// Latched input byte (`< 0` when empty).
    latch: i16,
    /// Decoded output symbol (`< 0` when empty).
    out: i16,
    /// Symbol bits - 1 at the full alphabet.
    width: u8,
    /// 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    phase: u8,
    model: u8,
    long: bool,
    reset: bool,
    need: bool,
    stalled: bool,
}

impl Default for Lane {
    fn default() -> Self {
        Self {
            x: 0,
            left: 0,
            latch: -1,
            out: -1,
            width: 0,
            phase: 0,
            model: 0,
            long: false,
            reset: false,
            need: false,
            stalled: false,
        }
    }
}

/// Branchless rANS symbol decode for `x > LOW`.
#[inline(always)]
fn decode_symbol(m: &Model, x: u32) -> (u8, u32) {
    let slot = x & 0xffff;
    // SAFETY: `slot <= 65535`, so `(slot >> 8) < 256`. `Tables::parse` guarantees
    // `m.start[used..=BUCKETS] == 65536 > slot` and `(m.bucket[w] & 0x7f) < used <= BUCKETS`,
    // so the loop stops with `b < BUCKETS` and `b + 1 <= BUCKETS`.
    unsafe {
        let raw_b = *m.bucket.get_unchecked((slot >> 8) as usize);
        let mut b = usize::from(raw_b & 0x7f);
        if raw_b >= 0x80 {
            while *m.start.get_unchecked(b + 1) <= slot {
                b += 1;
            }
        }
        let info = *m.info.get_unchecked(b);
        let offset = slot.wrapping_sub(u32::from(info.start));
        let q = ((u64::from(offset) * info.recip) >> 32) as u32;
        let f = u32::from(info.freq);
        let sym = info.base.wrapping_add(q as u8);
        // Exact mod-2^32 ring identity: (x>>16)*f + (offset - q*f) == ((x>>16) - q)*f + offset.
        (sym, (x >> 16).wrapping_sub(q).wrapping_mul(f).wrapping_add(offset))
    }
}

impl Lane {
    /// One step: take the latched byte (header, length or payload) and
    /// decode at most one symbol.
    #[inline(always)]
    fn step(&mut self, t: &Tables) {
        if self.phase != 3 {
            if self.latch < 0 {
                return;
            }
            let b = self.latch as u8;
            self.latch = -1;
            self.need = true;
            match self.phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = self.width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.reset = b & 0x10 != 0;
                    self.long = b & 0x20 != 0;
                    self.left = 0;
                    self.phase = 1;
                }
                1 if self.long => {
                    self.left = u32::from(b);
                    self.phase = 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.phase = 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.phase = 3;
                }
            }
            return;
        }
        let mut x = self.x;
        if x < LOW && self.latch >= 0 {
            self.left = self.left.wrapping_sub(1);
            x = (x << 8) | (self.latch as u32);
            self.latch = -1;
        }
        self.need = self.left != 0 && self.latch < 0;
        if x > LOW {
            let m = &t.models[usize::from(self.model)];
            let (sym, next_x) = decode_symbol(m, x);
            self.x = next_x;
            self.out = i16::from(sym);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.phase = 0;
            self.need = self.latch < 0;
            if self.reset {
                self.x = 0;
            }
        }
    }
}

/// The bucket holding `slot` (below 65536): the last whose start is at or
/// below it. The window table gives the bucket of the window's first slot;
/// later buckets that start inside the window are stepped over.
#[cfg(test)]
#[inline(always)]
fn bucket_of(t: &Tables, model: u8, slot: u32) -> usize {
    let m = &t.models[usize::from(model)];
    let raw_b = m.bucket[(slot >> 8) as usize];
    let mut b = usize::from(raw_b & 0x7f);
    if raw_b >= 0x80 {
        while m.start[b + 1] <= slot {
            b += 1;
        }
    }
    b
}

/// Monomorphized slice-backed lane decoder with a steady-state fast path for
/// when all `N` lanes are in payload phase (`phase == 3`, `left > 1`).
fn lanes_slice<const N: usize, const PER: usize>(
    t: &Tables,
    widths: &[u8; N],
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Result<(), Error> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("popcnt")
    {
        // SAFETY: `avx2`, `bmi1`, `bmi2`, and `popcnt` were checked at runtime above.
        return unsafe { lanes_slice_avx2::<N, PER>(t, widths, blocks, src, records) };
    }
    lanes_slice_impl::<N, PER>(t, widths, blocks, src, records)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt")]
unsafe fn lanes_slice_avx2<const N: usize, const PER: usize>(
    t: &Tables,
    widths: &[u8; N],
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Result<(), Error> {
    lanes_slice_impl::<N, PER>(t, widths, blocks, src, records)
}

#[inline(always)]
fn lanes_slice_impl<const N: usize, const PER: usize>(
    t: &Tables,
    widths: &[u8; N],
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Result<(), Error> {
    let stride = N * PER;
    let total_pos = blocks * PER;
    assert!(records.len() >= blocks * stride);
    let mut lanes = [Lane::default(); N];
    for j in 0..N {
        lanes[j].width = widths[j];
        lanes[j].need = true;
    }
    let full = (1u32 << N) - 1;
    let mut cursor = 0usize;
    let mut pos = 0usize;
    let mut at = 0usize;
    let mut sub = 0usize;
    let mut done = 0u32;
    let mut idle = 0u32;

    while pos < total_pos {
        // Steady-state fast path: while every lane is in phase 3 with `left > 1`
        // and `src` has at least `ROUND_BYTES` left, no lane can change model,
        // end its fragment, or hit EOF during the next `fast` rounds.
        let mut min_left = u32::MAX;
        for lane in &lanes {
            let l = if lane.phase == 3 { lane.left } else { 0 };
            if l < min_left {
                min_left = l;
            }
        }
        let avail_rounds = src.len().saturating_sub(cursor) / ROUND_BYTES;
        let mut fast = (min_left.saturating_sub(1) as usize).min(avail_rounds);
        if fast > 0 {
            let models: [&Model; N] =
                std::array::from_fn(|j| &t.models[usize::from(lanes[j].model)]);
            let mut x = [0u32; N];
            let mut left = [0u32; N];
            // Slot 16 is a dummy sink for the branchless 8-byte arbiter when <8 lanes want a byte.
            let mut latch = [0u8; 17];
            let mut out = [0u8; N];
            let mut rec = [0u8; N];
            let mut rec0 = [0u8; N];
            let mut has_latch = 0u32;
            let mut out_mask = 0u32;
            let mut need_mask = 0u32;
            let mut stalled_mask = 0u32;
            for j in 0..N {
                let bit = 1u32 << j;
                x[j] = lanes[j].x;
                left[j] = lanes[j].left;
                if lanes[j].latch >= 0 {
                    latch[j] = lanes[j].latch as u8;
                    has_latch |= bit;
                }
                if lanes[j].out >= 0 {
                    out[j] = lanes[j].out as u8;
                    out_mask |= bit;
                }
                if lanes[j].need {
                    need_mask |= bit;
                }
                if lanes[j].stalled {
                    stalled_mask |= bit;
                }
            }
            if PER == 2 && sub == 1 {
                for j in 0..N {
                    rec0[j] = records[at - 1 + j * 2];
                }
            }
            if done != 0 {
                for j in 0..N {
                    if (done & (1u32 << j)) != 0 {
                        rec[j] = records[at + j * PER];
                    }
                }
            }

            loop {
                while fast > 0 && pos < total_pos {
                    // Phase 1: step all active (non-stalled) lanes independently.
                    // Read from immutable `old_latch` so lanes have zero loop-carried
                    // dependency on each other's `x < LOW` decision.
                    let active = (!stalled_mask) & full;
                    let old_latch = has_latch;
                    let mut consumed_latch = 0u32;
                    let mut produced_out = 0u32;
                    let mut m = active;
                    while m != 0 {
                        let j = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let bit = 1u32 << j;
                        // SAFETY: `j < N <= 16` because `active <= full = (1 << N) - 1`.
                        unsafe {
                            let mut xj = *x.get_unchecked(j);
                            let take = (xj < LOW) & ((old_latch & bit) != 0);
                            xj = if take {
                                (xj << 8) | u32::from(*latch.get_unchecked(j))
                            } else {
                                xj
                            };
                            *left.get_unchecked_mut(j) -= u32::from(take);
                            consumed_latch |= if take { bit } else { 0 };
                            if xj > LOW {
                                let (sym, next_x) = decode_symbol(models.get_unchecked(j), xj);
                                *x.get_unchecked_mut(j) = next_x;
                                *out.get_unchecked_mut(j) = sym;
                                produced_out |= bit;
                            } else {
                                *x.get_unchecked_mut(j) = xj;
                            }
                        }
                    }
                    has_latch = old_latch & !consumed_latch;
                    out_mask |= produced_out;
                    need_mask = (need_mask & stalled_mask) | ((!has_latch) & active);

                    // Phase 2: zero-branch 8-byte round arbiter across all lanes.
                    let waiting_mask = out_mask & done;
                    let want_mask = need_mask & !waiting_mask;
                    let taken = (want_mask.count_ones() as usize).min(ROUND_BYTES);
                    let mut wm = want_mask;
                    // SAFETY: `fast <= avail_rounds` guarantees `cursor + ROUND_BYTES <= src.len()`.
                    // `(wm | (1 << 16)).trailing_zeros()` is in `0..N` when `wm != 0` and `16`
                    // (dummy sink slot in `latch: [u8; 17]`) when `wm == 0`.
                    unsafe {
                        let src_ptr = src.as_ptr().add(cursor);
                        for k in 0..ROUND_BYTES {
                            let j = (wm | (1u32 << 16)).trailing_zeros() as usize;
                            *latch.get_unchecked_mut(j) = *src_ptr.add(k);
                            wm &= wm.wrapping_sub(1);
                        }
                    }
                    cursor += taken;
                    has_latch |= want_mask ^ wm;
                    stalled_mask = wm | waiting_mask;

                    // Phase 3: zero-branch delivery into staging array + block flush.
                    let deliver_mask = out_mask & !stalled_mask;
                    for j in 0..N {
                        let bit = 1u32 << j;
                        rec[j] = if (deliver_mask & bit) != 0 {
                            out[j]
                        } else {
                            rec[j]
                        };
                    }
                    out_mask &= !deliver_mask;
                    done |= deliver_mask;

                    let moved = deliver_mask != 0;
                    if done == full {
                        pos += 1;
                        if PER == 1 {
                            // SAFETY: `pos - 1 < blocks` guarantees `at + N <= records.len()`.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    rec.as_ptr(),
                                    records.as_mut_ptr().add(at),
                                    N,
                                );
                            }
                            at += stride;
                        } else if sub == 0 {
                            rec0 = rec;
                            at += 1;
                            sub = 1;
                        } else {
                            // SAFETY: `at - 1 == block * stride` with `block < blocks`.
                            unsafe {
                                let dst = records.as_mut_ptr().add(at - 1);
                                for j in 0..N {
                                    *dst.add(j * 2) = rec0[j];
                                    *dst.add(j * 2 + 1) = rec[j];
                                }
                            }
                            at += stride - 1;
                            sub = 0;
                        }
                        done = 0;
                    }
                    idle = if taken > 0 || moved { 0 } else { idle + 1 };
                    if idle > 3 {
                        return Err(Error("the stream ends early or is corrupt"));
                    }
                    fast -= 1;
                }
                if pos >= total_pos {
                    break;
                }
                let mut ml = u32::MAX;
                for &l in &left {
                    if l < ml {
                        ml = l;
                    }
                }
                let ar = src.len().saturating_sub(cursor) / ROUND_BYTES;
                fast = (ml.saturating_sub(1) as usize).min(ar);
                if fast == 0 {
                    break;
                }
            }

            if PER == 2 && sub == 1 {
                for j in 0..N {
                    records[at - 1 + j * 2] = rec0[j];
                }
            }
            if done != 0 {
                for j in 0..N {
                    if (done & (1u32 << j)) != 0 {
                        records[at + j * PER] = rec[j];
                    }
                }
            }
            for j in 0..N {
                let bit = 1u32 << j;
                lanes[j].x = x[j];
                lanes[j].left = left[j];
                lanes[j].latch = if (has_latch & bit) != 0 {
                    i16::from(latch[j])
                } else {
                    -1
                };
                lanes[j].out = if (out_mask & bit) != 0 {
                    i16::from(out[j])
                } else {
                    -1
                };
                lanes[j].need = (need_mask & bit) != 0;
                lanes[j].stalled = (stalled_mask & bit) != 0;
            }
            continue;
        }

        // General round: handles headers, lengths, fragment endings, and EOF tail.
        let mut taken = 0usize;
        let done_before = done;
        for j in 0..N {
            let lane = &mut lanes[j];
            let bit = 1u32 << j;
            let before = taken;
            if !lane.stalled {
                lane.step(t);
            }
            let has_out = lane.out >= 0;
            let waiting = has_out && (done & bit) != 0;
            if lane.need && !waiting && taken < ROUND_BYTES && cursor < src.len() {
                lane.latch = i16::from(src[cursor]);
                cursor += 1;
                taken += 1;
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled && has_out {
                // SAFETY: `pos < blocks * PER` guarantees `at + j * PER < blocks * stride <= records.len()`.
                unsafe {
                    *records.get_unchecked_mut(at + j * PER) = lane.out as u8;
                }
                lane.out = -1;
                done |= bit;
            }
        }
        let moved = done != done_before;
        if done == full {
            pos += 1;
            if PER == 1 {
                at += stride;
            } else if sub == 0 {
                at += 1;
                sub = 1;
            } else {
                at += stride - 1;
                sub = 0;
            }
            done = 0;
        }
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}

/// Decodes `per` symbols per block on every lane into `records` (one record
/// of `widths.len() * per` symbols per block), taking input from `fetch`
/// (given the lane that takes the byte) in the order of the schedule: each
/// round the lanes step in order and take at most 8 bytes between them.
#[cfg(test)]
fn lanes(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    mut fetch: impl FnMut(usize) -> Option<u8>,
    records: &mut [u8],
) -> Result<(), Error> {
    let mut lanes = [Lane::default(); 16];
    let lanes = &mut lanes[..widths.len()];
    for (lane, &width) in lanes.iter_mut().zip(widths) {
        lane.width = width;
        lane.need = true;
    }
    let stride = widths.len() * per;
    let full = (1u32 << widths.len()) - 1;
    let (mut pos, mut done, mut idle) = (0, 0u32, 0);
    // Where symbol `pos` of lane 0 goes; lane j's is `per * j` further on.
    let mut at = 0;
    while pos < blocks * per {
        let (mut taken, mut moved) = (0, false);
        for (j, lane) in lanes.iter_mut().enumerate() {
            let bit = 1 << j;
            let before = taken;
            if !lane.stalled {
                lane.step(t);
            }
            let waiting = lane.out >= 0 && done & bit != 0;
            if lane.need && !waiting && taken < ROUND_BYTES {
                // Past the end nothing arrives; in a valid stream only finished lanes ask there.
                if let Some(b) = fetch(j) {
                    lane.latch = i16::from(b);
                    taken += 1;
                }
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled && lane.out >= 0 {
                records[at + j * per] = lane.out as u8;
                lane.out = -1;
                done |= bit;
                moved = true;
            }
        }
        if done == full {
            pos += 1;
            at += if pos.is_multiple_of(per) {
                stride - per + 1
            } else {
                1
            };
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}

// Blocks.

const SPREAD4: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut v = 0usize;
    while v < 16 {
        t[v] = ((v & 1) | ((v & 2) << 1) | ((v & 4) << 2) | ((v & 8) << 3)) as u8;
        v += 1;
    }
    t
};

const SPREAD8: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut v = 0usize;
    while v < 256 {
        let mut acc = 0u16;
        let mut k = 0;
        while k < 8 {
            acc |= (((v >> k) & 1) as u16) << (2 * k);
            k += 1;
        }
        t[v] = acc;
        v += 1;
    }
    t
};

const fn make_nibbles(odd_first: bool) -> [[i16; 2]; 256] {
    let mut t = [[0i16; 2]; 256];
    let mut b = 0usize;
    while b < 256 {
        let even_u = ((b & 1) | ((b >> 1) & 2) | ((b >> 2) & 4) | ((b >> 3) & 8)) as i16;
        let odd_u = (((b >> 1) & 1) | ((b >> 2) & 2) | ((b >> 3) & 4) | ((b >> 4) & 8)) as i16;
        let even = (even_u >> 1) ^ -(even_u & 1);
        let odd = (odd_u >> 1) ^ -(odd_u & 1);
        t[b] = if odd_first { [odd, even] } else { [even, odd] };
        b += 1;
    }
    t
}

const NIBBLES_EVEN: [[i16; 2]; 256] = make_nibbles(false);
const NIBBLES_ODD: [[i16; 2]; 256] = make_nibbles(true);

/// The record's endpoint and index references.
#[inline(always)]
fn references(s: &[u8], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (interleave(s[0], s[1]), interleave(s[2], s[3]))
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Two 4-bit symbols, the first on the even bits.
#[inline(always)]
fn interleave(a: u8, b: u8) -> u8 {
    SPREAD4[usize::from(a & 15)] | (SPREAD4[usize::from(b & 15)] << 1)
}

/// Two 8-bit symbols interleaved into a 16-bit word, `a` on even bits.
#[inline(always)]
fn wide(a: u8, b: u8) -> u16 {
    SPREAD8[usize::from(a)] | (SPREAD8[usize::from(b)] << 1)
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
#[inline(always)]
fn pick<'a, T>(i: usize, r: usize, history: &'a [T; HISTORY], dictionary: &'a [T], zero: &'a T) -> &'a T {
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) & (HISTORY - 1)],
        r => &dictionary[r - i - 1],
    }
}

/// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
#[inline(always)]
fn zz(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ (v & 1).wrapping_neg()) & mask
}

/// Element masks of the plain formats' endpoints and indices.
struct Widths {
    ep_mask: [u8; 8],
    ix_mask: [u8; 32],
}

const fn widths(ep: [u8; 8], ix3: usize, ix2: usize) -> Widths {
    let mut ep_mask = [0u8; 8];
    let mut k = 0;
    while k < 8 {
        ep_mask[k] = ((1u16 << ep[k]) - 1) as u8;
        k += 1;
    }
    let mut ix_mask = [0u8; 32];
    let mut k = 0;
    while k < ix3 + ix2 {
        ix_mask[k] = if k < ix3 { 7 } else { 3 };
        k += 1;
    }
    Widths { ep_mask, ix_mask }
}

const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0);

fn plain<'a, const PAIRED: bool, const EP_N: usize, const IX_N: usize>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[Plain],
    w: &Widths,
    residual: impl Fn(&[u8]) -> Plain,
    pack: fn(&Plain, &mut [u8]),
) -> Result<(), Error> {
    let mut history = [Plain::default(); HISTORY];
    let zero = Plain::default();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, PAIRED);
        let ep_ref = &pick(i, ep_ref, &history, dictionary, &zero).ep;
        let ix_ref = &pick(i, ix_ref, &history, dictionary, &zero).ix;
        let mut v = residual(s);
        for k in 0..EP_N {
            v.ep[k] = v.ep[k].wrapping_add(ep_ref[k]) & w.ep_mask[k];
        }
        for k in 0..IX_N {
            v.ix[k] = v.ix[k].wrapping_add(ix_ref[k]) & w.ix_mask[k];
        }
        history[i & (HISTORY - 1)] = v;
        pack(&v, block);
    }
    Ok(())
}

/// 2-bit index residuals: one symbol per 2x2 quad.
#[inline(always)]
fn quads(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    let v0 = rank[usize::from(s[0])];
    ix[0] = v0 & 3;
    ix[1] = (v0 >> 2) & 3;
    ix[4] = (v0 >> 4) & 3;
    ix[5] = (v0 >> 6) & 3;

    let v1 = rank[usize::from(s[1])];
    ix[2] = v1 & 3;
    ix[3] = (v1 >> 2) & 3;
    ix[6] = (v1 >> 4) & 3;
    ix[7] = (v1 >> 6) & 3;

    let v2 = rank[usize::from(s[2])];
    ix[8] = v2 & 3;
    ix[9] = (v2 >> 2) & 3;
    ix[12] = (v2 >> 4) & 3;
    ix[13] = (v2 >> 6) & 3;

    let v3 = rank[usize::from(s[3])];
    ix[10] = v3 & 3;
    ix[11] = (v3 >> 2) & 3;
    ix[14] = (v3 >> 4) & 3;
    ix[15] = (v3 >> 6) & 3;
}

/// 3-bit index residuals: one symbol per two pixels in raster order.
#[inline(always)]
fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = (v >> 4) & 7;
    }
}

#[inline(always)]
fn bc1_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain {
        ep: [
            zz(s[2], 5),
            zz(s[3], 6),
            zz(s[4], 5),
            zz(s[5], 5),
            zz(s[6], 6),
            zz(s[7], 5),
            0,
            0,
        ],
        ix: [0; 32],
    };
    quads(&t.rank1, &s[8..12], &mut v.ix[..16]);
    v
}

#[inline(always)]
fn bc3_residual(t: &Tables, s: &[u8]) -> Plain {
    // Colour lanes carry each channel's two endpoints together.
    let mut v = Plain {
        ep: [
            zz(s[14], 5),
            zz(s[16], 6),
            zz(s[18], 5),
            zz(s[15], 5),
            zz(s[17], 6),
            zz(s[19], 5),
            zz(s[4], 8),
            zz(s[5], 8),
        ],
        ix: [0; 32],
    };
    pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
    quads(&t.rank3c, &s[20..24], &mut v.ix[16..]);
    v
}

#[inline(always)]
fn bc4_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain {
        ep: [zz(s[2], 8), zz(s[3], 8), 0, 0, 0, 0, 0, 0],
        ix: [0; 32],
    };
    pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
    v
}

#[inline(always)]
fn bc5_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain {
        ep: [
            zz(s[4], 8),
            zz(s[5], 8),
            zz(s[6], 8),
            zz(s[7], 8),
            0,
            0,
            0,
            0,
        ],
        ix: [0; 32],
    };
    pairs(&t.rank5, &s[8..24], &mut v.ix);
    v
}

/// Indices along the endpoint line -> BC1 and BC4 codes.
const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

#[inline(always)]
fn bc1_color(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let ix: &[u8; 16] = ix[..16].try_into().unwrap();
    let mut bits = 0u32;
    for p in 0..16 {
        bits |= BC1_CODE[usize::from(ix[p] & 3)] << (2 * p);
    }
    let word = u64::from(c0) | (u64::from(c1) << 16) | (u64::from(bits) << 32);
    out[..8].copy_from_slice(&word.to_le_bytes());
}

/// BC4 half: endpoints are kept 128 below their stored value.
#[inline(always)]
fn bc4_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let ix: &[u8; 16] = ix[..16].try_into().unwrap();
    let mut bits = 0u64;
    for p in 0..16 {
        bits |= BC4_CODE[usize::from(ix[p] & 7)] << (16 + 3 * p);
    }
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
    out[..8].copy_from_slice(&v.to_le_bytes());
}

fn bc1_pack(v: &Plain, out: &mut [u8]) {
    bc1_color(&v.ep, &v.ix, out);
}

fn bc3_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
    bc1_color(&v.ep, &v.ix[16..], &mut out[8..]);
}

fn bc4_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix, out);
}

fn bc5_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
    bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
}

/// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
#[inline(always)]
fn nibbles(b: u8, odd_first: bool) -> [i16; 2] {
    if odd_first {
        NIBBLES_ODD[usize::from(b)]
    } else {
        NIBBLES_EVEN[usize::from(b)]
    }
}

/// Little-endian bit writer for one 128-bit block.
#[derive(Default)]
struct Bits {
    v: u128,
    at: u32,
}

impl Bits {
    #[inline(always)]
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 && self.at < 128 {
            let masked = if n >= 64 {
                value
            } else {
                value & ((1u64 << n) - 1)
            };
            self.v |= u128::from(masked) << self.at;
        }
        self.at += n;
    }
}

// BC7.

/// Public BC7 anchor indices: second subset of two; second and third of three.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
const BC7_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the primary and secondary sets per mode.
const BC7_INDEX: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];

fn pbits(mode: u16) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// Channel rotation bits; only BC7 modes 4 and 5 have a rotation field.
fn rotation(ctrl: u16) -> usize {
    if matches!(ctrl & 7, 4 | 5) {
        usize::from(ctrl >> 3 & 3)
    } else {
        0
    }
}

/// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
/// with the rotated channel.
fn bc7_precision(ctrl: u16) -> [u8; 4] {
    let (mode, rot) = (ctrl & 7, rotation(ctrl));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2") {
        // SAFETY: `avx2` and `bmi2` were checked at runtime above.
        return unsafe { bc7_avx2(records, out, t) };
    }
    bc7_impl(records, out, t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn bc7_avx2<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    bc7_impl(records, out, t)
}

#[inline(always)]
fn bc7_impl<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let mut history = [(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if ctrl == 0 {
            return Err(Error("unused BC7 control code"));
        }
        let mode = usize::from(ctrl & 7);
        let subsets = usize::from((ctrl & 0xff) >> 6);
        let prec = bc7_precision(ctrl);
        let r = &pick(i, ep_ref, &history, &t.bc7, &zero).0;
        let mut ep = Bc7Ep {
            ctrl,
            ..Default::default()
        };
        let columns = if prec[3] > 0 { 8 } else { 6 };
        if subsets == 1 {
            let r0 = &r.rows[0];
            for c in 0..columns {
                let p = prec[c / 2];
                if p == 0 {
                    continue;
                }
                let sh = 16 - p;
                let mask = 0xffffu16 << sh;
                let v = s[4 + c];
                let residual =
                    (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << sh;
                ep.rows[0][c] = (r0[c] & mask).wrapping_add(residual);
            }
        } else {
            let three = subsets == 3;
            let or0 = |v: u8| if three { 0 } else { v };
            let if3 = |v: u8| if three { v } else { 0 };
            let stored = [
                [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
                [
                    s[12],
                    s[13],
                    s[14],
                    s[15],
                    s[16],
                    s[17],
                    or0(s[18]),
                    or0(s[19]),
                ],
                [
                    s[20],
                    s[21],
                    if3(s[10]),
                    if3(s[11]),
                    if3(s[18]),
                    if3(s[19]),
                    0,
                    0,
                ],
            ];
            // Rows the reference lacks build on its first row.
            let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
            for c in 0..columns {
                let p = prec[c / 2];
                if p == 0 {
                    continue;
                }
                let sh = 16 - p;
                let mask = 0xffffu16 << sh;
                for (row, st) in stored.iter().enumerate().take(subsets) {
                    let v = st[c];
                    let residual =
                        (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << sh;
                    let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                    ep.rows[row][c] = base.wrapping_add(residual);
                }
            }
        }
        let bits = BC7_INDEX[mode];
        let rx = &pick(i, ix_ref, &history, &t.bc7, &zero).1;
        let mut ix = Bc7Ix {
            bits,
            ..Default::default()
        };
        for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
            let cur_bits = bits[set];
            if cur_bits == 0 {
                continue;
            }
            let sh = 16 - cur_bits;
            let ref_bits = rx.bits[set];
            let dst = &mut ix.ix[set];
            if ref_bits == 0 || ref_bits > 16 {
                for (k, o) in offsets.into_iter().enumerate() {
                    let lut = if k < 3 { &NIBBLES_ODD } else { &NIBBLES_EVEN };
                    let p0 = lut[usize::from(s[o])];
                    let p1 = lut[usize::from(s[o + 1])];
                    let base = 4 * k;
                    dst[base] = ((p0[0] as u16) << sh) as i16 >> sh;
                    dst[base + 1] = ((p0[1] as u16) << sh) as i16 >> sh;
                    dst[base + 2] = ((p1[0] as u16) << sh) as i16 >> sh;
                    dst[base + 3] = ((p1[1] as u16) << sh) as i16 >> sh;
                }
            } else {
                let ref_sh = 16 - ref_bits;
                let rix = &rx.ix[set];
                for (k, o) in offsets.into_iter().enumerate() {
                    let lut = if k < 3 { &NIBBLES_ODD } else { &NIBBLES_EVEN };
                    let p0 = lut[usize::from(s[o])];
                    let p1 = lut[usize::from(s[o + 1])];
                    let base = 4 * k;
                    dst[base] = (((p0[0] as u16) << sh)
                        .wrapping_add((rix[base] as u16) << ref_sh)
                        as i16)
                        >> sh;
                    dst[base + 1] = (((p0[1] as u16) << sh)
                        .wrapping_add((rix[base + 1] as u16) << ref_sh)
                        as i16)
                        >> sh;
                    dst[base + 2] = (((p1[0] as u16) << sh)
                        .wrapping_add((rix[base + 2] as u16) << ref_sh)
                        as i16)
                        >> sh;
                    dst[base + 3] = (((p1[1] as u16) << sh)
                        .wrapping_add((rix[base + 3] as u16) << ref_sh)
                        as i16)
                        >> sh;
                }
            }
        }
        history[i & (HISTORY - 1)] = (ep, ix);
        bc7_pack(&ep, &ix, prec, block);
    }
    Ok(())
}

#[inline(always)]
fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, prec: [u8; 4], out: &mut [u8]) {
    let ctrl = ep.ctrl;
    let mode = ctrl & 7;
    let subsets = usize::from((ctrl & 0xff) >> 6);
    let part = usize::from(ctrl >> 8 & 63);
    let rot = rotation(ctrl);
    let mut w = Bits::default();
    w.put(1 << mode, u32::from(mode) + 1);
    match mode {
        0 => w.put(part as u64, 4),
        1 | 2 | 3 | 7 => w.put(part as u64, 6),
        4 => {
            w.put(rot as u64, 2);
            w.put(u64::from(ctrl >> 5 & 1), 1);
        }
        5 => w.put(rot as u64, 2),
        _ => {}
    }
    let ends = 2 * subsets;
    let field = |row: usize, col: usize, p: u8| u64::from(ep.rows[row][col] >> (16 - p));
    let mut order = [0, 1, 2, 3];
    if rot != 0 {
        order.swap(3, rot - 1);
    }
    for ch in order {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        let drop = u32::from(ch == 0 && pbits(mode));
        for e in 0..ends {
            w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32::from(p) - drop);
        }
    }
    match mode {
        1 => (0..subsets).for_each(|row| w.put(field(row, 0, prec[0]) & 1, 1)),
        m if pbits(m) => (0..ends).for_each(|e| w.put(field(e / 2, e % 2, prec[0]) & 1, 1)),
        _ => {}
    }
    let anchor_mask: u16 = match subsets {
        2 => 1 | (1 << ANCHOR2[part]),
        3 => 1 | (1 << ANCHOR3A[part]) | (1 << ANCHOR3B[part]),
        _ => 1,
    };
    let mut indices = |set: usize| {
        let bits = u32::from(ix.bits[set]);
        let mut acc = 0u64;
        let mut shift = 0u32;
        for (p, &v) in ix.ix[set].iter().enumerate() {
            let n = bits - u32::from((anchor_mask >> p) & 1);
            acc |= (u64::from(v as u16) & ((1u64 << n) - 1)) << shift;
            shift += n;
        }
        w.put(acc, shift);
    };
    match mode {
        4 => {
            indices(1);
            indices(0);
        }
        5 => {
            indices(0);
            indices(1);
        }
        _ => indices(0),
    }
    out.copy_from_slice(&w.v.to_le_bytes());
}

// BC6H.

/// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
const BC6_BITS: [[u8; 4]; 15] = [
    [0; 4],
    [10, 5, 5, 5],
    [7, 6, 6, 6],
    [11, 5, 4, 4],
    [11, 4, 5, 4],
    [11, 4, 4, 5],
    [9, 5, 5, 5],
    [8, 6, 5, 5],
    [8, 5, 6, 5],
    [8, 5, 5, 6],
    [6, 6, 6, 6],
    [10, 10, 10, 10],
    [11, 9, 9, 9],
    [12, 8, 8, 8],
    [16, 4, 4, 4],
];
/// Mode code and its width, modes 1-14.
const BC6_CODE: [(u64, u32); 15] = [
    (0, 0),
    (0, 2),
    (1, 2),
    (2, 5),
    (6, 5),
    (10, 5),
    (14, 5),
    (18, 5),
    (22, 5),
    (26, 5),
    (30, 5),
    (3, 5),
    (7, 5),
    (11, 5),
    (15, 5),
];

/// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is
/// the partition) of endpoint `ep`, from bit `first` up.
#[derive(Clone, Copy)]
struct Run {
    ch: u8,
    ep: u8,
    first: u8,
    len: u8,
}

const fn r(ch: u8, ep: u8, first: u8, len: u8) -> Run {
    Run { ch, ep, first, len }
}

const R: u8 = 0;
const G: u8 = 1;
const B: u8 = 2;
const D: Run = r(3, 0, 0, 5);

/// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
/// high base bits in ascending order (what the console's packer does).
#[rustfmt::skip]
const BC6_LAYOUT: [&[Run]; 15] = [
    &[],
    &[
        r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
        r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(G, 2, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
        r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(B, 3, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
        r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
        r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
        r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
        r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
        r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
        r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
    ],
];

fn bc6_index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

fn bc6<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[(Bc6Ep, Bc6Ix)],
) -> Result<(), Error> {
    let mut history = [(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ..Default::default()
        },
    );
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error("BC6H mode out of range"));
        }
        let two = mode < 11;
        let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
        let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
        let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
        let (e2, e3) = if two {
            (
                [s[16], s[18], s[20]].map(|v| z(v.into())),
                [s[17], s[19], s[21]].map(|v| z(v.into())),
            )
        } else {
            (e0, e1)
        };
        let r = &pick(i, ep_ref, &history, dictionary, &zero).0;
        let mut ep = Bc6Ep::default();
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        let mask = |sh: u32| 0xffffu16 << sh;
        for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
            *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
        }
        for (k, res) in [e1, e2, e3]
            .iter()
            .enumerate()
            .take(if two { 3 } else { 1 })
        {
            for c in 0..3 {
                ep.e[k + 1][c] =
                    r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
            }
        }
        if !two {
            for c in 0..3 {
                ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
            }
            ep.e[3] = ep.e[1];
        }
        let rx = &pick(i, ix_ref, &history, dictionary, &zero).1;
        let bits = bc6_index_bits(mode);
        let sh = 16 - bits;
        let ref_bits = bc6_index_bits(rx.mode);
        let mut ix = Bc6Ix {
            mode,
            ..Default::default()
        };
        if ref_bits == 0 || ref_bits > 16 {
            for k in 0..8 {
                let pair = nibbles(s[22 + k], k < 6);
                ix.ix[2 * k] = ((pair[0] as u16) << sh) as i16 >> sh;
                ix.ix[2 * k + 1] = ((pair[1] as u16) << sh) as i16 >> sh;
            }
        } else {
            let ref_sh = 16 - ref_bits;
            for k in 0..8 {
                let pair = nibbles(s[22 + k], k < 6);
                let p0 = 2 * k;
                let p1 = p0 + 1;
                ix.ix[p0] = (((pair[0] as u16) << sh)
                    .wrapping_add((rx.ix[p0] as u16) << ref_sh) as i16)
                    >> sh;
                ix.ix[p1] = (((pair[1] as u16) << sh)
                    .wrapping_add((rx.ix[p1] as u16) << ref_sh) as i16)
                    >> sh;
            }
        }
        history[i & (HISTORY - 1)] = (ep, ix);
        // Fields: the working value has its top bit flipped.
        let field = |k: usize, c: usize| {
            let sh = if k == 0 { shift[0] } else { shift[1 + c] };
            u64::from((ep.e[k][c] ^ 0x8000) >> sh)
        };
        let mut w = Bits::default();
        let (code, n) = BC6_CODE[usize::from(mode)];
        w.put(code, n);
        let part = s[31] & 31;
        for run in BC6_LAYOUT[usize::from(mode)] {
            if run.ch == 3 {
                w.put(part.into(), 5);
            } else {
                w.put(
                    field(run.ep.into(), run.ch.into()) >> run.first,
                    run.len.into(),
                );
            }
        }
        let anchor_mask: u16 = if two {
            1 | (1 << ANCHOR2[usize::from(part)])
        } else {
            1
        };
        let bias = 1u16 << (bits - 1);
        let mut acc = 0u64;
        let mut shift = 0u32;
        for (p, &v) in ix.ix.iter().enumerate() {
            let n = u32::from(bits) - u32::from((anchor_mask >> p) & 1);
            let v = u64::from((v as u16).wrapping_add(bias)) & ((1u64 << n) - 1);
            acc |= v << shift;
            shift += n;
        }
        w.put(acc, shift);
        block.copy_from_slice(&w.v.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shipped_tables_parse() {
        assert!(super::Tables::builtin().is_ok());
    }

    use super::*;

    /// Tables we made: uniform models, identity ranks, a BC7 code per mode,
    /// empty dictionaries.
    fn tables() -> Vec<u8> {
        let mut t = Vec::new();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend((0..=255u8).map(|v| v & 0x77)); // rank1 (unused by the tests beyond 2-bit fields)
        t.extend(0..64u8); // rank3a
        t.extend(0..=255u8); // rank3c
        t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4)); // rank4: low 3 bits, next 3 bits
        t.extend(0..64u8); // rank5
        for code in 0..512u16 {
            // Code c: mode 6 (one subset), the only one the tests use.
            let ctrl: u16 = if code == 0 { 0x46 } else { 0 };
            t.extend_from_slice(&ctrl.to_le_bytes());
        }
        t.resize(TABLES_SIZE, 0);
        t
    }

    /// A lane's symbols as one fragment: uniform model, rANS from 2^24, no
    /// reset, so the lane keeps 2^24 when it ends.
    fn fragment(width: u8, symbols: &[u8]) -> Vec<u8> {
        let f = 1u32 << (15 - width);
        let mut x = LOW;
        let mut bytes = Vec::new();
        for &s in symbols.iter().rev() {
            let (freq, start) = if s == 0 {
                (f - 1, 1)
            } else {
                (f, u32::from(s) * f)
            };
            while x >= (freq << 16) {
                bytes.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << 16) + x % freq + start;
        }
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.reverse();
        let mut out = vec![0];
        if bytes.len() > 256 {
            out[0] |= 0x20;
            let n = bytes.len() - 257;
            out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
        } else {
            out.push((bytes.len() - 1) as u8);
        }
        out.extend(bytes);
        out
    }

    /// Interleaves per-lane fragments the way the decoder takes bytes. A lane
    /// that has finished still asks: a filler fragment (two payload bytes)
    /// brings its state to 2^24 with a byte latched, which parks it for good.
    fn stream(t: &Tables, widths: &[u8], per: usize, records: &[Vec<u8>]) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = fragment(widths[j], &syms);
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    const BC4_LANES: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];

    fn zigzag(v: i32) -> u8 {
        if v >= 0 {
            (2 * v) as u8
        } else {
            (-2 * v - 1) as u8
        }
    }

    #[test]
    fn bc4_blocks_with_and_without_references() {
        let t = Tables::parse(&tables()).unwrap();
        // Block 0: endpoints 10, -3 (+128 stored), indices p -> p & 7.
        let mut b0 = vec![0, 0, zigzag(10), zigzag(-3)];
        b0.extend((0..8).map(|k| ((2 * k) & 7) | ((2 * k + 1) & 7) << 3));
        // Block 1 refers to block 0 for both: residual endpoints +1, +1; indices +1.
        let mut b1 = vec![0, 1, zigzag(1), zigzag(1)];
        b1.extend((0..8).map(|_| 1 | 1 << 3));
        let src = stream(&t, &BC4_LANES, 1, &[b0, b1]);
        let mut dst = [0u8; 16];
        decode(4, &src, &mut dst, &t).unwrap();
        let block = |e0: u8, e1: u8, ix: &dyn Fn(usize) -> usize| {
            let bits = (0..16).fold(0u64, |a, p| a | BC4_CODE[ix(p)] << (16 + 3 * p));
            (u64::from(e0) | u64::from(e1) << 8 | bits).to_le_bytes()
        };
        assert_eq!(dst[..8], block(138, 125, &|p| p & 7));
        assert_eq!(dst[8..], block(139, 126, &|p| (p + 1) & 7));
    }

    #[test]
    fn lanes_take_bytes_in_schedule_order_and_long_fragments() {
        let t = Tables::parse(&tables()).unwrap();
        // Many blocks: long fragments (over 256 payload bytes) on the wide lanes.
        let records: Vec<Vec<u8>> = (0..400u32)
            .map(|i| {
                let mut r = vec![0, 0, (i * 7) as u8, (i * 13) as u8];
                r.extend((0..8).map(|k| ((i + k) * 5 % 64) as u8));
                r
            })
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut got = vec![0u8; records.len() * 12];
        let mut next = src.iter().copied();
        lanes(&t, &BC4_LANES, 1, records.len(), |_| next.next(), &mut got).unwrap();
        assert_eq!(got, records.concat());
    }

    #[test]
    fn bc7_mode6_block() {
        let t = Tables::parse(&tables()).unwrap();
        // Control code 0 -> mode 6; endpoints R 8 bits (p-bit low), G B A 7 bits.
        let mut s = vec![0u8; 32];
        s[4] = zigzag(0x55); // R0 = 0x55: field 0x2a, p-bit 1
        s[5] = zigzag(-1); // R1 = 0xff (8 bits)
        s[6] = zigzag(3);
        s[10] = zigzag(0x7f >> 1);
        let src = stream(
            &t,
            &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4],
            2,
            &[s],
        );
        let mut dst = [0u8; 16];
        decode(7, &src, &mut dst, &t).unwrap();
        let mut w = Bits::default();
        w.put(1 << 6, 7);
        for (v, n) in [
            (0x2a, 7),
            (0x7f, 7),
            (3, 7),
            (0, 7),
            (0, 7),
            (0, 7),
            (0x3f, 7),
            (0, 7),
        ] {
            w.put(v, n);
        }
        w.put(1, 1);
        w.put(1, 1);
        assert_eq!(dst, w.v.to_le_bytes());
    }

    #[test]
    fn bc7_rotation_applies_only_to_modes_4_and_5() {
        const BC7_WIDTHS: [u8; 16] = [3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4];
        const CTRL7_OFFSET: usize = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64;
        let tables_for = |ctrl: u16| {
            let mut raw = tables();
            raw[CTRL7_OFFSET..CTRL7_OFFSET + 2].copy_from_slice(&ctrl.to_le_bytes());
            let t = Tables::parse(&raw).unwrap();
            assert_eq!(t.ctrl7[0], ctrl);
            t
        };

        let mut record = vec![0u8; 32];
        for (i, &val) in [11, 23, 5, 19, 13, 29, 7, 17, 3, 25, 9, 21, 15, 31, 1, 12]
            .iter()
            .enumerate()
        {
            record[4 + i] = zigzag(val);
        }
        record[20] = zigzag(6);
        record[21] = zigzag(14);

        // Modes without a BC7 rotation field must ignore ctrl bits 4:3 instead of
        // panicking on shift overflow (modes 0/1/3 with rot=1) or corrupting channels.
        for (mode, subsets) in [(0u16, 3u16), (1, 2), (2, 3), (3, 2), (6, 1), (7, 2)] {
            let base_ctrl = (subsets << 6) | mode;
            let base_tables = tables_for(base_ctrl);
            let base_src = stream(&base_tables, &BC7_WIDTHS, 2, std::slice::from_ref(&record));
            let mut want = [0u8; 16];
            decode(7, &base_src, &mut want, &base_tables).unwrap();

            for rot in 1..=3u16 {
                let ctrl = base_ctrl | (rot << 3);
                let t = tables_for(ctrl);
                let src = stream(&t, &BC7_WIDTHS, 2, std::slice::from_ref(&record));
                let mut got = [0u8; 16];
                decode(7, &src, &mut got, &t).unwrap();
                assert_eq!(got, want, "mode {mode} rot {rot}");
                assert_eq!(
                    bc7_precision(ctrl),
                    bc7_precision(base_ctrl),
                    "mode {mode} rot {rot}"
                );
            }
        }

        // Modes 4 and 5 do have a 2-bit rotation field: rotation swaps alpha with channel rot-1.
        for (mode, base_bits) in [(4u16, [5u8, 5, 5, 6]), (5u16, [7u8, 7, 7, 8])] {
            for rot in 1..=3usize {
                let ctrl = (1 << 6) | ((rot as u16) << 3) | mode;
                let mut expected_prec = base_bits;
                expected_prec.swap(3, rot - 1);
                assert_eq!(bc7_precision(ctrl), expected_prec, "mode {mode} rot {rot}");
            }

            // Decode mode 4/5 with rot=1 (swap R and A) and check the packed BC7 block.
            let ctrl = (1 << 6) | (1 << 3) | mode;
            let t = tables_for(ctrl);
            let mut s = vec![0u8; 32];
            s[4] = zigzag(0x21); // stored R0 (holds alpha precision after rot=1)
            s[5] = zigzag(0x12); // stored R1
            s[6] = zigzag(0x05); // stored G0
            s[7] = zigzag(0x06); // stored G1
            s[8] = zigzag(0x09); // stored B0
            s[9] = zigzag(0x0a); // stored B1
            s[10] = zigzag(0x0d); // stored A0 (holds color precision after rot=1)
            s[11] = zigzag(0x0e); // stored A1
            let src = stream(&t, &BC7_WIDTHS, 2, &[s]);
            let mut dst = [0u8; 16];
            decode(7, &src, &mut dst, &t).unwrap();

            let color_bits = u32::from(base_bits[0]);
            let alpha_bits = u32::from(base_bits[3]);
            let mut w = Bits::default();
            w.put(1 << mode, u32::from(mode) + 1);
            w.put(1, 2); // rotation = 1
            if mode == 4 {
                w.put(0, 1); // indexSelectionBit = 0
            }
            // Endpoint order after rot=1 swaps indices 3 and 0: [3 (R), 1 (G), 2 (B), 0 (A)].
            for (v, n) in [
                (0x0d, color_bits),
                (0x0e, color_bits),
                (0x05, color_bits),
                (0x06, color_bits),
                (0x09, color_bits),
                (0x0a, color_bits),
                (0x21, alpha_bits),
                (0x12, alpha_bits),
            ] {
                w.put(v, n);
            }
            assert_eq!(dst, w.v.to_le_bytes(), "mode {mode} rot 1");
        }
    }

    #[test]
    fn malformed_input_is_an_error() {
        let t = Tables::parse(&tables()).unwrap();
        let records: Vec<Vec<u8>> = (0..64u8)
            .map(|i| vec![0, 0, i, i, 1, 2, 3, 4, 5, 6, 7, 8])
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut dst = vec![0u8; 64 * 8];
        decode(4, &src, &mut dst, &t).unwrap();
        // Truncated.
        assert!(decode(4, &src[..src.len() / 2], &mut dst, &t).is_err());
        // Headers promising more payload than the stream holds. (Other corrupt
        // headers decode to garbage: the format has no redundancy.)
        let headers = [[0x30; 8], [0xff; 8]].concat();
        assert!(decode(4, &headers, &mut dst[..8], &t).is_err());
        assert!(decode(4, &[], &mut dst, &t).is_err());
        // Destination not a whole number of blocks, empty, more blocks than coded.
        assert!(decode(4, &src, &mut dst[..12], &t).is_err());
        assert!(decode(4, &src, &mut [], &t).is_err());
        assert!(decode(4, &src, &mut vec![0u8; 65 * 8], &t).is_err());
        // Unsupported formats, bad tables.
        assert!(decode(2, &src, &mut dst, &t).is_err());
        assert!(Tables::parse(&[0; 16]).is_err());
        let mut no_tiling = tables();
        no_tiling[0] = 1;
        assert!(Tables::parse(&no_tiling).is_err());
    }

    #[test]
    fn noise_never_panics() {
        let t = Tables::parse(&tables()).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..300 {
            let len = round * 7 % 900;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            for bc in [1, 3, 4, 5, 6, 7] {
                let mut dst = vec![0u8; 16 * (1 + round % 40)];
                let _ = decode(bc, &src, &mut dst, &t);
            }
        }
    }

    /// The window table finds the bucket a scan of the starts finds, for
    /// every slot of every model, in the shipped tables and in ours (whose
    /// buckets are wide) and in tables whose buckets are one slot wide.
    #[test]
    fn bucket_lookup_matches_a_scan_of_the_starts() {
        let mut narrow = tables();
        for m in 0..MODELS {
            // Buckets 1.. of one slot each where they fit; bucket 0 takes the rest.
            let used = m / 16 + 2;
            let rest = (1u32 << 16) - (1..used).map(|b| 1u32 << (b - 1)).sum::<u32>();
            for b in 0..used {
                let f = if b == 0 { rest } else { 1 };
                narrow[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&(f as u16).to_le_bytes());
            }
        }
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
            Tables::parse(&narrow).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                let start = &t.models[usize::from(model)].start;
                for slot in 0..1u32 << 16 {
                    let scan = start[1..].iter().filter(|&&s| s <= slot).count();
                    assert_eq!(
                        bucket_of(&t, model, slot),
                        scan,
                        "model {model} slot {slot}"
                    );
                }
            }
        }
    }

    /// One rANS decode as first written: scan for the bucket, divide for the
    /// symbol, rebuild its frequency and start from the symbol.
    fn reference_decode(t: &Tables, model: u8, x: u32) -> (u8, u32) {
        let m = &t.models[usize::from(model)];
        let start = &m.start;
        let slot = x & 0xffff;
        let b = start[1..BUCKETS].iter().filter(|&&s| s <= slot).count();
        let sym = match b {
            0 => 0,
            1 => 1,
            _ => (1u32 << (b - 1)) + (slot - start[b]) / u32::from(m.info[b].freq),
        } as u8;
        let (f, c) = if sym == 0 {
            // `m.info[0].freq` already stores `raw_freq[0] - 1`.
            (u32::from(m.info[0].freq), 1)
        } else {
            let k = (8 - sym.leading_zeros()) as usize;
            let f = u32::from(m.info[k].freq);
            (
                f,
                ((u32::from(sym) - (1 << (k - 1))) * f + start[k]) & 0xffff,
            )
        };
        (
            sym,
            (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot),
        )
    }

    /// A payload step decodes the symbol and state the first version did, for
    /// every model and slot of the shipped tables and ours, at assorted states.
    #[test]
    fn payload_step_matches_the_reference_decode() {
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                for high in [0x100u32, 0x1234, 0xffff] {
                    for slot in 0..1u32 << 16 {
                        let x = high << 16 | slot;
                        if x <= LOW {
                            continue;
                        }
                        let mut lane = Lane {
                            phase: 3,
                            model,
                            x,
                            left: 1,
                            ..Default::default()
                        };
                        lane.step(&t);
                        let (sym, next) = reference_decode(&t, model, x);
                        assert_eq!(
                            (lane.out, lane.x),
                            (i16::from(sym), next),
                            "model {model} x {x:#x}"
                        );
                    }
                }
            }
        }
    }

    /// Pseudo-random records that fit each lane's alphabet; BC7 keeps control
    /// code 0 (mode 6), the only one the synthetic tables define.
    fn random_records(widths: &[u8], per: usize, blocks: usize, bc7: bool) -> Vec<Vec<u8>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..blocks)
            .map(|_| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let alphabet = 1u32 << (widths[k / per] + 1);
                        (seed as u32 % alphabet) as u8
                    })
                    .collect();
                if bc7 {
                    r[30] = 0;
                    r[31] = 0;
                }
                r
            })
            .collect()
    }

    /// The fast paths decode exactly what one lane step at a time does, on
    /// every format's lane layout, with short and long fragments.
    #[test]
    fn every_layout_round_trips_through_the_lanes() {
        let t = Tables::parse(&tables()).unwrap();
        let layouts: [(&[u8], usize); 6] = [
            (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
            (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
            (&BC4_LANES, 1),
            (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
            (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
            (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
        ];
        for (widths, per) in layouts {
            for blocks in [1, 2, 17, 300, 1000] {
                let records = random_records(widths, per, blocks, false);
                let src = stream(&t, widths, per, &records);
                let mut got = vec![0u8; records.concat().len()];
                let mut next = src.iter().copied();
                lanes(&t, widths, per, blocks, |_| next.next(), &mut got).unwrap();
                assert_eq!(got, records.concat(), "{widths:?} x{per}, {blocks} blocks");
            }
        }
    }

    /// Decode throughput on synthetic streams (no game data):
    /// `cargo test --release -p victory-formats bcpack_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn bcpack_throughput() {
        let t = Tables::parse(&tables()).unwrap();
        let cases: [(u8, &[u8], usize, usize); 3] = [
            (1, &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1, 8),
            (4, &BC4_LANES, 1, 8),
            (7, &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2, 16),
        ];
        for (bc, widths, per, block) in cases {
            // One fragment per lane holds at most 65792 bytes.
            let blocks = 1 << 14;
            let records = random_records(widths, per, blocks, bc == 7);
            let src = stream(&t, widths, per, &records);
            let mut dst = vec![0u8; blocks * block];
            decode(bc, &src, &mut dst, &t).unwrap();
            let runs = 40;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                decode(bc, std::hint::black_box(&src), &mut dst, &t).unwrap();
            }
            let secs = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "BC{bc}: {blocks} blocks, {:.1} MB out/s, {:.1} ns/block",
                (blocks * block) as f64 / secs / 1e6,
                secs * 1e9 / blocks as f64
            );
        }
    }

    /// Compares with reference outputs made outside the repository:
    /// `VICTORY_BCPACK_TABLES` (the table file) and `VICTORY_BCPACK_CASES`, a
    /// directory of `<name>.bc<n>` packed chunks with `<name>.out` beside them.
    #[test]
    #[ignore = "needs the format tables and reference chunks"]
    fn reference_chunks() {
        let tables = std::env::var_os("VICTORY_BCPACK_TABLES").expect("VICTORY_BCPACK_TABLES");
        let cases = std::env::var_os("VICTORY_BCPACK_CASES").expect("VICTORY_BCPACK_CASES");
        let t = Tables::parse(&std::fs::read(tables).unwrap()).unwrap();
        let mut n = 0;
        for entry in std::fs::read_dir(cases).unwrap() {
            let path = entry.unwrap().path();
            let Some(bc) = path
                .extension()
                .and_then(|e| e.to_str()?.strip_prefix("bc")?.parse().ok())
            else {
                continue;
            };
            let want = std::fs::read(path.with_extension("out")).unwrap();
            let mut got = vec![0u8; want.len()];
            decode(bc, &std::fs::read(&path).unwrap(), &mut got, &t).unwrap();
            assert!(got == want, "{}", path.display());
            n += 1;
        }
        assert!(n > 0);
    }

    /// A dictionary entry stands in for a block the decoder made: a set with
    /// 0 index bits holds no indices, and pixel 0 is the anchor (top bit clear).
    #[test]
    fn bc7_dictionary_indices_line_up() {
        let t = Tables::builtin().unwrap();
        for (k, (_, ix)) in t.bc7.iter().enumerate() {
            let anchor = 0..1 << (ix.bits[0] - 1);
            let ok = (ix.bits[1] != 0 || ix.ix[1] == [0; 16]) && anchor.contains(&ix.ix[0][0]);
            assert!(ok, "entry {k}: {:?}", ix.ix);
        }
    }

    #[test]
    fn builtin_tables_constant_parses() {
        assert_eq!(BUILTIN_TABLES.len(), TABLES_SIZE);
        let a = Tables::parse(BUILTIN_TABLES).unwrap();
        let b = Tables::builtin().unwrap();
        assert_eq!(a.ctrl7, b.ctrl7);
        assert_eq!(a.rank1, b.rank1);
        assert_eq!(a.bc7.len(), DICTIONARY);
    }

    #[test]
    fn const_luts_match_bit_loops() {
        for a in 0..16u8 {
            for b in 0..16u8 {
                let spread = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> k & 1) << (2 * k));
                assert_eq!(interleave(a, b), spread(a) | (spread(b) << 1));
            }
        }
        for a in [0u8, 1, 0x55, 0xaa, 0x0f, 0xf0, 0xff, 0x37] {
            for b in [0u8, 1, 0x55, 0xaa, 0x0f, 0xf0, 0xff, 0xc9] {
                let want = (0..8).fold(0u16, |acc, k| {
                    acc | (u16::from(a >> k & 1) << (2 * k))
                        | (u16::from(b >> k & 1) << (2 * k + 1))
                });
                assert_eq!(wide(a, b), want);
            }
        }
        for b in 0..=255u8 {
            let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
            let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
            let (even, odd) = (z(half(b)), z(half(b >> 1)));
            assert_eq!(nibbles(b, false), [even, odd]);
            assert_eq!(nibbles(b, true), [odd, even]);
        }
    }

    /// Encodes a lane as multiple fragments (alternating reset/no-reset and
    /// short/long payloads) to exercise phase transitions in `lanes_slice`.
    fn multi_fragment(width: u8, symbols: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut offset = 0;
        let mut chunk_idx = 0usize;
        while offset < symbols.len() {
            let take = match chunk_idx % 3 {
                0 => 13,
                1 => 300,
                _ => 47,
            }
            .min(symbols.len() - offset);
            let mut frag = fragment(width, &symbols[offset..offset + take]);
            // Reset state to 0 between fragments so the next fragment can load its initial x.
            if offset + take < symbols.len() {
                frag[0] |= 0x10;
            }
            out.extend_from_slice(&frag);
            offset += take;
            chunk_idx += 1;
        }
        out
    }

    fn stream_multi(t: &Tables, widths: &[u8], per: usize, records: &[Vec<u8>]) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = multi_fragment(widths[j], &syms);
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    #[test]
    fn lanes_slice_matches_reference_lanes_all_formats() {
        let t = Tables::parse(&tables()).unwrap();
        for blocks in [1, 2, 17, 300, 600] {
            // 12x1 (BC1, BC4)
            for widths in [&BC1_WIDTHS, &BC4_WIDTHS] {
                let recs = random_records(widths, 1, blocks, false);
                let src = stream_multi(&t, widths, 1, &recs);
                let mut want = vec![0u8; blocks * 12];
                let mut got = vec![0u8; blocks * 12];
                let mut next = src.iter().copied();
                lanes(&t, widths, 1, blocks, |_| next.next(), &mut want).unwrap();
                lanes_slice::<12, 1>(&t, widths, blocks, &src, &mut got).unwrap();
                assert_eq!(got, want, "12x1 blocks={blocks}");
            }
            // 12x2 (BC3, BC5)
            for widths in [&BC3_WIDTHS, &BC5_WIDTHS] {
                let recs = random_records(widths, 2, blocks, false);
                let src = stream_multi(&t, widths, 2, &recs);
                let mut want = vec![0u8; blocks * 24];
                let mut got = vec![0u8; blocks * 24];
                let mut next = src.iter().copied();
                lanes(&t, widths, 2, blocks, |_| next.next(), &mut want).unwrap();
                lanes_slice::<12, 2>(&t, widths, blocks, &src, &mut got).unwrap();
                assert_eq!(got, want, "12x2 blocks={blocks}");
            }
            // 16x2 (BC6, BC7)
            for widths in [&BC6_WIDTHS, &BC7_WIDTHS] {
                let recs = random_records(widths, 2, blocks, false);
                let src = stream_multi(&t, widths, 2, &recs);
                let mut want = vec![0u8; blocks * 32];
                let mut got = vec![0u8; blocks * 32];
                let mut next = src.iter().copied();
                lanes(&t, widths, 2, blocks, |_| next.next(), &mut want).unwrap();
                lanes_slice::<16, 2>(&t, widths, blocks, &src, &mut got).unwrap();
                assert_eq!(got, want, "16x2 blocks={blocks}");
            }
        }
    }

    #[test]
    fn all_six_bc_formats_history_wrap_and_dictionary_refs() {
        // Use real builtin tables' dictionaries and ctrl7/ranks, paired with
        // uniform rANS models so our synthetic `stream()` encoder can pack them.
        let mut raw = BUILTIN_TABLES.to_vec();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                raw[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&f.to_le_bytes());
            }
        }
        let t = Tables::parse(&raw).unwrap();

        let mut scratch = Vec::new();

        // Helper: test decode + decode_with_scratch match and produce non-trivial output
        // over >256 blocks (history wrap-around) with dictionary refs (block 0 referencing r=1..5).
        let split_byte = |v: u8| -> (u8, u8) {
            let mut even = 0u8;
            let mut odd = 0u8;
            for k in 0..4 {
                even |= ((v >> (2 * k)) & 1) << k;
                odd |= ((v >> (2 * k + 1)) & 1) << k;
            }
            (even, odd)
        };

        // 1. BC1: 300 blocks with dictionary ref on block 0 and history refs across >256.
        {
            let mut recs = random_records(&BC1_WIDTHS, 1, 300, false);
            // Block 0: ix_ref = 1 (dictionary entry 0), ep_ref = 2 (dictionary entry 1)
            recs[0][0] = zigzag(1);
            recs[0][1] = 1;
            // Block 260: references block 260 - 255 = 5 (wraps history ring)
            recs[260][0] = 0;
            recs[260][1] = 255;
            let src = stream(&t, &BC1_WIDTHS, 1, &recs);
            let mut dst1 = vec![0u8; 300 * 8];
            let mut dst2 = vec![0u8; 300 * 8];
            decode(1, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(1, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
            // Verify block 0 matches dictionary-seeded expectation.
            let mut ep = [0u8; 6];
            for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
                let mask = ((1u16 << bits) - 1) as u8;
                ep[k] = zz(recs[0][2 + k], bits).wrapping_add(t.bc1[1].ep[k]) & mask;
            }
            let mut ix = [0u8; 16];
            quads(&t.rank1, &recs[0][8..12], &mut ix);
            for k in 0..16 {
                ix[k] = ix[k].wrapping_add(t.bc1[0].ix[k]) & 3;
            }
            let mut want0 = [0u8; 8];
            bc1_color(&ep, &ix, &mut want0);
            assert_eq!(&dst1[..8], &want0);
        }

        // 2. BC3: 300 blocks with paired dictionary ref and history wrap-around.
        {
            let mut recs = random_records(&BC3_WIDTHS, 2, 300, false);
            let (d0, d1) = split_byte(zigzag(1));
            let (i0, i1) = split_byte(1);
            recs[0][0] = d0;
            recs[0][1] = d1;
            recs[0][2] = i0;
            recs[0][3] = i1;
            let src = stream(&t, &BC3_WIDTHS, 2, &recs);
            let mut dst1 = vec![0u8; 300 * 16];
            let mut dst2 = vec![0u8; 300 * 16];
            decode(3, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(3, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
        }

        // 3. BC4: 300 blocks with dictionary ref and history wrap-around.
        {
            let mut recs = random_records(&BC4_WIDTHS, 1, 300, false);
            recs[0][0] = 0;
            recs[0][1] = 3; // dictionary entry 2
            let src = stream(&t, &BC4_WIDTHS, 1, &recs);
            let mut dst1 = vec![0u8; 300 * 8];
            let mut dst2 = vec![0u8; 300 * 8];
            decode(4, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(4, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
        }

        // 4. BC5: 300 blocks with paired dictionary ref and history wrap-around.
        {
            let mut recs = random_records(&BC5_WIDTHS, 2, 300, false);
            let (d0, d1) = split_byte(0);
            let (i0, i1) = split_byte(2);
            recs[0][0] = d0;
            recs[0][1] = d1;
            recs[0][2] = i0;
            recs[0][3] = i1;
            let src = stream(&t, &BC5_WIDTHS, 2, &recs);
            let mut dst1 = vec![0u8; 300 * 16];
            let mut dst2 = vec![0u8; 300 * 16];
            decode(5, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(5, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
        }

        // 5. BC6H: all 14 modes (1..=14), dictionary refs, and >256 blocks history wrap-around.
        {
            let mut recs = random_records(&BC6_WIDTHS, 2, 300, false);
            for (i, r) in recs.iter_mut().enumerate() {
                r[30] = (i % 14) as u8; // mode = r[30] + 1 -> 1..=14
                r[31] = (i % 32) as u8; // partition 0..31
            }
            // Dictionary ref on block 0
            let (d0, d1) = split_byte(0);
            let (i0, i1) = split_byte(1);
            recs[0][0] = d0;
            recs[0][1] = d1;
            recs[0][2] = i0;
            recs[0][3] = i1;
            let src = stream(&t, &BC6_WIDTHS, 2, &recs);
            let mut dst1 = vec![0u8; 300 * 16];
            let mut dst2 = vec![0u8; 300 * 16];
            decode(6, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(6, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
            // Verify mode code bits in the output match BC6_CODE for every mode 1..=14.
            for mode in 1..=14usize {
                let block = u128::from_le_bytes(dst1[(mode - 1) * 16..mode * 16].try_into().unwrap());
                let (code, width) = BC6_CODE[mode];
                let mask = (1u128 << width) - 1;
                assert_eq!(block & mask, u128::from(code), "BC6H mode {mode}");
            }
            // Invalid BC6H mode (s[30] = 14 -> mode 15) returns Err.
            let mut bad = recs[..1].to_vec();
            bad[0][30] = 14;
            let bad_src = stream(&t, &BC6_WIDTHS, 2, &bad);
            assert!(decode(6, &bad_src, &mut [0u8; 16], &t).is_err());
        }

        // 6. BC7: all 8 modes (0..=7) with rotations, partitions, dictionary refs, >256 blocks.
        {
            // Collect valid control code indices for every BC7 mode 0..=7 from builtin tables.
            let mut mode_codes: [Vec<usize>; 8] = Default::default();
            for (idx, &ctrl) in t.ctrl7.iter().enumerate() {
                if ctrl != 0 {
                    mode_codes[usize::from(ctrl & 7)].push(idx);
                }
            }
            for mode in 0..8 {
                assert!(!mode_codes[mode].is_empty(), "builtin tables missing BC7 mode {mode}");
            }
            let valid_codes: Vec<usize> = t
                .ctrl7
                .iter()
                .enumerate()
                .filter_map(|(idx, &c)| (c != 0).then_some(idx))
                .collect();
            let mut recs = random_records(&BC7_WIDTHS, 2, 300, false);
            for (i, r) in recs.iter_mut().enumerate() {
                let code = if i < 8 {
                    mode_codes[i][0]
                } else {
                    valid_codes[i % valid_codes.len()]
                };
                r[30] = (code & 31) as u8;
                r[31] = (code >> 5) as u8;
            }
            // Dictionary ref on block 0
            let (d0, d1) = split_byte(0);
            let (i0, i1) = split_byte(1);
            recs[0][0] = d0;
            recs[0][1] = d1;
            recs[0][2] = i0;
            recs[0][3] = i1;
            let src = stream(&t, &BC7_WIDTHS, 2, &recs);
            let mut dst1 = vec![0u8; 300 * 16];
            let mut dst2 = vec![0u8; 300 * 16];
            decode(7, &src, &mut dst1, &t).unwrap();
            decode_with_scratch(7, &src, &mut dst2, &t, &mut scratch).unwrap();
            assert_eq!(dst1, dst2);
            // Verify mode bits in the packed output for modes 0..=7.
            for mode in 0..8usize {
                let block = u128::from_le_bytes(dst1[mode * 16..(mode + 1) * 16].try_into().unwrap());
                assert_eq!(block.trailing_zeros() as usize, mode, "BC7 mode {mode}");
            }
        }
    }
}
