//! Clean-up of a finished module before it is written: the lowering emits
//! one instruction after another as the console code names them, and
//! nothing stands between it and the driver's compiler. Every pass here is
//! exact: the module computes the same values and has the same effects for
//! every input.
//!
//! - [`DEAD`]: instructions without an effect whose result nothing reads
//!   are left out (the lowering computes every dword of every register a
//!   console instruction writes; a guarded write reads the register's old
//!   value whether anything reads the new one or not).
//! - [`FOLD`]: integer arithmetic on constants is done here, and identities
//!   (`x + 0`, `x & -1`, a select between a value and itself, a cast there
//!   and back, a join of one value) give their operand. Float arithmetic is
//!   left alone: `x + 0.0` and `x * 1.0` are not `x` for every `x`.
//! - [`REUSE`]: an instruction without an effect whose operands are those
//!   of an earlier one that every path to it has run gives that one's
//!   result (a descriptor handle made again in each block, an index clamped
//!   again, a register read as a float again).
//! - [`FACTS`]: in the blocks only one side of a branch leads to, the
//!   branch's condition is that side's constant, and so is what the
//!   condition was made of (both terms of an `and` that held, the opposite
//!   comparison). The lowering branches on a lane's condition and then
//!   guards each instruction behind the branch with the same condition.
//! - [`LOADS`]: within a block, a read of the same place as an earlier read
//!   gives that one's result while no store, atomic, barrier or other call
//!   with an effect stands between them (another thread's unsynchronised
//!   store may fall between any two reads or before both: the title's code
//!   cannot tell which).
//!
//! Which passes run is `emit::shader_opt()` (`VICTORY_SHADER_OPT`), part of
//! the translation and so of the shader cache's key.

use std::collections::HashMap;

use super::{
    Attr, Bin, Cast, Const, Func, MdDef, Module, Op, Term, Ty, TypeDef, V, for_each_operand, pred,
    successors,
};

/// Dead-code elimination.
pub const DEAD: u32 = 1;
/// Constants folded, identities removed.
pub const FOLD: u32 = 2;
/// One instruction for one value.
pub const REUSE: u32 = 4;
/// What a branch taken says about its condition.
pub const FACTS: u32 = 8;
/// One read for one place within a block, while nothing is stored.
pub const LOADS: u32 = 16;
/// Not a pass over the module: the lowering of a first-lane read in a
/// wave64 program run as two host waves skips the exchange that asks which
/// half has a lane where every lane is in EXEC (`lower.rs`,
/// `v_readfirstlane_b32`).
pub const FIRST: u32 = 32;
/// Not a pass over the module either: the lowering reads the dwords of one
/// console load (`s_buffer_load_dwordx2` to `x16` from an SRV, a
/// `global_load_dwordx2` to `x4`) by one call for every four of them, not
/// by a call each (`lower.rs`, `raw_dwords`). The same bytes are read.
/// From fxceless/victory-fork c36971b2. On an RX 7900 XTX it is worth nothing:
/// the yard's frame replayed turn by turn, dispatches and draws 18.47 -> 18.45
/// ms (12 loops) and 18.23 -> 18.16 (5 loops): the driver joins the loads.
pub const WIDE: u32 = 64;
/// The passes over the module.
pub const MODULE: u32 = DEAD | FOLD | REUSE | FACTS | LOADS;
/// Every pass.
pub const ALL: u32 = MODULE | FIRST | WIDE;
/// The passes that run when nothing says which: every one but [`WIDE`],
/// which no card has shown a gain for yet (`VICTORY_SHADER_OPT=127` asks
/// for it).
pub const DEFAULT: u32 = ALL & !WIDE;

/// What the instructions a module writes use, of the features a shader
/// declares by flag (the validator's "flags must match usage").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    /// A wave or quad operation.
    pub wave_ops: bool,
    /// A rasterizer ordered view.
    pub rovs: bool,
    /// A 64-bit integer, as a result or an operand.
    pub int64: bool,
    /// A typed load of several components from a UAV.
    pub typed_uav_loads: bool,
    /// A handle from the view heap.
    pub heap_views: bool,
    /// A handle from the sampler heap.
    pub heap_samplers: bool,
    /// A 64-bit atomic.
    pub atomic64: bool,
    /// Inline ray tracing (`RayQuery` or `RaytracingAccelerationStructure`).
    pub raytracing_tier_1_1: bool,
}

/// What makes two instructions without an effect the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Bin(u8, V, V),
    Cast(u8, Ty, V),
    Cmp(u8, V, V),
    Select(V, V, V),
    Extract(V, u32),
    Call(u32, Vec<V>),
    Gep(Ty, V, Vec<V>),
    Load(V),
}

/// The blocks reached from the entry, each after the blocks that dominate
/// it, and each block's children in the dominator tree.
struct Tree {
    order: Vec<usize>,
    children: Vec<Vec<usize>>,
    /// Per block: the reached blocks that branch to it, once per edge.
    preds: Vec<Vec<usize>>,
}

/// The comparison that holds exactly when `p` does not (floats 0-15,
/// integers 32-41).
fn opposite(p: u8) -> u8 {
    match p {
        0..=15 => 15 - p,
        pred::EQ => pred::NE,
        pred::NE => pred::EQ,
        pred::IUGT => pred::IULE,
        pred::IULE => pred::IUGT,
        pred::IUGE => pred::IULT,
        pred::IULT => pred::IUGE,
        pred::SGT => pred::SLE,
        pred::SLE => pred::SGT,
        pred::SGE => pred::SLT,
        _ => pred::SGE,
    }
}

/// The comparison with its operands swapped (`cmp(p, a, b) == cmp(swap_pred(p), b, a)`).
fn swap_pred(p: u8) -> u8 {
    match p {
        // Float predicates: bit 1 is GT, bit 2 is LT; swap them.
        0..=15 => (p & !6) | ((p & 2) << 1) | ((p & 4) >> 1),
        pred::EQ => pred::EQ,
        pred::NE => pred::NE,
        pred::IUGT => pred::IULT,
        pred::IULT => pred::IUGT,
        pred::IUGE => pred::IULE,
        pred::IULE => pred::IUGE,
        pred::SGT => pred::SLT,
        pred::SLT => pred::SGT,
        pred::SGE => pred::SLE,
        _ => pred::SGE,
    }
}

/// What holds in the block being walked: values that are a known bit there,
/// integer values known to equal a constant, and comparisons (by their
/// operands) with a known outcome.
#[derive(Default)]
struct Facts {
    bits: HashMap<V, bool>,
    compares: HashMap<(u8, V, V), bool>,
    values: HashMap<V, V>,
}

/// The facts a block added, to forget when its part of the tree is done.
#[derive(Default)]
struct Learned {
    bits: Vec<V>,
    compares: Vec<(u8, V, V)>,
    values: Vec<V>,
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

/// `x` of `bits` bits as a signed number.
fn signed(bits: u32, x: u64) -> i64 {
    if bits < 64 && x >> (bits - 1) & 1 != 0 {
        (x | !mask(bits)) as i64
    } else {
        x as i64
    }
}

/// What `v` stands for after the replacements so far.
fn resolve(repl: &[Option<V>], mut v: V) -> V {
    while let V::Inst(i) = v {
        match repl[i as usize] {
            Some(r) => v = r,
            None => break,
        }
    }
    v
}

/// Replaces every value `op` reads by `f` of it.
fn map_operands(op: &mut Op, mut f: impl FnMut(V) -> V) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => *v = f(*v),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            *a = f(*a);
            *b = f(*b);
            *c = f(*c);
        }
        Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
        Op::Phi(inc) => inc.iter_mut().for_each(|x| x.0 = f(x.0)),
        Op::Gep(_, p, idx) => {
            *p = f(*p);
            idx.iter_mut().for_each(|i| *i = f(*i));
        }
    }
}

/// What a call does besides returning its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Nothing, and the result depends on the arguments alone for the whole
    /// run of the shader (arithmetic, handles, inputs, constant buffers,
    /// resource sizes).
    Pure,
    /// Nothing, but the result depends on memory a store may change
    /// (buffer and texture reads).
    Reads,
    /// Nothing, but the result depends on which lanes run the instruction
    /// (wave and quad operations).
    Lanes,
    /// Stores, atomics, barriers, outputs, discards: always kept.
    Writes,
}

pub(super) fn effect(f: &Func) -> Effect {
    let name = f.name.strip_prefix("dx.op.").unwrap_or(&f.name);
    match f.attr {
        Some(Attr::ReadNone) => Effect::Pure,
        // A shader cannot write a constant buffer, resize a resource, or change
        // a thread's lane index or wave size.
        Some(Attr::ReadOnly)
            if name.starts_with("cbufferLoadLegacy")
                || name.starts_with("getDimensions")
                || name == "waveGetLaneIndex"
                || name == "waveGetLaneCount" =>
        {
            Effect::Pure
        }
        Some(Attr::ReadOnly) if name.starts_with("wave") => Effect::Lanes,
        Some(Attr::ReadOnly) => Effect::Reads,
        // Declared with the attribute of calls that have effects, and have
        // none: they read the other lanes' values.
        _ if name.starts_with("wave") || name.starts_with("quad") => Effect::Lanes,
        _ => Effect::Writes,
    }
}

impl Module {
    /// Runs the clean-up passes named in `passes` ([`ALL`]: every one).
    pub(crate) fn optimise(&mut self, passes: u32) {
        if passes & (FOLD | REUSE | FACTS | LOADS) != 0 {
            // Values merged in one round can make two joins the same in
            // the next (a loop's registers that only go round), and a
            // branch that goes one way leaves blocks with one way in.
            for _ in 0..8 {
                let mut changed = self.merge(passes);
                if passes & FOLD != 0 {
                    changed |= self.prune();
                }
                if !changed {
                    break;
                }
            }
        }
        if passes & DEAD != 0 {
            self.drop_dead();
        }
    }

    /// Per function: what a call of it does.
    pub(super) fn effects(&self) -> Vec<Effect> {
        self.funcs.iter().map(effect).collect()
    }

    /// The blocks reached from the entry.
    pub(super) fn reached(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        if self.blocks.is_empty() {
            return seen;
        }
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(b) = stack.pop() {
            for &s in successors(self.blocks[b].term).iter() {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
        }
        seen
    }

    fn dominator_tree(&self) -> Tree {
        let n = self.blocks.len();
        let succs = |b: usize| successors(self.blocks[b].term);
        // Reverse postorder, as `write` lays the blocks out.
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match succs(b).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let order = post;
        let mut number = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            number[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in succs(b).iter() {
                preds[s].push(b);
            }
        }
        // Immediate dominators (Cooper, Harvey and Kennedy's iteration).
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x] > number[y] {
                            x = idom[x];
                        }
                        while number[y] > number[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); n];
        for &b in &order[1..] {
            children[idom[b]].push(b);
        }
        Tree {
            order,
            children,
            preds,
        }
    }

    /// Notes that `v` (an i1) is `holds` in the block being walked, and what
    /// follows from it for the values `v` was made of.
    fn learn(&self, v: V, holds: bool, facts: &mut Facts, learned: &mut Learned, depth: u32) {
        let V::Inst(i) = v else { return };
        if facts.bits.contains_key(&v) {
            return;
        }
        facts.bits.insert(v, holds);
        learned.bits.push(v);
        if depth == 0 {
            return;
        }
        match self.insts[i as usize].op {
            // Both terms of an `and` that holds hold; neither of an `or`
            // that does not.
            Op::Bin(Bin::And, a, b) if holds => {
                self.learn(a, true, facts, learned, depth - 1);
                self.learn(b, true, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Or, a, b) if !holds => {
                self.learn(a, false, facts, learned, depth - 1);
                self.learn(b, false, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Xor, a, b) => {
                let (x, k) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((1, k))) => (a, k),
                    (Some((1, k)), None) => (b, k),
                    _ => return,
                };
                self.learn(x, holds != (k != 0), facts, learned, depth - 1);
            }
            // The opposite and operand-swapped comparisons of the same operands,
            // plus integer equality to a constant.
            Op::Cmp(p, a, b) => {
                let mut note_cmp = |key: (u8, V, V), val: bool| {
                    if let std::collections::hash_map::Entry::Vacant(e) = facts.compares.entry(key)
                    {
                        e.insert(val);
                        learned.compares.push(key);
                    }
                };
                note_cmp((opposite(p), a, b), !holds);
                let sp = swap_pred(p);
                note_cmp((sp, b, a), holds);
                note_cmp((opposite(sp), b, a), !holds);
                if p == pred::EQ || p == pred::NE {
                    let eq_holds = (p == pred::EQ) == holds;
                    let pair = match (a, b, self.int_k(a), self.int_k(b)) {
                        (V::Inst(_), _, None, Some((bits, k))) => Some((a, b, bits, k)),
                        (_, V::Inst(_), Some((bits, k)), None) => Some((b, a, bits, k)),
                        _ => None,
                    };
                    if let Some((var, kv, bits, k)) = pair {
                        if bits == 1 {
                            self.learn(var, (k != 0) == eq_holds, facts, learned, depth - 1);
                        } else if eq_holds
                            && let std::collections::hash_map::Entry::Vacant(e) =
                                facts.values.entry(var)
                        {
                            e.insert(kv);
                            learned.values.push(var);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn int_bits(&self, ty: Ty) -> Option<u32> {
        match self.types[ty.0 as usize] {
            TypeDef::Int(bits) => Some(bits),
            _ => None,
        }
    }

    /// An integer constant's width and value (the bits of that width).
    pub(crate) fn int_k(&self, v: V) -> Option<(u32, u64)> {
        let V::Const(i) = v else { return None };
        let (ty, Const::Int(x)) = &self.consts[i as usize] else {
            return None;
        };
        let bits = self.int_bits(*ty)?;
        Some((bits, *x as u64 & mask(bits)))
    }

    /// The bits of a 32-bit float that is a constant, given as one or as a
    /// cast of an integer constant.
    pub(crate) fn float_k(&self, v: V) -> Option<f32> {
        let (v, cast) = match v {
            V::Inst(i) => match self.insts[i as usize].op {
                Op::Cast(Cast::Bitcast, x) => (x, true),
                _ => return None,
            },
            _ => (v, false),
        };
        let V::Const(i) = v else { return None };
        match &self.consts[i as usize].1 {
            Const::Float(bits) if !cast => Some(f32::from_bits(*bits)),
            Const::Int(_) if cast => {
                let (bits, x) = self.int_k(v)?;
                (bits == 32).then_some(f32::from_bits(x as u32))
            }
            _ => None,
        }
    }

    /// For a float that is a number and not an infinity whatever the program
    /// reads: a power of two its magnitude does not pass.
    pub(crate) fn bounded(&self, v: V, depth: u32) -> Option<f32> {
        if let Some(k) = self.float_k(v) {
            return k.is_finite().then(|| k.abs().max(1e-30).log2().ceil());
        }
        let V::Inst(i) = v else { return None };
        let depth = depth.checked_sub(1)?;
        let of = |v: V| self.bounded(v, depth);
        // (A power that overflows, or might with rounding, is no bound.)
        let held = |b: f32| (b < 126.0).then_some(b);
        match &self.insts[i as usize].op {
            Op::Cast(Cast::UiToFp | Cast::SiToFp, _) => Some(32.0),
            Op::Cast(Cast::Bitcast, x) => match *x {
                V::Inst(j) => match self.insts[j as usize].op {
                    Op::Cast(Cast::Bitcast, y) if self.type_of(y) == self.type_of(v) => of(y),
                    _ => None,
                },
                _ => None,
            },
            Op::Bin(Bin::Mul, a, b) => held(of(*a)? + of(*b)?),
            Op::Bin(Bin::Add | Bin::Sub, a, b) => held(of(*a)?.max(of(*b)?) + 1.0),
            Op::Select(_, a, b) => Some(of(*a)?.max(of(*b)?)),
            Op::Call(_, args) => {
                let (_, code) = self.int_k(*args.first()?)?;
                match (code, &args[1..]) {
                    // Saturated, a fraction, a cosine or sine (of a number).
                    (7, [_]) => Some(0.0),
                    (6, [x]) => of(*x),
                    (22 | 12 | 13, [x]) => of(*x).map(|_| 0.0),
                    (26..=29, [x]) => held(of(*x)? + 1.0),
                    // Of two numbers; or one of them the other way round: the
                    // minimum (36) of a constant and a maximum of another.
                    (35 | 36, [a, b]) => match (of(*a), of(*b)) {
                        (Some(x), Some(y)) => Some(x.max(y)),
                        _ => {
                            let other = if code == 35 { 36 } else { 35 };
                            let (inner, k) = if self.float_k(*b).is_some() { (*a, *b) } else { (*b, *a) };
                            let k = self.float_k(k).and_then(|_| of(k))?;
                            let V::Inst(j) = inner else { return None };
                            let Op::Call(_, inner) = &self.insts[j as usize].op else { return None };
                            let (_, code) = self.int_k(*inner.first()?)?;
                            let low = inner[1..].iter().find(|x| self.float_k(**x).is_some())?;
                            (code == other).then_some(())?;
                            Some(k.max(of(*low)?))
                        }
                    },
                    (46, [a, b, c]) => held((of(*a)? + of(*b)?).max(of(*c)?) + 1.0),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The constant `x` of the integer type `ty`, kept sign-extended as
    /// [`Module::i32`] and [`Module::bool`] keep theirs.
    fn k(&mut self, ty: Ty, bits: u32, x: u64) -> V {
        self.int_const(ty, signed(bits, x & mask(bits)))
    }

    /// Bits (within `mask(bits)`) provably zero in `v` for every input.
    fn known_zero_bits(&self, v: V, bits: u32, depth: u32) -> u64 {
        let all = mask(bits);
        if let Some((_, k)) = self.int_k(v) {
            return !k & all;
        }
        let V::Inst(i) = v else { return 0 };
        let Some(depth) = depth.checked_sub(1) else {
            return 0;
        };
        match &self.insts[i as usize].op {
            Op::Bin(Bin::And, a, b) => {
                self.known_zero_bits(*a, bits, depth) | self.known_zero_bits(*b, bits, depth)
            }
            Op::Bin(Bin::Or | Bin::Xor, a, b) => {
                self.known_zero_bits(*a, bits, depth) & self.known_zero_bits(*b, bits, depth)
            }
            Op::Bin(Bin::Shl, a, b) => match self.int_k(*b) {
                Some((_, s)) if s < u64::from(bits) => {
                    let s = s as u32;
                    ((self.known_zero_bits(*a, bits, depth) << s) | mask(s)) & all
                }
                _ => 0,
            },
            Op::Bin(Bin::LShr, a, b) => match self.int_k(*b) {
                Some((_, s)) if s < u64::from(bits) => {
                    let s = s as u32;
                    (self.known_zero_bits(*a, bits, depth) >> s) | (all & !mask(bits - s))
                }
                _ => 0,
            },
            Op::Cast(Cast::ZExt, x) => match self.int_bits(self.type_of(*x)) {
                Some(fb) if fb < bits => {
                    self.known_zero_bits(*x, fb, depth) | (all & !mask(fb))
                }
                _ => 0,
            },
            Op::Select(_, t, f) => {
                self.known_zero_bits(*t, bits, depth) & self.known_zero_bits(*f, bits, depth)
            }
            Op::Call(f, args) => {
                let name = self.funcs[*f as usize].name.as_str();
                let Some((_, code)) = args.first().and_then(|&a| self.int_k(a)) else {
                    return 0;
                };
                if name == "dx.op.unaryBits.i32" && code == 31 && bits == 32 {
                    // Countbits of a 32-bit word is in 0..=32.
                    all & !0x3F
                } else if name.starts_with("dx.op.binary.i") && code == 40 && args.len() == 3 {
                    // UMin(a, b) <= min(a, b): bits above any constant bound are 0.
                    let both = self.known_zero_bits(args[1], bits, depth)
                        & self.known_zero_bits(args[2], bits, depth);
                    let k = self
                        .int_k(args[1])
                        .or_else(|| self.int_k(args[2]))
                        .map(|x| x.1);
                    match k {
                        Some(0) => all,
                        Some(k) => (all & !(u64::MAX >> k.leading_zeros())) | both,
                        None => both,
                    }
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// If `v` is an `i1` bitwise negation `x ^ true`, returns `x`.
    fn bit_not(&self, v: V) -> Option<V> {
        let V::Inst(i) = v else { return None };
        let Op::Bin(Bin::Xor, a, b) = self.insts[i as usize].op else {
            return None;
        };
        match (self.int_k(a), self.int_k(b)) {
            (None, Some((1, 1))) => Some(a),
            (Some((1, 1)), None) => Some(b),
            _ => None,
        }
    }

    /// Unwraps `v` to type `ty` if it is a `Bitcast` from `ty` or a 32-bit
    /// constant convertible to `ty` without inserting instructions.
    fn unwrap_bitcast(&mut self, v: V, ty: Ty) -> Option<V> {
        if let V::Inst(j) = v {
            if let Op::Cast(Cast::Bitcast, z) = self.insts[j as usize].op
                && self.type_of(z) == ty
            {
                return Some(z);
            }
            return None;
        }
        let V::Const(c) = v else { return None };
        let from = self.type_of(v);
        let to_float = matches!(self.types[ty.0 as usize], TypeDef::Float);
        match self.consts[c as usize].1 {
            Const::Int(bits) if to_float && self.int_bits(from) == Some(32) => {
                Some(self.f32(bits as u32))
            }
            Const::Float(bits) if self.int_bits(ty) == Some(32) => {
                Some(self.int_const(ty, i64::from(bits as i32)))
            }
            _ => None,
        }
    }

    /// If `v` is an integer `Op::Bin(op, x, k)` (commuted if commutative),
    /// returns `(op, x, k, k_on_lhs)`.
    fn inst_bin_k(&self, v: V) -> Option<(Bin, V, u64, bool)> {
        let V::Inst(j) = v else { return None };
        let Op::Bin(op, a, b) = self.insts[j as usize].op else {
            return None;
        };
        match (self.int_k(a), self.int_k(b)) {
            (None, Some((_, k))) => Some((op, a, k, false)),
            (Some((_, k)), None) => Some((op, b, k, true)),
            _ => None,
        }
    }

    /// The value instruction `i` is known to have without it, if any: a
    /// constant or an earlier value. May instead give `i` simpler operands
    /// (`changed` says so).
    fn simplify(&mut self, i: u32, changed: &mut bool) -> Option<V> {
        let ty = self.insts[i as usize].ty;
        match self.insts[i as usize].op {
            Op::Bin(op, a, b) => {
                // Floats: `x + 0.0`, `x * 1.0` and the like are not `x` for
                // every `x` (a negative zero, a signalling NaN).
                let bits = self.int_bits(ty)?;
                let (ka, kb) = (self.int_k(a).map(|k| k.1), self.int_k(b).map(|k| k.1));
                let all = mask(bits);
                if let (Some(x), Some(y)) = (ka, kb) {
                    // A shift by the width or more and a division by zero
                    // have no defined result: left as they are.
                    let amount = u32::try_from(y).ok().filter(|&y| y < bits);
                    let r = match op {
                        Bin::Add => x.wrapping_add(y),
                        Bin::Sub => x.wrapping_sub(y),
                        Bin::Mul => x.wrapping_mul(y),
                        Bin::And => x & y,
                        Bin::Or => x | y,
                        Bin::Xor => x ^ y,
                        Bin::Shl => x << amount?,
                        Bin::LShr => x >> amount?,
                        Bin::AShr => (signed(bits, x) >> amount?) as u64,
                        Bin::UDiv if y != 0 => x / y,
                        Bin::SDiv if y != 0 => {
                            let (sx, sy) = (signed(bits, x), signed(bits, y));
                            let smin = if bits == 64 { i64::MIN } else { -(1i64 << (bits - 1)) };
                            if sx == smin && sy == -1 {
                                return None;
                            }
                            sx.wrapping_div(sy) as u64
                        }
                        Bin::URem if y != 0 => x % y,
                        _ => return None,
                    };
                    return Some(self.k(ty, bits, r));
                }
                match (op, ka, kb) {
                    (
                        Bin::Add | Bin::Sub | Bin::Or | Bin::Xor | Bin::Shl | Bin::LShr | Bin::AShr,
                        _,
                        Some(0),
                    ) => return Some(a),
                    (Bin::Add | Bin::Or | Bin::Xor, Some(0), _) => return Some(b),
                    (Bin::Mul | Bin::UDiv | Bin::SDiv, _, Some(1)) => return Some(a),
                    (Bin::Mul, Some(1), _) => return Some(b),
                    (Bin::Mul | Bin::And, _, Some(0)) => return Some(b),
                    (Bin::Mul | Bin::And, Some(0), _) => return Some(a),
                    (Bin::And, _, Some(m)) if m == all => return Some(a),
                    (Bin::And, Some(m), _) if m == all => return Some(b),
                    (Bin::Or, _, Some(m)) if m == all => return Some(b),
                    (Bin::Or, Some(m), _) if m == all => return Some(a),
                    (Bin::And | Bin::Or, _, _) if a == b => return Some(a),
                    (Bin::Xor | Bin::Sub, _, _) if a == b => return Some(self.k(ty, bits, 0)),
                    _ => {}
                }
                // Non-constant algebraic identities (absorption, complement,
                // self-cancellation).
                let inst_bin = |v: V| match v {
                    V::Inst(j) => match self.insts[j as usize].op {
                        Op::Bin(iop, x, y) => Some((iop, x, y)),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    Bin::And => {
                        for (other, cand) in [(a, b), (b, a)] {
                            if let Some((iop, x, y)) = inst_bin(cand) {
                                // x & (x | y) -> x; x & (x & y) -> cand; x & ~x -> 0.
                                if iop == Bin::Or && (x == other || y == other) {
                                    return Some(other);
                                }
                                if iop == Bin::And && (x == other || y == other) {
                                    return Some(cand);
                                }
                                if iop == Bin::Xor
                                    && ((x == other && self.int_k(y).is_some_and(|k| k.1 == all))
                                        || (y == other
                                            && self.int_k(x).is_some_and(|k| k.1 == all)))
                                {
                                    return Some(self.k(ty, bits, 0));
                                }
                            }
                        }
                    }
                    Bin::Or => {
                        for (other, cand) in [(a, b), (b, a)] {
                            if let Some((iop, x, y)) = inst_bin(cand) {
                                // x | (x & y) -> x; x | (x | y) -> cand; x | ~x -> all.
                                if iop == Bin::And && (x == other || y == other) {
                                    return Some(other);
                                }
                                if iop == Bin::Or && (x == other || y == other) {
                                    return Some(cand);
                                }
                                if iop == Bin::Xor
                                    && ((x == other && self.int_k(y).is_some_and(|k| k.1 == all))
                                        || (y == other
                                            && self.int_k(x).is_some_and(|k| k.1 == all)))
                                {
                                    return Some(self.k(ty, bits, all));
                                }
                            }
                        }
                    }
                    Bin::Xor => {
                        for (other, cand) in [(a, b), (b, a)] {
                            if let Some((Bin::Xor, x, y)) = inst_bin(cand) {
                                if x == other {
                                    return Some(y);
                                }
                                if y == other {
                                    return Some(x);
                                }
                            }
                        }
                    }
                    Bin::Add => {
                        // (x - y) + y -> x
                        for (other, cand) in [(a, b), (b, a)] {
                            if let Some((Bin::Sub, x, y)) = inst_bin(cand)
                                && y == other
                            {
                                return Some(x);
                            }
                        }
                    }
                    Bin::Sub => {
                        // (x + y) - y -> x; (x + y) - x -> y
                        if let Some((Bin::Add, x, y)) = inst_bin(a) {
                            if y == b {
                                return Some(x);
                            }
                            if x == b {
                                return Some(y);
                            }
                        }
                        // x - (x - y) -> y
                        if let Some((Bin::Sub, x, y)) = inst_bin(b)
                            && x == a
                        {
                            return Some(y);
                        }
                    }
                    _ => {}
                }
                // Reassociation with constant operands and bit-level folding.
                if let Some((var, k2, k2_lhs)) = match (ka, kb) {
                    (None, Some(k)) => Some((a, k, false)),
                    (Some(k), None) => Some((b, k, true)),
                    _ => None,
                } {
                    match op {
                        Bin::Add => {
                            if let Some((iop, x, k1, k1_lhs)) = self.inst_bin_k(var) {
                                match iop {
                                    Bin::Add => {
                                        let k = k1.wrapping_add(k2) & all;
                                        if k == 0 {
                                            return Some(x);
                                        }
                                        let kv = self.k(ty, bits, k);
                                        self.insts[i as usize].op = Op::Bin(Bin::Add, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    Bin::Sub if !k1_lhs => {
                                        // (x - k1) + k2 -> x + (k2 - k1)
                                        let k = k2.wrapping_sub(k1) & all;
                                        if k == 0 {
                                            return Some(x);
                                        }
                                        let kv = self.k(ty, bits, k);
                                        self.insts[i as usize].op = Op::Bin(Bin::Add, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    Bin::Sub => {
                                        // (k1 - x) + k2 -> (k1 + k2) - x
                                        let kv = self.k(ty, bits, k1.wrapping_add(k2));
                                        self.insts[i as usize].op = Op::Bin(Bin::Sub, kv, x);
                                        *changed = true;
                                        return None;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Bin::Sub => {
                            if let Some((iop, x, k1, k1_lhs)) = self.inst_bin_k(var) {
                                match (k2_lhs, iop, k1_lhs) {
                                    (false, Bin::Add, _) => {
                                        // (x + k1) - k2 -> x + (k1 - k2)
                                        let k = k1.wrapping_sub(k2) & all;
                                        if k == 0 {
                                            return Some(x);
                                        }
                                        let kv = self.k(ty, bits, k);
                                        self.insts[i as usize].op = Op::Bin(Bin::Add, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    (false, Bin::Sub, false) => {
                                        // (x - k1) - k2 -> x - (k1 + k2)
                                        let k = k1.wrapping_add(k2) & all;
                                        if k == 0 {
                                            return Some(x);
                                        }
                                        let kv = self.k(ty, bits, k);
                                        self.insts[i as usize].op = Op::Bin(Bin::Sub, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    (false, Bin::Sub, true) => {
                                        // (k1 - x) - k2 -> (k1 - k2) - x
                                        let kv = self.k(ty, bits, k1.wrapping_sub(k2));
                                        self.insts[i as usize].op = Op::Bin(Bin::Sub, kv, x);
                                        *changed = true;
                                        return None;
                                    }
                                    (true, Bin::Add, _) => {
                                        // k2 - (x + k1) -> (k2 - k1) - x
                                        let kv = self.k(ty, bits, k2.wrapping_sub(k1));
                                        self.insts[i as usize].op = Op::Bin(Bin::Sub, kv, x);
                                        *changed = true;
                                        return None;
                                    }
                                    (true, Bin::Sub, false) => {
                                        // k2 - (x - k1) -> (k2 + k1) - x
                                        let kv = self.k(ty, bits, k2.wrapping_add(k1));
                                        self.insts[i as usize].op = Op::Bin(Bin::Sub, kv, x);
                                        *changed = true;
                                        return None;
                                    }
                                    (true, Bin::Sub, true) => {
                                        // k2 - (k1 - x) -> x + (k2 - k1)
                                        let k = k2.wrapping_sub(k1) & all;
                                        if k == 0 {
                                            return Some(x);
                                        }
                                        let kv = self.k(ty, bits, k);
                                        self.insts[i as usize].op = Op::Bin(Bin::Add, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Bin::Mul => {
                            if let Some((Bin::Mul, x, k1, _)) = self.inst_bin_k(var) {
                                let k = k1.wrapping_mul(k2) & all;
                                if k == 0 {
                                    return Some(self.k(ty, bits, 0));
                                }
                                if k == 1 {
                                    return Some(x);
                                }
                                let kv = self.k(ty, bits, k);
                                self.insts[i as usize].op = Op::Bin(Bin::Mul, x, kv);
                                *changed = true;
                                return None;
                            }
                        }
                        Bin::And => {
                            let live = all & !self.known_zero_bits(var, bits, 4);
                            if live & k2 == 0 {
                                return Some(self.k(ty, bits, 0));
                            }
                            if live & !k2 == 0 {
                                return Some(var);
                            }
                            if let Some((Bin::And, x, k1, _)) = self.inst_bin_k(var) {
                                let k = k1 & k2;
                                if k == 0 {
                                    return Some(self.k(ty, bits, 0));
                                }
                                if k == k1 {
                                    return Some(var);
                                }
                                let kv = self.k(ty, bits, k);
                                self.insts[i as usize].op = Op::Bin(Bin::And, x, kv);
                                *changed = true;
                                return None;
                            }
                            // (x | y) & k -> x & k when y has no bits in k (same for ^).
                            if let Some((Bin::Or | Bin::Xor, x, y)) = inst_bin(var) {
                                if (all & !self.known_zero_bits(y, bits, 3)) & k2 == 0 {
                                    let kv = self.k(ty, bits, k2);
                                    self.insts[i as usize].op = Op::Bin(Bin::And, x, kv);
                                    *changed = true;
                                    return None;
                                }
                                if (all & !self.known_zero_bits(x, bits, 3)) & k2 == 0 {
                                    let kv = self.k(ty, bits, k2);
                                    self.insts[i as usize].op = Op::Bin(Bin::And, y, kv);
                                    *changed = true;
                                    return None;
                                }
                            }
                        }
                        Bin::Or => {
                            if let Some((Bin::Or, x, k1, _)) = self.inst_bin_k(var) {
                                let k = k1 | k2;
                                if k == all {
                                    return Some(self.k(ty, bits, all));
                                }
                                if k == k1 {
                                    return Some(var);
                                }
                                let kv = self.k(ty, bits, k);
                                self.insts[i as usize].op = Op::Bin(Bin::Or, x, kv);
                                *changed = true;
                                return None;
                            }
                        }
                        Bin::Xor => {
                            if let Some((Bin::Xor, x, k1, _)) = self.inst_bin_k(var) {
                                let k = k1 ^ k2;
                                if k == 0 {
                                    return Some(x);
                                }
                                let kv = self.k(ty, bits, k);
                                self.insts[i as usize].op = Op::Bin(Bin::Xor, x, kv);
                                *changed = true;
                                return None;
                            }
                        }
                        Bin::Shl | Bin::LShr | Bin::AShr if !k2_lhs && k2 < u64::from(bits) => {
                            let s2 = k2 as u32;
                            if let Some((iop, x, k1, false)) = self.inst_bin_k(var)
                                && k1 < u64::from(bits)
                            {
                                let s1 = k1 as u32;
                                if iop == op {
                                    let sum = s1 + s2;
                                    if sum < bits {
                                        let kv = self.k(ty, bits, u64::from(sum));
                                        self.insts[i as usize].op = Op::Bin(op, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    if op == Bin::AShr {
                                        let max_s = bits - 1;
                                        if s1 == max_s {
                                            return Some(var);
                                        }
                                        let kv = self.k(ty, bits, u64::from(max_s));
                                        self.insts[i as usize].op = Op::Bin(Bin::AShr, x, kv);
                                        *changed = true;
                                        return None;
                                    }
                                    return Some(self.k(ty, bits, 0));
                                }
                                // (x << s) >>u s -> x & mask(bits - s);
                                // (x >>u s) << s -> x & !mask(s);
                                // (x << s) >>s s -> x when top s+1 bits are already sign bits.
                                if s1 == s2 && s1 > 0 {
                                    if iop == Bin::Shl && op == Bin::AShr {
                                        let live = all & !self.known_zero_bits(x, bits, 4);
                                        if live >> (bits - 1 - s1) == 0 {
                                            return Some(x);
                                        }
                                        if let Some((Bin::AShr, _, k0, false)) = self.inst_bin_k(x)
                                            && u64::from(s1) <= k0
                                            && k0 < u64::from(bits)
                                        {
                                            return Some(x);
                                        }
                                    }
                                    let m = match (iop, op) {
                                        (Bin::Shl, Bin::LShr) => Some(mask(bits - s1)),
                                        (Bin::LShr, Bin::Shl) => Some(all & !mask(s1)),
                                        _ => None,
                                    };
                                    if let Some(m) = m {
                                        let live = all & !self.known_zero_bits(x, bits, 4);
                                        if live & !m == 0 {
                                            return Some(x);
                                        }
                                        let mv = self.k(ty, bits, m);
                                        self.insts[i as usize].op = Op::Bin(Bin::And, x, mv);
                                        *changed = true;
                                        return None;
                                    }
                                }
                            }
                            if op == Bin::LShr {
                                let live = all & !self.known_zero_bits(var, bits, 4);
                                if live >> s2 == 0 {
                                    return Some(self.k(ty, bits, 0));
                                }
                                // (x | y) >>u s -> x >>u s when y's live bits all shift out.
                                if let Some((Bin::Or | Bin::Xor, x, y)) = inst_bin(var) {
                                    if (all & !self.known_zero_bits(y, bits, 3)) >> s2 == 0 {
                                        let sv = self.k(ty, bits, u64::from(s2));
                                        self.insts[i as usize].op = Op::Bin(Bin::LShr, x, sv);
                                        *changed = true;
                                        return None;
                                    }
                                    if (all & !self.known_zero_bits(x, bits, 3)) >> s2 == 0 {
                                        let sv = self.k(ty, bits, u64::from(s2));
                                        self.insts[i as usize].op = Op::Bin(Bin::LShr, y, sv);
                                        *changed = true;
                                        return None;
                                    }
                                }
                            } else if op == Bin::Shl {
                                let live = all & !self.known_zero_bits(var, bits, 4);
                                if (live << s2) & all == 0 {
                                    return Some(self.k(ty, bits, 0));
                                }
                                // `(x & m) << s` -> `x << s` when `m` preserves every surviving bit.
                                let keep = mask(bits - s2);
                                if let Some((Bin::And, x, m, _)) = self.inst_bin_k(var)
                                    && m & keep == keep
                                {
                                    let sv = self.k(ty, bits, u64::from(s2));
                                    self.insts[i as usize].op = Op::Bin(Bin::Shl, x, sv);
                                    *changed = true;
                                    return None;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                None
            }
            Op::Cmp(p, a, b) if p <= pred::TRUE => {
                // Two float constants compare one way (`v_mul_legacy` asks
                // whether its constant factor is zero).
                let (V::Const(x), V::Const(y)) = (a, b) else {
                    return None;
                };
                let (Const::Float(x), Const::Float(y)) =
                    (&self.consts[x as usize].1, &self.consts[y as usize].1)
                else {
                    return None;
                };
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                let unordered = x.is_nan() || y.is_nan();
                // Bits 0-2 of a predicate: equal, greater, less; bit 3:
                // or unordered.
                let holds = if unordered {
                    p & 8 != 0
                } else {
                    (p & 1 != 0 && x == y) || (p & 2 != 0 && x > y) || (p & 4 != 0 && x < y)
                };
                Some(self.bool(holds))
            }
            Op::Cmp(p, a, b) => {
                if !(pred::EQ..=pred::SLE).contains(&p) {
                    return None;
                }
                let eval_cmp = |pred_p: u8, x: u64, y: u64, bits: u32| {
                    let (sx, sy) = (signed(bits, x), signed(bits, y));
                    match pred_p {
                        pred::EQ => x == y,
                        pred::NE => x != y,
                        pred::IUGT => x > y,
                        pred::IUGE => x >= y,
                        pred::IULT => x < y,
                        pred::IULE => x <= y,
                        pred::SGT => sx > sy,
                        pred::SGE => sx >= sy,
                        pred::SLT => sx < sy,
                        _ => sx <= sy,
                    }
                };
                if let (Some((bits, x)), Some((_, y))) = (self.int_k(a), self.int_k(b)) {
                    return Some(self.bool(eval_cmp(p, x, y, bits)));
                }
                if a == b {
                    return Some(self.bool(eval_cmp(p, 0, 0, 32)));
                }
                let op_ty = self.type_of(a);
                let bits = self.int_bits(op_ty)?;
                let all = mask(bits);
                // Normalise constant to the RHS as `(var, k, p_norm)`.
                let norm = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((_, k))) => Some((a, k, p)),
                    (Some((_, k)), None) => Some((b, k, swap_pred(p))),
                    _ => None,
                };
                if let Some((var, k, pn)) = norm {
                    // i1 compared with a constant.
                    if bits == 1 && (pn == pred::EQ || pn == pred::NE) {
                        let same = (pn == pred::EQ) == (k != 0);
                        if same {
                            return Some(var);
                        }
                        if let Some(inner) = self.bit_not(var) {
                            return Some(inner);
                        }
                        let yes = self.bool(true);
                        self.insts[i as usize].op = Op::Bin(Bin::Xor, var, yes);
                        *changed = true;
                        return None;
                    }
                    // A bit widened and tested against 0 or 1.
                    if (pn == pred::EQ || pn == pred::NE)
                        && (k == 0 || k == 1)
                        && let V::Inst(z) = var
                        && let Op::Cast(Cast::ZExt, x) = self.insts[z as usize].op
                        && self.type_of(x) == ty
                    {
                        let same = (pn == pred::NE && k == 0) || (pn == pred::EQ && k == 1);
                        if same {
                            return Some(x);
                        }
                        if let Some(inner) = self.bit_not(x) {
                            return Some(inner);
                        }
                        let yes = self.bool(true);
                        self.insts[i as usize].op = Op::Bin(Bin::Xor, x, yes);
                        *changed = true;
                        return None;
                    }
                    // Comparing `select(c, k1, k2)` against constant `k`.
                    if let V::Inst(s) = var
                        && let Op::Select(sc, st, sf) = self.insts[s as usize].op
                        && let (Some((_, kt)), Some((_, kf))) = (self.int_k(st), self.int_k(sf))
                    {
                        let ht = eval_cmp(pn, kt, k, bits);
                        let hf = eval_cmp(pn, kf, k, bits);
                        match (ht, hf) {
                            (true, true) => return Some(self.bool(true)),
                            (false, false) => return Some(self.bool(false)),
                            (true, false) => return Some(sc),
                            (false, true) => {
                                if let Some(inner) = self.bit_not(sc) {
                                    return Some(inner);
                                }
                                let yes = self.bool(true);
                                self.insts[i as usize].op = Op::Bin(Bin::Xor, sc, yes);
                                *changed = true;
                                return None;
                            }
                        }
                    }
                    // `(x + k1) == k` -> `x == (k - k1)` (exact for modular EQ/NE).
                    if (pn == pred::EQ || pn == pred::NE)
                        && let Some((iop, x, k1, k1_lhs)) = self.inst_bin_k(var)
                    {
                        let new_k = match (iop, k1_lhs) {
                            (Bin::Add, _) => Some(k.wrapping_sub(k1)),
                            (Bin::Sub, false) => Some(k.wrapping_add(k1)),
                            (Bin::Sub, true) => Some(k1.wrapping_sub(k)),
                            (Bin::Xor, _) => Some(k ^ k1),
                            _ => None,
                        };
                        if let Some(nk) = new_k {
                            let kv = self.k(op_ty, bits, nk);
                            self.insts[i as usize].op = Op::Cmp(pn, x, kv);
                            *changed = true;
                            return None;
                        }
                    }
                    // Known-zero bit contradictions and unsigned/signed bounds.
                    let live = all & !self.known_zero_bits(var, bits, 4);
                    let smin = 1u64 << (bits - 1);
                    let smax = smin - 1;
                    match pn {
                        pred::EQ if k & !live != 0 => return Some(self.bool(false)),
                        pred::NE if k & !live != 0 => return Some(self.bool(true)),
                        pred::IULT if k == 0 => return Some(self.bool(false)),
                        pred::IUGE if k == 0 => return Some(self.bool(true)),
                        pred::IULT if live < k => return Some(self.bool(true)),
                        pred::IUGE if live < k => return Some(self.bool(false)),
                        pred::IULE if live <= k => return Some(self.bool(true)),
                        pred::IUGT if live <= k => return Some(self.bool(false)),
                        pred::SLT if k == smin => return Some(self.bool(false)),
                        pred::SGE if k == smin => return Some(self.bool(true)),
                        pred::SGT if k == smax => return Some(self.bool(false)),
                        pred::SLE if k == smax => return Some(self.bool(true)),
                        _ => {}
                    }
                }
                // `(x + y) == x` -> `y == 0`; `(x ^ y) == x` -> `y == 0`.
                if p == pred::EQ || p == pred::NE {
                    for (other, cand) in [(a, b), (b, a)] {
                        if let V::Inst(j) = cand
                            && let Op::Bin(Bin::Add | Bin::Xor, x, y) = self.insts[j as usize].op
                        {
                            let rest = if x == other {
                                Some(y)
                            } else if y == other {
                                Some(x)
                            } else {
                                None
                            };
                            if let Some(rest) = rest {
                                let z = self.k(op_ty, bits, 0);
                                self.insts[i as usize].op = Op::Cmp(p, rest, z);
                                *changed = true;
                                return None;
                            }
                        }
                    }
                }
                None
            }
            Op::Select(mut c, mut t, mut f) => {
                if let Some((_, c)) = self.int_k(c) {
                    return Some(if c != 0 { t } else { f });
                }
                let mut inner = false;
                if let Some(c2) = self.bit_not(c) {
                    c = c2;
                    std::mem::swap(&mut t, &mut f);
                    inner = true;
                }
                // Under the same condition an inner select is one of its
                // operands (two guarded writes of one register in a row).
                while let V::Inst(x) = t
                    && let Op::Select(c2, t2, _) = self.insts[x as usize].op
                    && c2 == c
                {
                    (t, inner) = (t2, true);
                }
                while let V::Inst(x) = f
                    && let Op::Select(c2, _, f2) = self.insts[x as usize].op
                    && c2 == c
                {
                    (f, inner) = (f2, true);
                }
                if t == f {
                    return Some(t);
                }
                if self.int_bits(ty) == Some(1) {
                    let kt = self.int_k(t).map(|k| k.1);
                    let kf = self.int_k(f).map(|k| k.1);
                    if kt == Some(1) && kf == Some(0) {
                        return Some(c);
                    }
                    if kt == Some(0) && kf == Some(1) {
                        let yes = self.bool(true);
                        self.insts[i as usize].op = Op::Bin(Bin::Xor, c, yes);
                        *changed = true;
                        return None;
                    }
                    if kt == Some(1) || t == c {
                        self.insts[i as usize].op = Op::Bin(Bin::Or, c, f);
                        *changed = true;
                        return None;
                    }
                    if kf == Some(0) || f == c {
                        self.insts[i as usize].op = Op::Bin(Bin::And, c, t);
                        *changed = true;
                        return None;
                    }
                }
                if inner {
                    self.insts[i as usize].op = Op::Select(c, t, f);
                    *changed = true;
                }
                None
            }
            Op::Cast(op, x) => {
                let from = self.type_of(x);
                let through = match x {
                    V::Inst(y) => match self.insts[y as usize].op {
                        Op::Cast(op2, z) => Some((op2, z)),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    Cast::Bitcast => {
                        if from == ty {
                            return Some(x);
                        }
                        // There and back, or chained bitcasts.
                        if let Some((Cast::Bitcast, z)) = through {
                            if self.type_of(z) == ty {
                                return Some(z);
                            }
                            self.insts[i as usize].op = Op::Cast(Cast::Bitcast, z);
                            *changed = true;
                            return None;
                        }
                        if let Some(v) = self.unwrap_bitcast(x, ty) {
                            return Some(v);
                        }
                        // `bitcast(select(c, bitcast(a), bitcast(b)))` -> `select(c, a, b)`.
                        if let V::Inst(s) = x
                            && let Op::Select(sc, st, sf) = self.insts[s as usize].op
                            && let Some(ut) = self.unwrap_bitcast(st, ty)
                            && let Some(uf) = self.unwrap_bitcast(sf, ty)
                        {
                            self.insts[i as usize].op = Op::Select(sc, ut, uf);
                            *changed = true;
                            return None;
                        }
                        None
                    }
                    Cast::ZExt | Cast::SExt | Cast::Trunc => {
                        let to = self.int_bits(ty)?;
                        if let Some((bits, k)) = self.int_k(x) {
                            let k = if op == Cast::SExt {
                                signed(bits, k) as u64
                            } else {
                                k
                            };
                            return Some(self.k(ty, to, k));
                        }
                        match (op, through) {
                            (Cast::ZExt, Some((Cast::ZExt, z)))
                            | (Cast::SExt, Some((Cast::SExt, z)))
                            | (Cast::Trunc, Some((Cast::Trunc, z))) => {
                                self.insts[i as usize].op = Op::Cast(op, z);
                                *changed = true;
                                return None;
                            }
                            (Cast::SExt, Some((Cast::ZExt, z)))
                                if self.int_bits(self.type_of(z)) < self.int_bits(from) =>
                            {
                                self.insts[i as usize].op = Op::Cast(Cast::ZExt, z);
                                *changed = true;
                                return None;
                            }
                            (Cast::Trunc, Some((op2 @ (Cast::ZExt | Cast::SExt), z))) => {
                                let zbits = self.int_bits(self.type_of(z))?;
                                if zbits == to {
                                    return Some(z);
                                }
                                let next = if zbits < to { op2 } else { Cast::Trunc };
                                self.insts[i as usize].op = Op::Cast(next, z);
                                *changed = true;
                                return None;
                            }
                            _ => {}
                        }
                        if op == Cast::Trunc
                            && let Some(fb) = self.int_bits(from)
                        {
                            let low = mask(to);
                            if self.known_zero_bits(x, fb, 4) & low == low {
                                return Some(self.k(ty, to, 0));
                            }
                            if let V::Inst(j) = x
                                && let Op::Bin(Bin::Or | Bin::Xor | Bin::Add, a, b) =
                                    self.insts[j as usize].op
                            {
                                if self.known_zero_bits(b, fb, 3) & low == low {
                                    self.insts[i as usize].op = Op::Cast(Cast::Trunc, a);
                                    *changed = true;
                                    return None;
                                }
                                if self.known_zero_bits(a, fb, 3) & low == low {
                                    self.insts[i as usize].op = Op::Cast(Cast::Trunc, b);
                                    *changed = true;
                                    return None;
                                }
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }
            Op::Call(f, ref args) => {
                // Exact integer DXIL intrinsics only; float calls stay untouched.
                let bits = self.int_bits(ty)?;
                let all = mask(bits);
                let name = self.funcs[f as usize].name.as_str();
                let (_, code) = self.int_k(*args.first()?)?;
                if name == "dx.op.unary.i32" && code == 30 && args.len() == 2 && bits == 32 {
                    let a = args[1];
                    if let Some((_, x)) = self.int_k(a) {
                        return Some(self.k(ty, 32, u64::from((x as u32).reverse_bits())));
                    }
                    if let V::Inst(j) = a
                        && let Op::Call(f2, ref a2) = self.insts[j as usize].op
                        && f2 == f
                        && a2.len() == 2
                        && self.int_k(a2[0]).is_some_and(|k| k.1 == 30)
                    {
                        return Some(a2[1]);
                    }
                    return None;
                }
                if name == "dx.op.unaryBits.i32" && args.len() == 2 && bits == 32 {
                    let (_, x) = self.int_k(args[1])?;
                    let u = x as u32;
                    let r = match code {
                        31 => u.count_ones(),
                        32 => {
                            if u == 0 {
                                u32::MAX
                            } else {
                                u.trailing_zeros()
                            }
                        }
                        33 => {
                            if u == 0 {
                                u32::MAX
                            } else {
                                u.leading_zeros()
                            }
                        }
                        34 => {
                            let s = u as i32;
                            if s == 0 || s == -1 {
                                u32::MAX
                            } else if s > 0 {
                                u.leading_zeros()
                            } else {
                                (!u).leading_zeros()
                            }
                        }
                        _ => return None,
                    };
                    return Some(self.k(ty, 32, u64::from(r)));
                }
                if name.starts_with("dx.op.binary.i")
                    && matches!(code, 37..=40)
                    && args.len() == 3
                {
                    let (op_k, a, b) = (args[0], args[1], args[2]);
                    if a == b {
                        return Some(a);
                    }
                    let combine = |x: u64, y: u64| match code {
                        37 => signed(bits, x).max(signed(bits, y)) as u64 & all,
                        38 => signed(bits, x).min(signed(bits, y)) as u64 & all,
                        39 => x.max(y),
                        _ => x.min(y),
                    };
                    let (ka, kb) = (self.int_k(a).map(|k| k.1), self.int_k(b).map(|k| k.1));
                    if let (Some(x), Some(y)) = (ka, kb) {
                        return Some(self.k(ty, bits, combine(x, y)));
                    }
                    let (var, k2, kv2) = match (ka, kb) {
                        (None, Some(k)) => (a, k, b),
                        (Some(k), None) => (b, k, a),
                        _ => return None,
                    };
                    let smin = 1u64 << (bits - 1);
                    let smax = smin - 1;
                    match code {
                        37 if k2 == smax => return Some(kv2),
                        37 if k2 == smin => return Some(var),
                        38 if k2 == smin => return Some(kv2),
                        38 if k2 == smax => return Some(var),
                        39 if k2 == all => return Some(kv2),
                        39 if k2 == 0 => return Some(var),
                        40 if k2 == 0 => return Some(kv2),
                        40 if k2 == all => return Some(var),
                        _ => {}
                    }
                    if code == 39 || code == 40 {
                        let live = all & !self.known_zero_bits(var, bits, 4);
                        if live <= k2 {
                            return Some(if code == 40 { var } else { kv2 });
                        }
                    }
                    // Nested min/max with constants: `UMin(UMin(x, k1), k2)`.
                    if let V::Inst(j) = var
                        && let Op::Call(f2, ref a2) = self.insts[j as usize].op
                        && f2 == f
                        && a2.len() == 3
                        && self.int_k(a2[0]).is_some_and(|k| k.1 == code)
                    {
                        let inner_pair = match (self.int_k(a2[1]), self.int_k(a2[2])) {
                            (None, Some((_, k1))) => Some((a2[1], k1)),
                            (Some((_, k1)), None) => Some((a2[2], k1)),
                            _ => None,
                        };
                        if let Some((x, k1)) = inner_pair {
                            let k = combine(k1, k2);
                            if k == k1 {
                                return Some(var);
                            }
                            let kv = self.k(ty, bits, k);
                            self.insts[i as usize].op = Op::Call(f, vec![op_k, x, kv]);
                            *changed = true;
                            return None;
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// What kind of instruction `i` is for merging.
    fn class(&self, effects: &[Effect], i: u32) -> Effect {
        match &self.insts[i as usize].op {
            Op::Bin(..)
            | Op::Cast(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Extract(..)
            | Op::Gep(..) => Effect::Pure,
            Op::Load(p) => {
                // A constant table reads the same whatever is stored.
                let base = match *p {
                    V::Inst(g) => match &self.insts[g as usize].op {
                        Op::Gep(_, base, _) => *base,
                        _ => *p,
                    },
                    _ => *p,
                };
                match base {
                    V::Global(g) if self.globals[g as usize].init.is_some() => Effect::Pure,
                    _ => Effect::Reads,
                }
            }
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => Effect::Writes,
            Op::Phi(_) => Effect::Lanes,
            Op::Call(f, _) => effects[*f as usize],
        }
    }

    fn key(&self, i: u32) -> Option<Key> {
        let inst = &self.insts[i as usize];
        Some(match &inst.op {
            Op::Bin(op, a, b) => {
                // Integer sums, products and bit operations whichever way
                // round; float ones as written (which NaN an operation of
                // two NaNs gives is the hardware's choice by operand).
                let commutes = matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor)
                    && self.int_bits(inst.ty).is_some();
                let (a, b) = if commutes && b < a {
                    (*b, *a)
                } else {
                    (*a, *b)
                };
                Key::Bin(*op as u8, a, b)
            }
            Op::Cast(op, v) => Key::Cast(*op as u8, inst.ty, *v),
            Op::Cmp(p, a, b) => {
                if b < a {
                    Key::Cmp(swap_pred(*p), *b, *a)
                } else {
                    Key::Cmp(*p, *a, *b)
                }
            }
            Op::Select(c, t, f) => Key::Select(*c, *t, *f),
            Op::Extract(a, n) => Key::Extract(*a, *n),
            Op::Call(f, args) => {
                let mut key_args = args.clone();
                if key_args.len() == 3
                    && self.funcs[*f as usize].name.starts_with("dx.op.binary.i")
                    && self.int_k(key_args[0]).is_some_and(|k| (37..=40).contains(&k.1))
                    && key_args[2] < key_args[1]
                {
                    key_args.swap(1, 2);
                }
                Key::Call(*f, key_args)
            }
            Op::Gep(ty, p, idx) => Key::Gep(*ty, *p, idx.clone()),
            Op::Load(p) => Key::Load(*p),
            _ => return None,
        })
    }

    /// [`FOLD`], [`REUSE`], [`FACTS`] and [`LOADS`]: one walk over the
    /// dominator tree. Returns whether anything changed.
    fn merge(&mut self, passes: u32) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let effects = self.effects();
        let tree = self.dominator_tree();
        let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut changed = false;
        // Joins of one value: until none is left (one gone can leave
        // another with one value).
        if passes & FOLD != 0 {
            let mut again = true;
            while again {
                again = false;
                for &b in &tree.order {
                    for &p in &self.blocks[b].phis {
                        if repl[p as usize].is_some() {
                            continue;
                        }
                        let Op::Phi(incoming) = &self.insts[p as usize].op else {
                            continue;
                        };
                        let mut one = None;
                        let mut same = true;
                        for &(v, _) in incoming {
                            let v = resolve(&repl, v);
                            if v == V::Inst(p) {
                                continue;
                            }
                            match one {
                                None => one = Some(v),
                                Some(o) if o == v => {}
                                Some(_) => same = false,
                            }
                        }
                        if let (true, Some(v)) = (same, one) {
                            repl[p as usize] = Some(v);
                            again = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // The values every path to the block being walked has computed
        // (those of the blocks above it in the tree, and its own so far).
        let mut known: HashMap<Key, V> = HashMap::new();
        // The reads of the block being walked since its last store, and each
        // block's exit reads for single-predecessor children.
        let mut read: HashMap<Key, (V, Effect)> = HashMap::new();
        let mut read_out: Vec<HashMap<Key, (V, Effect)>> = vec![HashMap::new(); self.blocks.len()];
        let mut facts = Facts::default();
        let (yes, no) = (self.bool(true), self.bool(false));
        enum Step {
            Enter(usize),
            Leave(Vec<Key>, Learned),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(keys, learned) => {
                    for k in keys {
                        known.remove(&k);
                    }
                    for v in learned.bits {
                        facts.bits.remove(&v);
                    }
                    for k in learned.compares {
                        facts.compares.remove(&k);
                    }
                    for v in learned.values {
                        facts.values.remove(&v);
                    }
                    continue;
                }
            };
            let mut added = Vec::new();
            // Reached from one block alone, by one side of its branch: the
            // condition is that side's, here and in every block below.
            let mut learned = Learned::default();
            if passes & FACTS != 0
                && let [p] = tree.preds[b][..]
                && let Some(Term::CondBr(c, t, f)) = self.blocks[p].term
                && t != f
            {
                self.learn(resolve(&repl, c), b == t, &mut facts, &mut learned, 8);
            }
            let known_val = |facts: &Facts, v: V| match facts.bits.get(&v) {
                Some(true) => yes,
                Some(false) => no,
                None => facts.values.get(&v).copied().unwrap_or(v),
            };
            read.clear();
            if passes & LOADS != 0
                && let [p] = tree.preds[b][..]
            {
                let same_lanes = matches!(self.blocks[p].term, Some(Term::Br(_)));
                for (k, &(v, eff)) in &read_out[p] {
                    if same_lanes || eff == Effect::Reads {
                        read.insert(k.clone(), (v, eff));
                    }
                }
            }
            // Propagate branch facts into phis dominated by the branch.
            if passes & FACTS != 0 && (!facts.bits.is_empty() || !facts.values.is_empty()) {
                for n in 0..self.blocks[b].phis.len() {
                    let p = self.blocks[b].phis[n];
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                        for (v, _) in incoming {
                            let r = resolve(&repl, *v);
                            let kv = known_val(&facts, r);
                            if kv != *v {
                                *v = kv;
                                changed = true;
                            }
                        }
                    }
                }
            }
            // Two joins of the same values from the same blocks are one.
            if passes & REUSE != 0 && self.blocks[b].phis.len() > 1 {
                let mut joins: HashMap<Vec<(usize, V)>, V> = HashMap::new();
                for &p in &self.blocks[b].phis {
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    let ty = self.insts[p as usize].ty;
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        continue;
                    };
                    let mut key: Vec<(usize, V)> = incoming
                        .iter()
                        .map(|&(v, from)| (from, resolve(&repl, v)))
                        .collect();
                    key.sort();
                    // The type too: an undefined value of one type is not
                    // that of another.
                    key.push((ty.0 as usize, V::Global(u32::MAX)));
                    match joins.get(&key) {
                        Some(&first) => {
                            repl[p as usize] = Some(first);
                            changed = true;
                        }
                        None => {
                            joins.insert(key, V::Inst(p));
                        }
                    }
                }
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                map_operands(&mut self.insts[i as usize].op, |v| {
                    let v = resolve(&repl, v);
                    let kv = known_val(&facts, v);
                    changed |= kv != v;
                    kv
                });
                if let Op::Cmp(p, x, y) = self.insts[i as usize].op
                    && let Some(&holds) = facts.compares.get(&(p, x, y))
                {
                    repl[i as usize] = Some(if holds { yes } else { no });
                    changed = true;
                    continue;
                }
                if passes & FOLD != 0
                    && let Some(v) = self.simplify(i, &mut changed)
                {
                    repl[i as usize] = Some(v);
                    changed = true;
                    continue;
                }
                let class = self.class(&effects, i);
                if passes & REUSE != 0
                    && class == Effect::Pure
                    && let Some(key) = self.key(i)
                {
                    if let Some(&first) = known.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    known.insert(key.clone(), V::Inst(i));
                    added.push(key);
                }
                // Reads of memory, and wave operations (the same lanes run
                // the whole block): the block's own earlier one.
                if passes & LOADS != 0
                    && matches!(class, Effect::Reads | Effect::Lanes)
                    && let Some(key) = self.key(i)
                {
                    if let Some(&(first, _)) = read.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    read.insert(key, (V::Inst(i), class));
                }
                if class == Effect::Writes {
                    read.clear();
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;
            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                let r = resolve(&repl, c);
                let c2 = known_val(&facts, r);
                changed |= c2 != r;
                self.blocks[b].term = Some(Term::CondBr(c2, t, f));
            }
            if passes & LOADS != 0 && !tree.children[b].is_empty() && !read.is_empty() {
                read_out[b] = std::mem::take(&mut read);
            }
            steps.push(Step::Leave(added, learned));
            for &c in tree.children[b].iter().rev() {
                steps.push(Step::Enter(c));
            }
        }
        if !changed {
            return false;
        }
        // Everything that read a replaced value reads its replacement
        // (joins read values from blocks walked after them).
        for blk in &mut self.blocks {
            blk.phis.retain(|&p| repl[p as usize].is_none());
            if let Some(Term::CondBr(c, t, f)) = blk.term {
                blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
            }
        }
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for blk in blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
            }
        }
        true
    }

    /// [`FOLD`]: a branch on a constant goes one way, a branch to one target
    /// drops its condition, and an inverted branch flips its targets. The side
    /// not taken loses the branching block's values at its joins, and so do
    /// the joins of blocks that nothing reaches any more. Returns whether a
    /// branch changed.
    fn prune(&mut self) -> bool {
        let mut changed = false;
        let mut pruned_edge = false;
        for b in 0..self.blocks.len() {
            let Some(Term::CondBr(c, t, f)) = self.blocks[b].term else {
                continue;
            };
            if t == f {
                let uniform = self.blocks[t].phis.iter().all(|&p| {
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        return true;
                    };
                    let mut first = None;
                    for &(v, from) in incoming {
                        if from == b {
                            match first {
                                None => first = Some(v),
                                Some(f) if f == v => {}
                                Some(_) => return false,
                            }
                        }
                    }
                    true
                });
                if uniform {
                    self.blocks[b].term = Some(Term::Br(t));
                    changed = true;
                    for n in 0..self.blocks[t].phis.len() {
                        let p = self.blocks[t].phis[n];
                        if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                            let mut seen = false;
                            incoming.retain(|&(_, from)| {
                                if from == b {
                                    let keep = !seen;
                                    seen = true;
                                    keep
                                } else {
                                    true
                                }
                            });
                        }
                    }
                }
                continue;
            }
            if let Some((_, k)) = self.int_k(c) {
                let (taken, not) = if k != 0 { (t, f) } else { (f, t) };
                self.blocks[b].term = Some(Term::Br(taken));
                changed = true;
                pruned_edge = true;
                for n in 0..self.blocks[not].phis.len() {
                    let p = self.blocks[not].phis[n];
                    if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                        incoming.retain(|&(_, from)| from != b);
                    }
                }
            } else if let Some(c2) = self.bit_not(c) {
                self.blocks[b].term = Some(Term::CondBr(c2, f, t));
                changed = true;
            }
        }
        if pruned_edge {
            let reached = self.reached();
            let (blocks, insts) = (&self.blocks, &mut self.insts);
            for (_, blk) in blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
                for &p in &blk.phis {
                    if let Op::Phi(incoming) = &mut insts[p as usize].op {
                        incoming.retain(|&(_, from)| reached[from]);
                    }
                }
            }
        }
        changed
    }

    /// Whether instruction `i` must stay whatever reads it.
    fn has_effect(&self, effects: &[Effect], i: u32) -> bool {
        match &self.insts[i as usize].op {
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
            Op::Call(f, _) => effects[*f as usize] == Effect::Writes,
            _ => false,
        }
    }

    /// [`DEAD`]: keeps the instructions with an effect, the branch
    /// conditions, and what they read, transitively; the rest leave their
    /// blocks. Blocks no branch reaches are emptied (they are not written).
    fn drop_dead(&mut self) {
        let effects = self.effects();
        let reached = self.reached();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        let mark = |live: &mut [bool], work: &mut Vec<u32>, v: V| {
            if let V::Inst(i) = v
                && !live[i as usize]
            {
                live[i as usize] = true;
                work.push(i);
            }
        };
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in &blk.body {
                if self.has_effect(&effects, i) {
                    mark(&mut live, &mut work, V::Inst(i));
                }
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                mark(&mut live, &mut work, c);
            }
        }
        while let Some(i) = work.pop() {
            for_each_operand(&self.insts[i as usize].op, |v| {
                mark(&mut live, &mut work, v)
            });
        }
        for (b, blk) in self.blocks.iter_mut().enumerate() {
            if reached[b] {
                blk.body.retain(|&i| live[i as usize]);
                blk.phis.retain(|&i| live[i as usize]);
            } else {
                blk.body.clear();
                blk.phis.clear();
            }
        }
        self.drop_unused_functions();
        self.drop_unused_tables();
    }

    /// See [`Usage`]: over the blocks reached from the entry.
    pub(crate) fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let reached = self.reached();
        let wide = |ty: Ty| self.types[ty.0 as usize] == TypeDef::Int(64);
        // The first field of a handle's annotation: its kind, with the UAV
        // bit; the second: the component type and count.
        let annotation = |h: V| -> Option<(i64, i64)> {
            let V::Inst(a) = h else { return None };
            let Op::Call(f, args) = &self.insts[a as usize].op else {
                return None;
            };
            if self.funcs[*f as usize].name != "dx.op.annotateHandle" {
                return None;
            }
            let V::Const(props) = *args.get(2)? else {
                return None;
            };
            let Const::Aggregate(fields) = &self.consts[props as usize].1 else {
                return None;
            };
            match (
                &self.consts[fields[0] as usize].1,
                &self.consts[fields[1] as usize].1,
            ) {
                (Const::Int(kind), Const::Int(components)) => Some((*kind, *components)),
                _ => None,
            }
        };
        for (_, blk) in self.blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &i in blk.phis.iter().chain(&blk.body) {
                let inst = &self.insts[i as usize];
                u.int64 |= wide(inst.ty);
                for_each_operand(&inst.op, |v| u.int64 |= wide(self.type_of(v)));
                let Op::Call(f, args) = &inst.op else {
                    continue;
                };
                let name = self.funcs[*f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                if name.starts_with("wave") || name.starts_with("quad") {
                    u.wave_ops = true;
                } else if name.starts_with("allocateRayQuery") || name.starts_with("rayQuery_") {
                    u.raytracing_tier_1_1 = true;
                } else if name == "createHandleFromHeap" {
                    match args.get(2).and_then(|&s| self.int_k(s)) {
                        Some((_, 0)) => u.heap_views = true,
                        _ => u.heap_samplers = true,
                    }
                } else if name == "annotateHandle" {
                    if let Some((kind, _)) = annotation(V::Inst(i)) {
                        u.rovs |= kind & 0x2000 != 0;
                        u.raytracing_tier_1_1 |= (kind & 0xff) == 16;
                    }
                } else if name.starts_with("atomic") && name.ends_with(".i64") {
                    u.atomic64 = true;
                } else if name.starts_with("textureLoad") || name.starts_with("bufferLoad") {
                    u.typed_uav_loads |= args.get(1).and_then(|&h| annotation(h)).is_none_or(
                        |(kind, components)| kind & 0x1000 != 0 && components >> 8 & 0xff > 1,
                    );
                }
            }
        }
        u
    }

    /// A constant table no remaining load reads leaves the module (the
    /// validator refuses an internal declaration nothing uses). Shared
    /// memory stays declared whether it is read or not.
    fn drop_unused_tables(&mut self) {
        let mut used: Vec<bool> = self.globals.iter().map(|g| g.init.is_none()).collect();
        if used.iter().all(|&u| u) {
            return;
        }
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Global(g) = v {
                        used[g as usize] = true;
                    }
                });
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Global(g)) = md {
                used[*g as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.globals.retain(|_| {
            old += 1;
            used[old - 1]
        });
        let renumber = |v: V| match v {
            V::Global(g) if used[g as usize] => V::Global(new[g as usize]),
            v => v,
        };
        for inst in &mut self.insts {
            map_operands(&mut inst.op, renumber);
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, v) = md {
                *v = renumber(*v);
            }
        }
    }

    /// Declarations no remaining call names leave the module (the validator
    /// refuses an external function nothing uses); the others keep their
    /// order, so `main` stays function 0.
    fn drop_unused_functions(&mut self) {
        let mut used: Vec<bool> = self.funcs.iter().map(|f| f.body).collect();
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                if let Op::Call(f, _) = self.insts[i as usize].op {
                    used[f as usize] = true;
                }
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                used[*f as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.funcs.retain(|_| {
            old += 1;
            used[old - 1]
        });
        self.func_ids = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i as u32))
            .collect();
        // Instructions that left their block keep a stale number: nothing
        // reads them again.
        for inst in &mut self.insts {
            if let Op::Call(f, _) = &mut inst.op
                && used[*f as usize]
            {
                *f = new[*f as usize];
            }
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                *f = new[*f as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `main` with one block and a function of each kind of effect.
    struct Kit {
        m: Module,
        b: usize,
        pure: V,
        reads: V,
        writes: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let pure = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        Kit {
            m,
            b,
            pure,
            reads,
            writes,
        }
    }

    fn body(m: &Module, b: usize) -> usize {
        m.blocks[b].body.len()
    }

    #[test]
    fn unread_results_without_effect_are_dropped() {
        let Kit {
            mut m,
            b,
            pure,
            reads,
            writes,
        } = kit();
        let (one, two) = (m.i32(1), m.i32(2));
        let used = m.call(b, reads, vec![one, two]);
        let dead = m.call(b, reads, vec![two, one]);
        let chain = m.bin(b, Bin::Add, dead, one);
        let _unread = m.call(b, pure, vec![chain, two]);
        // A store's result is unread and stays, with what it reads.
        let kept = m.bin(b, Bin::Add, used, two);
        m.call(b, writes, vec![kept, one]);
        m.terminate(b, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(body(&m, b), 3, "the load, the add and the store");
        m.write();
    }

    #[test]
    fn branch_conditions_and_phi_cycles() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let i32t = m.int(32);
        let (zero, one) = (m.i32(0), m.i32(1));
        let (head, exit) = (m.new_block(), m.new_block());
        m.terminate(b, Term::Br(head));
        // A counter that decides the branch, and a register that only
        // feeds itself round the loop.
        let count = m.phi(head, i32t);
        let idle = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, count, one);
        let idle_next = m.call(head, reads, vec![idle, one]);
        let more = m.cmp(head, pred::IULT, next, one);
        m.terminate(head, Term::CondBr(more, head, exit));
        m.set_phi(count, vec![(zero, b), (next, head)]);
        m.set_phi(idle, vec![(zero, b), (idle_next, head)]);
        m.call(exit, writes, vec![zero, one]);
        m.terminate(exit, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(m.blocks[head].phis.len(), 1, "the counter");
        assert_eq!(body(&m, head), 2, "the step and the test");
        assert_eq!(body(&m, exit), 1);
        m.write();
    }

    #[test]
    fn wave_operations_have_no_effect_of_their_own() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let first = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let barrier =
            k.m.function("dx.op.barrier", fty, Some(Attr::NoDuplicate), false);
        let one = k.m.i32(1);
        k.m.call(k.b, first, vec![one, one]);
        k.m.call(k.b, barrier, vec![one, one]);
        k.m.terminate(k.b, Term::Ret);
        k.m.optimise(DEAD);
        assert_eq!(body(&k.m, k.b), 1, "the barrier");
        // The declarations nothing calls any more are gone, and the call
        // that stayed names its function by the new number.
        let names: Vec<&str> = k.m.funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["main", "dx.op.barrier"]);
        let Op::Call(f, _) = k.m.insts[k.m.blocks[k.b].body[0] as usize].op else {
            panic!("a call");
        };
        assert_eq!(k.m.funcs[f as usize].name, "dx.op.barrier");
        k.m.write();
    }

    /// The operands of each store left in block `b`.
    fn stores(m: &Module, b: usize) -> Vec<Vec<V>> {
        m.blocks[b]
            .body
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(g, args) if m.funcs[*g as usize].name.contains("Store") => {
                    Some(args.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn constants_and_identities_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i32t, f32t) = (m.int(32), m.float());
        let (zero, one, all) = (m.i32(0), m.i32(1), m.i32(u32::MAX));
        let x = m.call(b, reads, vec![one, one]);
        // x + 0, -1 & x, x >> 0, x * 1, x | x: all x.
        let a = m.bin(b, Bin::Add, x, zero);
        let a = m.bin(b, Bin::And, all, a);
        let a = m.bin(b, Bin::LShr, a, zero);
        let a = m.bin(b, Bin::Mul, a, one);
        let a = m.bin(b, Bin::Or, a, a);
        // There and back.
        let f = m.cast(b, Cast::Bitcast, a, f32t);
        let a = m.cast(b, Cast::Bitcast, f, i32t);
        // Constants: (262168 >> 16) & 127 = 4, 7 & 31 = 7, -8 >>s 1 = -4.
        let k = m.i32(262168);
        let sixteen = m.i32(16);
        let k = m.bin(b, Bin::LShr, k, sixteen);
        let k127 = m.i32(127);
        let k = m.bin(b, Bin::And, k, k127);
        let minus8 = m.i32(-8i32 as u32);
        let s = m.bin(b, Bin::AShr, minus8, one);
        // A shift by the width is left alone.
        let k32 = m.i32(32);
        let wide = m.bin(b, Bin::Shl, one, k32);
        // A select between a value and itself, and on a constant.
        let c = m.cmp(b, pred::IULT, zero, one);
        let sel = m.select(b, c, a, wide);
        let same = m.select(b, c, sel, sel);
        m.call(b, writes, vec![same, k]);
        m.call(b, writes, vec![s, wide]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | DEAD);
        assert_eq!(body(&m, b), 4, "the read, the wide shift and two stores");
        let four = m.i32(4);
        let minus4 = m.i32(-4i32 as u32);
        let stores = stores(&m, b);
        assert_eq!(stores, [vec![x, four], vec![minus4, wide]]);
        m.write();
    }

    #[test]
    fn float_arithmetic_is_not_folded() {
        let Kit {
            mut m, b, writes, ..
        } = kit();
        let f32t = m.float();
        let fty = m.func_ty(f32t, vec![f32t, f32t]);
        let input = m.function("dx.op.loadInput.f32", fty, Some(Attr::ReadNone), false);
        let (zero, one, two) = (m.f32(0), m.f32(1.0f32.to_bits()), m.f32(2.0f32.to_bits()));
        let x = m.call(b, input, vec![zero, zero]);
        let y = m.call(b, input, vec![one, one]);
        let a = m.bin(b, Bin::Add, x, zero);
        let p = m.bin(b, Bin::Mul, a, one);
        // The same sum the other way round is another instruction.
        let xy = m.bin(b, Bin::Add, x, y);
        let yx = m.bin(b, Bin::Add, y, x);
        // Constants compare here: 1 < 2, and nothing equals a NaN.
        let nan = m.f32(f32::NAN.to_bits());
        let lt = m.cmp(b, pred::OLT, one, two);
        let eq = m.cmp(b, pred::OEQ, nan, nan);
        let une = m.cmp(b, pred::UNE, nan, one);
        let sel = m.select(b, lt, xy, yx);
        let sel = m.select(b, eq, p, sel);
        let sel = m.select(b, une, sel, p);
        m.call(b, writes, vec![sel, p]);
        m.terminate(b, Term::Ret);
        m.optimise(ALL);
        // Two inputs, x + 0.0, * 1.0, x + y and the store: y + x and the
        // selects are gone.
        assert_eq!(body(&m, b), 6);
        assert_eq!(stores(&m, b), [[xy, p]]);
        m.write();
    }

    /// entry: `c = x < y`, branch to `then` or `other`, which join.
    struct Diamond {
        k: Kit,
        x: V,
        y: V,
        c: V,
        then: usize,
        other: usize,
        join: usize,
    }

    fn diamond() -> Diamond {
        let mut k = kit();
        let (one, two) = (k.m.i32(1), k.m.i32(2));
        let x = k.m.call(k.b, k.reads, vec![one, one]);
        let y = k.m.call(k.b, k.reads, vec![two, two]);
        let c = k.m.cmp(k.b, pred::IULT, x, y);
        let (then, other, join) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(c, then, other));
        k.m.terminate(then, Term::Br(join));
        k.m.terminate(other, Term::Br(join));
        k.m.terminate(join, Term::Ret);
        Diamond {
            k,
            x,
            y,
            c,
            then,
            other,
            join,
        }
    }

    #[test]
    fn a_value_is_reused_below_its_block_and_not_beside_it() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    pure,
                    writes,
                    ..
                },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        // The entry's value again in a branch; a branch's value again in
        // the other branch and in the join (neither is on every path).
        let top = m.call(b, pure, vec![x, y]);
        m.call(b, writes, vec![top, top]);
        let again = m.call(then, pure, vec![x, y]);
        let side = m.bin(then, Bin::Add, y, x);
        m.call(then, writes, vec![again, side]);
        let beside = m.bin(other, Bin::Add, x, y);
        m.call(other, writes, vec![beside, beside]);
        let after = m.bin(join, Bin::Add, x, y);
        m.call(join, writes, vec![after, after]);
        m.optimise(REUSE);
        assert_eq!(stores(&m, then), [[top, side]]);
        assert_eq!(body(&m, then), 2);
        assert_eq!(body(&m, other), 2);
        assert_eq!(body(&m, join), 2);
        m.write();
    }

    #[test]
    fn a_branch_taken_says_what_its_condition_is() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        // Behind the branch: the condition again (a guard), its opposite
        // comparison, and a guarded write.
        let guard = m.bin(then, Bin::And, c, d);
        let old = m.select(then, guard, x, y);
        let not = m.cmp(then, pred::IUGE, x, y);
        let new = m.select(then, not, y, old);
        m.call(then, writes, vec![new, new]);
        let kept = m.select(other, c, x, y);
        m.call(other, writes, vec![kept, kept]);
        // After the join nothing is known.
        let open = m.select(join, c, x, y);
        m.call(join, writes, vec![open, open]);
        m.optimise(FOLD | FACTS);
        // then: select(d, x, y) and the store; other: y is stored.
        assert_eq!(body(&m, then), 2);
        let V::Inst(sel) = stores(&m, then)[0][0] else {
            panic!()
        };
        assert!(matches!(m.insts[sel as usize].op, Op::Select(g, a, z) if (g, a, z) == (d, x, y)));
        assert_eq!(stores(&m, other), [[y, y]]);
        assert_eq!(body(&m, join), 2);
        let _ = both;
        m.write();

        // Both terms of an `and` that held hold behind it.
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            ..
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        let (inner, out) = (m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(both, inner, out));
        let a = m.select(inner, c, x, y);
        let z = m.select(inner, d, a, y);
        m.call(inner, writes, vec![z, z]);
        m.terminate(inner, Term::Br(out));
        // Where the `and` did not hold either term may.
        let w = m.select(out, d, x, y);
        m.call(out, writes, vec![w, w]);
        m.terminate(out, Term::Ret);
        m.optimise(FOLD | FACTS);
        assert_eq!(stores(&m, inner), [[x, x]]);
        assert_eq!(body(&m, out), 2);
        m.write();
    }

    #[test]
    fn a_branch_on_a_constant_goes_one_way() {
        let Diamond {
            k: Kit {
                mut m, b, writes, ..
            },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let i32t = m.int(32);
        // Behind the branch the same test again: its other side is never
        // run, and the join behind it has one value.
        let (deep, never, meet) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(then, Term::CondBr(c, deep, never));
        m.terminate(deep, Term::Br(meet));
        m.call(never, writes, vec![y, y]);
        m.terminate(never, Term::Br(meet));
        let p = m.phi(meet, i32t);
        m.set_phi(p, vec![(x, deep), (y, never)]);
        m.call(meet, writes, vec![p, p]);
        m.terminate(meet, Term::Br(join));
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(p, meet), (y, other)]);
        m.call(join, writes, vec![q, q]);
        m.optimise(ALL);
        assert!(matches!(m.blocks[then].term, Some(Term::Br(t)) if t == deep));
        assert_eq!(stores(&m, meet), [[x, x]]);
        assert!(m.blocks[meet].phis.is_empty());
        assert_eq!(body(&m, never), 0, "emptied: nothing reaches it");
        let Op::Phi(incoming) = &m.insts[m.blocks[join].phis[0] as usize].op else {
            panic!()
        };
        assert_eq!(incoming, &vec![(x, meet), (y, other)]);
        let _ = b;
        m.write();
    }

    #[test]
    fn a_read_is_reused_within_its_block_until_a_store() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            then,
            ..
        } = diamond();
        let i32t = m.int(32);
        // The entry reads (1, 1) for x: again in the entry it is x; in the
        // next block it is read again.
        let one = m.i32(1);
        let again = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![again, y]);
        let after = m.call(b, reads, vec![one, one]);
        let twice = m.call(b, reads, vec![one, one]);
        let sum = m.bin(b, Bin::Add, after, twice);
        m.call(b, writes, vec![sum, sum]);
        let below = m.call(then, reads, vec![one, one]);
        m.call(then, writes, vec![below, below]);
        // A constant buffer cannot change: one load for the function.
        let cfty = m.func_ty(i32t, vec![i32t, i32t]);
        let cb = m.function(
            "dx.op.cbufferLoadLegacy.i32",
            cfty,
            Some(Attr::ReadOnly),
            false,
        );
        let top = m.call(b, cb, vec![one, one]);
        m.call(b, writes, vec![top, top]);
        let under = m.call(then, cb, vec![one, one]);
        m.call(then, writes, vec![under, under]);
        m.optimise(REUSE | LOADS);
        let stores = stores(&m, b);
        assert_eq!(stores[0], [x, y], "the second read is the first");
        let V::Inst(sum) = stores[1][0] else { panic!() };
        assert!(
            matches!(m.insts[sum as usize].op, Op::Bin(Bin::Add, a, z) if a == after && z == after),
            "read again after the store, once"
        );
        assert_eq!(stores[2], [top, top]);
        assert_eq!(
            super::tests::stores(&m, then),
            [vec![below, below], vec![top, top]]
        );
        m.write();
    }

    #[test]
    fn a_join_of_one_value_and_two_equal_joins() {
        let Diamond {
            k: Kit { mut m, writes, .. },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        let i32t = m.int(32);
        let one = m.phi(join, i32t);
        m.set_phi(one, vec![(x, then), (x, other)]);
        let p = m.phi(join, i32t);
        m.set_phi(p, vec![(x, then), (y, other)]);
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(y, other), (x, then)]);
        m.call(join, writes, vec![one, p]);
        let sum = m.bin(join, Bin::Add, p, q);
        m.call(join, writes, vec![sum, q]);
        m.optimise(FOLD | REUSE);
        assert_eq!(
            m.blocks[join].phis,
            [match p {
                V::Inst(i) => i,
                _ => unreachable!(),
            }]
        );
        let stores = stores(&m, join);
        assert_eq!(stores[0], [x, p]);
        assert_eq!(stores[1][1], p);
        m.write();
    }

    // ---- the passes against an interpreter, on random programs ----

    /// Runs `m` on `inputs`: the stores it makes, in order (and how it
    /// ended). Reads a value before its definition ran: a panic.
    fn run(m: &Module, inputs: &[u32]) -> Vec<(String, Vec<u64>)> {
        let mut vals: Vec<Option<u64>> = vec![None; m.insts.len()];
        let mut memory = [0u64; 16];
        let mut trace = Vec::new();
        let eval = |vals: &[Option<u64>], v: V| -> u64 {
            match v {
                V::Inst(i) => vals[i as usize].expect("a value defined before its use"),
                V::Const(c) => {
                    match (&m.consts[c as usize].1, m.int_bits(m.consts[c as usize].0)) {
                        (Const::Int(x), Some(bits)) => *x as u64 & mask(bits),
                        (Const::Float(bits), _) => u64::from(*bits),
                        _ => 0,
                    }
                }
                _ => panic!("a global or function as a value"),
            }
        };
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..4000 {
            let blk = &m.blocks[b];
            // Joins take their values together, before any is assigned.
            let joined: Vec<(u32, u64)> = blk
                .phis
                .iter()
                .map(|&p| {
                    let Op::Phi(incoming) = &m.insts[p as usize].op else {
                        panic!("a join")
                    };
                    let v = incoming
                        .iter()
                        .find(|x| x.1 == from)
                        .expect("a value for the edge taken");
                    (p, eval(&vals, v.0))
                })
                .collect();
            for (p, x) in joined {
                vals[p as usize] = Some(x);
            }
            for &i in &blk.body {
                let inst = &m.insts[i as usize];
                let bits = m.int_bits(inst.ty);
                let float = |x: u64| f32::from_bits(x as u32);
                let r = match &inst.op {
                    Op::Bin(op, x, y) => {
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        match bits {
                            Some(bits) => {
                                let r = match op {
                                    Bin::Add => x.wrapping_add(y),
                                    Bin::Sub => x.wrapping_sub(y),
                                    Bin::Mul => x.wrapping_mul(y),
                                    Bin::And => x & y,
                                    Bin::Or => x | y,
                                    Bin::Xor => x ^ y,
                                    Bin::Shl => x << y,
                                    Bin::LShr => x >> y,
                                    Bin::AShr => (signed(bits, x) >> y) as u64,
                                    _ => panic!("no divisions here"),
                                };
                                r & mask(bits)
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                let r = match op {
                                    Bin::Add => x + y,
                                    Bin::Sub => x - y,
                                    _ => x * y,
                                };
                                u64::from(r.to_bits())
                            }
                        }
                    }
                    Op::Cmp(p, x, y) => {
                        let wide = m.int_bits(m.type_of(*x));
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let holds = match wide {
                            Some(bits) => {
                                let (sx, sy) = (signed(bits, x), signed(bits, y));
                                match *p {
                                    pred::EQ => x == y,
                                    pred::NE => x != y,
                                    pred::IUGT => x > y,
                                    pred::IUGE => x >= y,
                                    pred::IULT => x < y,
                                    pred::IULE => x <= y,
                                    pred::SGT => sx > sy,
                                    pred::SGE => sx >= sy,
                                    pred::SLT => sx < sy,
                                    _ => sx <= sy,
                                }
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                if x.is_nan() || y.is_nan() {
                                    p & 8 != 0
                                } else {
                                    (p & 1 != 0 && x == y)
                                        || (p & 2 != 0 && x > y)
                                        || (p & 4 != 0 && x < y)
                                }
                            }
                        };
                        u64::from(holds)
                    }
                    Op::Select(c, t, f) => {
                        if eval(&vals, *c) != 0 {
                            eval(&vals, *t)
                        } else {
                            eval(&vals, *f)
                        }
                    }
                    Op::Cast(op, x) => {
                        let from_bits = m.int_bits(m.type_of(*x));
                        let x = eval(&vals, *x);
                        match (op, from_bits, bits) {
                            (Cast::Bitcast, _, _) => x,
                            (Cast::ZExt, _, _) => x,
                            (Cast::SExt, Some(f), Some(t)) => signed(f, x) as u64 & mask(t),
                            (Cast::Trunc, _, Some(t)) => x & mask(t),
                            _ => panic!("a cast the generator does not make"),
                        }
                    }
                    Op::Call(f, args) => {
                        let a: Vec<u64> = args.iter().map(|&v| eval(&vals, v)).collect();
                        let name = m.funcs[*f as usize].name.as_str();
                        match name {
                            "dx.op.loadInput.i32" => {
                                u64::from(inputs[(a[0] ^ a[1]) as usize % inputs.len()])
                            }
                            "dx.op.binary.i32" => a[0].min(a[1]),
                            "dx.op.cbufferLoadLegacy.i32" => u64::from(
                                inputs[a[0] as usize % inputs.len()].rotate_left(a[1] as u32 & 31),
                            ),
                            "dx.op.bufferLoad.i32" => memory[(a[0] ^ a[1]) as usize % 16],
                            "dx.op.waveReadLaneFirst.i32" => a[0] ^ a[1],
                            _ => {
                                memory[a[0] as usize % 16] = a[1];
                                trace.push((name.to_owned(), a));
                                0
                            }
                        }
                    }
                    _ => panic!("an instruction the generator does not make"),
                };
                vals[i as usize] = Some(r);
            }
            from = b;
            b = match blk.term.expect("a terminator") {
                Term::Ret => {
                    trace.push(("ret".into(), Vec::new()));
                    return trace;
                }
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if eval(&vals, c) != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        trace.push(("still running".into(), Vec::new()));
        trace
    }

    /// A random structured program over a few registers, as the lowering
    /// makes them: every register joined at every join, conditions tested
    /// again behind their branch, values computed again.
    struct Gen {
        k: Kit,
        rng: u64,
        regs: Vec<V>,
        conds: Vec<V>,
        cur: usize,
        input: V,
        table: V,
        wave: V,
        /// The operands of the block's last read (to read the place again,
        /// and to store to it).
        last: Option<(V, V)>,
    }

    impl Gen {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            (self.rng >> 11) % n
        }

        fn reg(&mut self) -> V {
            let n = self.next(self.regs.len() as u64) as usize;
            self.regs[n]
        }

        /// A register or a small constant.
        fn value(&mut self) -> V {
            match self.next(4) {
                0 => {
                    let k = [0u32, 1, 2, 31, u32::MAX, 0x8000_0000, 7][self.next(7) as usize];
                    self.k.m.i32(k)
                }
                _ => self.reg(),
            }
        }

        fn cond(&mut self) -> V {
            let c = self.cur;
            let made = match self.next(6) {
                0 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    return self.conds[n];
                }
                1 if self.conds.len() > 1 => {
                    let (a, b) = (
                        self.next(self.conds.len() as u64),
                        self.next(self.conds.len() as u64),
                    );
                    let op = [Bin::And, Bin::Or, Bin::Xor][self.next(3) as usize];
                    self.k
                        .m
                        .bin(c, op, self.conds[a as usize], self.conds[b as usize])
                }
                2 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    let t = self.k.m.bool(true);
                    self.k.m.bin(c, Bin::Xor, self.conds[n], t)
                }
                _ => {
                    let p = pred::EQ + self.next(10) as u8;
                    let (x, y) = (self.reg(), self.value());
                    self.k.m.cmp(c, p, x, y)
                }
            };
            self.conds.push(made);
            made
        }

        fn statement(&mut self) {
            let c = self.cur;
            let d = self.next(self.regs.len() as u64) as usize;
            let i32t = self.k.m.int(32);
            let f32t = self.k.m.float();
            let r = match self.next(14) {
                0..=3 => {
                    let op = [Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor]
                        [self.next(6) as usize];
                    let (x, y) = (self.value(), self.value());
                    self.k.m.bin(c, op, x, y)
                }
                4 => {
                    let op = [Bin::Shl, Bin::LShr, Bin::AShr][self.next(3) as usize];
                    let x = self.value();
                    let by = self.next(32) as u32;
                    let by = self.k.m.i32(by);
                    self.k.m.bin(c, op, x, by)
                }
                5 | 6 => {
                    // A guarded write: the old value where the guard fails.
                    let g = self.cond();
                    let x = self.value();
                    self.k.m.select(self.cur, g, x, self.regs[d])
                }
                7 => {
                    let (x, y) = (self.value(), self.value());
                    let f = [self.k.pure, self.input, self.table, self.wave][self.next(4) as usize];
                    self.k.m.call(c, f, vec![x, y])
                }
                8 | 9 => {
                    let again = self.next(2) == 0;
                    let (x, y) = match self.last {
                        Some(last) if again => last,
                        _ => (self.value(), self.value()),
                    };
                    self.last = Some((x, y));
                    self.k.m.call(c, self.k.reads, vec![x, y])
                }
                10 => {
                    // A store, half the time to the place read last.
                    let (mut x, y) = (self.value(), self.value());
                    let there = self.next(2) == 0;
                    if let Some((a, b)) = self.last
                        && there
                    {
                        x = self.k.m.bin(c, Bin::Xor, a, b);
                    }
                    self.k.m.call(c, self.k.writes, vec![x, y]);
                    return;
                }
                11 => {
                    // Float arithmetic, read back as bits.
                    let (x, y) = (self.value(), self.value());
                    let (x, y) = (
                        self.k.m.cast(c, Cast::Bitcast, x, f32t),
                        self.k.m.cast(c, Cast::Bitcast, y, f32t),
                    );
                    let op = [Bin::Add, Bin::Sub, Bin::Mul][self.next(3) as usize];
                    let f = self.k.m.bin(c, op, x, y);
                    if self.next(2) == 0 {
                        let p = self.next(16) as u8;
                        let t = self.k.m.cmp(c, p, f, y);
                        self.conds.push(t);
                    }
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
                12 => {
                    let g = self.cond();
                    self.k.m.cast(self.cur, Cast::ZExt, g, i32t)
                }
                _ => {
                    let x = self.value();
                    let f = self.k.m.cast(c, Cast::Bitcast, x, f32t);
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
            };
            self.regs[d] = r;
        }

        fn code(&mut self, depth: u32) {
            for _ in 0..1 + self.next(6) {
                match self.next(8) {
                    0 | 1 if depth > 0 => self.branch(depth - 1),
                    2 if depth > 0 => self.repeat(depth - 1),
                    _ => self.statement(),
                }
            }
        }

        /// if / else, every register joined.
        fn branch(&mut self, depth: u32) {
            let c = self.cond();
            let i32t = self.k.m.int(32);
            let (then, other, join) = (
                self.k.m.new_block(),
                self.k.m.new_block(),
                self.k.m.new_block(),
            );
            self.k.m.terminate(self.cur, Term::CondBr(c, then, other));
            let (before, known) = (self.regs.clone(), self.conds.len());
            self.cur = then;
            self.last = None;
            self.code(depth);
            let (then_regs, then_end) = (std::mem::replace(&mut self.regs, before), self.cur);
            self.k.m.terminate(then_end, Term::Br(join));
            self.conds.truncate(known);
            self.cur = other;
            self.last = None;
            if self.next(3) != 0 {
                self.code(depth);
            }
            let other_end = self.cur;
            self.k.m.terminate(other_end, Term::Br(join));
            self.conds.truncate(known);
            for (r, &t) in then_regs.iter().enumerate() {
                let p = self.k.m.phi(join, i32t);
                self.k
                    .m
                    .set_phi(p, vec![(t, then_end), (self.regs[r], other_end)]);
                self.regs[r] = p;
            }
            self.cur = join;
            self.last = None;
        }

        /// A loop that runs at most a few times, every register joined at
        /// its head.
        fn repeat(&mut self, depth: u32) {
            let i32t = self.k.m.int(32);
            let (head, exit) = (self.k.m.new_block(), self.k.m.new_block());
            let before = self.cur;
            self.k.m.terminate(before, Term::Br(head));
            let entering = self.regs.clone();
            let joins: Vec<V> = (0..self.regs.len())
                .map(|_| self.k.m.phi(head, i32t))
                .collect();
            let count = self.k.m.phi(head, i32t);
            self.regs = joins.clone();
            let known = self.conds.len();
            self.cur = head;
            self.last = None;
            self.code(depth);
            let one = self.k.m.i32(1);
            let next = self.k.m.bin(self.cur, Bin::Add, count, one);
            let limit = 1 + self.next(3) as u32;
            let limit = self.k.m.i32(limit);
            let mut more = self.k.m.cmp(self.cur, pred::IULT, next, limit);
            if self.next(2) == 0 {
                let c = self.cond();
                more = self.k.m.bin(self.cur, Bin::And, more, c);
            }
            let latch = self.cur;
            self.k.m.terminate(latch, Term::CondBr(more, head, exit));
            for (r, &p) in joins.iter().enumerate() {
                self.k
                    .m
                    .set_phi(p, vec![(entering[r], before), (self.regs[r], latch)]);
            }
            let zero = self.k.m.i32(0);
            self.k.m.set_phi(count, vec![(zero, before), (next, latch)]);
            self.conds.truncate(known);
            self.cur = exit;
            self.last = None;
        }
    }

    fn random_program(seed: u64) -> Module {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let input =
            k.m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        let table = k.m.function(
            "dx.op.cbufferLoadLegacy.i32",
            fty,
            Some(Attr::ReadOnly),
            false,
        );
        let wave = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let b = k.b;
        let mut g = Gen {
            k,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            regs: Vec::new(),
            conds: Vec::new(),
            cur: b,
            input,
            table,
            wave,
            last: None,
        };
        for r in 0..5 {
            let (x, y) = (g.k.m.i32(r), g.k.m.i32(r * 3));
            let v = g.k.m.call(b, input, vec![x, y]);
            g.regs.push(v);
        }
        g.code(3);
        g.code(2);
        // Every register is read at the end.
        for r in 0..g.regs.len() {
            let at = g.k.m.i32(r as u32);
            g.k.m.call(g.cur, g.k.writes, vec![at, g.regs[r]]);
        }
        g.k.m.terminate(g.cur, Term::Ret);
        g.k.m
    }

    #[test]
    fn every_pass_leaves_what_random_programs_store() {
        let mut removed = 0;
        for seed in 1..=400u64 {
            let plain = random_program(seed);
            let inputs: Vec<Vec<u32>> = (0..6u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| match n {
                            0 => 0,
                            1 => i,
                            2 => u32::MAX - i,
                            _ => {
                                (seed as u32 ^ n)
                                    .wrapping_mul(0x9E37_79B9)
                                    .rotate_left(i * 5)
                                    ^ (i << 28)
                            }
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run(&plain, i)).collect();
            let written: usize = plain
                .blocks
                .iter()
                .map(|b| b.body.len() + b.phis.len())
                .sum();
            for passes in [
                DEAD,
                FOLD,
                REUSE,
                FACTS,
                LOADS,
                FOLD | FACTS,
                FOLD | REUSE | DEAD,
                ALL,
            ] {
                let mut m = random_program(seed);
                m.optimise(passes);
                for (i, want) in inputs.iter().zip(&expected) {
                    assert_eq!(&run(&m, i), want, "seed {seed}, passes {passes}");
                }
                if passes == ALL {
                    let left: usize = m.blocks.iter().map(|b| b.body.len() + b.phis.len()).sum();
                    removed += written - left;
                }
                m.write();
            }
        }
        // The programs are of the wasteful kind: the passes find work.
        assert!(removed > 4000, "{removed} instructions removed");
    }

    #[test]
    fn rayquery_effects_and_usage_are_tracked() {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i32t = m.int(32);
        let alloc_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let proceed_ty = m.func_ty(i1, vec![i32t, i32t]);
        let state_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [alloc_ty, proceed_ty, state_ty, store_ty, main_ty] {
            m.ptr(ty);
        }
        let alloc = m.function("dx.op.allocateRayQuery", alloc_ty, Some(Attr::NoUnwind), false);
        let proceed = m.function(
            "dx.op.rayQuery_Proceed.i1",
            proceed_ty,
            Some(Attr::NoUnwind),
            false,
        );
        let state = m.function(
            "dx.op.rayQuery_StateScalar.i32",
            state_ty,
            Some(Attr::ReadOnly),
            false,
        );
        let store = m.function("dx.op.rawBufferStore.i32", store_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        let op_alloc = m.i32(178);
        let op_proceed = m.i32(180);
        let op_status = m.i32(184);
        let zero = m.i32(0);
        let rq1 = m.call(b, alloc, vec![op_alloc, zero]);
        let rq2 = m.call(b, alloc, vec![op_alloc, zero]);
        let s_before = m.call(b, state, vec![op_status, rq1]);
        let s_same = m.call(b, state, vec![op_status, rq1]);
        let _p = m.call(b, proceed, vec![op_proceed, rq1]);
        let s_after = m.call(b, state, vec![op_status, rq1]);
        let _dead_state = m.call(b, state, vec![op_status, rq2]);
        let sum = m.bin(b, Bin::Add, s_before, s_same);
        m.call(b, store, vec![sum, s_after]);
        m.terminate(b, Term::Ret);
        m.optimise(MODULE);
        assert!(m.usage().raytracing_tier_1_1);
        // Two distinct allocateRayQuery calls, one proceed, two kept state reads
        // (before and after proceed; s_same reused s_before, _dead_state dropped),
        // one add, and one store = 7 instructions.
        assert_eq!(body(&m, b), 7);
        let s = stores(&m, b);
        let V::Inst(add_i) = s[0][0] else { panic!() };
        assert!(
            matches!(m.insts[add_i as usize].op, Op::Bin(Bin::Add, x, y) if x == s_before && y == s_before),
            "identical state reads before proceed merge"
        );
        assert_ne!(s[0][1], s_before, "proceed invalidates cached state reads");
        m.write();
    }

    #[test]
    fn algebraic_reassociation_and_shift_masking_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (k0, k1, k3, k4, k5, k7, k8, k16, k20, k24, k31, k32) = (
            m.i32(0),
            m.i32(1),
            m.i32(3),
            m.i32(4),
            m.i32(5),
            m.i32(7),
            m.i32(8),
            m.i32(16),
            m.i32(20),
            m.i32(24),
            m.i32(31),
            m.i32(32),
        );
        let (kff, kffff, kall) = (m.i32(0xFF), m.i32(0xFFFF), m.i32(u32::MAX));
        let x = m.call(b, reads, vec![k1, k1]);
        let y = m.call(b, reads, vec![k3, k3]);

        // Self-cancellation and complement identities.
        let xxor = m.bin(b, Bin::Xor, x, x);
        let xsub = m.bin(b, Bin::Sub, x, x);
        let notx = m.bin(b, Bin::Xor, x, kall);
        let and_not = m.bin(b, Bin::And, x, notx);
        let or_not = m.bin(b, Bin::Or, x, notx);
        let xor_cancel = m.bin(b, Bin::Xor, x, y);
        let xor_cancel = m.bin(b, Bin::Xor, xor_cancel, x);
        let or_xy = m.bin(b, Bin::Or, x, y);
        let absorb = m.bin(b, Bin::And, x, or_xy);

        // Constant reassociation: (x + 5) - 5 -> x, (x + 5) + 7 -> x + 12,
        // (x - 3) + 7 -> x + 4, (x & 0xFF) & 0xFFFF -> x & 0xFF.
        let xp5 = m.bin(b, Bin::Add, x, k5);
        let back_x = m.bin(b, Bin::Sub, xp5, k5);
        let xp12 = m.bin(b, Bin::Add, xp5, k7);
        let xm3 = m.bin(b, Bin::Sub, x, k3);
        let xp4 = m.bin(b, Bin::Add, xm3, k7);
        let mask_ff = m.bin(b, Bin::And, x, kff);
        let red_mask = m.bin(b, Bin::And, mask_ff, kffff);

        // Shift reassociation, shift-pair masking, and redundant mask before shl.
        let sh7 = m.bin(b, Bin::Shl, x, k3);
        let sh7 = m.bin(b, Bin::Shl, sh7, k4);
        let lshr_zero = m.bin(b, Bin::LShr, x, k20);
        let lshr_zero = m.bin(b, Bin::LShr, lshr_zero, k20);
        let ashr_sign = m.bin(b, Bin::AShr, x, k20);
        let ashr_sign = m.bin(b, Bin::AShr, ashr_sign, k20);
        // UB shift >= width must NOT fold.
        let ub_sh = m.bin(b, Bin::Shl, x, k16);
        let ub_sh = m.bin(b, Bin::Shl, ub_sh, k32);
        // (x << 8) >>u 8 -> x & 0x00FF_FFFF; (x >>u 24) & 0xFF -> x >>u 24.
        let sh_pair = m.bin(b, Bin::Shl, x, k8);
        let sh_pair = m.bin(b, Bin::LShr, sh_pair, k8);
        let top_byte = m.bin(b, Bin::LShr, x, k24);
        let top_byte_m = m.bin(b, Bin::And, top_byte, kff);
        // Packed half: ((x & 0xFFFF) | ((y & 0xFFFF) << 16)) >>u 16 -> y & 0xFFFF,
        // and low half & 0xFFFF -> x & 0xFFFF.
        let lo16 = m.bin(b, Bin::And, x, kffff);
        let hi16_m = m.bin(b, Bin::And, y, kffff);
        let hi16 = m.bin(b, Bin::Shl, hi16_m, k16);
        let packed = m.bin(b, Bin::Or, lo16, hi16);
        let unp_lo = m.bin(b, Bin::And, packed, kffff);
        let unp_hi = m.bin(b, Bin::LShr, packed, k16);
        // Signed top-half bfe: (((x >>s 16) & 0xFFFF) << 16) >>s 16 -> x >>s 16.
        let ashr16 = m.bin(b, Bin::AShr, x, k16);
        let ashr16_m = m.bin(b, Bin::And, ashr16, kffff);
        let ashr16_shl = m.bin(b, Bin::Shl, ashr16_m, k16);
        let ashr16_back = m.bin(b, Bin::AShr, ashr16_shl, k16);

        // Known-zero unsigned range folding (bfe `width & 31 >= 32` is false).
        let w5 = m.bin(b, Bin::And, x, k31);
        let full = m.cmp(b, pred::IUGE, w5, k32);
        let sel_full = m.select(b, full, kall, w5);

        m.call(b, writes, vec![xxor, xsub]);
        m.call(b, writes, vec![and_not, or_not]);
        m.call(b, writes, vec![xor_cancel, absorb]);
        m.call(b, writes, vec![back_x, red_mask]);
        m.call(b, writes, vec![xp12, xp4]);
        m.call(b, writes, vec![sh7, lshr_zero]);
        m.call(b, writes, vec![ashr_sign, ub_sh]);
        m.call(b, writes, vec![sh_pair, top_byte_m]);
        m.call(b, writes, vec![unp_lo, unp_hi]);
        m.call(b, writes, vec![ashr16_back, sel_full]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | REUSE | DEAD);

        let st = stores(&m, b);
        assert_eq!(st[0], [k0, k0], "x ^ x and x - x fold to 0");
        assert_eq!(st[1], [k0, kall], "x & ~x = 0 and x | ~x = -1");
        assert_eq!(st[2], [y, x], "x ^ (x ^ y) = y and x & (x | y) = x");
        assert_eq!(st[3], [x, mask_ff], "(x + 5) - 5 = x and redundant mask dropped");
        let V::Inst(xp12_i) = st[4][0] else { panic!() };
        assert!(matches!(m.insts[xp12_i as usize].op, Op::Bin(Bin::Add, a, k) if a == x && m.int_k(k) == Some((32, 12))));
        let V::Inst(xp4_i) = st[4][1] else { panic!() };
        assert!(matches!(m.insts[xp4_i as usize].op, Op::Bin(Bin::Add, a, k) if a == x && m.int_k(k) == Some((32, 4))));
        let V::Inst(sh7_i) = st[5][0] else { panic!() };
        assert!(matches!(m.insts[sh7_i as usize].op, Op::Bin(Bin::Shl, a, k) if a == x && m.int_k(k) == Some((32, 7))));
        assert_eq!(st[5][1], k0, "(x >>u 20) >>u 20 folds to 0");
        let V::Inst(ashr_i) = st[6][0] else { panic!() };
        assert!(matches!(m.insts[ashr_i as usize].op, Op::Bin(Bin::AShr, a, k) if a == x && m.int_k(k) == Some((32, 31))));
        assert_eq!(st[6][1], ub_sh, "UB shift by >= width stays intact");
        let V::Inst(pair_i) = st[7][0] else { panic!() };
        assert!(matches!(m.insts[pair_i as usize].op, Op::Bin(Bin::And, a, k) if a == x && m.int_k(k) == Some((32, 0x00FF_FFFF))));
        assert_eq!(st[7][1], top_byte, "(x >>u 24) & 0xFF drops redundant mask");
        assert_eq!(st[8][0], lo16, "packed low half extracts cleanly");
        let V::Inst(unp_hi_i) = st[8][1] else { panic!() };
        assert!(matches!(m.insts[unp_hi_i as usize].op, Op::Bin(Bin::And, a, k) if a == y && m.int_k(k) == Some((32, 0xFFFF))));
        assert_eq!(st[9][0], ashr16, "signed top-half bfe collapses to x >>s 16");
        assert_eq!(st[9][1], w5, "(x & 31) >=u 32 folds to false");
        m.write();
    }

    #[test]
    fn cmp_select_bitcast_and_cast_chains_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i1t, i8t, i16t, i32t, i64t, f32t) =
            (m.int(1), m.int(8), m.int(16), m.int(32), m.int(64), m.float());
        let (k0, k1, k2, k3, k5, k10, yes, no) = (
            m.i32(0),
            m.i32(1),
            m.i32(2),
            m.i32(3),
            m.i32(5),
            m.i32(10),
            m.bool(true),
            m.bool(false),
        );
        let x = m.call(b, reads, vec![k1, k1]);
        let y = m.call(b, reads, vec![k2, k2]);
        let c = m.cmp(b, pred::IULT, x, y);
        let d = m.cmp(b, pred::NE, x, k0);

        // icmp on zext(i1) and select(c, k1, k2).
        let zc = m.cast(b, Cast::ZExt, c, i32t);
        let eq1 = m.cmp(b, pred::EQ, zc, k1);
        let eq0 = m.cmp(b, pred::EQ, zc, k0);
        let sel01 = m.select(b, c, k1, k0);
        let sel_ne0 = m.cmp(b, pred::NE, sel01, k0);
        let sel_eq0 = m.cmp(b, pred::EQ, sel01, k0);
        let sel_const = m.select(b, c, k5, k5);
        let sel_const_eq = m.cmp(b, pred::EQ, sel_const, k5);
        let r1 = m.select(b, eq1, x, y);
        let r2 = m.select(b, eq0, x, y);
        let r3 = m.select(b, sel_ne0, x, y);
        let r4 = m.select(b, sel_eq0, x, y);
        let r5 = m.select(b, sel_const_eq, x, y);

        // (x + 3) == 10 -> x == 7, but (x + 3) <u 10 MUST NOT fold (wrap-around).
        let xp3 = m.bin(b, Bin::Add, x, k3);
        let eq10 = m.cmp(b, pred::EQ, xp3, k10);
        let lt10 = m.cmp(b, pred::IULT, xp3, k10);
        let r6 = m.select(b, eq10, x, y);
        let r7 = m.select(b, lt10, x, y);

        // i1 selects: select(c, true, d) -> c | d, select(c, d, false) -> c & d.
        let sor = m.select(b, c, yes, d);
        let sand = m.select(b, c, d, no);
        let r8 = m.select(b, sor, x, y);
        let r9 = m.select(b, sand, x, y);

        // Bitcast through select: bitcast(select(c, bitcast(f1), 1.0_bits)) -> select(c, f1, 1.0).
        let f1 = m.cast(b, Cast::Bitcast, x, f32t);
        let bf1 = m.cast(b, Cast::Bitcast, f1, i32t);
        let one_bits = m.i32(1.0f32.to_bits());
        let sel_i = m.select(b, c, bf1, one_bits);
        let sel_f = m.cast(b, Cast::Bitcast, sel_i, f32t);
        let sel_f_back = m.cast(b, Cast::Bitcast, sel_f, i32t);

        // Cast chains: sext(zext(i8 -> i16) -> i32) -> zext(i8 -> i32),
        // trunc(zext(i1 -> i64) -> i32) -> zext(i1 -> i32).
        let x8 = m.cast(b, Cast::Trunc, x, i8t);
        let z16 = m.cast(b, Cast::ZExt, x8, i16t);
        let sz32 = m.cast(b, Cast::SExt, z16, i32t);
        let c64 = m.cast(b, Cast::ZExt, c, i64t);
        let c32 = m.cast(b, Cast::Trunc, c64, i32t);
        let _ = i1t;

        m.call(b, writes, vec![r1, r2]);
        m.call(b, writes, vec![r3, r4]);
        m.call(b, writes, vec![r5, r6]);
        m.call(b, writes, vec![r7, r8]);
        m.call(b, writes, vec![r9, sel_f_back]);
        m.call(b, writes, vec![sz32, c32]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | REUSE | DEAD);

        let st = stores(&m, b);
        assert_eq!(st[0][0], st[1][0], "eq1 and sel_ne0 both reduce to select(c, x, y)");
        assert_eq!(st[0][1], st[1][1], "eq0 and sel_eq0 both invert to select(c, y, x)");
        let V::Inst(r2_i) = st[0][1] else { panic!() };
        assert!(matches!(m.insts[r2_i as usize].op, Op::Select(cc, t, f) if cc == c && t == y && f == x));
        assert_eq!(st[2][0], x, "select(c, 5, 5) == 5 folds to true");
        let V::Inst(r6_i) = st[2][1] else { panic!() };
        let Op::Select(V::Inst(eq7_i), _, _) = m.insts[r6_i as usize].op else { panic!() };
        assert!(matches!(m.insts[eq7_i as usize].op, Op::Cmp(pred::EQ, a, k) if a == x && m.int_k(k) == Some((32, 7))));
        let V::Inst(r7_i) = st[3][0] else { panic!() };
        let Op::Select(V::Inst(lt_i), _, _) = m.insts[r7_i as usize].op else { panic!() };
        assert!(matches!(m.insts[lt_i as usize].op, Op::Cmp(pred::IULT, a, k) if a == xp3 && k == k10));
        let V::Inst(r8_i) = st[3][1] else { panic!() };
        let Op::Select(V::Inst(or_i), _, _) = m.insts[r8_i as usize].op else { panic!() };
        assert!(matches!(m.insts[or_i as usize].op, Op::Bin(Bin::Or, a, z) if a == c && z == d));
        let V::Inst(r9_i) = st[4][0] else { panic!() };
        let Op::Select(V::Inst(and_i), _, _) = m.insts[r9_i as usize].op else { panic!() };
        assert!(matches!(m.insts[and_i as usize].op, Op::Bin(Bin::And, a, z) if a == c && z == d));
        let V::Inst(sf_i) = st[4][1] else { panic!() };
        assert!(matches!(m.insts[sf_i as usize].op, Op::Select(cc, t, f) if cc == c && t == x && f == one_bits));
        let V::Inst(sz_i) = st[5][0] else { panic!() };
        assert!(matches!(m.insts[sz_i as usize].op, Op::Cast(Cast::ZExt, z) if z == x8));
        let V::Inst(c32_i) = st[5][1] else { panic!() };
        assert!(matches!(m.insts[c32_i as usize].op, Op::Cast(Cast::ZExt, z) if z == c));
        m.write();
    }

    #[test]
    fn pure_integer_dxop_calls_fold_and_cse() {
        let mut k = kit();
        let (i32t, f32t) = (k.m.int(32), k.m.float());
        let u_ty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let b_ty = k.m.func_ty(i32t, vec![i32t, i32t, i32t]);
        let fb_ty = k.m.func_ty(f32t, vec![i32t, f32t, f32t]);
        for ty in [u_ty, b_ty, fb_ty] {
            k.m.ptr(ty);
        }
        let unary_i = k.m.function("dx.op.unary.i32", u_ty, Some(Attr::ReadNone), false);
        let ubits = k.m.function("dx.op.unaryBits.i32", u_ty, Some(Attr::ReadNone), false);
        let bin_i = k.m.function("dx.op.binary.i32", b_ty, Some(Attr::ReadNone), false);
        let bin_f = k.m.function("dx.op.binary.f32", fb_ty, Some(Attr::ReadNone), false);
        let b = k.b;
        let (op30, op31, op32, op33, op34, op36, op37, op40) = (
            k.m.i32(30),
            k.m.i32(31),
            k.m.i32(32),
            k.m.i32(33),
            k.m.i32(34),
            k.m.i32(36),
            k.m.i32(37),
            k.m.i32(40),
        );
        let (k0, k1, k15, k50, k100, k2047) =
            (k.m.i32(0), k.m.i32(1), k.m.i32(15), k.m.i32(50), k.m.i32(100), k.m.i32(2047));
        let x = k.m.call(b, k.reads, vec![k1, k1]);
        let y = k.m.call(b, k.reads, vec![k15, k15]);

        // Constant bit ops + Bfrev(Bfrev(x)) -> x.
        let rev_k = k.m.i32(0x1234_5678);
        let rev_const = k.m.call(b, unary_i, vec![op30, rev_k]);
        let rev1 = k.m.call(b, unary_i, vec![op30, x]);
        let rev2 = k.m.call(b, unary_i, vec![op30, rev1]);
        let pop_arg = k.m.i32(0xF0F0_0001);
        let pop = k.m.call(b, ubits, vec![op31, pop_arg]);
        let fbl0 = k.m.call(b, ubits, vec![op32, k0]);
        let fbh_msb = k.m.i32(0x8000_0000);
        let fbh = k.m.call(b, ubits, vec![op33, fbh_msb]);
        let fbsh = k.m.call(b, ubits, vec![op34, fbh_msb]);

        // UMin/IMax identities, nested clamp, known-zero clamp elimination, commutative CSE.
        let masked = k.m.bin(b, Bin::And, x, k15);
        let clamp_noop = k.m.call(b, bin_i, vec![op40, masked, k2047]);
        let clamp1 = k.m.call(b, bin_i, vec![op40, x, k100]);
        let clamp2 = k.m.call(b, bin_i, vec![op40, clamp1, k50]);
        let umin_xy = k.m.call(b, bin_i, vec![op40, x, y]);
        let umin_yx = k.m.call(b, bin_i, vec![op40, y, x]);
        let imax_same = k.m.call(b, bin_i, vec![op37, x, x]);

        // Float FMin on constants must NOT fold.
        let f0 = k.m.f32(0);
        let fm0 = k.m.f32(0x8000_0000);
        let fmin = k.m.call(b, bin_f, vec![op36, f0, fm0]);
        let fmin_i = k.m.cast(b, Cast::Bitcast, fmin, i32t);

        k.m.call(b, k.writes, vec![rev_const, rev2]);
        k.m.call(b, k.writes, vec![pop, fbl0]);
        k.m.call(b, k.writes, vec![fbh, fbsh]);
        k.m.call(b, k.writes, vec![clamp_noop, clamp2]);
        k.m.call(b, k.writes, vec![umin_xy, umin_yx]);
        k.m.call(b, k.writes, vec![imax_same, fmin_i]);
        k.m.terminate(b, Term::Ret);
        k.m.optimise(FOLD | REUSE | DEAD);

        let st = stores(&k.m, b);
        let rev_want = k.m.i32(0x1234_5678u32.reverse_bits());
        assert_eq!(st[0], [rev_want, x]);
        let (k9, km1) = (k.m.i32(9), k.m.i32(u32::MAX));
        assert_eq!(st[1], [k9, km1]);
        assert_eq!(st[2], [k0, k1], "FirstbitHi(0x80000000)=0, FirstbitSHi(0x80000000)=1");
        assert_eq!(st[3][0], masked, "UMin(x & 15, 2047) drops clamp");
        let V::Inst(c2_i) = st[3][1] else { panic!() };
        assert!(matches!(&k.m.insts[c2_i as usize].op, Op::Call(_, a) if a[1] == x && a[2] == k50));
        assert_eq!(st[4][0], st[4][1], "UMin(x, y) and UMin(y, x) CSE");
        assert_eq!(st[5][0], x, "IMax(x, x) -> x");
        assert_eq!(st[5][1], fmin_i, "float FMin is left alone");
        k.m.write();
    }

    #[test]
    fn facts_propagate_integer_equality_and_cfg_cleanup() {
        let mut k = kit();
        let (i32t, f32t) = (k.m.int(32), k.m.float());
        let (k1, k3, k7, k10, yes) = (k.m.i32(1), k.m.i32(3), k.m.i32(7), k.m.i32(10), k.m.bool(true));
        let x = k.m.call(k.b, k.reads, vec![k1, k1]);
        let eq7 = k.m.cmp(k.b, pred::EQ, x, k7);
        let not_eq7 = k.m.bin(k.b, Bin::Xor, eq7, yes);
        let (then, other, join) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        // Branch on !eq7 -> flips to CondBr(eq7, then, other), and learns x == 7 in then.
        k.m.terminate(k.b, Term::CondBr(not_eq7, other, then));
        let sum = k.m.bin(then, Bin::Add, x, k3);
        k.m.call(then, k.writes, vec![sum, x]);
        // Identical targets CondBr(c, join, join) collapses to Br(join) and deduplicates phi.
        let dummy = k.m.cmp(then, pred::IUGT, x, k3);
        k.m.terminate(then, Term::CondBr(dummy, join, join));
        k.m.call(other, k.writes, vec![x, x]);
        k.m.terminate(other, Term::Br(join));
        let p = k.m.phi(join, i32t);
        k.m.set_phi(p, vec![(sum, then), (sum, then), (x, other)]);
        k.m.call(join, k.writes, vec![p, x]);
        k.m.terminate(join, Term::Ret);
        k.m.optimise(ALL);

        assert!(matches!(k.m.blocks[k.b].term, Some(Term::CondBr(c, t, f)) if c == eq7 && t == then && f == other));
        assert_eq!(stores(&k.m, then), [[k10, k7]], "x == 7 propagated in then");
        assert!(matches!(k.m.blocks[then].term, Some(Term::Br(t)) if t == join));
        assert_eq!(stores(&k.m, other), [[x, x]], "x unchanged in other");
        let Op::Phi(inc) = &k.m.insts[k.m.blocks[join].phis[0] as usize].op else { panic!() };
        assert_eq!(inc, &vec![(k10, then), (x, other)]);

        // Float OEQ(f, +0.0) must NOT substitute f with +0.0 (-0.0 has different sign bit).
        let mut k = kit();
        let f = k.m.call(k.b, k.reads, vec![k1, k1]);
        let ff = k.m.cast(k.b, Cast::Bitcast, f, f32t);
        let fp0 = k.m.f32(0);
        let feq = k.m.cmp(k.b, pred::OEQ, ff, fp0);
        let (t, o) = (k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(feq, t, o));
        let fb = k.m.cast(t, Cast::Bitcast, ff, i32t);
        k.m.call(t, k.writes, vec![fb, fb]);
        k.m.terminate(t, Term::Ret);
        k.m.terminate(o, Term::Ret);
        k.m.optimise(ALL);
        assert_eq!(stores(&k.m, t), [[f, f]], "float equality does not replace bits");
        k.m.write();
    }

    #[test]
    fn wave_lane_index_is_pure_and_single_pred_loads_propagate() {
        let mut k = kit();
        let (i1t, i32t) = (k.m.int(1), k.m.int(32));
        let lane_ty = k.m.func_ty(i32t, vec![i32t]);
        let helper_ty = k.m.func_ty(i1t, vec![i32t]);
        let wfirst_ty = k.m.func_ty(i32t, vec![i32t, i32t]);
        for ty in [lane_ty, helper_ty, wfirst_ty] {
            k.m.ptr(ty);
        }
        let lane_fn = k.m.function("dx.op.waveGetLaneIndex", lane_ty, Some(Attr::ReadOnly), false);
        let helper_fn = k.m.function("dx.op.isHelperLane.i1", helper_ty, Some(Attr::ReadOnly), false);
        let wfirst_fn = k.m.function("dx.op.waveReadLaneFirst.i32", wfirst_ty, Some(Attr::NoUnwind), false);
        let (op111, op118, op221, k1, k2) = (k.m.i32(111), k.m.i32(118), k.m.i32(221), k.m.i32(1), k.m.i32(2));

        let l1 = k.m.call(k.b, lane_fn, vec![op111]);
        let h1 = k.m.call(k.b, helper_fn, vec![op221]);
        let h1i = k.m.cast(k.b, Cast::ZExt, h1, i32t);
        k.m.call(k.b, k.writes, vec![l1, h1i]);
        // After the store: waveGetLaneIndex is Pure (reused), isHelperLane is Reads (not reused).
        let l2 = k.m.call(k.b, lane_fn, vec![op111]);
        let h2 = k.m.call(k.b, helper_fn, vec![op221]);
        let h2i = k.m.cast(k.b, Cast::ZExt, h2, i32t);
        let r1 = k.m.call(k.b, k.reads, vec![k1, k2]);
        let w1 = k.m.call(k.b, wfirst_fn, vec![op118, r1]);
        let cond = k.m.cmp(k.b, pred::NE, r1, k1);
        let (then, other) = (k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(cond, then, other));
        // Across CondBr into single-pred `then`: Reads propagates, Lanes does NOT.
        let r2 = k.m.call(then, k.reads, vec![k1, k2]);
        let w2 = k.m.call(then, wfirst_fn, vec![op118, r2]);
        k.m.call(then, k.writes, vec![l2, h2i]);
        k.m.call(then, k.writes, vec![r2, w2]);
        k.m.terminate(then, Term::Ret);
        k.m.call(other, k.writes, vec![w1, l2]);
        k.m.terminate(other, Term::Ret);
        k.m.optimise(REUSE | LOADS | DEAD);

        let st = stores(&k.m, then);
        assert_eq!(st[0][0], l1, "waveGetLaneIndex reused across store and branch");
        assert_ne!(st[0][1], h1i, "isHelperLane NOT reused across store/discard");
        assert_eq!(st[1][0], r1, "bufferLoad reused across single-pred CondBr without store");
        assert_ne!(st[1][1], w1, "waveReadLaneFirst NOT reused across CondBr");
        k.m.write();
    }
}
