//! The pass that uploads textures picked by index, on a thread of its own
//! (`VICTORY_HEAP_WORKER=1`; off by default; needs `VICTORY_LATE_WAIT`).
//!
//! [`crate::draw::State::heap_textures`] runs on whichever thread of the
//! title records the frame's first shader that picks textures by index: it
//! asks what of each due texture is mapped, makes the host texture, makes a
//! staging buffer and records the copy, up to 250 ms of every frame while a
//! backlog waits, and the threads waiting for that list wait with it. The
//! detiling of what it recorded then runs on the thread that submits. With
//! the switch the pass runs here, and the recording thread only wakes it.
//!
//! What keeps the picture as it was:
//! - the uploads are recorded into host lists of ours ([`Batch`]), which the
//!   next submission of the title's runs ahead of its own lists
//!   ([`take_ready`]): texture states are settled by submission order, as
//!   between any two lists;
//! - a texture's host slot is given its view at that submission, under the
//!   submission lock and ahead of `make_waiting` / `dxr::record_submission`,
//!   not before ([`ReadyBatch::publish`]): until then shaders see what the
//!   slot showed, never a texture without its contents. A slot the title
//!   wrote again meanwhile is left to the write's own turn in the queue;
//! - the detiling done here is checked at submission as any texture's is:
//!   memory the title wrote in between is read again;
//! - a title texture the pass works on is held ([`crate::resource::hold`]),
//!   and so is the title's heap;
//! - the uploads before the title's decoration bake, which must be there
//!   for that draw, still run on the recording thread; the worker makes way
//!   for them ([`urgent`]).
// shortcut: what a submission shows for a texture is up to a frame later
// than when the recording thread did it all; with a backlog it is earlier,
// since the pass has the whole frame instead of a slice of it.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

use crate::descriptor::Heap;
use crate::host::HostList;
use crate::object::This;
use crate::texture::Staging;

/// Whether the pass runs on its own thread now.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_WORKER").is_some_and(|v| v == "1")) && crate::host::late()
}

/// For the frame line, since the last one: nanoseconds the worker was at
/// work, of them detiling, lists given to submissions, textures published,
/// slots the title had written again, and textures in those lists whose
/// detiling was not done here.
pub(crate) static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// One upload list holds at most this much texture memory, or this many
/// milliseconds of work, so newly streamed mips flush promptly on scene cuts
/// without fragmenting into single-texture command lists.
const LIST_BYTES: u64 = 16 << 20;
const LIST_MS: u128 = 12;

/// Minimum textures in a flushed batch to fan out CPU detiling across the
/// dedicated background detile pool.
const PARALLEL_DETILE_MIN: usize = 8;

#[inline]
fn should_flush(bytes: u64, elapsed_ms: u128) -> bool {
    bytes >= LIST_BYTES || elapsed_ms >= LIST_MS
}

/// Bounded helper thread count for background CPU detiling (`victory-detile-*`),
/// sized to avoid oversubscribing the CPU or contending with `victory-submit`.
fn detile_helper_count(logical_cpus: usize) -> usize {
    if logical_cpus <= 2 {
        0
    } else {
        (logical_cpus / 4).clamp(1, 3)
    }
}

struct DetileWork {
    work: *const (dyn Fn(usize) + Sync),
    count: usize,
    next: AtomicUsize,
    running: AtomicUsize,
    panicked: AtomicBool,
}

// SAFETY: `work` is `Sync`, and `DetilePool::run` removes the slot and waits
// until `running == 0` before the caller's stack frame returns.
unsafe impl Send for DetileWork {}
unsafe impl Sync for DetileWork {}

struct DetileState {
    active: Vec<*const DetileWork>,
}

// SAFETY: raw pointers in `active` are only dereferenced while `running`
// pins their owning stack frame.
unsafe impl Send for DetileState {}

/// Persistent low-priority helper pool for burst CPU texture detiling so
/// `heap_worker` never spawns threads per burst and never steals workers from
/// `victory-submit`'s `WidePool`.
struct DetilePool {
    state: Mutex<DetileState>,
    more: Condvar,
    idle: Condvar,
    helpers: usize,
}

impl DetilePool {
    fn new(helpers: usize) -> &'static DetilePool {
        let pool: &'static DetilePool = Box::leak(Box::new(DetilePool {
            state: Mutex::new(DetileState { active: Vec::with_capacity(2) }),
            more: Condvar::new(),
            idle: Condvar::new(),
            helpers,
        }));
        for i in 0..helpers {
            let _ = std::thread::Builder::new()
                .name(format!("victory-detile-{i}"))
                .spawn(move || {
                    // Below-normal priority so background detiling never preempts
                    // `victory-submit`, `victory-wide-*`, or title threads.
                    use windows_sys::Win32::System::Threading::{
                        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
                    };
                    // SAFETY: sets priority of this helper thread only.
                    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
                    pool.help();
                });
        }
        pool
    }

    fn help(&self) {
        loop {
            let slot_ptr = {
                let mut state = lock(&self.state);
                loop {
                    let found = state.active.iter().rev().copied().find(|&ptr| {
                        // SAFETY: `ptr` remains live while listed in `state.active`.
                        unsafe { (*ptr).next.load(Relaxed) < (*ptr).count }
                    });
                    if let Some(ptr) = found {
                        // SAFETY: incrementing `running` under `state` pins `ptr` after removal.
                        unsafe { (*ptr).running.fetch_add(1, std::sync::atomic::Ordering::AcqRel) };
                        break ptr;
                    }
                    state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            };
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: pinned by `running` increment above.
                let slot = unsafe { &*slot_ptr };
                loop {
                    let i = slot.next.fetch_add(1, Relaxed);
                    if i >= slot.count {
                        break;
                    }
                    // SAFETY: `work` is valid until `running` drops to 0.
                    unsafe { (*slot.work)(i) };
                }
            }));
            if ran.is_err() {
                // SAFETY: still pinned by `running`.
                unsafe { (*slot_ptr).panicked.store(true, std::sync::atomic::Ordering::Release) };
            }
            // SAFETY: last dereference of `slot_ptr` before releasing the pin.
            let was = unsafe { (*slot_ptr).running.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) };
            if was == 1 {
                let _state = lock(&self.state);
                self.idle.notify_all();
            }
        }
    }

    fn run(&self, n: usize, work: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.helpers == 0 {
            for i in 0..n {
                work(i);
            }
            return;
        }
        // SAFETY: lifetime erased only for this stack frame; `run` removes
        // `slot_ptr` from `state.active` and waits until `running == 0` before returning.
        let work_ptr = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(work)
        };
        let slot = DetileWork {
            work: work_ptr,
            count: n,
            next: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        };
        let slot_ptr: *const DetileWork = &slot;
        {
            let mut state = lock(&self.state);
            state.active.push(slot_ptr);
        }
        if n > self.helpers {
            self.more.notify_all();
        } else {
            (1..n).for_each(|_| self.more.notify_one());
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let i = slot.next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                work(i);
            }
        }));
        {
            let mut state = lock(&self.state);
            if let Some(pos) = state.active.iter().rposition(|&p| p == slot_ptr) {
                state.active.remove(pos);
            }
            while slot.running.load(std::sync::atomic::Ordering::Acquire) > 0 {
                state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
        if let Err(panic) = ran {
            std::panic::resume_unwind(panic);
        }
        assert!(!slot.panicked.load(std::sync::atomic::Ordering::Acquire), "a detile worker panicked");
    }
}

fn detile_pool() -> &'static DetilePool {
    static POOL: OnceLock<&'static DetilePool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        DetilePool::new(detile_helper_count(cpus))
    })
}

/// Runs `refresh_one` across `items`, fanning out to `detile_pool()` when
/// `items.len() >= min_parallel`, and returning how many items were left
/// unrefreshed (returned `false` or skipped after `should_stop()` signaled).
fn refresh_batch_with<T: Sync>(
    items: &[T],
    min_parallel: usize,
    should_stop: &(dyn Fn() -> bool + Sync),
    refresh_one: &(dyn Fn(&T) -> bool + Sync),
) -> u64 {
    if items.len() < min_parallel {
        let mut left = 0u64;
        for item in items {
            if should_stop() || !refresh_one(item) {
                left += 1;
            }
        }
        return left;
    }
    let left = AtomicU64::new(0);
    detile_pool().run(items.len(), &|i| {
        if should_stop() || !refresh_one(&items[i]) {
            left.fetch_add(1, Relaxed);
        }
    });
    left.load(Relaxed)
}

/// A view to give a host slot once its texture's upload is in the queue:
/// the title's heap, the slot, the record's handle, and the record's key as
/// it was when the upload was recorded.
pub(crate) struct Publish {
    heap: usize,
    index: u32,
    handle: u64,
    key: u64,
}

/// Holds a reference to a title heap.
struct HeldHeap(This);
// SAFETY: a reference counted title object, of which only the count and the
// parts behind locks are touched.
unsafe impl Send for HeldHeap {}

impl HeldHeap {
    fn new(heap: &Heap) -> Self {
        let this = std::ptr::from_ref(heap).cast();
        crate::object::add_ref(this);
        HeldHeap(this)
    }

    fn heap(&self) -> &Heap {
        crate::descriptor::heap(self.0)
    }
}

impl Drop for HeldHeap {
    fn drop(&mut self) {
        crate::object::release_boxed::<Heap>(self.0);
    }
}

/// A closed upload list (if any uploads were recorded), and what is published
/// when it is submitted.
struct Ready {
    list: Option<HostList>,
    publish: Vec<Publish>,
    _heap: HeldHeap,
}
// SAFETY: D3D12 objects are free-threaded; the list is closed and touched
// by nobody until it is taken out.
unsafe impl Send for Ready {}

static READY: Mutex<Vec<Ready>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a pass on the worker's thread records into: one upload list after
/// the other, made when first needed.
pub(crate) struct Batch {
    heap: This,
    list: Option<HostList>,
    staging: Vec<Arc<Staging>>,
    publish: Vec<Publish>,
    bytes: u64,
    began: std::time::Instant,
}

impl Batch {
    fn new(heap: &Heap) -> Self {
        Batch {
            heap: std::ptr::from_ref(heap).cast(),
            list: None,
            staging: Vec::new(),
            publish: Vec::new(),
            bytes: 0,
            began: std::time::Instant::now(),
        }
    }

    /// The open upload list.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn list(&mut self) -> Option<&HostList> {
        if self.list.is_none() {
            self.list = crate::list::fresh_list();
            self.began = std::time::Instant::now();
        }
        self.list.as_ref()
    }

    /// Records a texture upload onto the open upload list while capturing any
    /// deferred [`Staging`] handles for parallel burst detiling on flush.
    pub(crate) fn record(&mut self, f: impl FnOnce(&HostList)) -> bool {
        if self.list.is_none() {
            self.list = crate::list::fresh_list();
            self.began = std::time::Instant::now();
        }
        let Some(list) = self.list.as_ref() else {
            return false;
        };
        crate::texture::capture_deferred_staging(&mut self.staging, || f(list));
        true
    }

    /// Slot `index` (its record at `handle`, then with `key`) shows its
    /// texture when the open list is submitted.
    pub(crate) fn publish(&mut self, index: u32, handle: u64, key: u64) {
        self.publish.push(Publish { heap: self.heap as usize, index, handle, key });
    }

    /// `bytes` more of texture memory in the open list, which is closed
    /// and handed over when it holds enough.
    pub(crate) fn uploaded(&mut self, bytes: u64) {
        self.bytes += bytes;
        if should_flush(self.bytes, self.began.elapsed().as_millis()) {
            self.flush();
        }
    }

    /// Detiles what the open list uploads, closes it and hands it to the
    /// next submission.
    fn flush(&mut self) {
        self.bytes = 0;
        let publish = std::mem::take(&mut self.publish);
        let staging = std::mem::take(&mut self.staging);
        let list = self.list.take();
        if list.is_none() && publish.is_empty() {
            return;
        }
        if let Some(list) = &list {
            let started = std::time::Instant::now();
            let left = if staging.len() >= PARALLEL_DETILE_MIN {
                refresh_batch_with(&staging, PARALLEL_DETILE_MIN, &make_way, &|s| s.refresh_ahead())
            } else {
                list.refresh_textures_ahead()
            };
            STATS[1].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
            STATS[5].fetch_add(left, Relaxed);
            list.close();
        }
        let heap = HeldHeap::new(crate::descriptor::heap(self.heap));
        lock(&READY).push(Ready { list, publish, _heap: heap });
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Ready upload lists and deferred slot publishes taken by a submission under
/// its submission lock: the lists are prepended ahead of the title's lists,
/// and [`PendingPublish::publish`] exposes their descriptors after
/// `heaps_synced` and before `make_waiting` / `dxr::record_submission` /
/// `ExecuteCommandLists`.
pub(crate) struct PendingPublish {
    publish: Vec<Publish>,
    _heaps: Vec<HeldHeap>,
}

impl PendingPublish {
    pub(crate) fn empty() -> Self {
        PendingPublish {
            publish: Vec::new(),
            _heaps: Vec::new(),
        }
    }

    /// Publishes every slot whose texture upload is included in this batch and
    /// drains remade texture views / ready copy-queue flips for the affected
    /// heaps so newly streamed mip levels are visible on this submission.
    pub(crate) fn publish(mut self) {
        for (i, held) in self._heaps.iter().enumerate() {
            let ptr = held.0 as usize;
            if self._heaps[..i].iter().any(|prev| prev.0 as usize == ptr) {
                continue;
            }
            if let Some(host) = held.heap().host() {
                host.texture_views_remade();
                host.texture_slots_flip();
            }
        }
        let mut batch: Vec<(u32, u64, u64)> = Vec::new();
        let mut cur_heap: Option<usize> = None;
        let flush_heap = |heap_ptr: usize, batch: &mut Vec<(u32, u64, u64)>| {
            if batch.is_empty() {
                return;
            }
            if let Some(host) = crate::descriptor::heap(heap_ptr as This).host() {
                let (shown, stale) = host.publish_batch(batch);
                STATS[3].fetch_add(shown, Relaxed);
                STATS[4].fetch_add(stale, Relaxed);
            }
            batch.clear();
        };
        for p in self.publish.drain(..) {
            if cur_heap.is_some_and(|h| h != p.heap) {
                flush_heap(cur_heap.unwrap(), &mut batch);
            }
            cur_heap = Some(p.heap);
            batch.push((p.index, p.handle, p.key));
        }
        if let Some(h) = cur_heap {
            flush_heap(h, &mut batch);
        }
    }
}

/// Takes all upload lists and slot publishes ready now for the submission
/// about to be made. Called under the submission lock so no other submission
/// can observe published slots before these upload lists execute on the card.
pub(crate) fn take_ready() -> (Vec<HostList>, PendingPublish) {
    // (Asked without the lock by every submission of a run without it.)
    if JOB.get().is_none() {
        return (Vec::new(), PendingPublish::empty());
    }
    let ready = std::mem::take(&mut *lock(&READY));
    if ready.is_empty() {
        return (Vec::new(), PendingPublish::empty());
    }
    let mut lists = Vec::with_capacity(ready.len());
    let mut publish = Vec::new();
    let mut heaps = Vec::with_capacity(ready.len());
    for r in ready {
        if let Some(list) = r.list {
            lists.push(list);
        }
        publish.extend(r.publish);
        heaps.push(r._heap);
    }
    STATS[2].fetch_add(lists.len() as u64, Relaxed);
    (lists, PendingPublish { publish, _heaps: heaps })
}

/// The heap the next pass is over, and what wakes the worker.
struct Job {
    heap: Mutex<Option<HeldHeap>>,
    wake: Condvar,
}

static JOB: OnceLock<Job> = OnceLock::new();
/// Held by whoever runs the pass.
static PASS: Mutex<()> = Mutex::new(());
/// A recording thread waits for [`PASS`].
static URGENT: AtomicBool = AtomicBool::new(false);

/// A pass over `heap` is due (once a frame). False if there is no worker:
/// the caller runs the pass itself.
pub(crate) fn kick(heap: &Heap) -> bool {
    if !on() {
        return false;
    }
    static STARTED: OnceLock<bool> = OnceLock::new();
    let started = *STARTED.get_or_init(|| {
        JOB.get_or_init(|| Job { heap: Mutex::new(None), wake: Condvar::new() });
        match std::thread::Builder::new().name("victory-heap".into()).spawn(worker) {
            Ok(_) => {
                tracing::info!("textures picked by index are uploaded by a thread of their own (VICTORY_HEAP_WORKER)");
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no thread for textures picked by index");
                false
            }
        }
    });
    let Some(job) = JOB.get().filter(|_| started) else {
        return false;
    };
    *lock(&job.heap) = Some(HeldHeap::new(heap));
    job.wake.notify_one();
    true
}

/// Takes the pass from the worker for the caller, which cannot wait for a
/// later submission: what the worker has recorded so far is handed over
/// first, so the caller's list runs behind it.
pub(crate) fn urgent() -> MutexGuard<'static, ()> {
    if let Ok(pass) = PASS.try_lock() {
        return pass;
    }
    URGENT.store(true, Relaxed);
    let pass = lock(&PASS);
    URGENT.store(false, Relaxed);
    pass
}

/// Whether the worker is to stop where it is ([`urgent`]).
pub(crate) fn make_way() -> bool {
    URGENT.load(Relaxed)
}

fn worker() {
    let Some(job) = JOB.get() else {
        return;
    };
    loop {
        let heap = {
            let mut next = lock(&job.heap);
            loop {
                match next.take() {
                    Some(heap) => break heap,
                    None => next = job.wake.wait(next).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        let started = std::time::Instant::now();
        {
            let _pass = lock(&PASS);
            let heap = heap.heap();
            if let Some(host) = heap.host() {
                let mut batch = Batch::new(heap);
                let _ = crate::draw::State::heap_textures(crate::draw::Into::Ahead(&mut batch), heap, host, None);
            }
        }
        STATS[0].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        // Whoever wants the pass now has it first.
        while make_way() {
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_ready_is_empty_without_worker() {
        let (lists, publish) = take_ready();
        assert!(lists.is_empty());
        publish.publish();
    }

    #[test]
    fn urgent_signals_make_way_until_pass_released() {
        let pass = lock(&PASS);
        let handle = std::thread::spawn(|| {
            let _urgent = urgent();
            assert!(!make_way());
        });
        while !make_way() {
            std::thread::yield_now();
        }
        assert!(make_way());
        drop(pass);
        handle.join().unwrap();
        assert!(!make_way());
    }

    #[test]
    fn flush_thresholds_match_16mib_and_12ms() {
        assert_eq!(LIST_BYTES, 16 << 20);
        assert_eq!(LIST_MS, 12);
        assert!(!should_flush(0, 0));
        assert!(!should_flush(LIST_BYTES - 1, LIST_MS - 1));
        assert!(should_flush(LIST_BYTES, 0));
        assert!(should_flush(0, LIST_MS));
        assert!(should_flush(LIST_BYTES + 1, LIST_MS + 1));
    }

    #[test]
    fn detile_helper_count_stays_bounded() {
        assert_eq!(detile_helper_count(1), 0);
        assert_eq!(detile_helper_count(2), 0);
        assert_eq!(detile_helper_count(4), 1);
        assert_eq!(detile_helper_count(8), 2);
        assert_eq!(detile_helper_count(12), 3);
        assert_eq!(detile_helper_count(32), 3);
    }

    #[test]
    fn refresh_batch_with_runs_sequential_and_parallel_and_counts_left() {
        let seen_small: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
        let left_small = refresh_batch_with(&seen_small, PARALLEL_DETILE_MIN, &|| false, &|slot| {
            slot.fetch_add(1, Relaxed);
            true
        });
        assert_eq!(left_small, 0);
        assert!(seen_small.iter().all(|c| c.load(Relaxed) == 1));

        let seen_burst: Vec<(usize, AtomicUsize)> = (0..24).map(|i| (i, AtomicUsize::new(0))).collect();
        let left_burst = refresh_batch_with(&seen_burst, PARALLEL_DETILE_MIN, &|| false, &|(i, slot)| {
            slot.fetch_add(1, Relaxed);
            i % 4 != 0
        });
        assert_eq!(left_burst, 6);
        assert!(seen_burst.iter().all(|(_, c)| c.load(Relaxed) == 1));
    }

    #[test]
    fn refresh_batch_with_respects_stop_signal() {
        let ran = AtomicUsize::new(0);
        let items: Vec<usize> = (0..16).collect();
        let left = refresh_batch_with(&items, PARALLEL_DETILE_MIN, &|| true, &|_| {
            ran.fetch_add(1, Relaxed);
            true
        });
        assert_eq!(ran.load(Relaxed), 0);
        assert_eq!(left, 16);
    }
}
