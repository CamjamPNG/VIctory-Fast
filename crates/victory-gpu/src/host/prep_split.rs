//! The title's buffer copies of a submission done beside the submitting
//! thread (`VICTORY_PREP_SPLIT=1`; off by default, like the other switches).
//!
//! What it moves. A list's steps are a chain: a copy into title memory, then
//! copies read from that memory, then work of ours ([`super::Step`]). But a
//! run of `Step::Copy` steps one after another, with nothing of the batch
//! left to do between them, has nothing in it that has to happen in order:
//! each writes its own bytes into the title's memory and notes them, and no
//! step between them reads what an earlier one wrote. Those copies are done
//! on the threads the refreshes already keep ([`crate::shadow::wide_each`])
//! instead of one after the other on the submitting thread, and the
//! submitting thread goes on to the next step once they are all done.
//!
//! What it does not move, and why ([`super::Host::submit`], in order):
//!
//! - the write-watch asking: the kernel is asked after everything the title
//!   wrote before the submission, and before anything reads what it holds;
//! - the steps done on the card ([`super::card_steps`]): they decide which
//!   copies the submission leaves to the card, which the lists' steps then
//!   skip, and their plan must be made while the flights they go behind are
//!   still in the air;
//! - the refreshes of host copies: they read title memory, so every copy
//!   and every step that writes it has to be done first;
//! - the list steps themselves, apart from the copies above: a `Fill` or a
//!   `Run` may read what an earlier copy wrote and may write what a later
//!   refresh reads, so they stay in order on one thread;
//! - the snapshots, the copies in the card's memory and the records of
//!   placed heaps: each is a function of the copies and refreshes done
//!   before it;
//! - the send (`ExecuteCommandLists` and its signal): the relay's order of
//!   sends to copies in the card's memory ([`super::QUEUE_ORDER`]) is a
//!   token its holder must give back, and a mutex guard cannot be handed to
//!   another thread (a Windows SRW lock must be released by the thread that
//!   took it), so a submission's send cannot be given to a thread of its
//!   own without redesigning that order — "ask for the next submission
//!   while this one is sent" stays where it is.
//!
//! Why the copies are safe beside: a copy is done beside only when the batch
//! has nothing left to do (so its barrier, `Batch::fill` and
//! `Batch::title_written`, would do nothing), and only when it shares no
//! 64 KiB block of title memory with any copy done beside at the same time
//! — as a destination with a destination, as a destination with a source, or
//! as a source with a destination. The blocks are the write-watch's own
//! (`crate::watch`), which a copy un-guards before the kernel writes it: two
//! copies in one block done at once would fight over that guard, which is
//! why disjointness is asked at that granularity and not at the copy's.
//! Everything a copy does besides writing (the journal, the notes, what the
//! card holds, the port journal) is per range too, and no two of these
//! copies hold a range in common.
//!
//! Nothing is sent ahead of the card, no lock is held across the copies, and
//! the copies complete before the step after them: the submission is the
//! same submission, with the same bytes in title memory, in the same order
//! of publication.

/// One of the title's buffer copies: destination, source, bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Job {
    pub dst: usize,
    pub src: usize,
    pub len: usize,
}

/// The blocks of the write watch: what a copy un-guards and the kernel
/// writes.
const BLOCK: usize = 64 << 10;

/// Copies done beside one after another before the batch is looked at again.
/// The check against the copies already beside is over all of them, and a
/// run of a list's copies is dozens long, not thousands.
const MOST: usize = 64;

/// Whether a submission's title copies may be done beside the submitting
/// thread, and on how many threads (with this one among them).
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PREP_SPLIT").is_some_and(|v| v == "1"))
}

/// How many threads the copies use: half the processors, as the refreshes'
/// threads are sized, and never more than three beside this one
/// (`VICTORY_PREP_SPLIT_THREADS` overrides; `1` copies them on this thread
/// alone, as without the switch).
pub(super) fn threads(setting: Option<&str>, cpus: Option<usize>) -> usize {
    match setting {
        Some("1") => 1,
        Some("auto") | None => cpus.map_or(2, |n| (n / 2).clamp(1, 3)),
        Some(other) => other.parse::<usize>().unwrap_or(2).clamp(1, 8),
    }
}

/// [`threads`] as this run has it.
pub(super) fn beside_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        threads(
            std::env::var("VICTORY_PREP_SPLIT_THREADS").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

/// The copies of one submission waiting to be done beside this thread, in
/// the order the lists asked for them.
#[derive(Default)]
pub(super) struct Beside {
    jobs: Vec<Job>,
}

impl Beside {
    pub(super) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.jobs.len()
    }

    pub(super) fn push(&mut self, job: Job) {
        self.jobs.push(job);
    }

    /// Takes the copies, leaving nothing: the caller does them all, in this
    /// order.
    pub(super) fn take(&mut self) -> Vec<Job> {
        std::mem::take(&mut self.jobs)
    }
}

/// Whether `job` may be done beside the submitting thread now: the switch is
/// on, nothing of the batch is left to do (so the barrier before the copy
/// would do nothing), there is room beside, and no copy beside it holds a
/// block it touches.
pub(super) fn may_defer(enabled: bool, batch_idle: bool, beside: &Beside, job: Job) -> bool {
    enabled
        && batch_idle
        && job.len > 0
        && job.dst != 0
        && beside.len() < MOST
        && beside.jobs.iter().all(|other| apart(other, &job))
}

/// The blocks an address and its bytes lie in, whole.
fn blocks(at: usize, len: usize) -> (usize, usize) {
    (at & !(BLOCK - 1), (at.saturating_add(len) | (BLOCK - 1)).saturating_add(1))
}

fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Whether two copies hold no block in common, neither as destinations nor
/// as source and destination: what makes it safe to do them at once.
fn apart(a: &Job, b: &Job) -> bool {
    let (a_dst, b_dst) = (blocks(a.dst, a.len), blocks(b.dst, b.len));
    let (a_src, b_src) = (blocks(a.src, a.len), blocks(b.src, b.len));
    !overlaps(a_dst, b_dst) && !overlaps(a_dst, b_src) && !overlaps(a_src, b_dst)
}

/// Copies done beside, runs of them, and the nanoseconds they took, since
/// the last frame line ([`super::memory_summary_line`]).
#[cfg(feature = "diagnostics")]
pub(super) static STATS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(test)]
mod tests {
    use super::*;

    fn job(dst: usize, len: usize) -> Job {
        Job { dst, src: 1 << 40, len }
    }

    #[test]
    fn threads_are_half_the_processors_and_never_more_than_three_beside() {
        assert_eq!(threads(None, Some(12)), 3);
        assert_eq!(threads(None, Some(24)), 3);
        assert_eq!(threads(None, Some(4)), 2);
        assert_eq!(threads(None, Some(1)), 1);
        assert_eq!(threads(None, None), 2);
        assert_eq!(threads(Some("1"), Some(12)), 1);
        assert_eq!(threads(Some("6"), Some(12)), 6);
        assert_eq!(threads(Some("auto"), Some(8)), 3);
        assert_eq!(threads(Some("nothing"), Some(4)), 2);
    }

    #[test]
    fn nothing_is_done_beside_with_the_switch_off() {
        // The off path: no copy is deferred, so none of the extra copies'
        // planning happens and the caller's allocations are as they were.
        let mut beside = Beside::default();
        for i in 0..1_000 {
            let j = job(0x1000_0000 + i * 0x1_0000, 0x1000);
            assert!(!may_defer(false, true, &beside, j));
            let _ = &mut beside;
        }
        assert!(beside.is_empty());
    }

    #[test]
    fn a_copy_waits_when_the_batch_has_something_to_do() {
        assert!(!may_defer(true, false, &Beside::default(), job(0x1000_0000, 0x1000)));
        assert!(may_defer(true, true, &Beside::default(), job(0x1000_0000, 0x1000)));
    }

    #[test]
    fn copies_in_the_same_block_are_not_done_at_once() {
        let mut beside = Beside::default();
        beside.push(job(0x1000_0000, 0x100));
        // Same 64 KiB block: the guards are per block.
        assert!(!may_defer(true, true, &beside, job(0x1000_0800, 0x100)));
        assert!(!may_defer(true, true, &beside, job(0x1000_ffff, 0x2)));
        // The next block: apart.
        assert!(may_defer(true, true, &beside, job(0x1001_0000, 0x100)));
    }

    #[test]
    fn a_copy_whose_source_is_another_destination_waits() {
        let mut beside = Beside::default();
        beside.push(job(0x2000_0000, 0x1000));
        let reads_it = Job { dst: 0x3000_0000, src: 0x2000_0100, len: 0x100 };
        assert!(!may_defer(true, true, &beside, reads_it));
        // And the other way: the later copy writes what an earlier one reads.
        let mut beside = Beside::default();
        beside.push(Job { dst: 0x3000_0000, src: 0x2000_0100, len: 0x100 });
        assert!(!may_defer(true, true, &beside, job(0x2000_0000, 0x1000)));
    }

    #[test]
    fn no_more_than_most_are_beside_at_once() {
        let mut beside = Beside::default();
        for i in 0..MOST {
            let j = job(0x1000_0000 + i * 0x1_0000, 0x100);
            assert!(may_defer(true, true, &beside, j));
            beside.push(j);
        }
        assert!(!may_defer(true, true, &beside, job(0x1000_0000 + MOST * 0x1_0000, 0x100)));
    }

    #[test]
    fn empty_copies_are_not_deferred() {
        assert!(!may_defer(true, true, &Beside::default(), Job { dst: 0x1000, src: 0x2000, len: 0 }));
    }

    #[test]
    fn taking_the_copies_keeps_every_one_and_takes_them_all() {
        // A deferred copy still runs: what is taken is every copy pushed,
        // in the order it was pushed, and nothing is left behind.
        let mut beside = Beside::default();
        let jobs: Vec<Job> = (0..5).map(|i| job(0x1000_0000 + i * 0x1_0000, 0x2000)).collect();
        for j in jobs.iter().copied() {
            assert!(may_defer(true, true, &beside, j));
            beside.push(j);
        }
        assert_eq!(beside.take(), jobs);
        assert!(beside.is_empty());
        assert_eq!(beside.take(), Vec::new());
    }
}
