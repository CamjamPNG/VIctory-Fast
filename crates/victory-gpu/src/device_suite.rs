//! The device suite: the `#[ignore]`d tests of this crate, each in the one
//! configuration it needs, with one command per group. `docs/device-suite.md`
//! is the list a person reads before running anything.
//!
//! `#[ignore]` was doing four jobs at once, and they cannot be done in one
//! run. Some of these tests want the relay at a particular stage, some want a
//! diagnostic switch that is off by default, some want a process of their
//! own, and some want nothing at all and only print what something cost. So
//! `cargo test -p victory-gpu -- --ignored`, with no environment set, was not
//! one suite: the tests that want switches panicked on their own guards, and
//! a guard that reads like an assertion reads like eleven bugs.
//!
//! Here every ignored test is in exactly one group, a group is one
//! configuration, and a group's tests each run in a process of their own,
//! started with that configuration. `every_ignored_test_is_classified` fails
//! if a test is added, renamed, un-ignored, or moved between groups without
//! this file and its `#[ignore]` reason agreeing, so the table cannot drift
//! away from the code.
//!
//! Why the tests are re-executed instead of set up in place: the layer reads
//! its switches into process-wide statics the first time anything asks for
//! one ([`crate::host::relay_exact`], [`crate::picked::on`],
//! [`crate::host::torn::on`]), so a test cannot set them for itself after the
//! fact — it has to be started with them. `indirect::tests`'s
//! `new_blocks_under_readers` already starts `current_exe` with an
//! environment for the same reason. Separate test targets (`[[test]]` per
//! configuration) cannot see the library's own unit tests at all, and a
//! runner script would put the switches somewhere other than the test, where
//! nothing keeps them right. Re-executing keeps `cargo test` as the interface
//! and keeps the switches beside the tests they belong to.
//!
//! A group's tests are its members, and a red line names one test. A member
//! that cannot run on this machine says so and does not count as a failure:
//! without a hardware adapter the whole suite says so once and stops, which
//! is what a runner without a GPU gets.
//!
//! What this file cannot do: make a test that is run bare, outside its group,
//! report as skipped. A Rust test either passes or fails, and reporting a
//! missing configuration as a pass would be a lie. So every guard site asks
//! [`require`] instead of asserting, and a bare run fails saying which group
//! the test belongs to, which switches it wants, what this process has, and
//! the command that provides them — a configuration problem, never a bug.

use std::process::Command;

/// One configuration, and the tests that need it.
pub(crate) struct Group {
    /// This module's own path, from [`module_path`]: `victory_gpu::device_suite::check::relay`.
    module: &'static str,
    /// What a green run of the group says about the layer.
    proves: &'static str,
    /// The switches the layer must be started with, and their values.
    env: &'static [(&'static str, &'static str)],
    /// The tests in the group, by name. Names are unique in this crate.
    members: &'static [&'static str],
}

impl Group {
    /// The group's name, as its command selects it: `check::relay`.
    pub(crate) fn name(&self) -> &'static str {
        self.module.split_once("device_suite::").map_or(self.module, |(_, name)| name)
    }

    /// The command that runs this group, as a person types it.
    pub(crate) fn command(&self) -> String {
        format!("cargo nextest run -p victory-gpu --lib device_suite::{}", self.name())
    }

    /// The group's switches, written as they are set.
    fn switches(&self) -> String {
        if self.env.is_empty() {
            return "no switches".to_owned();
        }
        self.env.iter().map(|(name, value)| format!("{name}={value}")).collect::<Vec<_>>().join(" ")
    }
}

/// Every switch this suite reads, and the four that turn the machinery these
/// tests look at off from a distance: `VICTORY_GPU_TIMING` makes
/// [`crate::host`]'s `late_stage` 0 and so takes the relay out,
/// `VICTORY_KEEP_SNAPSHOTS` makes `shadow::relay` answer no,
/// `VICTORY_LATE_WAIT` chooses which stage of it a run is at, and
/// `VICTORY_SAFE_COPY=0` turns the safe copy off.
///
/// A member's environment is these removed and then its group's set, so a
/// group runs the same however the caller's shell was set up.
const SCRUBBED: &[&str] = &[
    "VICTORY_GPU_TIMING",
    "VICTORY_KEEP_SNAPSHOTS",
    "VICTORY_LATE_WAIT",
    "VICTORY_SAFE_COPY",
    "VICTORY_RELAY",
    "VICTORY_RELAY_COPY",
    "VICTORY_CARD_STEPS",
    "VICTORY_FETCH_CHECK",
    "VICTORY_UPCOPY_CHECK",
    "VICTORY_PICKED_WRITES",
    "VICTORY_PICKED_MB",
    "VICTORY_TILES_TEST_CARD",
];

/// The relay at its second stage, with the copies that go up relayed.
pub(crate) const RELAY: &[(&str, &str)] = &[("VICTORY_RELAY", "2"), ("VICTORY_RELAY_COPY", "1")];

/// [`RELAY`], and the fetch check that puts a result caught in between right.
pub(crate) const RELAY_FETCH: &[(&str, &str)] =
    &[("VICTORY_RELAY", "2"), ("VICTORY_RELAY_COPY", "1"), ("VICTORY_FETCH_CHECK", "1")];

/// [`RELAY`], and the up-copy check that names a write nobody declared.
pub(crate) const UPCOPY: &[(&str, &str)] =
    &[("VICTORY_RELAY", "2"), ("VICTORY_RELAY_COPY", "1"), ("VICTORY_UPCOPY_CHECK", "1")];

/// [`RELAY`], and the picked writes, looked at a megabyte a frame.
pub(crate) const PICKED: &[(&str, &str)] = &[
    ("VICTORY_RELAY", "2"),
    ("VICTORY_RELAY_COPY", "1"),
    ("VICTORY_PICKED_WRITES", "1"),
    ("VICTORY_PICKED_MB", "1"),
];

/// The steps a copy made on the card goes in, with the relay under them.
pub(crate) const CARD_STEPS: &[(&str, &str)] = &[("VICTORY_RELAY", "2"), ("VICTORY_CARD_STEPS", "2")];

/// Declares a check group: its configuration, what it proves, and one entry
/// point per test in it, so that a red line is one test and not a group.
macro_rules! check_group {
    ($proves:literal, $env:expr, [$($member:ident),* $(,)?]) => {
        pub(crate) static GROUP: $crate::device_suite::Group = $crate::device_suite::Group {
            module: module_path!(),
            proves: $proves,
            env: $env,
            members: &[$(stringify!($member)),*],
        };
        $(
            /// This group's configuration, in a process of its own.
            #[test]
            fn $member() {
                $crate::device_suite::dispatch(&GROUP, stringify!($member))
            }
        )*
    };
}

pub(crate) mod check {
    /// The layer's own work on a real adapter: memory, descriptors,
    /// pipelines, placement, detiling and the tiles a run on the card gets.
    pub(crate) mod card {
        check_group!(
            "the layer does what it says on a real adapter",
            &[],
            [
                gpu_detile_batched_prelude_layout_and_zero_subresources,
                host_list_budget_starts_empty_at_each_submission,
                a_mesh_pipeline_loads_from_the_library,
                depth_bounds_pipeline_loads_and_filters_draws,
                a_placed_texture_holds_its_heap,
                gpu_detile_end_to_end_and_partial_mip_upload_match_cpu,
                a_placed_texture_is_filled_ahead_of_the_lists,
                drain_waits_until_then_callbacks_have_run,
                card_wrote_failure_does_not_poison_card_mask,
            ]
        );
    }

    /// What a clamps batch charges, on the software adapter. Its own group
    /// because it does not want the card: a runner's WARP loses its virtual
    /// device part way through, which is why it is ignored for CI.
    pub(crate) mod warp {
        check_group!(
            "a clamps batch charges every execution it was given, on the software adapter",
            &[],
            [warp_batch_charges_every_execution]
        );
    }

    /// The levels of a tiled texture arriving as the card has room, with the
    /// debug layer naming anything malformed.
    pub(crate) mod tiles {
        check_group!(
            "a tiled texture's levels arrive on demand on the card, and a full card keeps new ones waiting",
            &[("VICTORY_TILES_TEST_CARD", "1")],
            [levels_on_demand_on_the_card]
        );
    }

    /// The second stage of the relay: what a submission reads is what its own
    /// flight ended with, whichever flight is beside it.
    pub(crate) mod relay {
        check_group!(
            "a copy sent up holds what the flight before it ended with, not what was there in between",
            crate::device_suite::RELAY,
            [
                each_submission_reads_its_own_copy_on_the_card,
                each_submission_reads_its_own_copy_in_pc_memory,
                copies_stand_where_the_list_has_them_on_the_card,
                copies_stand_where_the_list_has_them_mixed,
                copies_stand_where_the_list_has_them_in_pc_memory,
                copy_draw_copy_draw,
                declared_writes_beside_the_titles_own_stay,
            ]
        );
    }

    /// [`relay`]'s tests, plus the fetch check: text on one of these in the
    /// crate is `VICTORY_FETCH_CHECK=1`, which this group is the one that
    /// needs (the test's own assertion says so).
    pub(crate) mod relay_fetch {
        check_group!(
            "a result caught while the next flight is at the same bytes is put right",
            crate::device_suite::RELAY_FETCH,
            [a_result_caught_in_between_is_put_right]
        );
    }

    /// [`relay`], plus the up-copy check: a shader's write nobody declared is
    /// found by the look at rest and the next send up undoes it.
    pub(crate) mod upcopy {
        check_group!(
            "the up-copy check finds an undeclared write and says so",
            crate::device_suite::UPCOPY,
            [the_check_finds_an_undeclared_write_undone]
        );
    }

    /// [`relay`], plus picked writes: a view a shader picks by an index it
    /// computed is found and what it wrote comes back.
    pub(crate) mod picked {
        check_group!(
            "what a shader writes through a view it picked comes back to title memory",
            crate::device_suite::PICKED,
            [writes_through_picked_views_come_back]
        );
    }

    /// The steps a copy made on the card goes in, behind a flight.
    pub(crate) mod card_steps {
        check_group!(
            "a copy done on the card behind a flight reads what the flight wrote",
            crate::device_suite::CARD_STEPS,
            [a_copy_behind_a_flight_reads_what_the_flight_wrote]
        );
    }
}

/// The ignored tests that are not checks: each prints what something cost, or
/// what the card does at an edge, and a green or red line from one would say
/// nothing. They are listed here so that the classification of every ignored
/// test is complete; run one by name when its number is wanted, as
/// `docs/device-suite.md` says.
///
/// On this machine (`measure::cpu`): no adapter, and they print times.
/// On the card (`measure::card`): they want a real adapter and print what it
/// costs or what it gives. As a benchmark (`measure::release`): the number is
/// only worth reading from a release build. `a_frame_of_a_titles_size` wants a
/// title's dump and gigabytes of card memory and disk
/// (`cargo test -p victory-gpu --features diagnostics --release --test replay_big -- --ignored --nocapture`).
const MEASUREMENTS: &[&str] = &[
    "bytes_under_overlapping_ranges",
    "frame_of_refreshes",
    "hash_costs",
    "runs_of_equal_pages",
    "what_landing_costs",
    "small_buffers_cost",
    "copy_costs",
    "card_compare_costs",
    "raw_loads_past_views",
    "descriptors_written_under_readers",
    "textures_read_in_another_dimension",
    "loads_far_outside_views",
    "new_blocks_under_readers",
    "batch_costs",
    "bare_command_cost",
    "look_costs",
    "cold_texture_costs",
    "allocation_costs",
    "gpu_bcpack_bench",
    "gpu_detile_bench",
    "texture_checks_per_submission",
    "bench_indirect_draw_barriers",
    "benchmark_private_data_release",
    "a_frame_of_a_titles_size",
];

/// Every check group, for the table's own checks.
fn groups() -> Vec<&'static Group> {
    vec![
        &check::card::GROUP,
        &check::warp::GROUP,
        &check::tiles::GROUP,
        &check::relay::GROUP,
        &check::relay_fetch::GROUP,
        &check::upcopy::GROUP,
        &check::picked::GROUP,
        &check::card_steps::GROUP,
    ]
}

/// Whether the layer is in `group`'s configuration, for a test's own guard.
///
/// A test that wants `VICTORY_RELAY=2` and is started without it must not fail
/// as though the layer were wrong. This says what the process has instead, and
/// the command that would have provided it.
pub(crate) fn require(group: &'static Group, holds: bool) {
    if holds {
        return;
    }
    let now = group
        .env
        .iter()
        .map(|(name, _)| format!("{name}={}", std::env::var(name).unwrap_or_else(|_| "unset".to_owned())))
        .collect::<Vec<_>>()
        .join(" ");
    panic!(
        "configuration missing, and nothing is wrong with the layer: this test is in the {} group, which wants {}; \
         this process has {now}. Run it in its group: {}",
        group.name(),
        group.switches(),
        group.command()
    );
}

/// Runs one member of `group` in a process of its own, with the group's
/// switches. What a member cannot do on this machine it says and skips.
pub(crate) fn dispatch(group: &'static Group, member: &str) {
    if !adapter() {
        eprintln!(
            "device suite skipped {member}: no adapter of its own here (the first adapter is a software one, has no \
             memory of its own, or there is none), so it cannot run. It is one of the tests that prove {}: {}",
            group.proves,
            group.command()
        );
        return;
    }
    let said = run(group, member);
    for line in said.lines().filter(|line| line.contains("skip")) {
        eprintln!("{}: {}", member, line.trim());
    }
}

/// Whether this machine has an adapter of its own to run these on: the first
/// one is not a software adapter and has memory of its own.
///
/// The tests check for a device again where they make one
/// ([`crate::host::host`]), and skip without it; this is so the suite says it
/// once, in words, instead of one line per test. A runner's software adapter
/// reports no dedicated memory and is what this keeps the suite off, because
/// it loses its virtual device part way through a run rather than failing
/// cleanly. A machine that has a card and still says it skipped here can run
/// a member by name with its group's switches, as `docs/device-suite.md`
/// says.
fn adapter() -> bool {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        use windows::core::Interface;
        // SAFETY: a plain query of the adapters. Nothing is created.
        unsafe {
            let factory: IDXGIFactory4 = match CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) {
                Ok(factory) => factory,
                Err(_) => return false,
            };
            match factory.EnumAdapters(0).ok().and_then(|adapter| adapter.cast::<IDXGIAdapter1>().ok()) {
                Some(adapter) => adapter.GetDesc1().is_ok_and(|desc| {
                    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 && desc.DedicatedVideoMemory > 0
                }),
                None => false,
            }
        }
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// One member in a process of its own, with `group`'s switches, bounded as
/// `.config/nextest.toml` bounds a test. The member's own words come back.
fn run(group: &'static Group, member: &str) -> String {
    let bound = std::time::Duration::from_secs(120);
    let exe = std::env::current_exe().expect("the test binary's own path");
    let log = std::env::temp_dir().join(format!("device-suite-{}-{member}.log", std::process::id()));
    let file = std::fs::File::create(&log).expect("a log beside the temp folder");
    let mut command = Command::new(&exe);
    command
        .args(["--ignored", "--nocapture", "--test-threads=1", member])
        .stdin(std::process::Stdio::null())
        .stdout(file.try_clone().expect("a second handle on the log"))
        .stderr(file);
    for switch in SCRUBBED {
        command.env_remove(switch);
    }
    for (switch, value) in group.env {
        command.env(switch, value);
    }
    let mut child = command
        .spawn()
        .unwrap_or_else(|e| panic!("{} needs the test binary to start again, and it will not: {e}", group.command()));
    let started = std::time::Instant::now();
    let status = loop {
        match child.try_wait().expect("the child answers") {
            Some(status) => break Some(status),
            None if started.elapsed() > bound => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };
    let said = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let ran = tests_run(&said);
    let tail = said.lines().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
    assert_eq!(
        ran,
        Some(1),
        "{} should be one test in this binary and ran as {ran:?}; it was renamed, un-ignored or moved: the table in \
         device_suite.rs and its #[ignore] reason must be kept with it.\n{tail}",
        member
    );
    let Some(status) = status else {
        panic!("{} did not finish within {bound:?}, so it was killed (as .config/nextest.toml bounds it).\n{tail}", member)
    };
    assert!(
        status.success(),
        "{member} failed in its own process ({status}) with the {} its group sets. Its group proves {}; a failure here \
         is a failure of the layer or of the test, not of the configuration, which this file set itself.\n{tail}",
        group.switches(),
        group.proves
    );
    said
}

/// How many tests the child reported running: libtest says `running N tests`.
/// A filter that matches nothing exits 0 without running anything, and a
/// member that silently does not exist would be a pass that proves nothing.
fn tests_run(said: &str) -> Option<usize> {
    said.lines()
        .find_map(|line| line.trim().strip_prefix("running ")?.split_whitespace().next()?.parse().ok())
}

/// Every `#[ignore]`d test in this crate's sources, with the reason its
/// attribute gives.
#[derive(Debug)]
struct Ignored {
    /// `src/host/upcheck.rs:755`.
    at: String,
    name: String,
    /// What stood between `#[ignore` and the end of the line, as written.
    reason: String,
}

/// Every ignored test in `src/` and `tests/`, in file order.
fn ignored() -> Vec<Ignored> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for folder in ["src", "tests"] {
        sources(&root.join(folder), &mut files);
    }
    files.sort();
    let mut found = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let shown = file.strip_prefix(root).unwrap_or(&file).display().to_string();
        let lines: Vec<&str> = text.lines().collect();
        for (n, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("#[ignore") else {
                continue;
            };
            let name = lines[n + 1..]
                .iter()
                .take(6)
                .find_map(|line| test_name(line))
                .unwrap_or_default();
            found.push(Ignored {
                at: format!("{shown}:{}", n + 1),
                name,
                reason: rest.trim().to_owned(),
            });
        }
    }
    found
}

/// Every `.rs` file under `folder`.
fn sources(folder: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            into.push(path);
        }
    }
}

/// The name of the test a line declares, if it declares one: `fn x(` or
/// `pub fn x(`, and nothing in a comment or an attribute.
fn test_name(line: &str) -> Option<String> {
    let line = line.trim_start();
    if line.starts_with("//") || line.starts_with('#') {
        return None;
    }
    let rest = line.strip_prefix("fn ").or_else(|| line.strip_prefix("pub fn "))?;
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    (!name.is_empty()).then_some(name)
}

/// Every ignored test in this crate is in exactly one group or in
/// [`MEASUREMENTS`], every name here is one, every check member's `#[ignore]`
/// reason names every switch its group sets, and no `#[ignore]` is bare.
///
/// This runs everywhere the crate's tests do, needs no adapter, and is the
/// reason the table can be trusted as the list of what the suite runs.
#[test]
fn every_ignored_test_is_classified() {
    let found = ignored();
    let known = groups();
    let mut problems = Vec::new();
    for test in &found {
        let in_groups: Vec<String> =
            known.iter().filter(|group| group.members.contains(&test.name.as_str())).map(|group| group.command()).collect();
        let measured = MEASUREMENTS.contains(&test.name.as_str());
        match (in_groups.len(), measured) {
            (1, false) | (0, true) => {}
            (0, false) => problems
                .push(format!("{} ({}): ignored, and in no group and not in MEASUREMENTS", test.name, test.at)),
            (_, true) => problems.push(format!(
                "{} ({}): in MEASUREMENTS and in {} group(s)",
                test.name,
                test.at,
                in_groups.len()
            )),
            (n, false) => problems.push(format!("{} ({}): in {n} groups", test.name, test.at)),
        }
        if test.reason.is_empty() {
            problems.push(format!(
                "{} ({}): #[ignore] with no reason; it must say what it wants",
                test.name, test.at
            ));
        }
        for group in known.iter().filter(|group| group.members.contains(&test.name.as_str())) {
            for (switch, value) in group.env {
                let wanted = format!("{switch}={value}");
                if !test.reason.contains(&wanted) {
                    problems.push(format!(
                        "{} ({}): its #[ignore] reason does not name {wanted}, which its group sets",
                        test.name, test.at
                    ));
                }
            }
        }
    }
    for name in known.iter().flat_map(|group| group.members.iter()).chain(MEASUREMENTS.iter()) {
        if !found.iter().any(|test| test.name == *name) {
            problems.push(format!("{name}: in this table, and not an #[ignore]d test of this crate"));
        }
    }
    assert!(problems.is_empty(), "the table and the sources disagree:\n  {}", problems.join("\n  "));
}
