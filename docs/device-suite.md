# The device suite

`victory-gpu`'s tests that want a real adapter, a switch, or a process of their
own are `#[ignore]`d, because a CI runner can satisfy none of those. They are
not one suite and never were: running all of them with no environment set gave
**31 passed, 14 failed**, and eleven of those fourteen were the tests' own
guards firing on a configuration nobody had set. A guard reads like an
assertion, so it reads like eleven bugs.

So they are grouped now, by the configuration each one needs, and each group
is one command. A group's tests each run in a process of their own, started
with that configuration and with every other switch removed first, so a run
does not depend on what the caller's shell had set. The table below is
checked against the sources by
`device_suite::every_ignored_test_is_classified`: a test added, renamed,
un-ignored or moved between groups without this file and its `#[ignore]`
reason agreeing is a red build, so the two cannot drift apart.

The machinery is `crates/victory-gpu/src/device_suite.rs`. It re-executes the
test binary rather than setting switches in place, because the layer reads
each switch into a process-wide static the first time anything asks for it:
they cannot be set after the fact. `indirect::tests`'s
`new_blocks_under_readers` already starts `current_exe` with an environment
for the same reason.

## The commands

Every check, one test at a time (a 1660 SUPER's 6 GB does not want twelve
device tests at once, and `--no-fail-fast` shows every red rather than the
first):

```powershell
cargo nextest run -p victory-gpu --lib --test-threads 1 --no-fail-fast device_suite::check
```

One group, when a red line has already named the group:

```powershell
cargo nextest run -p victory-gpu --lib --test-threads 1 device_suite::check::card
```

Without nextest, the same selection with the same process-per-test shape:

```powershell
cargo test -p victory-gpu --lib device_suite::check -- --test-threads=1
```

The table's own check — every ignored test is classified — needs no adapter
and runs on a runner:

```powershell
cargo test -p victory-gpu --lib device_suite::every_ignored_test_is_classified
```

## The groups

| Group | Switches | What a green run says |
|---|---|---|
| `check::card` | none — a real adapter | The layer does what it says on a real adapter: memory, descriptors, pipelines, placement, detiling, tiles. |
| `check::warp` | none — the software adapter | A clamps batch charges every execution it was given. |
| `check::tiles` | `VICTORY_TILES_TEST_CARD=1` | A tiled texture's levels arrive on demand on the card, and a full card keeps new ones waiting. |
| `check::relay` | `VICTORY_RELAY=2 VICTORY_RELAY_COPY=1` | A copy sent up holds what the flight before it ended with, not what was there in between. |
| `check::relay_fetch` | `VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_FETCH_CHECK=1` | A result caught while the next flight is at the same bytes is put right. |
| `check::upcopy` | `VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_UPCOPY_CHECK=1` | The up-copy check finds an undeclared write and says so. |
| `check::picked` | `VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_PICKED_WRITES=1 VICTORY_PICKED_MB=1` | What a shader writes through a view it picked comes back to title memory. |
| `check::card_steps` | `VICTORY_RELAY=2 VICTORY_CARD_STEPS=2` | A copy done on the card behind a flight reads what the flight wrote. |

A group skips, saying so, when this machine has no adapter of its own: the
first adapter is a software one, has no dedicated memory, or there is none.
That is what a CI runner gets, and it is why these tests stay `#[ignore]`d.
A machine that has a card and still says it skipped can run one member by
name with its group's switches:

```powershell
$env:VICTORY_RELAY='2'; $env:VICTORY_RELAY_COPY='1'
cargo test -p victory-gpu --lib copy_draw_copy_draw -- --ignored --test-threads=1 --nocapture
```

### What is in each group

`check::card` — the nine that want the host's device (`crate::host::host`):

- `gpu_detile::tests::gpu_detile_batched_prelude_layout_and_zero_subresources`
- `indirect::tests::host_list_budget_starts_empty_at_each_submission` — **red today: CAM-122** (left 1, right 0)
- `pipeline::host::tests::a_mesh_pipeline_loads_from_the_library` — also skips without mesh shaders and `dxc.exe`
- `pipeline::host::tests::depth_bounds_pipeline_loads_and_filters_draws`
- `pool::tests::a_placed_texture_holds_its_heap` — **red today: CAM-122** ("a placed texture holds no reference to its heap")
- `texture::tests::gpu_detile_end_to_end_and_partial_mip_upload_match_cpu`
- `texture::ahead::tests::a_placed_texture_is_filled_ahead_of_the_lists`
- `texture::ahead::tests::drain_waits_until_then_callbacks_have_run`
- `tiles::tests::card_wrote_failure_does_not_poison_card_mask`

`check::warp` — one, on the software adapter:

- `indirect::tests::warp_batch_charges_every_execution` — **red today: CAM-122** (left 0, right 134217728)

`check::tiles` — one:

- `tiles::tests::levels_on_demand_on_the_card`

`check::relay` — seven:

- `host::upcheck::tests::each_submission_reads_its_own_copy_on_the_card`
- `host::upcheck::tests::each_submission_reads_its_own_copy_in_pc_memory`
- `host::upcheck::tests::copies_stand_where_the_list_has_them_on_the_card`
- `host::upcheck::tests::copies_stand_where_the_list_has_them_mixed`
- `host::upcheck::tests::copies_stand_where_the_list_has_them_in_pc_memory`
- `host::upcheck::tests::copy_draw_copy_draw`
- `host::upcheck::tests::declared_writes_beside_the_titles_own_stay`

`check::relay_fetch` — one:

- `host::upcheck::tests::a_result_caught_in_between_is_put_right`

`check::upcopy` — one:

- `host::upcheck::tests::the_check_finds_an_undeclared_write_undone`

`check::picked` — one:

- `host::upcheck::tests::writes_through_picked_views_come_back`

`check::card_steps` — one:

- `host::card_steps::tests::a_copy_behind_a_flight_reads_what_the_flight_wrote`

## Running one bare

A test run outside its group has no way to report itself as skipped: a Rust
test passes or fails, and reporting a missing configuration as a pass would be
a lie. So each guard asks `device_suite::require`, and a bare run fails in
words that cannot be read as a bug:

```
configuration missing, and nothing is wrong with the layer: this test is in
the check::relay group, which wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1; this
process has VICTORY_RELAY=unset VICTORY_RELAY_COPY=unset. Run it in its group:
cargo nextest run -p victory-gpu --lib device_suite::check::relay
```

## What is not a check

Twenty-four of the ignored tests print a number or what the card does at an
edge, and a green or red line from one would say nothing. They are listed as
measurements in `device_suite.rs`, so the classification is complete, and run
by name when the number is wanted:

| What | Ignored test | Run it |
|---|---|---|
| The merge under overlapping ranges, 174 MB | `host::tests::bytes_under_overlapping_ranges` | `cargo test -p victory-gpu --lib bytes_under_overlapping_ranges -- --ignored --nocapture` |
| A frame's refreshes, 174 MB | `host::tests::frame_of_refreshes` | as above, `frame_of_refreshes` |
| Hashing 96 MB beside a compare | `host::torn::tests::hash_costs` | as above, `hash_costs` |
| Scanning 64 KiB for equal runs | `shadow::tests::runs_of_equal_pages` | as above, `runs_of_equal_pages` |
| Landing, with other threads at work | `land::tests::what_landing_costs` | as above, `what_landing_costs` |
| 1,000 small buffers, card and PC memory | `host::place::tests::small_buffers_cost` | as above, needs a card |
| Copies by the kernel and here | `safecopy::tests::copy_costs` | as above, needs a card |
| The passes over copies on the card | `indirect::tests::card_compare_costs` | as above, needs a card; `VICTORY_BENCH_CS=<file>` times another source |
| A raw load past its view | `indirect::tests::raw_loads_past_views` | as above, needs a card; `VICTORY_BENCH_WARP=1` runs on WARP |
| A descriptor half written under its readers | `indirect::tests::descriptors_written_under_readers` | as above, needs a card; `VICTORY_BENCH_WARP=1`, `VICTORY_BENCH_S` |
| A texture read in another dimension | `indirect::tests::textures_read_in_another_dimension` | as above, needs a card |
| Loads far outside their views | `indirect::tests::loads_far_outside_views` | as above, needs a card, and may lose the device |
| Blocks made under readers | `indirect::tests::new_blocks_under_readers` | as above, needs a card; each way runs in a process of its own; `VICTORY_BENCH_S`, `VICTORY_BENCH_N` |
| A frame's clamps, one by one and batched | `indirect::tests::batch_costs` | as above, needs a card |
| A bare command's cost | `indirect::tests::bare_command_cost` | as above, needs a card |
| Texture looks | `texture::tests::look_costs` | as above |
| Cold textures | `texture::tests::cold_texture_costs` | as above; `VICTORY_BENCH_FILE` |
| Allocation | `texture::tests::allocation_costs` | as above |
| BCPack on the card | `gpu_bcpack::tests::gpu_bcpack_bench` | as above |
| Detiling on the card | `gpu_detile::tests::gpu_detile_bench` | as above |
| Texture checks per submission | `host::tests::texture_checks_per_submission` | as above |
| Indirect draws and barriers | `indirect::bench::bench_indirect_draw_barriers` | as above |
| Releasing private data | `object::tests::benchmark_private_data_release` | `--release --ignored --nocapture` |
| A frame of a title's size | `replay_big::a_frame_of_a_titles_size` | `cargo test -p victory-gpu --features diagnostics --release --test replay_big -- --ignored --nocapture`; takes gigabytes of card memory and disk |

## Two changes to the tests themselves

- `draw::tests::spans_of_a_pass` was `#[ignore]`d with no reason and needs no
  adapter: it builds the spans of 4,400 copies from 72,000 slots both ways and
  asserts both find the 4,400. It runs in CI now, where it is a check of the
  two paths agreeing rather than a printed time.
- `host::upcheck::tests::a_result_caught_in_between_is_put_right`'s `#[ignore]`
  reason named `VICTORY_RELAY` and `VICTORY_RELAY_COPY` and not
  `VICTORY_FETCH_CHECK=1`, which its own assertion and its own doc comment
  both require: without the fetch check it fails with "without
  VICTORY_FETCH_CHECK=1 most are wrong". Its group sets all three.
