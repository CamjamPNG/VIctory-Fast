# Which switch is in which build

Every `VICTORY_*` variable the title reads is listed in the comments at the
site that reads it, and most work in any build. A few do not: their code is
inside `cfg(feature = "diagnostics")`, so in a plain release build the switch
is not read at all and setting it does nothing, silently. This page says which
those are and which command makes each live. It is for anyone measuring on
hardware, so that a switch missing from a log is a fact and not a guess.

The rule the repository enforces: **a diagnostics build of `cargo build -p
victory` compiles in every crate that gates on the feature, and no plain build
compiles one in.** `cargo xtask features` prints the graph and fails when the
rule breaks (it also runs as a test, so CI runs it).

## The two builds

| Build | Command | Artifact |
|---|---|---|
| Release | `cargo build --release --locked -p victory` | `target/release/victory.exe` — no diagnostic path at all |
| Diagnostics | `cargo build --profile diag --locked -p victory --features diagnostics` | `target/diag/victory.exe` — `release` codegen plus every diagnostic path |
| Everything at once (CI's clippy pass) | `cargo clippy --workspace --all-targets --all-features` | compiles every crate's diagnostics |

`--profile diag` is `release` with `lto = false` and `codegen-units = 16`, so it
is the same optimised code with a faster link. It is not the same binary as the
release one: the feature changes what is compiled, so timings from the two are
not interchangeable.

## The graph

`cargo build -p victory --features diagnostics` turns on:

| Crate | Gates | Named switches that need this build |
|---|---|---|
| `victory` (the binary) | frame capture hooks, the crash watcher, the scripted poke | `VICTORY_WATCH`, `VICTORY_TIMER_LOG`, `VICTORY_POKE`, `VICTORY_CRASH_SAVE` |
| `victory-core` | the log cap, `INFO` in a plain build and `TRACE` here | `VICTORY_LOG` above `info` |
| `victory-hle` | XMA input capture, protection-change notes, per-import lines, GPU frame phase and replay | `VICTORY_XMA_DUMP` |
| `victory-loader` | console-instruction site counters, scheduled stack dumps and profiling, memory saved on a crash | counters written when `VICTORY_WATCH` or `VICTORY_TIMER_LOG` are set (with `victory-core`) |
| `victory-gpu` | the debug layer and device-removed data, WARP, GPU timings, per-draw and per-list counters, the frame recorder, the arenas' dump, the indirect budget's default | `VICTORY_D3D12_DEBUG`, `VICTORY_D3D12_DRED`, `VICTORY_D3D12_GBV`, `VICTORY_D3D12_WARP`, `VICTORY_GPU_TIMING`, `VICTORY_MEMORY_DUMP`, `VICTORY_PORT_RT`, `VICTORY_PORT_GUARD`, `VICTORY_PORT_JOURNAL`, `VICTORY_SKIP_PIPELINES`, `VICTORY_POOL_SMALL`, `VICTORY_SPANS_QUICK`, `VICTORY_SYNC_IN_PLACE`, `VICTORY_PLACED_ORDER`, `VICTORY_LIST_SLOTS` |
| `victory-shader` | the loop limit's override, and one byte of the translation fingerprint | `VICTORY_SHADER_LOOP_LIMIT` |
| `victory-formats`, `victory-game`, `victory-pe`, `victory-pipeline` | nothing: they have no `diagnostics` feature and gate no code on it | — |

There is no diagnostic path below `victory-formats`: those four crates are the
same code in both builds. Do not add a `diagnostics` feature to one of them and
forget the forwarding — `cargo xtask features` fails on it, naming the crate and
the manifest that should have forwarded it.

The forwarding runs `victory` → `victory-hle` → `victory-gpu`, `victory-loader`
→ `victory-core`, and `victory` → `victory-core`, `victory-shader`. The binary
does not list `victory-gpu` as a dependency (it does not call it directly), so
its diagnostics arrive through `victory-hle`, which is where the device layer
is. Cargo refuses `"victory-gpu/diagnostics"` in `apps/victory`'s feature list
for that reason: *feature `diagnostics` includes `victory-gpu/diagnostics`, but
`victory-gpu` is not a dependency*. A test therefore pins the reachability
instead of the direct name.

`victory-gpu`'s `diagnostics` deliberately does **not** forward
`victory-shader/diagnostics`: that one changes what the emitter produces (the
loop limit) and one byte of the cache key, and a GPU diagnostics build is used
to measure commands, not to translate differently. `cargo xtask features` knows
this pair by name and the reason sits beside it in `xtask/src/features.rs`.

## Switches read only in a diagnostics build

Each of these has no read site outside `cfg(feature = "diagnostics")`: in a
plain release build, setting it changes nothing and says nothing.

| Switch | Where | What it does |
|---|---|---|
| `VICTORY_D3D12_DEBUG` | `victory-gpu/src/host.rs` | enables the D3D12 debug layer (only the validator half works without the feature: `victory-shader` reads the same variable for `translate --validate`) |
| `VICTORY_D3D12_DRED` / `light` / `faults` | `victory-gpu/src/host.rs` | device-removed data and breadcrumbs |
| `VICTORY_D3D12_GBV` | `victory-gpu/src/host.rs` | GPU-based validation, inside the debug layer's branch, so it needs `VICTORY_D3D12_DEBUG=1` as well |
| `VICTORY_D3D12_WARP` | `victory-gpu/src/host.rs` | the software adapter for the host device |
| `VICTORY_GPU_TIMING` | `victory-gpu/src/host.rs` | per-list timings on the card, one line a list |
| `VICTORY_MEMORY_DUMP` | `victory-gpu/src/memory.rs` | the arena dump when a file `go` appears |
| `VICTORY_PORT_RT`, `VICTORY_PORT_JOURNAL`, `VICTORY_PORT_GUARD` | `victory-gpu/src/{port_rt,watch}.rs` | the port's render targets and journal |
| `VICTORY_SKIP_PIPELINES`, `VICTORY_POOL_SMALL`, `VICTORY_SPANS_QUICK`, `VICTORY_SYNC_IN_PLACE`, `VICTORY_PLACED_ORDER`, `VICTORY_LIST_SLOTS` | `victory-gpu/src/…` | A/B experiments inside the diagnostic paths |
| `VICTORY_SHADER_LOOP_LIMIT` | `victory-shader/src/emit/mod.rs` | a new bound for translated loops |
| `VICTORY_XMA_DUMP` | `victory-hle/src/modules/audio.rs` | XMA input written to files |
| `VICTORY_WATCH`, `VICTORY_TIMER_LOG`, `VICTORY_POKE`, `VICTORY_CRASH_SAVE` | `apps/victory/src/run.rs` | the crash watcher, periodic timers, the scripted poke, memory saved on a crash |

## Switches whose default changes, and those that work anywhere

These three are read in every build but default differently, so the same run
differs between the two builds with no variable set:

| Switch | Plain build | Diagnostics build |
|---|---|---|
| `VICTORY_INDIRECT_BUDGET` | off; `=1` turns it on | on; `=0` turns it off |
| `VICTORY_TRAIL` | off; `=1` turns it on | on; `=0` turns it off |
| `VICTORY_LOG` | `info` and above, whatever the filter says | `trace` and above |

And two worth knowing because their names promise more than a release build
does:

- `VICTORY_GPU_TIMING` still has one effect in a release build: any value at all
  makes `late_stage()` return 0 (`victory-gpu/src/host.rs`), taking the late-wait
  relay out. The timing it names is not compiled in, so a release run with it
  set measures something other than what the name says.
- `VICTORY_INDIRECT_BUDGET=1` works in a release build; it is the counting
  entries in the clamp shader that a diagnostics build adds by default.

Everything else in the repository — `VICTORY_THIN`, `VICTORY_AMBIENT`,
`VICTORY_PORT_CONSTANTS`, the residency and texture switches, the shader
lowering switches — is read in both builds. `cargo xtask features` lists the
crates; the comment at each read site is the authority on what it does.

## Commands for the crates on their own

`--features diagnostics` on one package enables that package's feature only.
What each gives:

| Command | Which crates' diagnostics |
|---|---|
| `cargo build --profile diag -p victory --features diagnostics` | all five, as the table above |
| `cargo test -p victory-gpu --features diagnostics` | `victory-gpu` only (the shader's is spared on purpose) |
| `cargo test -p victory-hle --features diagnostics` | `victory-hle`, `victory-gpu`, `victory-loader`, `victory-core` |
| `cargo test -p victory-loader --features diagnostics` | `victory-loader`, `victory-core` |
| `cargo test --workspace` (CI) | none: no crate is built with the feature |
