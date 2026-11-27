# The weak target

`Play-VIctory.cmd weak` is the configuration for one machine: a Ryzen 5 3600
(6 cores, 12 threads, Zen 2, no AVX-512), a GTX 1660 SUPER (TU116: 6 GB, D3D12
feature level 12_1, no RT cores), 32 GB of DDR4 at 2133 MT/s, Windows 11,
PCIe 3.0 x16. The other two presets, `pareto-residency` and `native-rt`, are
unchanged and stay as they were.

Its job is to make a session on that machine measurable and repeatable, so that
what the programme does to the layer (work packages W2 to W5) is argued with
rows of `measurements.csv` and not with impressions. It makes no claim about
frames per second by itself: the bar the work is read against is 30 fps at a
540p internal render, from `CAM-110`, and every row here is a data point toward
or against it.

The preset runs the title as `--console series-s`, as both other presets do
(the launcher's command line, unchanged): the Series S profile is the 1080p one,
and it is the profile whose memory figures this card can hold.

## The two hard limits

- **No DXR.** TU116 has no RT cores, so `VICTORY_NATIVE_RT` and the
  `XboxBvhBuilder.dll` path cannot run, and the console's ray-traced bounce
  light and shadows do not exist here. The layer's stand-in for that light is
  what the image has instead: `VICTORY_AMBIENT` (a share of an estimate of the
  light in the frame) and `VICTORY_AMBIENT_AIR` (a bound on what the air
  scatters), plus `VICTORY_AMBIENT_REFLECT`'s cap on RT-off reflections, which
  is on by default at 0.5.
- **6 GB of VRAM** against a design that on a 24 GB card sat at 11 to 16 GB
  and relied on the card having room. Everything below that touches memory is
  about keeping the render targets and the title's buffer copies in that 6 GB.

## The switches, and what each is expected to trade

Every line is one the code reads at the site named, and each sets a value the
code would otherwise choose for itself. Nothing here is set if you already have
it set, so your own overrides win.

| Switch | Value | Expected trade |
|---|---|---|
| `VICTORY_OUTPUT` | `1920x1080` | The swap chain and the display. One of the five sizes the title has a display mode for (`victory-loader/src/resolution.rs`, `MODES`). |
| `VICTORY_RENDER_SCALE` | `0.5` | Image sharpness for GPU time: the title renders 960x540 and its own upscaler stretches that to 1080p. **Read the clamp below**: below 2/3 the title's *largest* render size stays at 2/3, so this may render 1280x720 instead. |
| `VICTORY_UPSCALER` | `6` | The title's own upscaler instead of FidelityFX. On Series S the picker returns FidelityFX, which reads its own scale constant, so `VICTORY_RENDER_SCALE` would do nothing; 6 is the picker that honours it. `VICTORY_FFX_SCALE` is not used: it is documented unsafe at 0.5 (the title divided by a zero jitter phase count). |
| `VICTORY_TEXTURE_BUDGET_MB` | `3200` | Texture detail and eviction churn for room. A cap on the bytes of host copies of textures the title fills; past it the queue makes no more, which leaves ~2.9 GB of the card for render targets and buffer copies. Without it the policy allows up to nine tenths of the budget for textures alone, with nothing reserved. |
| `VICTORY_TEXTURE_MAX` | `512` | Distant texture detail for memory, pinned: a streamed texture skips levels until no side passes 512. The default follows pressure (upstream's bands: 2048 below 85 % of the target, 1024 at 85-92 %, 512 at 92-97 %, 256 above), so it would keep changing step on this card; a fixed value keeps the copy each view reads the same across a run. The pin also means the pinned cap already does everything a texture-quality tier would: with no side above 512 the chain has no level 0 left to drop. |
| `VICTORY_TEXTURE_STAGE_MB` | `96` | Load time for texture memory: the per-frame upload budget for textures picked by index. 96 MiB is exactly what the code picks on a `<= 6 GiB` budget, so this only removes a dependence on what the driver reports. |
| `VICTORY_CARD_BUFFERS_MAX_MB` | `512` | Buffer latency on the card for memory: the ceiling on card copies of title buffers. 512 MiB is the default for any budget up to 12 GiB, set here so it does not move if the reported budget does. |
| `VICTORY_NATIVE_RT` | `0` | Nothing to trade: the switch only ever turns DXR on (`dxr.rs`, `requested()`). Stated so a run of this preset cannot quietly inherit it from a window that set it. |
| `VICTORY_BVH_BUILDER` | `0` | Same: builder support on or off. Off is also what leaves the ray-resolve programs out of the pipeline queue (`pipeline/make.rs`, `left_out`), which is GPU and CPU time saved on a card that cannot use them. |
| `VICTORY_THIN_THREADS` | `4` | List translation: fewer threads for the work that was worth eight on a 24-thread CPU (`list/thin.rs`, default 8). Four of them on six cores, leaving the submitting thread and the title their own. |
| `VICTORY_VIEW_THREADS` | `3` | Descriptor work: 3 threads each for the two pools (views picked by index, and heap records copied as bytes) whose defaults here are 4 and 6 (`land.rs`, `each_view`). Fewer threads making records means submissions wait slightly longer for them; that is the trade against core oversubscription. |
| `VICTORY_UNDER_THREADS` | `4` | The comparison of title bytes under shader writes (6 by default, `land.rs`) — a couple of milliseconds a frame either way, for 2 fewer threads on 6 cores. |
| `VICTORY_LAND_THREADS` | `3` | Landing what shaders wrote, on the relay's own threads: the default is a quarter of the logical processors (3 here), pinned so it cannot become 4 or more on a machine that reports more processors. |
| `VICTORY_WIDE_THREADS` | `4` | The copyback workers in `shadow.rs` (snapshots of shader-written ranges and the compares that find them, 6 by default under `VICTORY_WIDE_BACK=1`). They run inside the frame, so 4 leaves the submitting thread a core; a longer compare wall is the cost. |
| `VICTORY_PIPELINE_WORKERS` | `4` | First-run stalls for frame time: pipeline making happens below the title's own threads, 8 workers by default on this machine's 12 logical processors. Four means a cold cache takes longer to fill; it also means fewer threads stealing frames while it fills. |
| `VICTORY_JOIN_PARTS` | `4` | How many parts of a submission are prepared as one (8 in `pareto-residency`): fewer parts joined is less work stolen from the frame, and a little more per-submission latency. |
| `VICTORY_VRAM_TARGET_MB` | (not set) | How much of the card this layer lets itself hold. Unset, the code computes 5,200 MB here (`resource.rs`, `VRAM_CEILING`) and measures admission, the mip caps, eviction and the empty blocks it keeps against that figure instead of the 5.4-6.1 GiB the driver reports. Set it to move the line: it is clamped to between 1 GiB and the reported budget less the desktop's share, so it cannot oversubscribe the card. Turn it down if the log shows the card full and frames waiting; turn it up if textures are refused while the card has room. |

### The 2/3 clamp, and `weak-540`

`VICTORY_RENDER_SCALE` sets the title's own upscaler's scale, but the same module records
that below 2/3 the title's *largest* render size (`+0x94`, `max(2/3, scale)`) stays at
2/3. So `0.5` on a 1920x1080 output may render 1280x720 rather than 960x540, which the
programme's own arithmetic puts at roughly 50 ms against 38 ms of card time here — enough
to decide whether a 33 ms frame is reachable. Whether the title clamps is not known until
there is a row for it, and this page states no measured figure of its own.

A clamped row is recognisable without the panel: its `peak_local_vram_mb` and its fps sit
within noise of a run at render scale 1.0, where two rows that really differ by 35 % of
the picture cannot. On screen, F3's `FRAME` number stays within a few milliseconds of what
the 1.0 run showed.

For a true 960x540 there is `Play-VIctory.cmd weak-540`: the same preset with
`VICTORY_OUTPUT=1280x720` and `VICTORY_RENDER_SCALE=0.75`. At a 720p output the clamp is
`max(2/3, 0.75)` = 0.75, so 960x540 is rendered and stretched to a 720p swap chain, which
Windows or the monitor then stretches to 1080p. Two things about the pair are untested
until a row exists: that the title's own upscaler takes 0.75 at a 720p output, and how that
image looks stretched. Measure both presets in one sitting:

```powershell
scripts\measure.ps1 -Exe $env:VICTORY_GAME_EXE -Tag r1080 -Preset weak
scripts\measure.ps1 -Exe $env:VICTORY_GAME_EXE -Tag r540 -Preset weak-540
```

### Kept from `pareto-residency`

The rest of that preset is inherited unchanged, because none of it depends on
memory size or core count: `VICTORY_RELAY=2`, `VICTORY_THIN=1`,
`VICTORY_CARD_STEPS=2`, `VICTORY_SHOW_AFTER=1`, `VICTORY_RELAY_COPY=1`,
`VICTORY_JOIN_COPIES=1`, `VICTORY_WIDE_BACK=1`, `VICTORY_CARD_REHASH_RUNS=1`,
`VICTORY_INDIRECT_WINDOW=1`, `VICTORY_INDIRECT_BATCH=2`,
`VICTORY_TEXTURE_NO_WAIT=1`, `VICTORY_CARD_AWAKE=1`,
`VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_LEVELS_ON_DEMAND=1`,
`VICTORY_UPLOAD_AHEAD=1`, `VICTORY_MAP_THREADS=1`, `VICTORY_AMBIENT=0.8`,
`VICTORY_AMBIENT_AIR=1.5`, and the cache folder.

### Where its files go

The preset keeps everything off the system SSD: `VICTORY_CACHE_DIR=D:\VIctory\cache`
(the shader and pipeline cache, and the converted images) and `VICTORY_DATA_DIR=D:\VIctory\data`
(saves, settings, `crashes.log`), both made if they are not there, and both only if you have
not set them yourself. The measurements go to `D:\VIctory\measurements` the same way
(`-Out` says otherwise). If `D:\` cannot be written, the preset says so once and leaves both
at their previous defaults under `%LOCALAPPDATA%` rather than failing the launch.

**The checkout belongs on D: too.** Build `target\` follows the repository, so a clone under
`D:\` keeps a release build of this workspace (and a 60-100 GB title dump) off a 256 GB
system SSD that Windows shares. The trade is real: D: is a mechanical drive, so reading the
shader cache and streaming the dump is slower than from the SSD. If space on C: frees up
later, moving the cache back is one line — set `VICTORY_CACHE_DIR` before the preset runs.

### Deliberately not set

- **`VICTORY_TEXTURE_KEEP`** (default: on). It lets idle textures go early.
  The default already switches to its 5 s and 1.5 s rules once the card passes
  91 % of its target and trims back to 88 %, which on 6 GB is most of a
  session; forcing `0` would only add the re-upload churn the comment there
  counts.
- **`VICTORY_TEXTURE_ROOM_PERCENT`** (default 93 upstream, 90 in what this
  page was written against). `VICTORY_TEXTURE_BUDGET_MB`
  caps those copies' bytes, but the card's own line is still read under it
  once `VICTORY_VRAM_TARGET_MB` or the computed target applies, so this
  percentage is no longer bypassed: the default holds the queue a few per cent
  of the target below the line. `VICTORY_TEXTURE_BUDGET_MB` pins the bytes
  first, so the percentage only decides the case under that cap.
- **`VICTORY_FAST_CORES`** (default 2). It moves threads onto the fastest
  efficiency class where a CPU has two; a 3600 has one, and the code returns
  without asking.
- **`VICTORY_CARD_ROOM_FOR_TARGETS_MB`** (default 2048 upstream, 3072 before
  the merge). It only feeds the room test, which is read under
  `VICTORY_TEXTURE_BUDGET_MB` as well, so on this preset it is read: it is the
  room the queue leaves for render targets and the card copies of title
  buffers, less what they already hold.
- **`VICTORY_FRAME_INTERVAL_MS`**, **`VICTORY_WINDOW_MAX`**: they change what a
  frame *is* rather than how fast it is made, and this preset is the baseline.

### The VRAM hold, and the line to keep

On this card the layer does not budget against the 5.4-6.1 GiB the driver
reports, because Windows keeps a slice of that for the desktop and the rest is
where the system starts moving the card's memory into the PC's over PCIe 3.0.
It computes a target instead — 5,200 MB here (`resource.rs`, `VRAM_CEILING`),
what is left of the reported budget after the desktop's eighth, whichever is
lower — and every VRAM share is measured against that figure: admission to the
texture queue, the mip caps, the idle-keep bands and the empty blocks the pool
retains. `VICTORY_VRAM_TARGET_MB` replaces the computed figure.

Once a second the memory summary line carries it, and a second line follows
with what it cost:

```
memory summary: RAM private <n> MB, working set <n> MB; VRAM local <used>/<reported> MB, SHR <used>/<budget> MB (holding to <target> MB, peak <peak> MB); buffers: ...
vram target: <target> MB of the <reported> MB the system reports (computed: ceiling 5200 MB less the desktop's share), local in use <used> MB, peak <peak> MB; texture admissions refused <n> times, host copies of textures the title fills let go <n> (<n> MB) since the start; the card is over its target
```

`peak` is the highest local figure the driver reported all session; refusals are
textures the queue deferred rather than made; the let-go counts are the sweep's.
If `peak` sits above the target for long stretches, or refusals grow while the
card is under it, raise `VICTORY_VRAM_TARGET_MB` by 200 and run the same scene.

## How to measure

The log has to be at info level: `VICTORY_LOG=warn` (the launcher's default)
hides the two lines the harness reads. `scripts\measure.ps1` sets it for the
run and records every `VICTORY_*` variable it ran with.

```powershell
# build first: cargo build --release --locked -p victory
$env:VICTORY_GAME_EXE = 'D:\Games\MyDump\title.exe'
scripts\measure.ps1 -Exe $env:VICTORY_GAME_EXE -Tag water -Seconds 90
```

That starts `Play-VIctory.cmd weak`, runs it 90 seconds, closes it, writes the
log to `D:\VIctory\measurements\<stamp>-<tag>.log`, and appends one row to
`D:\VIctory\measurements\measurements.csv` (`-Out` says elsewhere): build commit and
binary hash, median fps, preparation and
copyback milliseconds, peak local VRAM and commit, page faults per frame, and
whether the process exited cleanly. `-Preset pareto-residency` runs the old
configuration through the same harness, which is what makes the two
comparable.

Two things the log does not have. There is no per-frame time series, so
`p99_frame_ms` is written as `n/a`; press F3 in the title and type the `FRAME`
and `LONGEST` numbers the panel shows into that cell. (`LONGEST` is the longest
frame of the last half second, not a p99 — `host/panel.rs`.) And scenes differ
from run to run, so start each run from the same place, and read differences
under about 5 % as nothing.

Nothing the harness writes is tracked in the repository; it all lives under
`D:\VIctory\measurements` unless `-Out` says otherwise.
