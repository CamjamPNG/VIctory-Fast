# VIctory-DXR

A fork of [VIctory](https://github.com/Marvisak/VIctory), a Direct3D 12 and runtime compatibility layer for running Xbox Series X|S binaries natively on Windows using files supplied by the user.

**VIctory-DXR focuses on native hardware DirectX Raytracing (DXR 1.1 `RayQuery`)** — replacing slow software BVH traversal loops and emulated BVH build shaders with real host GPU acceleration structures and hardware ray-tracing cores (tested on AMD RDNA 4 / Radeon RX 9070 XT).

> **Legal & Asset Policy**
>
> **This repository contains no copyrighted game files, executables, shaders, textures, audio, or assets.** You must supply your own legally obtained dump of any title you run. This project is an independent open-source compatibility layer under **GPL-3.0** and is not affiliated with or endorsed by any console manufacturer, game developer, or publisher. See [docs/asset-policy.md](docs/asset-policy.md).

---

## Key Features

### Native Hardware DXR 1.1 Ray Tracing (`VICTORY_NATIVE_RT=1`)
- **`XboxBvhBuilder.dll` HLE (`crates/victory-hle/src/modules/bvh_builder.rs`):** Intercepts console BVH builder APIs directly rather than emulating dozens of BVH-building compute shaders in software.
- **Host DXR Acceleration Structure Manager (`crates/victory-gpu/src/dxr.rs`):** Builds and updates native D3D12 multi-geometry static BLASes, dynamic BLASes, and `PREFER_FAST_TRACE` TLASes directly on host RT hardware with topology validation, safe deferred resource retention, and null-TLAS fallback handling.
- **RDNA 2 → DXIL SM 6.6 `RayQuery` Lifter (`crates/victory-shader/src/emit/rayquery_lift.rs`, `lower.rs`):** Detects `image_bvh*_intersect_ray` traversal loops (opaque, front/back-track, and alpha-tested any-hit loops), coalesces `AllocateRayQuery` handles across disjoint loops, and lifts them into native DXIL Shader Model 6.6 `RayQuery` (`TraceRayInline` / `Proceed` / `Committed*` / `Candidate*`) instructions. Closest-hit `AnyHit` loops default to the inline fallback path on RDNA 4 (opt-in via `VICTORY_NATIVE_RT_ANYHIT=1`) so opaque/shadow passes capture the full hardware speedup without divergent `Proceed()` stalls.
- **AMD RDNA 4 Stability (`amdxc64.dll`):** Fixes buffer `getDimensions` lowering crashes and validates lifted DXR shaders through `dxil.dll` and host D3D12 PSO creation.

---

## Performance & Benchmarks

Measured on **AMD Ryzen 7 7800X3D · AMD Radeon RX 9070 XT (16 GB, RDNA 4) · 32 GB RAM · Windows 11**.

### Native DXR 1.1 vs. Software BVH (Head-to-Head `VICTORY_GPU_TIMING=1` + `DRED`, Series S Profile)

| Metric | Software BVH (`VICTORY_BVH_BUILDER=1`) | Native DXR 1.1 (`VICTORY_NATIVE_RT=1`) | Delta |
|---|---:|---:|---:|
| Total GPU card time / frame | `128.6 ms` | **`80.6 ms`** | **-37.3% (-48.0 ms)** |
| Indirect compute time / frame | `105.4 ms` | **`64.2 ms`** | **-39.1% (-41.2 ms)** |
| Shadow RT pass (`f607780aa8e502d0`) | `8.20 ms` | **`~0.1 ms`** | **~80× faster** |
| RT pass (`caf96a3229145fa6`) | `21.04 ms` | **`10.39 ms`** | **-50.6%** |
| RT pass (`36f04a4ed0ad1d9b`) | `5.01 ms` | **`2.19 ms`** | **-56.3%** |
| RT pass (`3592426e3e9142d8`) | `6.01 ms` | **`3.30 ms`** | **-45.1%** |
| Resident VRAM / faults | ~12–16 GB | **`~10.2 GB` (`0` faults)** | **~2+ GB lower** |

### Configuration FPS Summary (Measured)

| Configuration | Build | Median / Typical FPS | Peak FPS | Notes |
|---|---|---:|---:|---|
| **Series S (1080p) — RT Off** | `--release` (measured, `7d2f44f`) | **~30.4 fps** settled (`28.6` overall med) | **32.4 fps** | `VICTORY_RDNA_WAVE32=2`: `32.9 ms/f`, **`23.9 ms`** prep, **`0.0 ms`** GPU wait, `14.7 ms` copyback, `16.7 GB` RAM / `8.2 GB` VRAM / `2.09 GB` SHR, `0` faults |
| **Series X (Native 1440p) — RT Off** | `--release` (measured, `7d2f44f`) | **~29.8 fps** settled (`30.7` final) | **30.8 fps** | `VICTORY_RDNA_WAVE32=2`: `33.6 ms/f`, **`20.8 ms`** prep, **`0.0 ms`** GPU wait, `13.9 ms` copyback, `19.5 GB` RAM / `9.6 GB` VRAM / `1.75 GB` SHR, `0` faults |
| **Series X (1440p, FSR 4.1.2) — RT Off** | `--release` (measured, `3b7f7fd`) | **~30.1 fps** settled (**32.6** steady) | **32.9 fps** | `1080p → 1440p` PC FSR 4.1.2 + `VICTORY_RDNA_WAVE32=2`: `35.1 ms/f` (`30.7 ms` steady), `26.1 ms` prep, **`0.0 ms`** GPU wait, `18.5 ms` copyback, `11.3 GB` VRAM / `2.29 GB` SHR, `0` faults |
| **Series S — Software RT** | `--release` (measured, `d1686f1`, 165s) | **~11.7 fps** settled (`12.0 fps` overall med) | **18.8 fps** | `1.4 ms` med / `4.7 ms` settled GPU wait, `54.4 ms` prep, `11.3 GB` max VRAM (`128.6 ms` timed `diag`) |
| **Series X (1440p) — Software RT** | `--release` (measured) | **~8.2 fps** | **19.9 fps** | Software BVH emulation (`175.8 ms` GPU in timed `diag`) |
| **Series S — Native DXR 1.1 RT** | `--release` (measured, `6a4ee3c`, parity default) | **~11.6 fps** settled (`10.9` med; `13.5` opt-in FBT) | **17.9 fps** (`25.1` peak `c9de853`) | Visual-parity default: `86.6 ms/f`, `36.9 ms` prep, `36.1 ms` GPU wait, `20.9 ms` copyback, **`13.4 GB`** max VRAM, `0` faults |
| **Series X (1440p) — Native DXR 1.1 RT** | `--release` (measured, `14d91db`, VRAM-fit) | **~7.5 fps** settled (`7.3` mean / `6.4` med) | **19.0 fps** | `133.8 ms/frame`, `46.0 ms` prep, `55.1 ms` GPU wait, `25.4 ms` copyback, strictly inside 16 GB local VRAM (`0` faults with `VICTORY_TEXTURE_BUDGET_MB=3072`) |

*Note: `diag` runs with `VICTORY_GPU_TIMING=1` and `VICTORY_D3D12_DRED=1` incur heavy instrumentation overhead (~3–6 fps) and are used strictly for apples-to-apples per-shader GPU timing. Use `--release` without timing/DRED flags for gameplay.*

### Renderer, Residency & Frame-Time Improvements (Inherited from Upstream Forks)
- **Async list recording & parallel submission prep:** `VICTORY_THIN`, `VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`, AVX2 page comparisons, and parallel descriptor/refresh passes.
- **Background GPU read-back & on-card steps:** `VICTORY_RELAY`, `VICTORY_LAND`, `VICTORY_CARD_STEPS`, and texture read-back.
- **32 GB RAM residency & two-plane composition:** Fallback heaps, commit-pressure backoff, `TablePlan` caching, Wave32 lowering support, and Series S two-plane composition.
- **Frame record & replay diagnostics:** Per-pass GPU timing (`VICTORY_GPU_TIMING=1`) and replay tooling ([docs/replay.md](docs/replay.md)).

---

## Requirements

- **OS:** Windows 10/11 x64
- **CPU:** x86-64 with AVX2
- **GPU:** Direct3D 12 GPU with **Shader Model 6.6**, **Mesh Shaders**, and **DXR Tier 1.1** (for `VICTORY_NATIVE_RT=1`)
- **Build tools:** Rust toolchain (pinned in `rust-toolchain.toml`) and MSVC C++ Build Tools with Windows SDK
- **Title files:** Your own dump stored **outside** this repository

---

## Building & Running

```powershell
# Release build (recommended for gameplay)
cargo build --release --locked -p victory

# Diagnostics build (with frame/GPU timing instrumentation)
cargo build --profile diag --locked -p victory --features diagnostics
```

Configure local paths outside the repository (see `victory.example.toml`):

```powershell
copy victory.example.toml victory.toml
```

Run with the PowerShell launcher:

```powershell
# Standard run
scripts\play.ps1 -Exe <path-to-your-title-executable> -Build release

# With ray tracing + native DXR 1.1 hardware acceleration
scripts\play.ps1 -Exe <path-to-your-title-executable> -Build release -RayTracing -NativeRT

# Safe fallback mode (disables experimental optimizations)
scripts\play.ps1 -Exe <path-to-your-title-executable> -Safe
```

Or with `Play-VIctory.cmd`:

```powershell
$env:VICTORY_GAME_EXE = 'D:\Games\MyDump\title.exe'
.\Play-VIctory.cmd native-rt
```

Optional ahead-of-time shader/pipeline warm-up:

```powershell
target\release\victory.exe shaders <path-to-shader-archives>
target\release\victory.exe warm --used
```

---

## Key Environment Switches

| Variable | Effect |
|---|---|
| `VICTORY_NATIVE_RT=1` | Enable native D3D12 DXR 1.1 `RayQuery` lifting & host AS management |
| `VICTORY_NATIVE_RT_ANYHIT=1` | Force closest-hit `AnyHit` loops onto `RayQuery` `Proceed()` (default off on RDNA 4) |
| `VICTORY_BVH_BUILDER=1` | Enable BVH builder support (paired with `VICTORY_NATIVE_RT=1` or software fallback) |
| `VICTORY_D3D12_DRED=1` / `light` | Enable D3D12 DRED breadcrumbs for GPU fault diagnostics |
| `VICTORY_GPU_TIMING=1` | Log per-command GPU execution times (pair with `diag` build + DRED) |
| `VICTORY_CACHE_DIR` | Override local shader/pipeline cache directory (default under `%LOCALAPPDATA%`) |

For additional switches and architecture notes, see [docs/architecture.md](docs/architecture.md), [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md), [docs/findings.md](docs/findings.md), and [docs/README-fxceless.md](docs/README-fxceless.md).

---

## Credits & License

Licensed under the **GNU General Public License v3.0** ([LICENSE](LICENSE)).

Built upon the work of the VIctory community:
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — core loader, runtime HLE, D3D12 backend, and shader recompiler.
- **fxceless**, **avatarmanz**, **codec-d**, **DexNom**, **Belobug**, **romainhedouin**, **ZJOU**, **myokex** (with Spexivus, joroyt, tightfall), **Joshua Odom**, **Geckomacabre**, and **cxy0072019-sketch** — upstream renderer, residency, read-back, Wave32, and AMD/NVIDIA improvements.
