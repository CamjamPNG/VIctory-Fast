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
- **Host DXR Acceleration Structure Manager (`crates/victory-gpu/src/dxr.rs`):** Builds and updates native D3D12 multi-geometry static BLASes, dynamic BLASes, and TLASes directly on host RT hardware with topology validation, safe deferred resource retention, and null-TLAS fallback handling.
- **RDNA 2 → DXIL SM 6.6 `RayQuery` Lifter (`crates/victory-shader/src/emit/rayquery_lift.rs`, `lower.rs`):** Detects `image_bvh*_intersect_ray` traversal loops (opaque, front/back-track, and alpha-tested any-hit loops) and lifts them into native DXIL Shader Model 6.6 `RayQuery` (`TraceRayInline` / `Proceed` / `Committed*` / `Candidate*`) instructions.
- **AMD RDNA 4 Stability (`amdxc64.dll`):** Fixes buffer `getDimensions` lowering crashes and validates lifted DXR shaders through `dxil.dll` and host D3D12 PSO creation.

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
