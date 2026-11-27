@echo off
setlocal EnableExtensions DisableDelayedExpansion
if not "%~1"=="" if /i not "%~1"=="pareto-residency" if /i not "%~1"=="native-rt" if /i not "%~1"=="weak" if /i not "%~1"=="weak-540" (
  echo Usage: Play-VIctory.cmd [pareto-residency^|native-rt^|weak^|weak-540] 1>&2
  exit /b 2
)
if not defined VICTORY_GAME_EXE (
  echo Set VICTORY_GAME_EXE to the full path of your own title executable. 1>&2
  exit /b 2
)
if not exist "%VICTORY_GAME_EXE%" (
  echo Executable not found: %VICTORY_GAME_EXE% 1>&2
  exit /b 2
)
set "runner=%~dp0target\release\victory.exe"
if not exist "%runner%" set "runner=%~dp0target\diag\victory.exe"
if not exist "%runner%" (
  echo Build first with: cargo build --release --locked -p victory 1>&2
  exit /b 2
)
if /i "%~1"=="native-rt" (
  if not defined VICTORY_NATIVE_RT set "VICTORY_NATIVE_RT=1"
  if not defined VICTORY_BVH_BUILDER set "VICTORY_BVH_BUILDER=1"
)
rem weak: for a 6 GiB / 6-core PC (GTX 1660 SUPER, Ryzen 5 3600): 1080p output, no DXR, and
rem every pool sized for six cores. weak-540 renders 960x540 for real at a 720p output, past
rem the title's 2/3 clamp on the render scale. Why each line: docs/weak-target.md.
if /i "%~1"=="weak" set "victory_weak=1"
if /i "%~1"=="weak-540" set "victory_weak=1"
if /i "%~1"=="weak-540" (
  if not defined VICTORY_OUTPUT set "VICTORY_OUTPUT=1280x720"
  if not defined VICTORY_RENDER_SCALE set "VICTORY_RENDER_SCALE=0.75"
)
if defined victory_weak (
  if not defined VICTORY_OUTPUT set "VICTORY_OUTPUT=1920x1080"
  if not defined VICTORY_RENDER_SCALE set "VICTORY_RENDER_SCALE=0.5"
  if not defined VICTORY_UPSCALER set "VICTORY_UPSCALER=6"
  if not defined VICTORY_TEXTURE_BUDGET_MB set "VICTORY_TEXTURE_BUDGET_MB=3200"
  if not defined VICTORY_TEXTURE_MAX set "VICTORY_TEXTURE_MAX=512"
  if not defined VICTORY_TEXTURE_STAGE_MB set "VICTORY_TEXTURE_STAGE_MB=96"
  if not defined VICTORY_CARD_BUFFERS_MAX_MB set "VICTORY_CARD_BUFFERS_MAX_MB=512"
  if not defined VICTORY_NATIVE_RT set "VICTORY_NATIVE_RT=0"
  if not defined VICTORY_BVH_BUILDER set "VICTORY_BVH_BUILDER=0"
  if not defined VICTORY_THIN_THREADS set "VICTORY_THIN_THREADS=4"
  if not defined VICTORY_VIEW_THREADS set "VICTORY_VIEW_THREADS=3"
  if not defined VICTORY_UNDER_THREADS set "VICTORY_UNDER_THREADS=4"
  if not defined VICTORY_LAND_THREADS set "VICTORY_LAND_THREADS=3"
  if not defined VICTORY_WIDE_THREADS set "VICTORY_WIDE_THREADS=4"
  if not defined VICTORY_PIPELINE_WORKERS set "VICTORY_PIPELINE_WORKERS=4"
  if not defined VICTORY_JOIN_PARTS set "VICTORY_JOIN_PARTS=4"
  rem Off the system SSD (C:), which Windows shares: see docs/weak-target.md.
  if not defined VICTORY_CACHE_DIR (
    if not exist "D:\VIctory\cache\" mkdir "D:\VIctory\cache" >nul 2>nul
    if exist "D:\VIctory\cache\" set "VICTORY_CACHE_DIR=D:\VIctory\cache"
  )
  if not defined VICTORY_DATA_DIR (
    if not exist "D:\VIctory\data\" mkdir "D:\VIctory\data" >nul 2>nul
    if exist "D:\VIctory\data\" set "VICTORY_DATA_DIR=D:\VIctory\data"
  )
  if not defined VICTORY_CACHE_DIR echo VIctory: D:\VIctory\cache could not be made; the cache stays under %LOCALAPPDATA% 1>&2
  if not defined VICTORY_DATA_DIR echo VIctory: D:\VIctory\data could not be made; saves stay under %LOCALAPPDATA% 1>&2
  set "victory_weak="
)
if not defined VICTORY_AMBIENT_AIR set "VICTORY_AMBIENT_AIR=1.5"
if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%LOCALAPPDATA%\VIctory\cache-pareto"
if not defined VICTORY_TEXTURE_NO_WAIT set "VICTORY_TEXTURE_NO_WAIT=1"
if not defined VICTORY_RELAY set "VICTORY_RELAY=2"
if not defined VICTORY_WIDE_BACK set "VICTORY_WIDE_BACK=1"
if not defined VICTORY_CARD_REHASH_RUNS set "VICTORY_CARD_REHASH_RUNS=1"
if not defined VICTORY_JOIN_COPIES set "VICTORY_JOIN_COPIES=1"
if not defined VICTORY_INDIRECT_WINDOW set "VICTORY_INDIRECT_WINDOW=1"
if not defined VICTORY_RELAY_COPY set "VICTORY_RELAY_COPY=1"
if not defined VICTORY_CARD_STEPS set "VICTORY_CARD_STEPS=2"
if not defined VICTORY_INDIRECT_BATCH set "VICTORY_INDIRECT_BATCH=2"
if not defined VICTORY_MAP_THREADS set "VICTORY_MAP_THREADS=1"
if not defined VICTORY_CARD_AWAKE set "VICTORY_CARD_AWAKE=1"
if not defined VICTORY_SHOW_AFTER set "VICTORY_SHOW_AFTER=1"
if not defined VICTORY_AMBIENT set "VICTORY_AMBIENT=0.8"
if not defined VICTORY_THIN set "VICTORY_THIN=1"
if not defined VICTORY_JOIN_PARTS set "VICTORY_JOIN_PARTS=8"
if not defined VICTORY_UPLOAD_AHEAD set "VICTORY_UPLOAD_AHEAD=1"
if not defined VICTORY_LEVELS_ON_DEMAND set "VICTORY_LEVELS_ON_DEMAND=1"
if not defined VICTORY_CARD_BUFFERS_MIN_KB set "VICTORY_CARD_BUFFERS_MIN_KB=64"
if not defined VICTORY_LOG if not defined RUST_LOG set "VICTORY_LOG=warn"
"%runner%" run "%VICTORY_GAME_EXE%" --console series-s --execute
exit /b %errorlevel%
