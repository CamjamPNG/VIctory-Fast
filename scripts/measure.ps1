# One session, one row of measurements.csv: docs/weak-target.md says how.
#   scripts\measure.ps1 -Exe <path of your own copy's executable> -Tag water
#
# -Exe       the title's executable (what VICTORY_GAME_EXE takes)
# -Seconds   how long the title runs before it is closed (default 90)
# -Preset    weak (the default), pareto-residency or native-rt, as Play-VIctory.cmd takes it
# -Save      a folder or file of your own save, put back before the run: the save in the
#            data folder is copied aside first, under measurements\
# -Tag       what this run is, for the row (default: <preset>-<time>)
#
# The log has to be at info: VICTORY_LOG=warn (Play-VIctory.cmd's default) hides the
# per-second "N fps, M ms/frame (...)" line and the "memory summary: ..." line this
# reads, so VICTORY_LOG is set to info here and that is recorded with the run.
# Every other VICTORY_* variable in this window is left as it is and recorded too:
# Play-VIctory.cmd sets nothing it finds already defined, so yours win.
# Run it from a normal window, not as administrator (the driver then keeps its shader
# cache elsewhere), and from the same spot in the title each time: the row is the
# whole session, loading included.
param(
    [Parameter(Mandatory)][string]$Exe,
    [ValidateRange(5, 3600)][int]$Seconds = 90,
    [string]$Tag = '',
    [ValidateSet('weak', 'pareto-residency', 'native-rt')][string]$Preset = 'weak',
    [string]$Save = ''
)
$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function Round-Str([double]$value, [int]$digits) {
    if ([double]::IsNaN($value)) { return 'n/a' }
    $value.ToString("F$digits", $culture)
}

function Median([double[]]$values) {
    if ($values.Count -eq 0) { return [double]::NaN }
    $sorted = $values | Sort-Object
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return $sorted[$middle] }
    ($sorted[$middle - 1] + $sorted[$middle]) / 2
}

$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$launcher = Join-Path $root 'Play-VIctory.cmd'
$csv = Join-Path $root 'measurements.csv'
$exePath = (Resolve-Path -LiteralPath $Exe).Path
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $Tag) { $Tag = "$Preset-$stamp" }
$out = Join-Path $root 'measurements'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "$stamp-$Tag.log"

if (-not (Test-Path $launcher)) { throw "No $launcher." }
# What Play-VIctory.cmd will run (its own choice: release, else diag).
$runner = Join-Path $root 'target\release\victory.exe'
if (-not (Test-Path $runner)) { $runner = Join-Path $root 'target\diag\victory.exe' }
if (-not (Test-Path $runner)) { throw "Build it first: cargo build --release --locked -p victory" }

# The build: the commit, whether the tree it was built from was clean, and the file.
$sha = (& git -C $root rev-parse HEAD 2>$null)
if (-not $sha) { $sha = 'n/a' }
$dirty = (& git -C $root status --porcelain 2>$null | Measure-Object).Count -gt 0
$buildSha = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash.ToLowerInvariant()

# -Save: put the owner's save back, keeping the one there first.
if ($Save) {
    $dataDir = if ($env:VICTORY_DATA_DIR) { $env:VICTORY_DATA_DIR } else { Join-Path $env:LOCALAPPDATA 'victory\data' }
    $stores = @(Get-ChildItem -LiteralPath $dataDir -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'storage' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Sort-Object { (Get-Item -LiteralPath $_).LastWriteTime } -Descending)
    if ($stores.Count -eq 0) { throw "No save folder under $dataDir. Run the title once first." }
    $storage = $stores[0]
    Write-Host "save           $storage (backed up, then what -Save has is put there)"
    $backup = Join-Path $out "$stamp-$Tag-save-before"
    Copy-Item -LiteralPath $storage -Destination $backup -Recurse -Force
    $from = (Resolve-Path -LiteralPath $Save).Path
    if (Test-Path -LiteralPath $from -PathType Container) {
        Copy-Item -Path (Join-Path $from '*') -Destination $storage -Recurse -Force
    } else {
        Copy-Item -LiteralPath $from -Destination $storage -Force
    }
}

$env:VICTORY_GAME_EXE = $exePath
$env:VICTORY_LOG = 'info'

$info = @(
    "timestamp      $(Get-Date -Format o)",
    "tag            $Tag",
    "preset         $Preset",
    "seconds        $Seconds",
    "exe            $exePath",
    "runner         $runner",
    "git sha        $sha$(if ($dirty) { ' (tree not clean)' })",
    "build sha256   $buildSha",
    "log            $log",
    ''
)
'' | Out-File -LiteralPath $log -Encoding utf8
$info | Out-File -LiteralPath $log -Encoding utf8 -Append
# Every switch this run has, so a row can be reproduced.
Get-ChildItem Env: |
    Where-Object { $_.Name -like 'VICTORY_*' -or $_.Name -eq 'RUST_LOG' } |
    Sort-Object Name |
    ForEach-Object { "$($_.Name)=$($_.Value)" } |
    Out-File -LiteralPath $log -Encoding utf8 -Append
'' | Out-File -LiteralPath $log -Encoding utf8 -Append

# Run it through the preset, both streams into the log.
Write-Host "running        Play-VIctory.cmd $Preset for $Seconds s -> $log"
$command = 'call "{0}" {1} 1>>"{2}" 2>&1' -f $launcher, $Preset, $log
$shell = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $command) -WorkingDirectory $root -NoNewWindow -PassThru
$gameName = [System.IO.Path]::GetFileNameWithoutExtension($runner)
$game = $null
$began = Get-Date
for ($i = 0; $i -lt 60 -and -not $game; $i++) {
    Start-Sleep -Seconds 1
    $game = Get-Process -Name $gameName -ErrorAction SilentlyContinue |
        Where-Object { try { $_.StartTime -ge $began.AddSeconds(-1) } catch { $false } } |
        Select-Object -First 1
}
if (-not $game) {
    Write-Warning "No $gameName process started: the title did not get as far as running. See $log"
}
$exit = 'n/a'
$code = 'n/a'
if ($game) {
    # Run for the seconds asked for, unless the title ends first (a crash, or a quit).
    while (((Get-Date) - $began).TotalSeconds -lt $Seconds) {
        Start-Sleep -Milliseconds 500
        if ($game.HasExited) { $exit = 'exited-early'; break }
    }
    if (-not $game.HasExited) {
        # Closing the window is the title's own way out; a kill is the fallback.
        $closed = $game.CloseMainWindow()
        if ($closed -and $game.WaitForExit(20000)) {
            $exit = 'closed'
        } else {
            Stop-Process -Id $game.Id -Force
            $exit = 'forced'
        }
    }
    try { $game.Refresh(); $code = $game.ExitCode } catch { $code = 'n/a' }
}
if (-not $shell.WaitForExit(15000)) { Stop-Process -Id $shell.Id -Force }
Add-Content -LiteralPath $log -Value '' -Encoding utf8
Add-Content -LiteralPath $log -Value "# exit: $exit ($code) after $([int]((Get-Date) - $began).TotalSeconds) s" -Encoding utf8

# What the layer reported. The frame line is host.rs's "N fps, M ms/frame (prep: copies
# .. ms .. MB, textures .. ms .. MB; GPU; copying back .. ms .. MB)"; the memory line is
# host.rs's "memory summary: RAM private .. VRAM local used/budget, SHR used/budget ...
# OS page faults N/frame". Neither has a per-frame or a p99 figure: those are n/a.
$text = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue
if (-not $text) { $text = '' }
$frameRe = [regex]'(?m)^.*?(\d+(?:\.\d+)?) fps, (\d+(?:\.\d+)?) ms/frame \((\d+(?:\.\d+)?) preparing: copies (\d+(?:\.\d+)?) ms [^;]*; (\d+(?:\.\d+)?) on the GPU; (\d+(?:\.\d+)?) copying back'
$memRe = [regex]'(?m)^.*?memory summary: RAM private (\d+) MB, working set (\d+) MB; VRAM local (\d+)/(\d+) MB, SHR (\d+)/(\d+) MB'
$faultRe = [regex]'(?m)^.*?memory summary:.*?OS page faults (\d+(?:\.\d+)?)/frame'
$frames = @($frameRe.Matches($text))
$mems = @($memRe.Matches($text))
$faults = @($faultRe.Matches($text))

$fps = @($frames | ForEach-Object { [double]$_.Groups[1].Value })
$longest = @($frames | ForEach-Object { [double]$_.Groups[2].Value })
$prep = @($frames | ForEach-Object { [double]$_.Groups[3].Value })
$copy = @($frames | ForEach-Object { [double]$_.Groups[4].Value })
$wait = @($frames | ForEach-Object { [double]$_.Groups[5].Value })
$back = @($frames | ForEach-Object { [double]$_.Groups[6].Value })
$localUsed = @($mems | ForEach-Object { [double]$_.Groups[3].Value })
$localBudget = @($mems | ForEach-Object { [double]$_.Groups[4].Value })
$shrUsed = @($mems | ForEach-Object { [double]$_.Groups[5].Value })
$commit = @($mems | ForEach-Object { [double]$_.Groups[1].Value })
$faultRate = @($faults | ForEach-Object { [double]$_.Groups[1].Value })

$peak = { param($values) if ($values.Count -eq 0) { [double]::NaN } else { ($values | Measure-Object -Maximum).Maximum } }
$medianFps = Median $fps
$row = [pscustomobject]@{
    timestamp        = (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')
    tag              = $Tag
    preset           = $Preset
    sha              = if ($dirty) { "$($sha.Substring(0, [math]::Min(12, $sha.Length)))*" } else { $sha.Substring(0, [math]::Min(12, $sha.Length)) }
    build_sha256     = $buildSha
    median_fps       = Round-Str $medianFps 1
    p99_frame_ms     = 'n/a'
    prep_ms          = Round-Str (Median $prep) 1
    copyback_ms      = Round-Str (Median $back) 1
    peak_local_vram_mb = Round-Str (& $peak $localUsed) 0
    peak_commit_mb   = Round-Str (& $peak $commit) 0
    faults_per_frame = Round-Str (Median $faultRate) 1
    exit             = $exit
    seconds          = $frames.Count
    gpu_wait_ms      = Round-Str (Median $wait) 1
    peak_shr_mb      = Round-Str (& $peak $shrUsed) 0
    local_budget_mb  = Round-Str (& $peak $localBudget) 0
    exit_code        = $code
    log              = if ($log.StartsWith($root)) { $log.Substring($root.Length).TrimStart('\', '/') } else { $log }
}
$columns = @($row.PSObject.Properties.Name)
if (Test-Path $csv) {
    $header = (Get-Content -LiteralPath $csv -TotalCount 1) -split ',' | ForEach-Object { $_.Trim('"') }
    if (($header -join ',') -ne ($columns -join ',')) {
        throw "$csv has other columns ($($header -join ',')); move it aside and run again."
    }
}
$row | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding utf8 -Append

if ($frames.Count -eq 0) {
    Write-Warning "The log has no frame line: the title never reached a frame line once a second, or it was not at info. See $log"
}
Write-Host ''
Write-Host "measurements.csv:"
$row | Format-List
Write-Host 'The log has no per-frame times, so p99_frame_ms is n/a. For it, and to check the'
Write-Host 'VRAM figure, press F3 in the title (VICTORY_PANEL shows the box without it) and read,'
Write-Host 'in its upper left, at the end of a run in a steady scene:'
Write-Host '  FPS <n>  FRAME <ms> MS  LONGEST <ms>      <- type FRAME and LONGEST into the row'
Write-Host '  VRAM <used>/<budget> GB  SHARED <n>      <- the panel is rounded to 0.1 GB'
Write-Host 'LONGEST is the longest frame of the last half second, not a p99 (host/panel.rs).'
