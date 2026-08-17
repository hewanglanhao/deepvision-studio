param(
  [switch]$Check,
  [switch]$Force,
  [ValidateRange(1, 8)]
  [int]$Concurrency = 4,
  [string]$SourceBase = "https://poloclub.github.io/transformer-explainer"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$frontendRoot = Join-Path $repoRoot "frontend"
$setupScript = Join-Path $frontendRoot "scripts\setup-mode-d-assets.mjs"

if (-not (Test-Path -LiteralPath $setupScript)) {
  throw "Mode D setup script not found: $setupScript"
}

$nodeArgs = @($setupScript)
if ($Check) { $nodeArgs += "--check" }
if ($Force) { $nodeArgs += "--force" }
$nodeArgs += @("--concurrency", $Concurrency, "--source-base", $SourceBase)

& node @nodeArgs
if ($LASTEXITCODE -ne 0) {
  throw "Mode D asset setup failed with exit code $LASTEXITCODE"
}
