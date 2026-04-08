# Helper script to run `openenv push` with UTF-8 encoding on Windows

$ErrorActionPreference = "Stop"

# Force Python and console to use UTF-8 so emojis/logging don't break
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
} catch {
    # If setting console encoding fails, continue anyway
}

# Change to test_env directory relative to this script
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envDir = Join-Path $repoRoot "test_env"
Set-Location $envDir

# Run openenv push for this environment
openenv push --repo-id anshraz27/icu-env
