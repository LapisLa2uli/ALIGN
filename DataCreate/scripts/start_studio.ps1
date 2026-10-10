param(
    [int]$Port = 8765,
    [string]$Python = 'python',
    [string]$KeyFile,
    [string]$FeedbackConfig
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $KeyFile) { $KeyFile = Join-Path $projectRoot 'llmauth.txt' }
if (-not $FeedbackConfig) { $FeedbackConfig = $env:ALIGN_FEEDBACK_CONFIG }
if (-not $FeedbackConfig) { $FeedbackConfig = Join-Path $projectRoot 'DataCreate/config/feedback.fish.yaml' }
$FeedbackConfig = (Resolve-Path -LiteralPath $FeedbackConfig).Path
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }

# Validate the provider and extract only the credential, never the complete file.
$authText = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $KeyFile).Path)
if ($authText -notmatch 'https://api\.ssstoken\.net(?:/|\s|$)') {
    throw 'The key file must identify https://api.ssstoken.net as its provider.'
}
$keyMatches = [regex]::Matches($authText, 'sk-[A-Za-z0-9_-]+')
if ($keyMatches.Count -ne 1) { throw 'Expected exactly one API key in the ssstoken key file.' }

$previousKey = $env:SSSTOKEN_API_KEY
$previousConfig = $env:ALIGN_FEEDBACK_CONFIG
$previousPythonPath = $env:PYTHONPATH
$previousNumbaCache = $env:NUMBA_CACHE_DIR
try {
    $env:SSSTOKEN_API_KEY = $keyMatches[0].Value
    $env:ALIGN_FEEDBACK_CONFIG = $FeedbackConfig
    $sourceDir = Join-Path $projectRoot 'DataCreate/src'
    $env:PYTHONPATH = if ($previousPythonPath) { "$sourceDir;$previousPythonPath" } else { $sourceDir }
    # Avoid Windows profile-cache permission stalls while importing the model.
    if (-not $env:NUMBA_CACHE_DIR) {
        $env:NUMBA_CACHE_DIR = Join-Path $projectRoot 'DataCreate/work/studio-numba-cache'
        New-Item -ItemType Directory -Force -Path $env:NUMBA_CACHE_DIR | Out-Null
    }
    Write-Output 'Starting studio with the saved ssstoken credential.'
    Write-Output "Feedback configuration: $FeedbackConfig"
    Write-Output "Practice studio: http://127.0.0.1:$Port/studio"
    # Run in this terminal so Ctrl+C stops the server; do not create a hidden process.
    & $Python -m datacreate.cli serve --port $Port
} finally {
    $env:SSSTOKEN_API_KEY = $previousKey
    $env:ALIGN_FEEDBACK_CONFIG = $previousConfig
    $env:PYTHONPATH = $previousPythonPath
    $env:NUMBA_CACHE_DIR = $previousNumbaCache
    $authText = $null
    $keyMatches = $null
}
