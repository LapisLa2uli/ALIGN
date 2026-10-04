param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runtimeDir = Join-Path $projectRoot '.local/fish'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$sourceDir = Join-Path $runtimeDir 'fish-speech-1.5.1'
if (-not (Test-Path -LiteralPath $sourceDir)) {
    $archive = Join-Path $runtimeDir 'source-v1.5.1.zip'
    Invoke-WebRequest 'https://codeload.github.com/fishaudio/fish-speech/zip/refs/tags/v1.5.1' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeDir
}
$pythonPath = Join-Path $runtimeDir 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & $Python -m venv (Join-Path $runtimeDir 'venv')
    if ($LASTEXITCODE) { throw 'Failed to create Fish environment. Use Python 3.11.' }
}
& $pythonPath -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed.' }
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'fish_local_requirements.txt')
if ($LASTEXITCODE) { throw 'Fish inference dependency installation failed.' }
& $pythonPath (Join-Path $PSScriptRoot 'download_fish_local.py') --output (Join-Path $runtimeDir 'checkpoints/fish-speech-1.5')
if ($LASTEXITCODE) { throw 'Fish weights download or verification failed.' }
Write-Output 'Installed. Start with DataCreate/scripts/start_fish_local.ps1.'
