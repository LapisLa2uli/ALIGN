param([switch]$Foreground)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runtimeDir = Join-Path $projectRoot '.local/fish'
$pythonPath = Join-Path $runtimeDir 'venv/Scripts/python.exe'
$serverPath = Join-Path $PSScriptRoot 'fish_local_server.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Local Fish runtime is missing. See docs/local_fish.md.' }
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8081/v1/health' -TimeoutSec 3
    if ($health.backend -eq 'fish-speech-1.5') { Write-Output 'Fish Speech is already running on localhost:8081.'; exit 0 }
    throw 'Port 8081 belongs to another service.'
} catch {
    if ($_.Exception.Message -eq 'Port 8081 belongs to another service.') { throw }
}
if ($Foreground) {
    & $pythonPath $serverPath
    exit $LASTEXITCODE
}
$process = Start-Process -FilePath $pythonPath -ArgumentList @('"' + $serverPath + '"') `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log')
Write-Output "Loading Fish Speech (launcher PID $($process.Id))..."
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8081/v1/health' -TimeoutSec 2
        if ($health.backend -eq 'fish-speech-1.5') {
            $health.pid | Set-Content -LiteralPath (Join-Path $runtimeDir 'server.pid')
            Write-Output "Fish Speech ready at http://127.0.0.1:8081 (server PID $($health.pid))."
            exit 0
        }
    } catch { }
    if ($process.HasExited) { throw 'Fish server exited. Check .local/fish/server.stderr.log.' }
}
throw 'Fish startup timed out. Check .local/fish/server.stderr.log before starting another process.'
