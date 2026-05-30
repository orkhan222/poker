param(
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot $VenvDir
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ModelPath = Join-Path $ProjectRoot "models\poker_policy.json"

Set-Location $ProjectRoot

if (!(Test-Path $ModelPath)) {
    Write-Error "Bundled model not found: $ModelPath"
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (!$PythonCommand) {
    Write-Error "Python was not found. Install Python 3.11+ and run this installer again."
}

Write-Host "Creating virtual environment..." -ForegroundColor Green
if (!(Test-Path $VenvPython)) {
    & $PythonCommand.Source -m venv $VenvPath
}

Write-Host "Installing Python dependencies..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host "Start the app with:"
Write-Host ".\run_server.ps1"
Write-Host ""
Write-Host "Open:"
Write-Host "http://127.0.0.1:8001/predict"
