$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot "env\Scripts\python.exe"
$ModelPath = "C:\Users\user\Desktop\AllFile\poker_policy.json"

if (!(Test-Path $Python)) {
    Write-Error "Python env not found: $Python"
}

if (!(Test-Path $ModelPath)) {
    Write-Error "Model not found: $ModelPath. Run scripts\train_policy.py first."
}

$env:POKER_POLICY_PATH = $ModelPath
Set-Location $ProjectRoot

Write-Host "Starting Poker Agent API..."
Write-Host "Docs: http://127.0.0.1:8000/docs"
& $Python -m uvicorn poker_agent.service:app --host 127.0.0.1 --port 8000

