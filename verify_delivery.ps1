param(
    [switch]$RequireGatePass
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "env\Scripts\python.exe")
)
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += $PythonCommand.Source
}

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -and (Test-Path $Candidate)) {
        & $Candidate -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $Python = $Candidate
            break
        }
    }
}
if (!$Python) {
    Write-Error "Working Python was not found. Run .\install.ps1 first."
}

Set-Location $ProjectRoot
$ArgsList = @(
    "scripts\verify_delivery.py",
    "--project-root", $ProjectRoot,
    "--model", (Join-Path $ProjectRoot "models\poker_policy.joblib"),
    "--zip", (Join-Path $ProjectRoot "release\poker-decision-agent.zip"),
    "--json-out", (Join-Path $ProjectRoot "reports\delivery_verification.json")
)
if ($RequireGatePass) {
    $ArgsList += "--require-gate-pass"
}

& $Python @ArgsList
