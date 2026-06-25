param(
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot $VenvDir
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvConfig = Join-Path $VenvPath "pyvenv.cfg"
$ModelPath = Join-Path $ProjectRoot "models\poker_policy.joblib"

Set-Location $ProjectRoot

if (!(Test-Path $ModelPath)) {
    Write-Error "Bundled model not found: $ModelPath"
}

$PythonCandidates = @()
$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    $PythonCandidates += @{
        executable = $PyLauncher.Source
        prefix = @("-3.11")
    }
}
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonCandidates += @{
        executable = $PythonCommand.Source
        prefix = @()
    }
}
if (!$PythonCandidates) {
    Write-Error "Python was not found. Install Python 3.11+ and run this installer again."
}

$PythonRuntime = $null
foreach ($Candidate in $PythonCandidates) {
    & $Candidate.executable @($Candidate.prefix) -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 12) else 1)" *> $null
    if ($LASTEXITCODE -eq 0) {
        $PythonRuntime = $Candidate
        break
    }
}
if (!$PythonRuntime) {
    Write-Error "Python 3.11 is required by the bundled model runtime."
}

$VenvIsComplete = (Test-Path $VenvPython) -and (Test-Path $VenvConfig)
if ((Test-Path $VenvPath) -and !$VenvIsComplete) {
    $ResolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
    $ResolvedParent = (Resolve-Path -LiteralPath (Split-Path -Parent $VenvPath)).Path
    if (!$ResolvedParent.Equals($ResolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Error "Refusing to repair a virtual environment outside the project root: $VenvPath"
    }
    $BackupPath = Join-Path ([IO.Path]::GetTempPath()) "poker-agent-venv-corrupt-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Write-Host "Incomplete virtual environment detected. Moving it to $BackupPath" -ForegroundColor Yellow
    Move-Item -LiteralPath $VenvPath -Destination $BackupPath
}

Write-Host "Creating virtual environment..." -ForegroundColor Green
if (!(Test-Path $VenvPython)) {
    & $PythonRuntime.executable @($PythonRuntime.prefix) -m venv $VenvPath
    if ($LASTEXITCODE -ne 0 -or !(Test-Path $VenvPython)) {
        Write-Error "Virtual environment creation failed: $VenvPath"
    }
}

Write-Host "Installing Python dependencies..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host "Start the app with:"
Write-Host ".\run_server.ps1"
Write-Host "CMD activation: activate_env.cmd"
Write-Host "PowerShell activation: .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Open:"
Write-Host "http://127.0.0.1:8001/predict"
