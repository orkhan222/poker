param(
    [string]$ImageName = "poker-decision-agent:latest",
    [string]$ContainerName = "poker-decision-agent-check",
    [int]$HostPort = 8011
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$DockerArgs
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @DockerArgs
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    if ($ExitCode -ne 0) {
        throw "Docker command failed: docker $($DockerArgs -join ' ')"
    }
}

function Test-DockerDaemon {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker info *> $null
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    return $ExitCode -eq 0
}

if (!(Test-DockerDaemon)) {
    Write-Error "Docker is installed, but the Docker daemon is not reachable or healthy. Start/restart Docker Desktop, wait until it is ready, then run .\verify_docker.ps1 again."
}

Write-Host "Building Docker image..." -ForegroundColor Green
Invoke-Docker build -t $ImageName .

$ExistingContainer = docker ps -aq --filter "name=^/$ContainerName$"
if ($LASTEXITCODE -ne 0) {
    throw "Docker command failed: docker ps -aq --filter name=^/$ContainerName$"
}
if ($ExistingContainer) {
    Invoke-Docker rm -f $ContainerName | Out-Null
}

Write-Host "Starting Docker container on host port $HostPort..." -ForegroundColor Green
Invoke-Docker run -d --name $ContainerName -p "${HostPort}:8001" $ImageName | Out-Null

try {
    $HealthUrl = "http://127.0.0.1:$HostPort/wsl --shutdown"
    $PredictUrl = "http://127.0.0.1:$HostPort/predict"
    $Ready = $false

    for ($Attempt = 1; $Attempt -le 20; $Attempt++) {
        try {
            $Health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
            if ($Health.status -eq "ok") {
                $Ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (!$Ready) {
        Write-Error "Docker container did not become healthy."
    }

    $Body = @{
        position = "BTN"
        street = "preflop"
        hole_cards = @("Ah", "Kd")
        board_cards = @()
        pot = 2.5
        to_call = 1.0
        stack = 100.0
        min_raise = 2.0
        player_count = 6
    } | ConvertTo-Json

    $Prediction = Invoke-RestMethod -Method Post -Uri $PredictUrl -ContentType "application/json" -Body $Body

    Write-Host ""
    Write-Host "Docker verification passed." -ForegroundColor Green
    Write-Host "Health: $HealthUrl"
    Write-Host "Predicted action: $($Prediction.action)"
}
finally {
    docker rm -f $ContainerName *> $null
}
