$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot "env\Scripts\python.exe"
$Dataset = "C:\Users\user\Desktop\AllFile\dataset"
$Model = "C:\Users\user\Desktop\AllFile\poker_policy.json"

Set-Location $ProjectRoot

& $Python scripts\train_policy.py `
    --dataset $Dataset `
    --model-out $Model `
    --epochs 3 `
    --max-examples 50000

& $Python scripts\evaluate_policy.py `
    --dataset $Dataset `
    --model $Model `
    --max-examples 50000

