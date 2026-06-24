@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    exit /b 0
)

if exist "env\Scripts\activate.bat" (
    call "env\Scripts\activate.bat"
    exit /b 0
)

echo A complete virtual environment was not found.
echo Run this command first:
echo powershell -ExecutionPolicy Bypass -File .\install.ps1
exit /b 1
