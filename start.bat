@echo off
TITLE Remember Me AI - Sovereign Shell
COLOR 0A

echo ====================================================
echo    REMEMBER ME AI: SOVEREIGN TRINITY BOOTLOADER
echo ====================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

:: Check for virtualenv or just install deps
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate

echo [SETUP] Checking dependencies...
pip install -r requirements.txt >nul 2>&1
pip install pypdf >nul 2>&1

echo [BOOT] Launching Sovereign Web Interface...
streamlit run src/interface.py

echo.
echo [SHUTDOWN] System halted.
pause
