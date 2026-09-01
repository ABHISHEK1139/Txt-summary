@echo off
setlocal enabledelayedexpansion
title AI Document Summarizer
cls

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║        AI Document Summarizer - Auto Setup          ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ----------------------------------------------------------
:: 1. Find Python
:: ----------------------------------------------------------
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not on PATH.
    echo.
    echo  Download from: https://www.python.org/downloads/
    echo  IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] %PYVER%

:: ----------------------------------------------------------
:: 2. Create virtual environment (first run only)
:: ----------------------------------------------------------
if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo.
    echo  ────────────────────────────────────────────────────
    echo   FIRST RUN - Full setup (this takes a few minutes)
    echo  ────────────────────────────────────────────────────
    echo.
    echo  [1/5] Creating virtual environment...
    python -m venv "%~dp0.venv"
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK]  Virtual environment created.
) else (
    echo  [OK] Virtual environment found.
)

:: ----------------------------------------------------------
:: 3. Activate virtual environment
:: ----------------------------------------------------------
call "%~dp0.venv\Scripts\activate.bat"

:: ----------------------------------------------------------
:: 4. Install Python packages
:: ----------------------------------------------------------
set "STAMP=%~dp0.venv\.deps_installed"
set "REQS=%~dp0requirements.txt"

set NEED_INSTALL=0
if not exist "!STAMP!" set NEED_INSTALL=1

if !NEED_INSTALL!==0 (
    for %%R in ("!REQS!") do set REQ_DATE=%%~tR
    for %%S in ("!STAMP!") do set STAMP_DATE=%%~tS
    if "!REQ_DATE!" gtr "!STAMP_DATE!" set NEED_INSTALL=1
)

if !NEED_INSTALL!==1 (
    echo.
    echo  [2/5] Installing Python packages...
    echo        (fastapi, torch, transformers, PyMuPDF, etc.)
    echo.
    pip install --upgrade pip >nul 2>&1
    pip install -r "%~dp0requirements.txt"
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Package installation failed. Check errors above.
        pause
        exit /b 1
    )
    echo. > "!STAMP!"
    echo.
    echo  [OK]  All Python packages installed.
) else (
    echo  [OK] Python packages up to date.
)

:: ----------------------------------------------------------
:: 5. Download NLTK data
:: ----------------------------------------------------------
echo  [3/5] Checking NLTK data...
python -c "import nltk, os; d=os.path.join(os.path.expanduser('~'),'nltk_data'); nltk.download('punkt', quiet=True, download_dir=d); nltk.download('punkt_tab', quiet=True, download_dir=d); nltk.download('stopwords', quiet=True, download_dir=d); print('  [OK]  NLTK data ready.')"
if %errorlevel% neq 0 (
    echo  [WARN] NLTK download had issues - app will retry on first use.
)

:: ----------------------------------------------------------
:: 6. Ensure model is available
:: ----------------------------------------------------------
echo  [4/5] Checking AI model...
if exist "%~dp0models\final\config.json" (
    echo  [OK]  Local fine-tuned model found.
) else (
    echo  [!!]  No local model in models\final\
    echo        Downloading fallback model (t5-small ~240MB)...
    echo.
    python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; print('        Downloading tokenizer...'); t=AutoTokenizer.from_pretrained('t5-small'); print('        Downloading model weights...'); m=AutoModelForSeq2SeqLM.from_pretrained('t5-small'); import os; p=os.path.join('%~dp0'.rstrip('\\'),'models','final'); os.makedirs(p,exist_ok=True); t.save_pretrained(p); m.save_pretrained(p); print('  [OK]  t5-small saved to models\\final\\')"
    if %errorlevel% neq 0 (
        echo  [WARN] Model download failed - app will try again on first use.
    )
)

:: ----------------------------------------------------------
:: 7. Create uploads folder
:: ----------------------------------------------------------
if not exist "%~dp0uploads" mkdir "%~dp0uploads"

:: ----------------------------------------------------------
:: 8. Launch
:: ----------------------------------------------------------
echo  [5/5] Starting server...
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║                                                      ║
echo  ║   App running at: http://127.0.0.1:8000              ║
echo  ║   Browser will open automatically.                   ║
echo  ║                                                      ║
echo  ║   Press Ctrl+C in this window to stop.               ║
echo  ║                                                      ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: Open browser after server has time to start
python -c "import webbrowser,threading;threading.Thread(target=lambda:(__import__('time').sleep(3),webbrowser.open('http://127.0.0.1:8000'))).start()"

:: Start server
python -m uvicorn src.server:app --reload

echo.
echo  Server stopped.
pause
