@echo off
title Document Summarizer
cls

echo ========================================================
echo       Starting AI Document Summarizer System
echo ========================================================
echo.

:: Activate virtual environment if it exists
if exist "%~dp0.venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call "%~dp0.venv\Scripts\activate.bat"
) else (
    echo [!] Virtual environment not found at %~dp0.venv
    echo     Attempting to run using system Python...
)

echo.
echo [*] Opening Application in Browser shortly...
:: Use Python to open the browser after a short delay so the server has time to start
python -c "import webbrowser, time, threading; threading.Thread(target=lambda: (time.sleep(3), webbrowser.open('http://127.0.0.1:8000'))).start()"

echo [*] Starting Backend Server...
echo     (Press Ctrl+C in this window to stop the server)
echo.

:: Start the server in the current window
cd "%~dp0app"
python -m uvicorn server:app --reload

echo.
echo [*] Server stopped.
pause
