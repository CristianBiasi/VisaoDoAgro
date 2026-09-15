@echo off
setlocal
cd /d "%~dp0"
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo AgroSafe Vision is already running on port 8000.
  echo Open https://localhost:8000 in your browser.
  exit /b 0
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH. Install Python 3.11+ and enable Add Python to PATH.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)
set "PYTHON=%~dp0.venv\Scripts\python.exe"
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
if not exist .certs mkdir .certs
if not exist .certs\cert.pem (
  echo Generating local HTTPS certificate...
  "%PYTHON%" generate_cert.py
)
echo.
echo AGROSAFE VISION
if exist .certs\cert.pem (
  echo Dashboard: https://localhost:8000
  echo Phone Camera: use the QR Code shown in the dashboard
) else (
  echo Dashboard: http://localhost:8000
)
echo.
if exist .certs\cert.pem (
  "%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile .certs\key.pem --ssl-certfile .certs\cert.pem
) else (
  echo No OpenSSL certificate found. Install mkcert or use a HTTPS tunnel for phone camera access.
  "%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000
)
