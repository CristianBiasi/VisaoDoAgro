@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente Python ausente. Execute start.bat para instalar as dependencias.
  exit /b 1
)
".venv\Scripts\python.exe" -m training.train %*
exit /b %errorlevel%
