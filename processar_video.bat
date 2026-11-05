@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" processar_video.py %*
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python nao encontrado. Execute start.bat para configurar o ambiente.
        pause
        exit /b 1
    )
    python processar_video.py %*
)

pause

