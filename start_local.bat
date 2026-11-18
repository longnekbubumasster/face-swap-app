@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PORT=7860
set MODEL_DIR=%CD%\.models
set FACE_MODEL=buffalo_s
set PRELOAD_MODELS=true
if not defined OPEN_BROWSER set OPEN_BROWSER=true
set GRADIO_ANALYTICS_ENABLED=False
set OMP_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

if errorlevel 1 (
    echo.
    echo Khong the khoi dong. Hay chay setup_local.bat truoc.
    pause
)
