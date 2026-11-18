@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] Tao moi truong Python...
python -m venv .venv
if errorlevel 1 goto :error

echo [2/4] Cap nhat pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo [3/4] Cai cac thu vien local...
".venv\Scripts\python.exe" -m pip install -r requirements-local.txt
if errorlevel 1 goto :error

echo [4/4] Tai va kiem tra model AI...
set MODEL_DIR=%CD%\.models
set FACE_MODEL=buffalo_s
".venv\Scripts\python.exe" -m scripts.download_models
if errorlevel 1 goto :error

echo.
echo Cai dat thanh cong. Hay double-click start_local.bat de chay.
pause
exit /b 0

:error
echo.
echo Cai dat that bai. Kiem tra ket noi Internet va Python 3.12.
pause
exit /b 1
