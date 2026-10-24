@echo off
chcp 65001 >nul
rem 启动实时翻译。额外参数会传给程序，例如: start.bat --target en
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 还没安装，请先双击 install.bat
    pause
    exit /b 1
)
set MODEL=translategemma:12b
if exist config.bat call config.bat
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" live_translate.py --model %MODEL% %*
if errorlevel 1 pause
