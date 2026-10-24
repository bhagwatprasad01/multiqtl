@echo off
REM Opens the MultiQTL interface in your web browser. Close this window to stop it.
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat || (echo Run install_windows.bat first. & pause & exit /b 1)
python -m multiqtl gui
pause
