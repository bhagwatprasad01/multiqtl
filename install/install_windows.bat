@echo off
REM ---- MultiQTL installer for Windows ------------------------------------------------
REM Double-click this file. Requires Python 3.9+ from https://www.python.org/downloads/
REM (tick "Add python.exe to PATH" during Python installation).
cd /d "%~dp0\.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python not found. Install it from python.org and re-run. & pause & exit /b 1)
echo Creating a private environment in .venv ...
%PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m multiqtl --version && echo. && echo Installation complete. Double-click install\start_multiqtl_windows.bat to open MultiQTL.
pause
