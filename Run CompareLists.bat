@echo off
REM Launch the CompareLists web app (Windows). Double-click this file.
cd /d "%~dp0"

if not exist ".venv" (
  echo CompareLists is not installed yet. Running the installer first...
  echo.
  call "Install CompareLists.bat"
)

echo Starting CompareLists... your browser will open shortly.
echo Keep this window open while you use the app. Press Ctrl-C here to stop it.
echo.
set COMPARELISTS_OPEN_BROWSER=1
uv run comparelists-web
pause
