@echo off
REM CompareLists installer (Windows). Double-click this file.
cd /d "%~dp0"

echo ===============================================
echo   CompareLists - installation
echo ===============================================
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv ^(the Python tool runner^)...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  echo.
  echo If uv is not found after this, close this window, reopen it, and run again.
)

echo Installing dependencies ^(a couple of minutes the first time^)...
uv sync
if errorlevel 1 (
  echo.
  echo Installation failed. See the messages above.
  pause
  exit /b 1
)

echo.
echo Downloading the model ^(first time only^)...
uv run comparelists --prefetch

echo.
echo ===============================================
echo   Done. Double-click "Run CompareLists.bat" to start.
echo ===============================================
pause
