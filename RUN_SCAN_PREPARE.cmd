@echo off
setlocal
cd /d "%~dp0"
echo H2LM - LOCAL SCAN PREPARATION. NOT OCR OR A TRAINED MODEL.
echo The original PDF remains unchanged. No documents are uploaded.
if not exist ".venv-scan\Scripts\python.exe" (
  py -3.11 -m venv .venv-scan
  if errorlevel 1 goto failed
)
".venv-scan\Scripts\python.exe" -m pip install -e ".[scan]"
if errorlevel 1 goto failed
if "%~1"=="" (
  ".venv-scan\Scripts\python.exe" -m h2lm.scans.cli
) else (
  ".venv-scan\Scripts\python.exe" -m h2lm.scans.cli --pdf "%~1"
)
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Failed. Requires Python 3.11 x64 and py launcher. See local logs, do not use partial results.
pause
exit /b 1
