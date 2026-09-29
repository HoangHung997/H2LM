@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag the FOLDER containing the three original PDFs onto this script.
  echo The filenames and checksums must match data/scan_seed_20260929/draft_labels.json.
  pause
  exit /b 1
)
if not exist ".venv-scan\Scripts\python.exe" (
  py -3.11 -m venv .venv-scan
  if errorlevel 1 goto failed
)
".venv-scan\Scripts\python.exe" -m pip install -e ".[scan]"
if errorlevel 1 goto failed
".venv-scan\Scripts\python.exe" -m h2lm.scans.review_seed --seed "data/scan_seed_20260929/draft_labels.json" --input "%~1" --output "artifacts/real-scan-review-%RANDOM%-%RANDOM%"
if errorlevel 1 goto failed
echo Draft review only. No neural training, no document upload. See the local review.html.
pause
exit /b 0
:failed
echo Failed. Do not use incomplete output. Requires Python 3.11 x64 and the py launcher.
pause
exit /b 1
