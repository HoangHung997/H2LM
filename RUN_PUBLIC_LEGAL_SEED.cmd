@echo off
setlocal
cd /d "%~dp0"
echo H2LM - PUBLIC LEGAL TOKENIZER SEED - NOT A TRAINED NEURAL MODEL
echo Downloads 8 selected public Gazette documents. No GPU or paid API.
echo Requires Python 3.11 or compatible Python 3.10+ and Internet.
if not exist ".venv-seed\Scripts\python.exe" (
  python -m venv .venv-seed
  if errorlevel 1 goto failed
)
".venv-seed\Scripts\python.exe" -m pip install -e ".[public_seed]"
if errorlevel 1 goto failed
".venv-seed\Scripts\python.exe" scripts\public_legal_seed.py --allow-network
if errorlevel 1 goto failed
echo Completed. Inspect artifacts\public-legal-seed and README.
pause
exit /b 0
:failed
echo Failed. Do not relabel incomplete or unverified data as approved.
pause
exit /b 1
