@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.11 x64 and enable Add to PATH.
  set "RESULT=2"
  goto :done
)
if not exist ".venv-tokenizer\Scripts\python.exe" (
  python -m venv .venv-tokenizer
  if errorlevel 1 (
    set "RESULT=2"
    goto :done
  )
)
".venv-tokenizer\Scripts\python.exe" -m pip install -e ".[tokenizer,dev]"
if errorlevel 1 (
  set "RESULT=2"
  goto :done
)
".venv-tokenizer\Scripts\python.exe" scripts\corpus_benchmark_demo.py %*
set "RESULT=%ERRORLEVEL%"
:done
if not "%H2LM_NO_PAUSE%"=="1" pause
exit /b %RESULT%
