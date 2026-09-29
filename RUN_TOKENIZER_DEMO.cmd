@echo off
setlocal
cd /d "%~dp0"
set "VENV=.venv-tokenizer"
echo H2LM M1 - CPU TOKENIZER DEMO - NOT A TRAINED AI
if exist "%VENV%\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto fallback
py -3.11 -m venv "%VENV%"
if not errorlevel 1 goto install
:fallback
python -m venv "%VENV%"
if errorlevel 1 goto failed
:install
"%VENV%\Scripts\python.exe" -m pip install -e ".[tokenizer]"
if errorlevel 1 goto failed
"%VENV%\Scripts\python.exe" scripts\tokenizer_demo.py
if errorlevel 1 goto failed
echo Completed. See artifacts\tokenizer. This is not legal reasoning or PDF OCR.
pause
exit /b 0
:failed
echo FAILED. Keep this window and copy the error above. Python 3.11 is recommended.
pause
exit /b 1
