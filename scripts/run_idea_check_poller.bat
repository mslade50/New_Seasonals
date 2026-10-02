@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

REM Idea Check poller. Runs every minute from Task Scheduler (see
REM register_idea_check_task.ps1). One pass: pick up new requests from R2,
REM run the /idea-check skill headlessly, upload the verdict. Exits quietly
REM when another pass holds the lock or the queue is empty.
REM
REM The agent session uses bypassPermissions and writes only under
REM scratch\idea_checks\<id>\ by skill rule. It places no orders.

if not defined IDEA_CHECK_MODEL set "IDEA_CHECK_MODEL=opus"
if not defined IDEA_CHECK_EFFORT set "IDEA_CHECK_EFFORT=high"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

set "LOG=%REPO%\scripts\logs\idea_check_last_run.log"
if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"
echo [agent: model %IDEA_CHECK_MODEL%, effort %IDEA_CHECK_EFFORT%] >> "%LOG%"
python "%REPO%\idea_check_poller.py" --once >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [poller exit code: %RC%] >> "%LOG%"
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
