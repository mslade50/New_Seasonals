@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

REM Daily Seasonal morning run. Mirrors run_daily_pitch.bat step for step
REM (docs/claude_ref/daily_seasonal.md):
REM   0. refresh the caches from R2 (same set as the pitch, which includes the
REM      root atr_seasonal_ranks.parquet)
REM   1. grade the seasonal journal so today's email footer carries a scoreboard
REM   2. assemble today's seasonal state (ranks, calendar cells, board, tape)
REM   3. hand the state to the /daily-seasonal skill, which surveys, falsifies,
REM      composes and publishes via daily_pitch.py --product seasonal
REM   4. verify something was actually delivered, so a quiet failure is loud
REM
REM Scheduled weekdays 04:30 local, before the 5:10 Daily Pitch (owner
REM decision 2026-10-01, was 07:00). Today's pitch has not published yet,
REM so the dedup against the pitch journal covers prior days only.         
REM
REM Permissions, model and effort: identical to the pitch and for the same
REM reasons (see run_daily_pitch.bat). The session can write in this repo and
REM send the Daily Seasonal email. It cannot place orders: nothing reads the
REM Seasonal Agent tab for placement in v1.

set "PITCH_MODEL=opus"
set "PITCH_EFFORT=xhigh"
set "AGENT_TIMEOUT_SECONDS=6300"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

set "LOG=%REPO%\scripts\logs\daily_seasonal_last_run.log"
if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"

python "%REPO%\scripts\pull_scan_caches.py" --set pitch >> "%LOG%" 2>&1
echo [pull pitch caches exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\grade_pitch_journal.py" --product seasonal >> "%LOG%" 2>&1
echo [grade_pitch_journal --product seasonal exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\build_seasonal_state.py" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] state assembly failed; not running the seasonal agent. >> "%LOG%"
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

echo [agent: model %PITCH_MODEL%, effort %PITCH_EFFORT%] >> "%LOG%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\invoke_daily_seasonal_agent.ps1" -ClaudeExe "%CLAUDE_EXE%" -Model "%PITCH_MODEL%" -Effort "%PITCH_EFFORT%" -TimeoutSeconds %AGENT_TIMEOUT_SECONDS% >> "%LOG%" 2>&1
set CLAUDE_RC=%ERRORLEVEL%
echo [claude exit code: %CLAUDE_RC%] >> "%LOG%"

python "%REPO%\scripts\check_pitch_delivered.py" --product seasonal --require-r2 >> "%LOG%" 2>&1
set DELIVERY_RC=%ERRORLEVEL%
echo [delivery check exit code: %DELIVERY_RC%] >> "%LOG%"
set RC=%DELIVERY_RC%
if not "%CLAUDE_RC%"=="0" set RC=%CLAUDE_RC%
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
