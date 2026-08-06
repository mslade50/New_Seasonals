@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Daily Pitch morning run. Three steps:
REM   1. grade yesterday's ideas so today's email footer carries a scoreboard
REM   2. assemble today's state (calendar, tape, dials, book, research index)
REM   3. hand the state to the /daily-pitch skill, which invents, falsifies,
REM      composes and publishes
REM   4. verify something was actually delivered, so a quiet failure is loud
REM
REM Scheduled weekdays 7:00 AM ET: after the ~4:47 AM scan chain has refreshed
REM every input, and well before the 9:05 approval runner.
REM
REM NOTE ON PERMISSIONS: the agent step runs unattended, so it uses
REM --permission-mode bypassPermissions. That session can write files in this
REM repo and send the pitch email. It cannot place orders: order placement
REM lives in pitch_moo.py on the trading machine, behind its own approval gate
REM and activation flag.
REM
REM MODEL AND EFFORT ARE PINNED HERE ON PURPOSE. Without the flags the run
REM inherits whatever ~/.claude/settings.json happens to say, so switching
REM models in an interactive session one afternoon would quietly change every
REM following morning's pitch with nothing in the email to show it. Opus at
REM xhigh is the right tier for this job: stage C writes and interprets real
REM empirical checks, and the spec is explicit that the falsification stage
REM must not be truncated to save tokens. Subagents inherit these, so the
REM verifier fan-out runs at the same tier as the composer.

set "PITCH_MODEL=opus"
set "PITCH_EFFORT=xhigh"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

set "LOG=%REPO%\scripts\logs\daily_pitch_last_run.log"
if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"

python "%REPO%\scripts\grade_pitch_journal.py" >> "%LOG%" 2>&1
echo [grade_pitch_journal exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\build_pitch_state.py" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] state assembly failed; not running the pitch. >> "%LOG%"
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

echo [agent: model %PITCH_MODEL%, effort %PITCH_EFFORT%] >> "%LOG%"
call claude -p "/daily-pitch" --model %PITCH_MODEL% --effort %PITCH_EFFORT% --permission-mode bypassPermissions >> "%LOG%" 2>&1
echo [claude exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\check_pitch_delivered.py" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [delivery check exit code: %RC%] >> "%LOG%"
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
