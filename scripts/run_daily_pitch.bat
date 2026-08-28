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
REM Scheduled weekdays 5:10 AM ET. NOT 5:00: the 4:47 scan takes 13-16 min
REM (measured over five AM runs), so it lands 5:00-5:04 and a 5:00 start would
REM race the Order_Staging/Overflow tabs and exposure_state.json that the
REM mandatory `overlap` field is written from. 5:10 clears it with margin and
REM still delivers ~6:20, well before the 9:05 approval runner.
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

REM Absolute path: a Task Scheduler session does not necessarily
REM inherit the interactive PATH. Falls back to PATH if absent.
set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

set "LOG=%REPO%\scripts\logs\daily_pitch_last_run.log"
if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"

REM ---- 0. Refresh the inputs from R2. The local-primary AM pipeline publishes
REM master prices, earnings, fragility, environment, exposure and CBOE state
REM there before this task starts. R2 is also the private site's authoritative
REM production boundary, so the pitch no longer depends on generated-state bot
REM commits or mutates the user's dirty development checkout with git restore.
REM Best effort by design: a stale/missing cache surfaces as a freshness WARNING
REM in pitch_state.json, which daily_pitch renders in a red box at the top.
python "%REPO%\scripts\pull_scan_caches.py" --set pitch >> "%LOG%" 2>&1
echo [pull pitch caches exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\grade_pitch_journal.py" >> "%LOG%" 2>&1
echo [grade_pitch_journal exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\build_pitch_state.py" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] state assembly failed; not running the pitch. >> "%LOG%"
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

echo [agent: model %PITCH_MODEL%, effort %PITCH_EFFORT%] >> "%LOG%"
call "%CLAUDE_EXE%" -p "/daily-pitch" --model %PITCH_MODEL% --effort %PITCH_EFFORT% --permission-mode bypassPermissions >> "%LOG%" 2>&1
echo [claude exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\check_pitch_delivered.py" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [delivery check exit code: %RC%] >> "%LOG%"
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
