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

REM ---- 0. Refresh the inputs. Without this the pitch reasons about whatever
REM happens to be on disk. The AM chain publishes to two different places:
REM   committed to main  - rd2_fragility, rd2_environment, exposure_state,
REM                        cboe_putcall (risk_report.yml + daily_screener.yml
REM                        + update_cboe_putcall.yml all push back)
REM   R2                 - master_prices, earnings_calendar
REM `git restore --worktree` takes just those paths from origin without
REM touching the branch or the (very dirty) working tree, so this never
REM rebases and never stages anything.
git fetch origin main --quiet >> "%LOG%" 2>&1
git restore --source=origin/main --worktree -- data/rd2_fragility.parquet data/rd2_environment.json data/exposure_state.json data/cboe_putcall.parquet >> "%LOG%" 2>&1
echo [git restore of committed state exit code: %ERRORLEVEL%] >> "%LOG%"

REM Best effort by design: a stale price cache surfaces as a freshness WARNING
REM in pitch_state.json, which daily_pitch renders in a red box at the top of
REM the email. Aborting here would trade a visible warning for a silent
REM no-delivery.
python "%REPO%\scripts\pull_scan_caches.py" >> "%LOG%" 2>&1
echo [pull_scan_caches exit code: %ERRORLEVEL%] >> "%LOG%"

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
