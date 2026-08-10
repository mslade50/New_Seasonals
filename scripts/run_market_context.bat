@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Market Context evening run. Five steps:
REM   0. pull the price cache from R2
REM   1. sweep (build_context_state.py); FATAL if it cannot run at all
REM   1b. if the freshest bar is behind the session being written about, wait
REM       out the price cron and pull once more before degrading
REM   2. hand the sweep to the /market-context skill, which maps the cells,
REM      drills the interesting ones, composes the brief and posts it
REM   3. verify something actually reached Slack, so a quiet failure is loud
REM
REM Scheduled Sun-Thu 18:30 ET. NOT 18:00: the denali report card runs at
REM 18:00 and the two would contend for the interactive Claude session. The
REM report card typically finishes well inside 30 minutes; if that stops being
REM true, move this later rather than overlapping them.
REM
REM PERMISSIONS. Unlike the daily pitch, this run needs NO web access, so it
REM uses a scoped allowlist instead of --permission-mode bypassPermissions.
REM The session can read this repo, write check scripts under
REM scratch/context_checks/ and post one Slack message. It cannot reach the
REM broker, the book, or the denali dashboard.
REM
REM MODEL AND EFFORT ARE PINNED HERE ON PURPOSE. Without the flags the run
REM inherits whatever ~/.claude/settings.json happens to say, so changing
REM models in an interactive session one afternoon would quietly change every
REM following evening's brief with nothing in the output to show it. Both are
REM stamped into every journal record.

if "%CONTEXT_MODEL%"=="" set "CONTEXT_MODEL=opus"
if "%CONTEXT_EFFORT%"=="" set "CONTEXT_EFFORT=xhigh"

REM Absolute path: a Task Scheduler session does not necessarily inherit the
REM interactive PATH. Falls back to PATH if absent.
set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
set "LOG=%REPO%\scripts\logs\market_context_last_run.log"
REM powershell, not wmic: wmic is removed on current Win11 builds.
for /f %%D in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%D"
set "DATEDLOG=%REPO%\scripts\logs\market_context_%TODAY%.log"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"
echo [model %CONTEXT_MODEL%, effort %CONTEXT_EFFORT%] >> "%LOG%"

REM ---- 0. prices
python "%REPO%\scripts\pull_context_prices.py" >> "%LOG%" 2>&1
echo [pull_context_prices exit code: %ERRORLEVEL%] >> "%LOG%"

REM ---- 1. sweep
python "%REPO%\scripts\build_context_state.py" --require-fresh >> "%LOG%" 2>&1
set SWEEP=%ERRORLEVEL%
echo [build_context_state exit code: %SWEEP%] >> "%LOG%"

if %SWEEP% EQU 1 (
    echo [CRITICAL] sweep failed; not running the brief. >> "%LOG%"
    copy /y "%LOG%" "%DATEDLOG%" >nul
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

REM ---- 1b. one retry on a stale bar.
REM
REM NOT a cron-timing fix. update_master_prices.yml writes R2 at 21:10 UTC
REM (17:10 ET EDT / 16:10 ET EST), which clears 18:30 in both halves of the
REM year. The spec's "the cron lands before the EST close" hazard was read
REM off a CLAUDE.md line that still said 20:30 UTC after the workflow had
REM already been moved; both are corrected now.
REM
REM What DOES leave a stale bar here is the PM job not having run at all.
REM GitHub sheds scheduled workflows under load and never backfills a missed
REM cron, so the job that never STARTS leaves no failure notification behind.
REM Ten minutes buys one window for a late run; a real miss ships the
REM scheduled lane with the stale banner, which is the right outcome.
if %SWEEP% EQU 2 (
    echo [prices stale; waiting 10 min for the R2 pull, then retrying once] >> "%LOG%"
    timeout /t 600 /nobreak >nul
    python "%REPO%\scripts\pull_context_prices.py" >> "%LOG%" 2>&1
    python "%REPO%\scripts\build_context_state.py" >> "%LOG%" 2>&1
    REM !ERRORLEVEL!, not %ERRORLEVEL%: inside a parenthesised block cmd
    REM expands % at PARSE time, so the percent form would log the value from
    REM before the block ran and quietly report the wrong exit code on the one
    REM path that only fires when something has already gone wrong.
    echo [rebuild exit code: !ERRORLEVEL!] >> "%LOG%"
)

REM ---- 2. the brief
call "%CLAUDE_EXE%" -p "/market-context" --model %CONTEXT_MODEL% --effort %CONTEXT_EFFORT% --settings "%REPO%\scripts\context_headless_settings.json" >> "%LOG%" 2>&1
echo [claude exit code: %ERRORLEVEL%] >> "%LOG%"

REM ---- 3. did it deliver
python "%REPO%\scripts\check_context_delivered.py" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [delivery check exit code: %RC%] >> "%LOG%"
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
copy /y "%LOG%" "%DATEDLOG%" >nul
endlocal & exit /b %RC%
