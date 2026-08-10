@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Daily Posts evening run. Five steps:
REM   0. pull the price cache from R2 (reuses the context brief's puller -
REM      both products read exactly one file)
REM   1. grade any idea drafts whose windows closed (posted AND unposted -
REM      the unposted bucket is how the scoreboard measures the filter)
REM   2. assemble state (build_posts_state.py) - also ingests yesterday's
REM      queue Posted marks, the one capture window
REM   3. hand it to the /daily-posts skill: select material, write check
REM      scripts, draft the queue, lint + journal
REM   4. verify a queue actually landed, so a quiet failure is loud
REM
REM Scheduled Sun-Fri 8:00 PM ET. Well clear of the PM price cron (21:10
REM UTC = 5:10 PM EDT) and of the 6:30 PM market-context session; Saturday
REM skipped (no session to preview - weekend posting draws on the backlog).
REM
REM PERMISSIONS: scoped allowlist (posts_headless_settings.json), not
REM bypassPermissions. This run needs no web, no git, no broker.
REM
REM MODEL AND EFFORT ARE PINNED HERE ON PURPOSE (pitch convention): an
REM interactive /model change must not silently re-tier every following
REM evening. Cut draft count before cutting tier.

if "%POSTS_MODEL%"=="" set "POSTS_MODEL=fable"
if "%POSTS_EFFORT%"=="" set "POSTS_EFFORT=high"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
set "LOG=%REPO%\scripts\logs\daily_posts_last_run.log"
for /f %%D in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%D"
set "DATEDLOG=%REPO%\scripts\logs\daily_posts_%TODAY%.log"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"
echo [model %POSTS_MODEL%, effort %POSTS_EFFORT%] >> "%LOG%"

REM ---- 0. prices
python "%REPO%\scripts\pull_context_prices.py" >> "%LOG%" 2>&1
echo [pull_context_prices exit code: %ERRORLEVEL%] >> "%LOG%"

REM ---- 1. grade due ideas
python "%REPO%\scripts\posts_scoreboard.py" >> "%LOG%" 2>&1
echo [posts_scoreboard exit code: %ERRORLEVEL%] >> "%LOG%"

REM ---- 2. state (fatal if the price cache is unusable - nothing to draft from)
python "%REPO%\scripts\build_posts_state.py" >> "%LOG%" 2>&1
set STATE=%ERRORLEVEL%
echo [build_posts_state exit code: %STATE%] >> "%LOG%"
if %STATE% NEQ 0 (
    echo [CRITICAL] state build failed; not drafting. >> "%LOG%"
    copy /y "%LOG%" "%DATEDLOG%" >nul
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

REM ---- 3. draft
call "%CLAUDE_EXE%" -p "/daily-posts" --model %POSTS_MODEL% --effort %POSTS_EFFORT% --settings "%REPO%\scripts\posts_headless_settings.json" >> "%LOG%" 2>&1
echo [claude exit code: %ERRORLEVEL%] >> "%LOG%"

REM ---- 4. did it deliver
python "%REPO%\scripts\check_posts_delivered.py" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [delivery check exit code: %RC%] >> "%LOG%"
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
copy /y "%LOG%" "%DATEDLOG%" >nul
endlocal & exit /b %RC%
