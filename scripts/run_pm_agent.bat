@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

REM PM Weekly (market-only weekly brief with graded forecasts). Steps:
REM   1. grade: resolve last week's forecasts, rebuild the scoreboard
REM   2. build the state (syncs the PM's allowlisted R2 market objects)
REM   3. hand the state to the /pm-agent skill, which surveys, forecasts and
REM      publishes through weekly_pm_agent.py
REM   4. verify a brief or stand-down was journaled AND emailed
REM
REM Scheduled Sundays 16:00 ET on the trading desktop. Everything the PM
REM writes lives in PM_AGENT_HOME (outside this checkout) and R2 pm_agent/,
REM which the Risk Agent's allowlist denies. Doc: docs/claude_ref/pm_agent.md
REM
REM PERMISSIONS: scoped allowlist (scripts/pm_agent_headless_settings.json),
REM not bypassPermissions. MODEL AND EFFORT ARE PINNED HERE ON PURPOSE (see
REM run_risk_agent.bat); weekly_pm_agent.py stamps them on every record.
REM No auto-retry: a retry cannot tell "died before publishing" from
REM "published, then the delivery check failed".

set "PM_AGENT_MODEL=opus"
set "PM_AGENT_EFFORT=xhigh"
set "AGENT_TIMEOUT_SECONDS=5400"
if not defined PM_AGENT_HOME set "PM_AGENT_HOME=%USERPROFILE%\.pm_agent"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

if not exist "%PM_AGENT_HOME%\logs" mkdir "%PM_AGENT_HOME%\logs"
set "LOG=%PM_AGENT_HOME%\logs\pm_agent_last_run.log"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"

python "%REPO%\scripts\grade_pm_agent.py" >> "%LOG%" 2>&1
echo [grade_pm_agent exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\build_pm_state.py" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] state assembly failed; not running the PM Weekly. >> "%LOG%"
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

echo [agent: model %PM_AGENT_MODEL%, effort %PM_AGENT_EFFORT%, home %PM_AGENT_HOME%] >> "%LOG%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\invoke_pm_agent.ps1" -ClaudeExe "%CLAUDE_EXE%" -Model "%PM_AGENT_MODEL%" -Effort "%PM_AGENT_EFFORT%" -PmHome "%PM_AGENT_HOME%" -TimeoutSeconds %AGENT_TIMEOUT_SECONDS% >> "%LOG%" 2>&1
set CLAUDE_RC=%ERRORLEVEL%
echo [claude exit code: %CLAUDE_RC%] >> "%LOG%"

python "%REPO%\scripts\check_pm_agent_delivered.py" --require-r2 >> "%LOG%" 2>&1
set DELIVERY_RC=%ERRORLEVEL%
echo [delivery check exit code: %DELIVERY_RC%] >> "%LOG%"
set RC=%DELIVERY_RC%
if not "%CLAUDE_RC%"=="0" set RC=%CLAUDE_RC%
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
