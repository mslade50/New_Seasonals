@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

REM Risk Agent nightly run (independent $200k paper sleeve). Steps:
REM   1. sync the allowed R2 objects (risk_agent_data)
REM   2. grade: fill pending paper orders, mark, exit, settle options
REM   3. build the compact state the agent reads (abort on failure)
REM   4. hand the state to the /risk-agent skill, which surveys, forecasts,
REM      decides and publishes through daily_risk_agent.py
REM   5. verify a decision or stand-down was journaled AND emailed
REM
REM Scheduled weekdays 18:15 local, after the 17:10 postclose job has rebuilt
REM shared/site_risk.json and prices.
REM
REM NOTE ON PERMISSIONS: the agent step runs unattended with
REM --permission-mode bypassPermissions. It can write files in this repo and
REM send the Risk Agent email. It cannot place orders: nothing here talks to a
REM broker, and the sleeve is paper only.
REM
REM MODEL AND EFFORT ARE PINNED HERE ON PURPOSE. Without the flags the run
REM inherits whatever ~/.claude/settings.json says, so switching models in an
REM interactive session one afternoon would quietly change every following
REM night's decision with nothing in the email to show it. Opus at xhigh is the
REM right tier: the agent writes and interprets real empirical checks before
REM it may open a position. daily_risk_agent.py stamps these on every journal
REM record, so the scoreboard can split by tier.
REM
REM No auto-retry: a retry cannot tell "died before publishing" from
REM "published, then the delivery check failed".

set "RISK_AGENT_MODEL=opus"
set "RISK_AGENT_EFFORT=xhigh"
set "AGENT_TIMEOUT_SECONDS=5400"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

set "LOG=%REPO%\scripts\logs\risk_agent_last_run.log"
if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"

python -c "import risk_agent_data as d; d.sync()" >> "%LOG%" 2>&1
echo [risk_agent_data sync exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\grade_risk_agent.py" >> "%LOG%" 2>&1
echo [grade_risk_agent exit code: %ERRORLEVEL%] >> "%LOG%"

python "%REPO%\scripts\build_risk_agent_state.py" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] state assembly failed; not running the risk agent. >> "%LOG%"
    echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
    endlocal & exit /b 1
)

echo [agent: model %RISK_AGENT_MODEL%, effort %RISK_AGENT_EFFORT%] >> "%LOG%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\invoke_risk_agent.ps1" -ClaudeExe "%CLAUDE_EXE%" -Model "%RISK_AGENT_MODEL%" -Effort "%RISK_AGENT_EFFORT%" -TimeoutSeconds %AGENT_TIMEOUT_SECONDS% >> "%LOG%" 2>&1
set CLAUDE_RC=%ERRORLEVEL%
echo [claude exit code: %CLAUDE_RC%] >> "%LOG%"

python "%REPO%\scripts\check_risk_agent_delivered.py" --require-r2 >> "%LOG%" 2>&1
set DELIVERY_RC=%ERRORLEVEL%
echo [delivery check exit code: %DELIVERY_RC%] >> "%LOG%"
set RC=%DELIVERY_RC%
if not "%CLAUDE_RC%"=="0" set RC=%CLAUDE_RC%
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
