@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

set "RESEARCH_MODEL=opus"
set "RESEARCH_EFFORT=xhigh"
set "AGENT_TIMEOUT_SECONDS=9000"
set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"
set "LOG=%REPO%\artifacts\strategy_research_agent\last_run.log"
set "MARKER=%REPO%\artifacts\strategy_research_agent\current_run.marker"
if not exist "%REPO%\artifacts\strategy_research_agent" mkdir "%REPO%\artifacts\strategy_research_agent"
cd /d "%REPO%"

echo ===== RUN START %DATE% %TIME% ===== > "%LOG%"
echo %DATE% %TIME% > "%MARKER%"
python "%REPO%\scripts\collect_strategy_sources.py" collect >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [CRITICAL] source collection failed; no research email can be sent. >> "%LOG%"
    python "%REPO%\scripts\send_strategy_research_failure_email.py" --phase source_collection --summary "Source collection failed before a complete capture was available." --send >> "%LOG%" 2>&1
    endlocal & exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\invoke_strategy_research_agent.ps1" -ClaudeExe "%CLAUDE_EXE%" -Model "%RESEARCH_MODEL%" -Effort "%RESEARCH_EFFORT%" -TimeoutSeconds %AGENT_TIMEOUT_SECONDS% >> "%LOG%" 2>&1
set AGENT_RC=%ERRORLEVEL%
echo [claude exit code: %AGENT_RC%] >> "%LOG%"

python "%REPO%\scripts\check_strategy_research_run.py" --marker "%MARKER%" >> "%LOG%" 2>&1
set CHECK_RC=%ERRORLEVEL%
echo [completion check exit code: %CHECK_RC%] >> "%LOG%"
set RC=%CHECK_RC%
if not "%AGENT_RC%"=="0" set RC=%AGENT_RC%
if not "%RC%"=="0" (
    if not "%AGENT_RC%"=="0" (
        python "%REPO%\scripts\send_strategy_research_failure_email.py" --phase research_agent --summary "The research agent did not complete successfully." --send >> "%LOG%" 2>&1
    ) else (
        python "%REPO%\scripts\send_strategy_research_failure_email.py" --phase completion_check --summary "The run lacked a confirmed email or explicit NO_EMAIL decision." --send >> "%LOG%" 2>&1
    )
)
echo ===== RUN END %DATE% %TIME% ===== >> "%LOG%"
endlocal & exit /b %RC%
