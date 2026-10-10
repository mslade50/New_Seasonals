@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM PM daily check-in (code only, no LLM): weekdays 08:30 ET on the trading
REM desktop. Journals a check_in record every session and emails only when an
REM exception fires. Doc: docs/claude_ref/pm_agent.md

if not defined PM_AGENT_HOME set "PM_AGENT_HOME=%USERPROFILE%\.pm_agent"
set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"
if not exist "%PM_AGENT_HOME%\logs" mkdir "%PM_AGENT_HOME%\logs"
set "LOG=%PM_AGENT_HOME%\logs\pm_daily_check.log"
cd /d "%REPO%"

echo ===== CHECK START %DATE% %TIME% ===== >> "%LOG%"
python "%REPO%\scripts\pm_daily_check.py" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ===== CHECK END %DATE% %TIME% (exit %RC%) ===== >> "%LOG%"
endlocal & exit /b %RC%
