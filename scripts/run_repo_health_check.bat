@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Repo health check. Runs the /repo-health-check skill headless: the skill
REM runs scripts/repo_health_check.py, investigates failures and writes a
REM verdict to scripts/logs/health_check_<date>.md.
REM
REM Model/effort pinned here (market-context convention) so an interactive
REM /model change never silently re-tiers the scheduled run.

if "%HEALTH_MODEL%"=="" set "HEALTH_MODEL=sonnet"
if "%HEALTH_EFFORT%"=="" set "HEALTH_EFFORT=high"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if not exist "%CLAUDE_EXE%" set "CLAUDE_EXE=claude"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
for %%I in ("%DIR%\..") do set "REPO=%%~fI"

if not exist "%REPO%\scripts\logs" mkdir "%REPO%\scripts\logs"
set "LOG=%REPO%\scripts\logs\health_check_last_run.log"
for /f %%D in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%D"
set "DATEDLOG=%REPO%\scripts\logs\health_check_%TODAY%.log"
cd /d "%REPO%"

echo ===== HEALTH CHECK START %DATE% %TIME% ===== > "%LOG%"
echo [model %HEALTH_MODEL%, effort %HEALTH_EFFORT%] >> "%LOG%"

call "%CLAUDE_EXE%" -p "/repo-health-check" --model %HEALTH_MODEL% --effort %HEALTH_EFFORT% --settings "%REPO%\scripts\health_headless_settings.json" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo [claude exit code: %RC%] >> "%LOG%"
echo ===== HEALTH CHECK END %DATE% %TIME% ===== >> "%LOG%"
copy /y "%LOG%" "%DATEDLOG%" >nul
endlocal & exit /b %RC%
