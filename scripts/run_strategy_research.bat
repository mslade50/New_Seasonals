@echo off
rem Compatibility for the retained cmd task; prefer its prepared interpreter.
if exist "%~dp0..\artifacts\strategy_research_runtime\.venv\Scripts\python.exe" (
    "%~dp0..\artifacts\strategy_research_runtime\.venv\Scripts\python.exe" "%~dp0run_strategy_research.py"
) else (
    python "%~dp0run_strategy_research.py"
)
exit /b %ERRORLEVEL%
