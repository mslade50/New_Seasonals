@echo off
rem Compatibility entry point. Task Scheduler uses an absolute Python action.
python "%~dp0run_strategy_research.py"
exit /b %ERRORLEVEL%
