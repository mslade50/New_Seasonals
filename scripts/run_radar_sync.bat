@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Weekly momentum-radar sync. Two steps, in this order for a reason:
REM
REM   1. upload_radar_recs.py  - git-pulls the radar clone, then publishes the
REM      week's plans to R2 for the site's Radar tab.
REM   2. radar_trail_sync.py   - reads that SAME clone and moves live stops to
REM      the trail/breakeven the radar's book engine stepped to.
REM
REM Step 2 does NOT pull. It reads whatever the clone holds, so running it
REM without step 1 first can apply LAST week's stops - which, since the job
REM never lowers a stop, silently under-raises instead of failing. Keep them
REM together and keep this order.
REM
REM Scheduled Mondays 8:50 AM ET: after the radar's weekend screen + cloud
REM agent have committed (Sun/Mon 08:00 and 11:00 UTC), before the open so a
REM raised stop is in place for a gap, and clear of the 9:05 event / 9:10 OLV
REM exit / 9:31 order-entry tasks.
REM
REM ARMING. Step 2 previews by default and transmits NOTHING. It sends real
REM modify commands only when radar_trail_enabled.flag exists next to
REM radar_trail_sync.py - same flag convention as pitch_moo / event_moo.
REM Delete the flag to disarm without unregistering the task.

set REPO=%~dp0..
set IBKR=%USERPROFILE%\OneDrive\trading_ibkr
set LOGDIR=%~dp0logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
REM One rolling log, same convention as run_daily_pitch.bat. Parsing %DATE% for
REM a filename is locale-dependent and produced "18-Tue-08" here.
set "LOG=%LOGDIR%\radar_sync_last_run.log"

echo ============================================================ > "%LOG%"
echo radar sync %DATE% %TIME% >> "%LOG%"

echo [1/2] publishing recs to R2 >> "%LOG%"
python "%REPO%\scripts\upload_radar_recs.py" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo   WARNING: publish step returned %errorlevel% - the clone may be stale >> "%LOG%"
)

echo [2/2] syncing trail stops >> "%LOG%"
if exist "%IBKR%\radar_trail_enabled.flag" (
  echo   flag present - APPLY mode >> "%LOG%"
  python "%IBKR%\radar_trail_sync.py" --apply >> "%LOG%" 2>&1
) else (
  echo   no radar_trail_enabled.flag - PREVIEW only >> "%LOG%"
  python "%IBKR%\radar_trail_sync.py" >> "%LOG%" 2>&1
)

echo done rc=%errorlevel% >> "%LOG%"
endlocal
