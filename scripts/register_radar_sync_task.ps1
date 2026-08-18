# Registers the Monday 8:50 AM ET momentum-radar sync.
#
# What it does each week: pulls the radar clone, publishes the week's plans to
# R2 for the site's Radar tab, then reconciles live stops against the trail the
# radar's book engine stepped to.
#
# Registration alone places NOTHING. The trail step previews unless
# radar_trail_enabled.flag exists beside radar_trail_sync.py, the same flag
# convention pitch_moo and event_moo use. Registering and arming are two
# deliberate steps; do them on different days.
#
# House convention, and it applies here: run scripts\run_radar_sync.bat by hand
# and read the log for at least one weekend where a radar position is actually
# open BEFORE arming. As of 2026-08-18 the radar book has no fills at all, so
# the job has nothing to act on and every run is a no-op - which makes this a
# safe thing to register early and a pointless thing to arm early.

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_radar_sync.bat'
$taskName = 'Radar Weekly Sync'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

# 8:50 AM Monday. The radar's screen commits Sun/Mon 08:00 UTC and its cloud
# agent runs 11:00 UTC (7 AM ET), so by 8:50 the clone has this week's recs.
# Before the 9:30 open so a raised stop is live for a gap, and clear of the
# 9:05 event sleeve, 9:10 OLV exits and 9:31 order entry.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 8:50AM

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew

# StartWhenAvailable matters: this is WEEKLY, so a machine that was off at 8:50
# would otherwise skip the entire week rather than run late.

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description 'Publish the momentum radar recs to R2 and reconcile live trail stops.' -Force

Write-Host ""
Write-Host "Registered '$taskName' - Mondays 8:50 AM."
Write-Host "It runs in PREVIEW: no orders are modified."
Write-Host "To arm later:  New-Item '$env:USERPROFILE\OneDrive\trading_ibkr\radar_trail_enabled.flag' -ItemType File"
Write-Host "Logs:          $dir\logs\radar_sync_<date>.log"
