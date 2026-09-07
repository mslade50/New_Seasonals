# Register the daily 12:30 AM strategy-research collection and email pipeline.
# This file is inert until deliberately run by the operator.
$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $dir 'run_strategy_research.bat'
$taskName = 'Strategy Research (agent)'
if (-not (Test-Path -LiteralPath $bat -PathType Leaf)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" `
    -WorkingDirectory (Split-Path -Parent $dir)
$trigger = New-ScheduledTaskTrigger -Daily -At '12:30AM'
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 180)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host "Registered task '$taskName' -> daily 12:30 AM"
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
