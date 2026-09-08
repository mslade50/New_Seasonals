# Register the daily 12:30 AM strategy-research collection and email pipeline.
# This file is inert until deliberately run by the operator.
[CmdletBinding()]
param([string]$PythonExe = (Get-Command python.exe -ErrorAction Stop).Source)
$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $dir 'run_strategy_research.py'
$taskName = 'Strategy Research (agent)'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Cannot find $runner" }
if (-not [IO.Path]::IsPathRooted($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw 'PythonExe must be an absolute existing interpreter'
}
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -in @('Running', 'Queued')) { throw 'Research is running; refusing to replace its task' }

$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$runner`"" `
    -WorkingDirectory (Split-Path -Parent $dir)
$trigger = New-ScheduledTaskTrigger -Daily -At '12:30AM'
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 180)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.Actions.Execute -ne $PythonExe -or $task.Actions.Arguments -ne "`"$runner`"" -or
    [string]$task.Principal.LogonType -ne 'S4U' -or -not $task.Settings.WakeToRun) {
    throw 'Installed research task does not match the requested definition'
}
$info = $task | Get-ScheduledTaskInfo
Write-Host "Registered task '$taskName' -> daily 12:30 AM"
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
