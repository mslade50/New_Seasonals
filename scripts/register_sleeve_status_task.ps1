[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Install,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$ConfigRoot,
    [Parameter(Mandatory = $true)][string]$ExecutorRoot
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script = Join-Path $PSScriptRoot 'run_sleeve_status.ps1'
$name = 'New Seasonals Sleeve Status'
foreach ($path in @($PythonExe, $ConfigRoot, $ExecutorRoot, $script)) {
    if (-not [IO.Path]::IsPathRooted($path) -or -not (Test-Path -LiteralPath $path)) { throw "Required absolute path does not exist: $path" }
    if ($path.Contains('"')) { throw 'Quotes are not supported in task paths' }
}
$shell = Join-Path $env:WINDIR 'System32/WindowsPowerShell/v1.0/powershell.exe'
$arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$script`" -PythonExe `"$PythonExe`" -ConfigRoot `"$ConfigRoot`" -ExecutorRoot `"$ExecutorRoot`""
Write-Output "Read-only sleeve observer: every 30 minutes, writing only ops/sleeve_runtime_status.json."
if (-not $Install) { Write-Output 'Preview only. No task was registered.'; return }
if ($repoRoot -match '(?i)[\\/]artifacts[\\/](?:task_)?worktrees[\\/]') {
    throw 'Install from the stable checkout after release, not a task worktree.'
}
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw "Task already exists: $name. Inspect it before changing its configuration." }
if ($PSCmdlet.ShouldProcess($name, 'Register a read-only sleeve status collector')) {
    $action = New-ScheduledTaskAction -Execute $shell -Argument $arguments -WorkingDirectory $repoRoot
    # Repeats indefinitely. Does not wake the computer just to publish status.
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30)
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
    Write-Output "Registered $name"
}
