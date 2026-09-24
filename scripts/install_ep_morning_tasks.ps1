param(
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedCommit,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$EnvFile,
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
$RuntimeRoot = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
$EnvFile = (Resolve-Path -LiteralPath $EnvFile).Path
$runner = Join-Path $RuntimeRoot 'scripts\run_ep_morning_task.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw 'EP runtime task runner missing' }
# Explicitly bind wall-clock triggers to this machine's Eastern zone. Do not
# silently install 08:20 in another local time zone or use a fixed UTC offset.
if ((Get-TimeZone).Id -ne 'Eastern Standard Time') { throw 'EP task installation requires Eastern local time' }
& $Python (Join-Path $RuntimeRoot 'scripts\validate_ep_runtime.py') --expected-commit $ExpectedCommit
if ($LASTEXITCODE -ne 0) { throw 'EP runtime integrity check failed' }
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$plans = @(
    @{Name='Seasonals EP Morning Preparation'; Phase='Prepare'; Times=@('08:20','08:30','08:40','08:50','09:00','09:10','09:20')},
    @{Name='Seasonals EP Morning Deadline'; Phase='Deadline'; Times=@('09:40','09:45')}
)
foreach ($plan in $plans) {
    $description = 'EP research-only deterministic ' + $plan.Phase + '; pinned runtime; no orders or broker access.'
    $existing = Get-ScheduledTask -TaskName $plan.Name -ErrorAction SilentlyContinue
    if ($existing -and -not $existing.Description.StartsWith('EP research-only deterministic ')) {
        throw 'An unrelated task already uses the requested task name'
    }
    $arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $runner + '" -Phase ' + $plan.Phase +
        ' -ExpectedCommit ' + $ExpectedCommit + ' -Python "' + $Python + '" -EnvFile "' + $EnvFile + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $RuntimeRoot
    $triggers = @($plan.Times | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ })
    if ($Install) {
        Register-ScheduledTask -TaskName $plan.Name -Action $action -Trigger $triggers `
            -Principal $principal -Settings $settings -Description $description -Force | Out-Null
    }
    [pscustomobject]@{Task=$plan.Name; Installed=[bool]$Install; Times=$plan.Times; Commit=$ExpectedCommit} | ConvertTo-Json -Compress
}
