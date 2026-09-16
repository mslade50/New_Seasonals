param(
    [switch]$Install,
    [switch]$Replace,
    [switch]$Live,
    [string[]]$Accounts = @("primary"),
    [string]$LiveAcknowledgement = ""
)

$ErrorActionPreference = "Stop"
$Accounts = @(
    $Accounts |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
)
if ($Accounts.Count -eq 0 -or @($Accounts | Where-Object { $_ -notin @("primary", "pa") }).Count -gt 0) {
    throw "Accounts must contain only exact labels: primary, pa."
}
if (@($Accounts | Select-Object -Unique).Count -ne $Accounts.Count) {
    throw "Accounts must not contain duplicates."
}
$repoRoot = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $PSScriptRoot "run_legend_etf_task.ps1"
$powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$pythonExe = (Get-Command python -ErrorAction Stop).Source
if ((Get-TimeZone).Id -ne "Eastern Standard Time") {
    throw "Legend ETF tasks require the Windows host timezone Eastern Standard Time."
}
$signalName = "NewSeasonals-LegendETF-Signals"
$sessionName = "NewSeasonals-LegendETF-Session"
$watchdogName = "NewSeasonals-LegendETF-Watchdog"
$accountArgs = ($Accounts -join ",")
$watchdogTimes = $(
    if ($Live) {
        @("10:40", "12:00", "14:00", "15:45", "16:10")
    }
    else {
        @("10:40")
    }
)

$definitions = @(
    [pscustomobject]@{
        Name = $signalName
        At = "08:45"
        Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -Mode signals -PythonExe `"$pythonExe`""
    },
    [pscustomobject]@{
        Name = $sessionName
        At = "09:28"
        Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -Mode session -Accounts `"$accountArgs`" -PythonExe `"$pythonExe`"" + $(if ($Live) { " -Live" } else { " -ShadowThroughExit" })
    },
    [pscustomobject]@{
        Name = $watchdogName
        At = $watchdogTimes
        Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`" -Mode watchdog -Accounts `"$accountArgs`" -PythonExe `"$pythonExe`"" + $(if ($Live) { " -Live" } else { "" })
    }
)

foreach ($definition in $definitions) {
    Write-Host "$($definition.Name) @ $(@($definition.At) -join ', ')"
    Write-Host "  $($definition.Arguments)"
}
if (-not $Install) {
    $previewMode = $(if ($Live) { "dated-gated live" } else { "shadow" })
    Write-Host "Preview only ($previewMode). Re-run with -Install to register."
    exit 0
}
if ($repoRoot -match '(?i)[\\/]artifacts[\\/](?:worktrees|task_worktrees)[\\/]') {
    throw "Refusing to install scheduled tasks from a disposable task worktree. Merge into the stable production checkout first."
}
if ($Live -and $LiveAcknowledgement -ne "REGISTER_DATED_GATED_LIVE_TASK") {
    throw "Live registration requires -LiveAcknowledgement REGISTER_DATED_GATED_LIVE_TASK"
}

$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

# Resolve every name conflict before the first Task Scheduler mutation.
$conflicts = @()
foreach ($definition in $definitions) {
    $existing = Get-ScheduledTask -TaskName $definition.Name -ErrorAction SilentlyContinue
    if ($null -ne $existing -and -not $Replace) {
        $conflicts += $definition.Name
    }
}
if ($conflicts.Count -gt 0) {
    throw "Existing task(s) require review and explicit -Replace: $($conflicts -join ', ')"
}

# Construct every task definition successfully before registering any of them.
$planned = @()
foreach ($definition in $definitions) {
    $action = New-ScheduledTaskAction `
        -Execute $powershellExe `
        -Argument $definition.Arguments `
        -WorkingDirectory $repoRoot
    $triggers = @()
    foreach ($at in @($definition.At)) {
        $firstRun = [DateTime]::Today.Add([TimeSpan]::Parse($at))
        if ($firstRun -le (Get-Date)) { $firstRun = $firstRun.AddDays(1) }
        $triggers += New-ScheduledTaskTrigger `
            -Weekly `
            -WeeksInterval 1 `
            -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
            -At $firstRun
    }
    $task = New-ScheduledTask `
        -Action $action `
        -Trigger $triggers `
        -Principal $principal `
        -Settings $settings `
        -Description $(if ($Live) { "Dedicated dated-gated live Legend EMA ETF task" } else { "Dedicated shadow Legend EMA ETF task" })
    $planned += [pscustomobject]@{
        Definition = $definition
        Task = $task
    }
}

$registered = @()
try {
    foreach ($item in $planned) {
        Register-ScheduledTask `
            -TaskName $item.Definition.Name `
            -InputObject $item.Task `
            -Force:$Replace | Out-Null
        $registered += $item.Definition.Name
    }
}
catch {
    throw "Task registration failed after [$($registered -join ', ')]. No automatic task deletion was attempted; review this partial set before retrying. $($_.Exception.Message)"
}

$missing = @(
    $definitions |
        Where-Object {
            $null -eq (Get-ScheduledTask -TaskName $_.Name -ErrorAction SilentlyContinue)
        } |
        ForEach-Object { $_.Name }
)
if ($missing.Count -gt 0) {
    throw "Registration verification failed; missing task(s): $($missing -join ', ')"
}

Write-Host "Registered Legend ETF tasks. Live orders still require the independent daily environment gate."
