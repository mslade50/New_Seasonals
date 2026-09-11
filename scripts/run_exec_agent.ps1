# Execution-bridge Task Scheduler boundary. Keep failures visible to the scheduler.
param(
    [string]$RuntimeDirectory = 'C:\Users\McKinley Slade\OneDrive\trading_ibkr',
    [string]$PythonPath = 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe'
)

$ErrorActionPreference = 'Stop'
$log = Join-Path $RuntimeDirectory 'exec_agent_last_run.log'
function Write-AgentLog([string]$Message) {
    "=== exec_agent $(Get-Date -Format o) $Message ===" |
        Out-File -LiteralPath $log -Append -Encoding utf8
}

try {
    Set-Location -LiteralPath $RuntimeDirectory
    Get-Content -LiteralPath (Join-Path $RuntimeDirectory 'exec_agent.env') | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)\s*=\s*(.+?)\s*$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    # Match exec_agent.py's existing local-time window, including its overrides.
    $startHour = 5
    $endHour = 21
    if ($env:AGENT_START_HOUR) { $startHour = [int]$env:AGENT_START_HOUR }
    if ($env:AGENT_END_HOUR) { $endHour = [int]$env:AGENT_END_HOUR }
    $hour = (Get-Date).Hour
    if ($hour -lt $startHour -or $hour -ge $endHour) {
        Write-AgentLog 'outside run window; skipped'
        exit 0
    }
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw 'Python executable unavailable'
    }
    $agentPath = Join-Path $RuntimeDirectory 'exec_agent.py'
    if (-not (Test-Path -LiteralPath $agentPath -PathType Leaf)) {
        throw 'Agent script unavailable'
    }
    Write-AgentLog "launch wrapper_pid=$PID"
    # Windows PowerShell treats native stderr as ErrorRecords. Log it without
    # interrupting the wait; use the actual native exit code to classify failure.
    $ErrorActionPreference = 'Continue'
    & $PythonPath -u $agentPath 2>&1 | ForEach-Object {
        $_.ToString() | Out-File -LiteralPath $log -Append -Encoding utf8 -ErrorAction Stop
    }
    $agentExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $hour = (Get-Date).Hour
    $unexpectedExit = $hour -ge $startHour -and $hour -lt $endHour
    Write-AgentLog "exit python_code=$agentExitCode unexpected=$unexpectedExit"
    if ($null -eq $agentExitCode -or $agentExitCode -ne 0 -or $unexpectedExit) {
        exit 1
    }
    exit 0
} catch {
    # Do not echo environment values or a potentially secret-bearing exception.
    try { Write-AgentLog ('launcher failure: ' + $_.Exception.GetType().Name) } catch {}
    exit 1
}
