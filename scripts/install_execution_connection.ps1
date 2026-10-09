param([Parameter(Mandatory=$true)][string]$CandidateDirectory)
$ErrorActionPreference = 'Stop'
# Operator-run promotion only. This script never stops/starts a task or agent.
$runtime = 'C:\Users\McKinley Slade\OneDrive\trading_ibkr'
$candidate = (Resolve-Path -LiteralPath $CandidateDirectory).Path
$manifest = Get-Content -LiteralPath (Join-Path $candidate 'manifest.json') -Raw | ConvertFrom-Json
$eastern = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$now = [TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $eastern)
if (($now.Hour -ge 5 -and $now.Hour -lt 21) -or ($now.Hour -eq 4 -and $now.Minute -ge 55) -or ($now.Hour -eq 21 -and $now.Minute -lt 5)) {
    throw 'Install after the normal 21:00 ET exit, from 21:05 to 04:55 ET; no restart is performed.'
}
$task = Get-ScheduledTask -TaskName 'ExecAgent' -TaskPath '\' -ErrorAction Stop
if ($task.State -ne 'Ready') { throw 'ExecAgent is not Ready; do not promote over an active/unknown service.' }
$agent = Join-Path $runtime 'exec_agent.py'
if ((Get-FileHash -LiteralPath $agent -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.source_agent_sha256) {
    throw 'Runtime source changed since review; prepare and review a new candidate.'
}
$names = @('execution_connection.py', 'exec_agent.py')
foreach ($name in $names) {
    $source = Join-Path $candidate $name
    $expected = $manifest.files.$name
    if (-not $expected -or (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Candidate hash mismatch: $name"
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($name in $names) {
    $target = Join-Path $runtime $name
    if (Test-Path -LiteralPath $target) {
        if ($name -eq 'execution_connection.py' -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.files.$name) {
            throw 'An existing connection module differs; review before replacing it.'
        }
        Copy-Item -LiteralPath $target -Destination "$target.before-connection-$stamp" -ErrorAction Stop
    }
}
# Complete both staged files before moving the agent into place. Fixed filenames
# stay within the explicitly named runtime; no recursive move/delete is used.
foreach ($name in $names) {
    Copy-Item -LiteralPath (Join-Path $candidate $name) -Destination (Join-Path $runtime "$name.connection-candidate") -ErrorAction Stop
}
if ((Get-ScheduledTask -TaskName 'ExecAgent' -TaskPath '\').State -ne 'Ready') {
    throw 'Agent task changed state while staging; no agent source has been replaced.'
}
foreach ($name in $names) {
    Move-Item -LiteralPath (Join-Path $runtime "$name.connection-candidate") -Destination (Join-Path $runtime $name) -Force -ErrorAction Stop
}
foreach ($name in $names) {
    if ((Get-FileHash -LiteralPath (Join-Path $runtime $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.files.$name) {
        throw "Installed hash mismatch: $name"
    }
}
Write-Output 'Installed connection source only. Existing 05:00 ET launch will load it; no task/agent was started or restarted.'
