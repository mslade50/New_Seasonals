param(
    [Parameter(Mandatory=$true)][ValidateSet('Prepare','Deadline')][string]$Phase,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedCommit,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$EnvFile
)
$ErrorActionPreference = 'Stop'
$runtimeRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $runtimeRoot
$artifactRoot = Join-Path $runtimeRoot 'artifacts\episodic_pivot\task_logs'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$logPath = Join-Path $artifactRoot ((Get-Date -Format 'yyyyMMddTHHmmssfff') + '-' + $Phase + '.log')
& $Python -u (Join-Path $PSScriptRoot 'validate_ep_runtime.py') --expected-commit $ExpectedCommit *> $logPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Phase -eq 'Prepare') {
    & $Python -u (Join-Path $PSScriptRoot 'prepare_ep_morning.py') --capture *>> $logPath
} else {
    & $Python -u (Join-Path $PSScriptRoot 'finish_ep_morning.py') --env-file $EnvFile --send *>> $logPath
}
exit $LASTEXITCODE
