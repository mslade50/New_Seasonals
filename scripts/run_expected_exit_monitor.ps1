[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RuntimeRoot,
    [Parameter(Mandatory = $true)][string]$ConfigRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$ExecEnv,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$PinnedSha,
    [switch]$ValidateOnly
)

# Dedicated observation publisher: no command-agent launch, broker mutation,
# inventory refresh, fill harvesting, scan, or email delivery is available here.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
foreach ($path in @($RuntimeRoot, $ConfigRoot, $Python, $ExecEnv)) {
    if (-not [IO.Path]::IsPathRooted($path) -or -not (Test-Path -LiteralPath $path)) {
        throw 'Monitor configuration requires existing absolute paths'
    }
}
$RuntimeRoot = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$ConfigRoot = (Resolve-Path -LiteralPath $ConfigRoot).Path
$scriptRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $scriptRoot.Equals($RuntimeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Expected-exit script and pinned runtime paths differ'
}
$safeRoot = $RuntimeRoot.Replace('\', '/')
$head = (& git -c "safe.directory=$safeRoot" -C $RuntimeRoot rev-parse HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $head.Equals($PinnedSha, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Expected-exit runtime does not match the tested commit'
}
$changes = (& git -c "safe.directory=$safeRoot" -C $RuntimeRoot status --porcelain --untracked-files=no | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $changes) {
    throw 'Expected-exit runtime has tracked changes or cannot be verified'
}
$runner = Join-Path $PSScriptRoot 'run_expected_exit_monitor.py'
if ($ValidateOnly) {
    & $Python $runner --help
    if ($LASTEXITCODE -ne 0) { throw 'Expected-exit Python runner is unavailable' }
    Write-Output 'Expected-exit pinned runtime is valid; no observation was collected or published'
    exit 0
}
$stateRoot = Join-Path $ConfigRoot 'artifacts\expected-exits'
$logs = Join-Path $stateRoot 'logs'
[IO.Directory]::CreateDirectory($logs) | Out-Null
$log = Join-Path $logs ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N') + '.log')
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
& $Python -u $runner --config-root $ConfigRoot --exec-env $ExecEnv `
    --state (Join-Path $stateRoot 'state.json') --artifacts (Join-Path $stateRoot 'runs') --upload `
    2>&1 | Out-File -LiteralPath $log -Encoding utf8
$result = $LASTEXITCODE
if ($result -ne 0) { Write-Error "Expected-exit producer failed; inspect $log" -ErrorAction Continue }
exit $result
