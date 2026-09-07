param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$ConfigRoot,
    [Parameter(Mandatory = $true)][string]$ExecutorRoot
)
$ErrorActionPreference = 'Stop'
try {
    & $PythonExe (Join-Path $PSScriptRoot 'publish_sleeve_runtime_status.py') --config-root $ConfigRoot --executor-root $ExecutorRoot --upload
    exit $LASTEXITCODE
} catch {
    Write-Error 'Sleeve status collector could not run; previous report must age out.'
    exit 1
}
