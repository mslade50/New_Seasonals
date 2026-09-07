[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ClaudeExe,
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Effort,
    [ValidateRange(60, 10620)][int]$TimeoutSeconds = 9000
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifactRoot = Join-Path $repoRoot 'artifacts\strategy_research_agent'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $artifactRoot "$stamp.stdout.log"
$stderrPath = Join-Path $artifactRoot "$stamp.stderr.log"
$exitCode = 1
$process = $null

try {
    $arguments = @(
        '-p', '/strategy-research',
        '--model', $Model,
        '--effort', $Effort,
        '--permission-mode', 'bypassPermissions'
    )
    $process = Start-Process -FilePath $ClaudeExe -ArgumentList $arguments `
        -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Write-Output "[CRITICAL] Strategy Research exceeded $TimeoutSeconds seconds; terminating its process tree."
        $exitCode = 124
        $killer = Start-Process -FilePath "$env:SystemRoot\System32\taskkill.exe" `
            -ArgumentList @('/PID', $process.Id, '/T', '/F') -WindowStyle Hidden -PassThru
        $killer.WaitForExit(10000) | Out-Null
    }
    else {
        $process.Refresh()
        $exitCode = $process.ExitCode
    }
}
catch {
    Write-Output "[CRITICAL] Strategy Research launcher failed: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Encoding UTF8 }
    if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Encoding UTF8 }
    Write-Output "[agent stdout: $stdoutPath]"
    Write-Output "[agent stderr: $stderrPath]"
}
exit $exitCode
