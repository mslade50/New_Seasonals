[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ClaudeExe,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$Effort,

    [Parameter(Mandatory = $true)]
    [string]$PmHome,

    [ValidateRange(60, 7140)]
    [int]$TimeoutSeconds = 5400
)

# Same launcher shape as invoke_risk_agent.ps1 (timeout, process-tree kill, no
# retry), but with the scoped allowlist instead of bypassPermissions, and with
# PM_AGENT_HOME added as a working directory so the session can write there.
# Logs go to PM_AGENT_HOME\logs, outside the repo.

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$settings = Join-Path $PSScriptRoot 'pm_agent_headless_settings.json'
$artifactRoot = Join-Path $PmHome 'logs'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $artifactRoot "$stamp.stdout.log"
$stderrPath = Join-Path $artifactRoot "$stamp.stderr.log"
$exitCode = 1
$process = $null

try {
    $arguments = @(
        '-p', "`"/pm-agent $PmHome`"",
        '--model', $Model,
        '--effort', $Effort,
        '--settings', "`"$settings`"",
        '--add-dir', "`"$PmHome`""
    )
    $process = Start-Process -FilePath $ClaudeExe `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Write-Output "[CRITICAL] PM Weekly exceeded $TimeoutSeconds seconds; terminating its process tree."
        $exitCode = 124
        try {
            $killProcess = Start-Process `
                -FilePath "$env:SystemRoot\System32\taskkill.exe" `
                -ArgumentList @('/PID', $process.Id, '/T', '/F') `
                -WindowStyle Hidden -PassThru
            if (-not $killProcess.WaitForExit(10000)) {
                Stop-Process -Id $killProcess.Id -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            Write-Output "[CRITICAL] Could not terminate the timed-out agent tree: $($_.Exception.Message)"
        }
    }
    else {
        $process.Refresh()
        $exitCode = $process.ExitCode
    }
}
catch {
    Write-Output "[CRITICAL] PM Weekly launcher failed: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath -Encoding UTF8
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Encoding UTF8
    }
    Write-Output "[agent stdout: $stdoutPath]"
    Write-Output "[agent stderr: $stderrPath]"
}

exit $exitCode
