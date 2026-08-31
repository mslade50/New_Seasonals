[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ClaudeExe,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$Effort,

    [ValidateRange(60, 7140)]
    [int]$TimeoutSeconds = 6300
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifactRoot = Join-Path $repoRoot 'artifacts\daily_pitch_agent'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $artifactRoot "$stamp.stdout.log"
$stderrPath = Join-Path $artifactRoot "$stamp.stderr.log"
$exitCode = 1
$process = $null

try {
    $arguments = @(
        '-p', '/daily-pitch',
        '--model', $Model,
        '--effort', $Effort,
        '--permission-mode', 'bypassPermissions'
    )
    $process = Start-Process -FilePath $ClaudeExe `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Write-Output "[CRITICAL] Daily Pitch agent exceeded $TimeoutSeconds seconds; terminating its process tree."
        $exitCode = 124
        try {
            $killProcess = Start-Process `
                -FilePath "$env:SystemRoot\System32\taskkill.exe" `
                -ArgumentList @('/PID', $process.Id, '/T', '/F') `
                -WindowStyle Hidden -PassThru
            if (-not $killProcess.WaitForExit(10000)) {
                Write-Output "[CRITICAL] taskkill itself exceeded its 10-second deadline."
                Stop-Process -Id $killProcess.Id -Force -ErrorAction SilentlyContinue
            }
            else {
                $killProcess.Refresh()
                if ($killProcess.ExitCode -ne 0) {
                    Write-Output "[CRITICAL] taskkill exited $($killProcess.ExitCode) for timed-out agent PID $($process.Id)."
                }
            }
            if (-not $process.WaitForExit(10000)) {
                Write-Output "[CRITICAL] Timed-out agent PID $($process.Id) is still running after the 10-second termination grace period."
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
    Write-Output "[CRITICAL] Daily Pitch agent launcher failed: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath
    }
    Write-Output "[agent stdout: $stdoutPath]"
    Write-Output "[agent stderr: $stderrPath]"
}

exit $exitCode
