[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ClaudeExe,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$Effort,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$')]
    [string]$RequestId,

    [ValidateRange(60, 7140)]
    [int]$TimeoutSeconds = 1200
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$artifactRoot = Join-Path $repoRoot 'scratch\idea_checks\_state\agent_logs'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$stdoutPath = Join-Path $artifactRoot "$RequestId.stdout.log"
$stderrPath = Join-Path $artifactRoot "$RequestId.stderr.log"
$exitCode = 1
$process = $null

try {
    # Only the request id goes on the command line, never the idea text.
    $arguments = @(
        '-p', "`"/idea-check $RequestId`"",
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
        Write-Output "[CRITICAL] Idea Check agent exceeded $TimeoutSeconds seconds; terminating its process tree."
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
    Write-Output "[CRITICAL] Idea Check agent launcher failed: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    Write-Output "[agent stdout: $stdoutPath]"
    Write-Output "[agent stderr: $stderrPath]"
}

exit $exitCode
