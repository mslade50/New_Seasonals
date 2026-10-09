param(
    [Parameter(Mandatory = $true)][string]$Session,
    [string]$Repo = 'C:\Users\McKinley Slade\dev\New_Seasonals',
    [int]$TimeoutSeconds = 5400
)
$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe'
$Runs = Join-Path $Repo 'artifacts\open_breakout_runs'
& $Python (Join-Path $Repo 'scripts\open_breakout_launch_monitor.py') `
    --runs $Runs --session $Session --timeout-seconds $TimeoutSeconds
exit $LASTEXITCODE
