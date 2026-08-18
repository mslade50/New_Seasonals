[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')]
    [string]$Task,

    [string]$BaseBranch = 'main',

    [string]$WorktreeRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoName = Split-Path $repoRoot -Leaf

if (-not $WorktreeRoot) {
    $WorktreeRoot = Join-Path (Split-Path $repoRoot -Parent) "$repoName-worktrees"
}

$branchName = "codex/$Task"
$target = Join-Path $WorktreeRoot $Task
$safeDirectory = $repoRoot.Replace('\', '/')

if (Test-Path -LiteralPath $target) {
    throw "Worktree target already exists: $target"
}

& git -c "safe.directory=$safeDirectory" show-ref --verify --quiet "refs/heads/$branchName"
if ($LASTEXITCODE -eq 0) {
    throw "Branch already exists: $branchName"
}
if ($LASTEXITCODE -ne 1) {
    throw "Unable to check whether branch exists: $branchName"
}

if ($PSCmdlet.ShouldProcess($target, "Create worktree on $branchName from $BaseBranch")) {
    New-Item -ItemType Directory -Path $WorktreeRoot -Force | Out-Null
    & git -c "safe.directory=$safeDirectory" worktree add -b $branchName $target $BaseBranch
    if ($LASTEXITCODE -ne 0) {
        throw "git worktree add failed with exit code $LASTEXITCODE"
    }
    Write-Output "Worktree: $target"
    Write-Output "Branch:   $branchName"
}
