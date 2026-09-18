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

$requiredRoot = Join-Path (Split-Path $repoRoot -Parent) "$repoName-worktrees"

if (-not $WorktreeRoot) {
    $WorktreeRoot = $requiredRoot
}

function Get-FullNormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $provider = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [System.IO.Path]::GetFullPath($provider).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
}

function Assert-OutsideRepo {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $full = Get-FullNormalizedPath $Path
    $repoFull = Get-FullNormalizedPath $repoRoot
    $inside = $full.Equals($repoFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $full.StartsWith($repoFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
    if ($inside) {
        throw "$Label is inside the repository: $full. Worktrees must live under the sibling root $requiredRoot, never under artifacts/ or anywhere below $repoFull."
    }
    return $full
}

$WorktreeRoot = Assert-OutsideRepo -Path $WorktreeRoot -Label 'Worktree root'

$branchName = "codex/$Task"
$target = Join-Path $WorktreeRoot $Task
$target = Assert-OutsideRepo -Path $target -Label 'Worktree target'
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
