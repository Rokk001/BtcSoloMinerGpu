<#
Safe release script for SatoshiRig
- Prevents creating duplicate tags (local or remote)
- Requires clean working directory by default (no auto-commit)
- Optionally uses CHANGELOG.md as release notes (if gh is available)

Usage examples:
# Interactive: prompt for tag
pwsh .\scripts\release.ps1

# Provide an explicit semver tag
pwsh .\scripts\release.ps1 -Tag v1.2.3

# Auto-commit local changes (NOT recommended)
pwsh .\scripts\release.ps1 -Tag v1.2.3 -AutoCommit
#>

param(
    [string]$Tag,
    [switch]$AutoCommit,
    [string]$Message = "Release",
    [switch]$UseChangelog
)

function Write-ErrAndExit($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# ensure we're in a git repo
if (-not (Test-Path .git)) {
    Write-ErrAndExit "Not a git repository (no .git folder found). Run this from the repo root."
}

# ensure git exists
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-ErrAndExit "git not available in PATH"
}

# determine tag if not supplied - use seconds to avoid accidental dupes
if (-not $Tag) {
    $Tag = "v" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

Write-Host "Preparing release tag: $Tag"

# check if tag exists locally
$localTagExists = (& git rev-parse -q --verify "refs/tags/$Tag" 2>$null) -ne $null
if ($localTagExists) {
    Write-ErrAndExit "Local tag '$Tag' already exists. Aborting to avoid duplicate releases."
}

# check if tag exists on origin
$remoteTags = & git ls-remote --tags origin 2>$null
if ($remoteTags -and $remoteTags -match [regex]::Escape($Tag)) {
    Write-ErrAndExit "Remote tag '$Tag' already exists on origin. Aborting to avoid duplicate releases."
}

# ensure working tree is clean unless AutoCommit
$porcelain = (& git status --porcelain)
if ($porcelain -ne "" -and -not $AutoCommit) {
    Write-ErrAndExit "Working directory is not clean. Commit or stash changes, or run with -AutoCommit to auto-commit (not recommended)."
}

if ($porcelain -ne "" -and $AutoCommit) {
    Write-Host "Auto-committing local changes before tagging..."
    & git add -A
    $tempMsg = "chore(release): auto-commit before tagging ($Tag)"
    & git commit -m $tempMsg
}

# ensure main branch
$currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne "main") {
    Write-Host "Switching to main branch"
    & git checkout main
    & git pull origin main
}
else {
    & git pull origin main
}

# create annotated tag
Write-Host "Creating annotated tag: $Tag"
& git tag -a $Tag -m "$Message $Tag"

# push tag
Write-Host "Pushing tag to origin: $Tag"
$pushOut = & git push origin $Tag 2>&1
Write-Host $pushOut
if ($LASTEXITCODE -ne 0) {
    Write-ErrAndExit "Failed to push tag to origin"
}

# create GitHub release if gh is present
if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "gh CLI detected - creating GitHub release for $Tag"
    if ($UseChangelog -and (Test-Path CHANGELOG.md)) {
        gh release create $Tag --title $Tag --notes-file CHANGELOG.md
    }
    else {
        gh release create $Tag --title $Tag --notes "Automated release $Tag"
    }
}
else {
    Write-Host "gh CLI not found - skipping GitHub release creation"
}

Write-Host "Release $Tag created successfully."
