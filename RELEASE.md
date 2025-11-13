# Release Process

This document outlines the steps to create a new release of SatoshiRig.

## Pre-Release Checklist

- [ ] All changes are committed
- [ ] Version number updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with new features and fixes
- [ ] `README.md` updated if needed
- [ ] `DEPLOY.md` updated if needed
- [ ] All tests pass (if applicable)
- [ ] Documentation is up to date

## Release Steps

### 1. Update Version and Documentation

The version should already be updated in `pyproject.toml` (currently v2.17.2).

### 2. Commit and Push Changes

```bash
# Stage all changes
git add .

# Commit with release message
git commit -m "Release v2.17.2: Wallet address DB-only, never from config.toml"

# Push to main branch
git push origin main
```

### 3. Create Git Tag

```bash
# Create annotated tag
git tag -a v2.17.2 -m "Release v2.17.2

Changes:
- Wallet address is now stored and loaded exclusively from database
- Wallet address is never read from or written to config.toml
- Wallet address is always loaded from database, even if empty

See CHANGELOG.md for full details."

# Push tag to remote
git push origin v2.17.2
```

### 4. Build and Publish Docker Image

**Option A: Automatic (via GitHub Actions)**

The Docker image will be automatically built and published when you push the tag:

1. Go to GitHub → Actions → "Build and Publish Docker Image"
2. Click "Run workflow"
3. Select the tag `v2.17.2` (or leave empty to use the latest tag)
4. Click "Run workflow"

The workflow will:
- Build the Docker image
- Push to `ghcr.io/rokk001/satoshirig:latest` and `ghcr.io/rokk001/satoshirig:v2.17.2`
- Automatically make the package public

**Option B: Manual Build**

```bash
# Build locally
docker build -t satoshirig:2.17.2 .

# Tag for GHCR
docker tag satoshirig:2.17.2 ghcr.io/rokk001/satoshirig:2.17.2
docker tag satoshirig:2.17.2 ghcr.io/rokk001/satoshirig:latest

# Push to GHCR (requires authentication)
docker push ghcr.io/rokk001/satoshirig:2.17.2
docker push ghcr.io/rokk001/satoshirig:latest
```

### 5. Create GitHub Release

**Option A: Automatic (via GitHub Actions)**

1. Go to GitHub → Actions → "Create GitHub Release"
2. Click "Run workflow"
3. Enter tag: `v2.17.2`
4. Click "Run workflow"

This will create a GitHub release with auto-generated release notes.

**Option B: Manual**

1. Go to GitHub → Releases → "Draft a new release"
2. Choose tag: `v2.17.2`
3. Title: `v2.17.2`
4. Description: Copy from `CHANGELOG.md` for version 2.17.2
5. Click "Publish release"

## Post-Release

- [ ] Verify Docker image is available: `docker pull ghcr.io/rokk001/satoshirig:latest`
- [ ] Verify GitHub release is published
- [ ] Update any external documentation if needed
- [ ] Announce release (if applicable)

## Quick Release Command Sequence

```bash
# 1. Commit and push
git add .
git commit -m "Release v2.17.2: Wallet address DB-only, never from config.toml"
git push origin main

# 2. Create and push tag
git tag -a v2.17.2 -m "Release v2.17.2 - See CHANGELOG.md"
git push origin v2.17.2

# 3. Trigger GitHub Actions manually:
# - Go to Actions → "Build and Publish Docker Image" → Run workflow
# - Go to Actions → "Create GitHub Release" → Run workflow with tag v2.17.2
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Current version: **2.17.2**

