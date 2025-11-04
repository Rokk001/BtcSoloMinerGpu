# How to Make Docker Package Public on GitHub Container Registry

If the automated workflow didn't set the package to public, follow these steps:

## Method 1: Via GitHub Web UI (Easiest)

1. Go to your GitHub profile: https://github.com/Rokk001
2. Click on the "Packages" tab (or go directly to: https://github.com/Rokk001?tab=packages)
3. Find the `satoshirig` package
4. Click on it to open the package page
5. Click on "Package settings" (on the right side)
6. Scroll down to the "Danger Zone" section
7. Click "Change visibility"
8. Select "Public"
9. Type the package name to confirm
10. Click "I understand, change visibility"

## Method 2: Via GitHub CLI

```bash
# Login to GitHub CLI
gh auth login

# List packages to find the exact name
gh api user/packages -q '.[] | select(.name == "satoshirig") | .name'

# Make it public (replace with actual package name if different)
gh api -X PATCH user/packages/container/satoshirig -f visibility=public
```

## Method 3: Via API Call

```bash
curl -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/user/packages/container/satoshirig \
  -d '{"visibility":"public"}'
```

## Verify

After making it public, verify by accessing:
```
https://ghcr.io/v2/rokk001/satoshirig/manifests/latest
```

This should return the manifest JSON without authentication errors.

