#!/bin/sh
# Create https://github.com/s-broda/RND if needed and push main.
set -e
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export PATH
if ! command -v gh >/dev/null; then
  echo "Install GitHub CLI (gh) and retry." >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Log in to GitHub (one-time). A browser window will open."
  gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key
fi
if ! git ls-remote git@github.com:s-broda/RND.git >/dev/null 2>&1; then
  gh repo create s-broda/RND --public --source=. --remote=origin --description "Closed-form risk-neutral density from option prices"
fi
git push -u origin main
echo "https://github.com/s-broda/RND"
