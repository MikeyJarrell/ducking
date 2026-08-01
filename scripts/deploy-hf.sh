#!/bin/bash
# Deploy the current committed state of this repo to the Hugging Face Space.
#
# Why this script exists instead of a plain `git push hf main`:
#   Hugging Face refuses binary files stored the ordinary way — they must go
#   through Git LFS. Our icon.png is an ordinary binary in this repo's history,
#   and HF scans the ENTIRE history of whatever you push, so the push is
#   rejected even if the current files are fine. Removing the icon from the
#   newest commit does not help either; the old commit still carries it.
#
# So we don't push this repo's history at all. We build ONE fresh snapshot
# commit in a temp directory, with icon.png tracked by LFS the way HF wants,
# and force-push that. Consequences, all intended:
#   - The Space's git log is just a list of deploy snapshots. Fine: GitHub holds
#     the real history.
#   - This repo is never touched, rewritten, or switched off main.
#
# Usage:  bash scripts/deploy-hf.sh
# Requires: git-lfs installed, and a Hugging Face token at
#           ~/.cache/huggingface/token (check with `huggingface-cli whoami`).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_URL="https://huggingface.co/spaces/MikeyJarrell/ducking"
TOKEN_FILE="$HOME/.cache/huggingface/token"

command -v git-lfs >/dev/null || { echo "ERROR: git-lfs is not installed (brew install git-lfs)."; exit 1; }
[ -f "$TOKEN_FILE" ] || { echo "ERROR: no Hugging Face token at $TOKEN_FILE. Run: huggingface-cli login"; exit 1; }

# Refuse to deploy a dirty tree — what ships must be what is committed.
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  echo "ERROR: working tree has uncommitted changes. Commit or stash first."
  git -C "$REPO" status --short
  exit 1
fi

SRC_COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Export the committed tree (not the working directory) into the staging dir.
git -C "$REPO" archive HEAD | tar -x -C "$STAGE"

cd "$STAGE"
git init -q
git lfs install --local >/dev/null

# Track the binaries HF objects to. The repo already ships a .gitattributes
# from the Space template; append rather than replace it.
{ echo "*.png filter=lfs diff=lfs merge=lfs -text"
  echo "*.icns filter=lfs diff=lfs merge=lfs -text"; } >> .gitattributes

git add -A
git -c user.name="$(git -C "$REPO" config user.name)" \
    -c user.email="$(git -C "$REPO" config user.email)" \
    commit -q -m "Deploy $SRC_COMMIT

Snapshot of GitHub main $SRC_COMMIT. Binaries stored via Git LFS."

echo "Staged deploy snapshot from $SRC_COMMIT:"
git lfs ls-files | sed 's/^/  LFS: /'

git remote add hf "$SPACE_URL"
git -c credential.helper='!f() { echo "username=MikeyJarrell"; echo "password=$(cat '"$TOKEN_FILE"')"; }; f' \
    push --force hf HEAD:refs/heads/main

echo
echo "Deployed $SRC_COMMIT to $SPACE_URL"
echo "The Space rebuilds automatically; give it a couple of minutes."
