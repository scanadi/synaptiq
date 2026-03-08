#!/usr/bin/env bash
set -euo pipefail

# ── Release script for Synaptiq ──────────────────────────────────────────────
# Usage:
#   ./scripts/release.sh patch       # 0.5.0 → 0.5.1
#   ./scripts/release.sh minor       # 0.5.0 → 0.6.0
#   ./scripts/release.sh major       # 0.5.0 → 1.0.0
#   ./scripts/release.sh 0.7.0       # explicit version
#   ./scripts/release.sh patch --dry  # preview without making changes
# ─────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$ROOT/pyproject.toml"
INIT_PY="$ROOT/src/synaptiq/__init__.py"

DRY_RUN=false

# ── Parse args ───────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <patch|minor|major|X.Y.Z> [--dry]"
    exit 1
fi

BUMP="$1"
shift
for arg in "$@"; do
    [[ "$arg" == "--dry" ]] && DRY_RUN=true
done

# ── Read current version ─────────────────────────────────────────────────────

CURRENT=$(grep -m1 '^version' "$PYPROJECT" | sed 's/.*"\(.*\)"/\1/')
IFS='.' read -r MAJOR MINOR PATCH_NUM <<< "$CURRENT"

echo "Current version: $CURRENT"

# ── Compute new version ──────────────────────────────────────────────────────

case "$BUMP" in
    patch) NEW_VERSION="$MAJOR.$MINOR.$((PATCH_NUM + 1))" ;;
    minor) NEW_VERSION="$MAJOR.$((MINOR + 1)).0" ;;
    major) NEW_VERSION="$((MAJOR + 1)).0.0" ;;
    *)
        if [[ "$BUMP" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            NEW_VERSION="$BUMP"
        else
            echo "Error: invalid bump type '$BUMP'. Use patch, minor, major, or X.Y.Z"
            exit 1
        fi
        ;;
esac

echo "New version:     $NEW_VERSION"
TAG="v$NEW_VERSION"

# ── Dry run exit ─────────────────────────────────────────────────────────────

if $DRY_RUN; then
    echo ""
    echo "[dry run] Would update:"
    echo "  $PYPROJECT  → version = \"$NEW_VERSION\""
    echo "  $INIT_PY    → __version__ = \"$NEW_VERSION\""
    echo "  git commit  → chore: bump version to $NEW_VERSION"
    echo "  git tag     → $TAG"
    echo "  git push    → origin main + $TAG"
    exit 0
fi

# ── Guard: clean working tree ────────────────────────────────────────────────

if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    echo "Error: working tree is dirty. Commit or stash changes first."
    exit 1
fi

# ── Guard: on main branch ───────────────────────────────────────────────────

BRANCH=$(git -C "$ROOT" branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: releases must be made from the 'main' branch (currently on '$BRANCH')."
    exit 1
fi

# ── Guard: up to date with remote ───────────────────────────────────────────

git -C "$ROOT" fetch origin main --quiet
LOCAL=$(git -C "$ROOT" rev-parse HEAD)
REMOTE=$(git -C "$ROOT" rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "Error: local main is not up to date with origin/main. Pull first."
    exit 1
fi

# ── Guard: tag doesn't already exist ────────────────────────────────────────

if git -C "$ROOT" tag -l "$TAG" | grep -q "$TAG"; then
    echo "Error: tag $TAG already exists."
    exit 1
fi

# ── Update version in pyproject.toml ─────────────────────────────────────────

sed -i '' "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"

# ── Update version in __init__.py ────────────────────────────────────────────

sed -i '' "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW_VERSION\"/" "$INIT_PY"

# ── Verify updates ──────────────────────────────────────────────────────────

echo ""
echo "Updated files:"
grep -n "version" "$PYPROJECT" | head -1
grep -n "__version__" "$INIT_PY"

# ── Commit, tag, push ───────────────────────────────────────────────────────

echo ""
git -C "$ROOT" add "$PYPROJECT" "$INIT_PY"
git -C "$ROOT" commit -m "chore: bump version to $NEW_VERSION"
git -C "$ROOT" tag -a "$TAG" -m "Release $NEW_VERSION"

echo ""
echo "Pushing to origin..."
git -C "$ROOT" push origin main
git -C "$ROOT" push origin "$TAG"

echo ""
echo "Done! Release $TAG pushed."
echo "GitHub Actions will build and publish to PyPI."
echo "Monitor: https://github.com/scanadi/synaptiq/actions"
