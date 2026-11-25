#!/usr/bin/env bash
# release.sh — one-command release for TERAG
#
# Usage:
#   ./scripts/release.sh 0.9.0          # release a specific version
#   ./scripts/release.sh patch          # auto-bump patch (0.8.0 → 0.8.1)
#   ./scripts/release.sh minor          # auto-bump minor (0.8.0 → 0.9.0)
#   ./scripts/release.sh major          # auto-bump major (0.8.0 → 1.0.0)
#   ./scripts/release.sh --dry-run 0.9.0  # validate everything without pushing
#
# What it does:
#   1. Checks prerequisites (clean tree, tools installed)
#   2. Bumps version in pyproject.toml and __init__.py
#   3. Cleans old builds
#   4. Runs tests
#   5. Builds sdist + wheel
#   6. Validates with twine check --strict
#   7. Test-installs the wheel in a temp venv
#   8. Commits the version bump
#   9. Tags vX.Y.Z and pushes (triggers GitHub Actions publish.yml)
#
# Prerequisites:
#   - PYPI_API_TOKEN secret configured in GitHub repo settings
#   - python3, pip, build, twine, pytest available

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYPROJECT="pyproject.toml"
INIT_PY="terag/terag/__init__.py"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[release]${NC} $*"; }
warn()  { echo -e "${YELLOW}[release]${NC} $*"; }
error() { echo -e "${RED}[release]${NC} $*" >&2; }
fatal() { error "$@"; exit 1; }

# --- Parse flags ---
DRY_RUN=false
SKIP_TESTS=false
LOCAL_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        --local)      LOCAL_UPLOAD=true; shift ;;
        -*)           fatal "Unknown flag: $1" ;;
        *)            VERSION_ARG="$1"; shift ;;
    esac
done

VERSION_ARG="${VERSION_ARG:-}"

# --- Read current version ---
current_version() {
    grep '^version' "$PYPROJECT" | head -1 | sed 's/.*= *"\(.*\)"/\1/'
}

CURRENT=$(current_version)
info "Current version: ${BOLD}$CURRENT${NC}"

# --- Compute target version ---
bump_version() {
    local ver="$1" part="$2"
    IFS='.' read -r major minor patch <<< "$ver"
    case "$part" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
    esac
}

if [[ -z "$VERSION_ARG" ]]; then
    fatal "Usage: $0 [--dry-run] [--skip-tests] [--local] <version|patch|minor|major>"
fi

case "$VERSION_ARG" in
    major|minor|patch) NEW_VERSION=$(bump_version "$CURRENT" "$VERSION_ARG") ;;
    [0-9]*)            NEW_VERSION="$VERSION_ARG" ;;
    *)                 fatal "Invalid version: $VERSION_ARG" ;;
esac

TAG="v${NEW_VERSION}"
info "Target version: ${BOLD}$NEW_VERSION${NC} (tag: $TAG)"

if $DRY_RUN; then
    warn "DRY RUN: will validate but not commit, tag, or push"
fi

# --- Preflight checks ---
info "Running preflight checks..."

command -v git >/dev/null || fatal "git not found"

# Prefer the project .venv if it exists and has python
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    info "Using project venv: $PYTHON"
else
    PYTHON="python3"
    info "Using system python: $PYTHON"
fi
command -v "$PYTHON" >/dev/null || fatal "python not found"

# Check tools, install if missing (into whichever python we picked)
for tool in build twine pytest; do
    if ! "$PYTHON" -m "$tool" --version &>/dev/null 2>&1; then
        warn "$tool not found, installing..."
        "$PYTHON" -m pip install --quiet "$tool"
    fi
done

# Clean git tree
if [[ -n "$(git status --porcelain)" ]]; then
    fatal "Working tree is dirty. Commit or stash changes first."
fi

# Check the tag doesn't already exist
if git rev-parse "$TAG" &>/dev/null; then
    fatal "Tag $TAG already exists"
fi

# Check we're on main
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    warn "You are on branch '$BRANCH', not 'main'. Proceed? (y/N)"
    read -r answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
fi

info "Preflight checks passed"

# --- Bump version ---
info "Bumping version $CURRENT -> $NEW_VERSION"

# Update pyproject.toml
sed -i '' "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"

# Update __init__.py fallback
sed -i '' "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW_VERSION\"/" "$INIT_PY"

# Verify the bump took
NEW_CHECK=$(current_version)
if [[ "$NEW_CHECK" != "$NEW_VERSION" ]]; then
    fatal "Version bump failed: pyproject.toml shows '$NEW_CHECK'"
fi

info "Version bumped in pyproject.toml and __init__.py"

# --- Run tests ---
if $SKIP_TESTS; then
    warn "Skipping tests (--skip-tests)"
else
    info "Running tests..."
    "$PYTHON" -m pytest -q || {
        # Revert the version bump on test failure
        sed -i '' "s/^version = \"$NEW_VERSION\"/version = \"$CURRENT\"/" "$PYPROJECT"
        sed -i '' "s/__version__ = \"$NEW_VERSION\"/__version__ = \"$CURRENT\"/" "$INIT_PY"
        fatal "Tests failed. Version bump reverted."
    }
    info "Tests passed"
fi

# --- Clean and build ---
info "Cleaning old builds..."
rm -rf dist/ build/ terag/terag.egg-info/ terag/*.egg-info

info "Building sdist and wheel..."
"$PYTHON" -m build --outdir dist/

info "Build artifacts:"
ls -lh dist/

# --- Validate ---
info "Running twine check..."
"$PYTHON" -m twine check --strict dist/* || {
    sed -i '' "s/^version = \"$NEW_VERSION\"/version = \"$CURRENT\"/" "$PYPROJECT"
    sed -i '' "s/__version__ = \"$NEW_VERSION\"/__version__ = \"$CURRENT\"/" "$INIT_PY"
    fatal "twine check failed. Version bump reverted."
}

# --- Test install ---
info "Test-installing wheel in temp venv..."
TMPVENV=$(mktemp -d)
"$PYTHON" -m venv "$TMPVENV"
"$TMPVENV/bin/python" -m pip install --quiet dist/*.whl
"$TMPVENV/bin/python" -c "
from terag import TERAG, TERAGConfig, __version__
assert __version__ == '$NEW_VERSION', f'Version mismatch: {__version__}'
print(f'terag {__version__} installed and importable')
" || {
    rm -rf "$TMPVENV"
    sed -i '' "s/^version = \"$NEW_VERSION\"/version = \"$CURRENT\"/" "$PYPROJECT"
    sed -i '' "s/__version__ = \"$NEW_VERSION\"/__version__ = \"$CURRENT\"/" "$INIT_PY"
    fatal "Test install failed. Version bump reverted."
}
rm -rf "$TMPVENV"
info "Test install passed"

# --- Dry run stops here ---
if $DRY_RUN; then
    warn "DRY RUN complete. Reverting version bump."
    sed -i '' "s/^version = \"$NEW_VERSION\"/version = \"$CURRENT\"/" "$PYPROJECT"
    sed -i '' "s/__version__ = \"$NEW_VERSION\"/__version__ = \"$CURRENT\"/" "$INIT_PY"
    rm -rf dist/
    info "Everything looks good. Run without --dry-run to publish."
    exit 0
fi

# --- Commit, tag, push ---
info "Committing version bump..."
git add "$PYPROJECT" "$INIT_PY"
git commit -m "Release $TAG"

info "Tagging $TAG..."
git tag -a "$TAG" -m "Release $NEW_VERSION"

info "Pushing to origin..."
git push origin main
git push origin "$TAG"

# --- Optional local upload ---
if $LOCAL_UPLOAD; then
    info "Uploading to PyPI directly (--local mode)..."
    "$PYTHON" -m twine upload dist/*
else
    info "Tag $TAG pushed. GitHub Actions will publish to PyPI."
    echo ""
    echo -e "  ${BOLD}Monitor the workflow:${NC}"
    echo -e "  https://github.com/rudranaik/terag/actions"
    echo ""
fi

echo ""
info "Release $NEW_VERSION complete."
echo -e "  ${BOLD}Verify:${NC} pip install terag==$NEW_VERSION"
echo ""
