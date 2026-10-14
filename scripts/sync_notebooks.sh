#!/usr/bin/env bash
# =============================================================================
# SYNC_NOTEBOOKS.SH — Bidirectional Colab ↔ Local Notebook Sync
# =============================================================================
# Usage:
#   ./scripts/sync_notebooks.sh push <notebook.ipynb>     # Local → Colab (gist)
#   ./scripts/sync_notebooks.sh pull <gist_id> [dest]     # Colab → Local
#   ./scripts/sync_notebooks.sh status                    # Show sync status
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOKS_DIR="$REPO_ROOT/notebooks"

ACTION="${1:-}"
TARGET="${2:-}"
DEST="${3:-}"

command -v gh >/dev/null || { echo "❌ GitHub CLI (gh) not installed"; exit 1; }
command -v jq >/dev/null || { echo "❌ jq not installed"; exit 1; }

# Ensure we're in a git repo
cd "$REPO_ROOT"

case "$ACTION" in
    push)
        [[ -z "$TARGET" ]] && { echo "Usage: $0 push <notebook.ipynb>"; exit 1; }
        [[ -f "$NOTEBOOKS_DIR/$TARGET" ]] || { echo "❌ Not found: $NOTEBOOKS_DIR/$TARGET"; exit 1; }

        echo "=== Pushing $TARGET to GitHub Gist ==="
        
        # Create gist (private)
        GIST_ID=$(gh gist create "$NOTEBOOKS_DIR/$TARGET" --private -d "ML Project: $TARGET" --json id -q .id)
        
        if [[ -n "$GIST_ID" ]]; then
            GIST_URL="https://gist.github.com/$GIST_ID"
            echo "✅ Gist created: $GIST_URL"
            echo "Colab URL: https://colab.research.google.com/gist/$GIST_ID"
            
            # Save mapping
            mkdir -p "$REPO_ROOT/.sync"
            echo "$TARGET $GIST_ID" >> "$REPO_ROOT/.sync/gist_map.txt"
        else
            echo "❌ Failed to create gist"
            exit 1
        fi
        ;;

    pull)
        [[ -z "$TARGET" ]] && { echo "Usage: $0 pull <gist_id> [dest.ipynb]"; exit 1; }
        
        DEST_FILE="${DEST:-$(basename "$TARGET").ipynb}"
        DEST_PATH="$NOTEBOOKS_DIR/$DEST_FILE"
        
        echo "=== Pulling gist $TARGET to $DEST_PATH ==="
        
        gh gist view "$TARGET" --raw > "$DEST_PATH"
        
        if [[ -s "$DEST_PATH" ]]; then
            echo "✅ Saved to $DEST_PATH"
            echo "Add to git: git add $DEST_PATH"
        else
            echo "❌ Failed to pull gist"
            exit 1
        fi
        ;;

    status)
        echo "=== Sync Status ==="
        if [[ -f "$REPO_ROOT/.sync/gist_map.txt" ]]; then
            echo "Tracked notebooks:"
            while read -r notebook gist_id; do
                echo "  $notebook → gist:$gist_id"
            done < "$REPO_ROOT/.sync/gist_map.txt"
        else
            echo "No sync history found"
        fi
        
        echo ""
        echo "Local notebooks:"
        ls -la "$NOTEBOOKS_DIR"/*.ipynb 2>/dev/null | head -20
        ;;

    *)
        echo "Usage: $0 {push|pull|status} [args]"
        echo "  push <notebook.ipynb>     Create private gist for Colab"
        echo "  pull <gist_id> [dest]     Download gist to local"
        echo "  status                    Show sync mappings"
        exit 1
        ;;
esac
