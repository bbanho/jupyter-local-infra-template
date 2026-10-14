#!/usr/bin/env bash
# =============================================================================
# BACKUP_MODELS.SH — Export Ollama Models
# =============================================================================
# Usage: ./scripts/backup_models.sh [model_name] [output_dir]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$REPO_ROOT/infra/docker-compose.jupyter.yaml"
PROJECT_NAME=$(grep -E '^PROJECT_NAME=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 || echo "ml-project")

MODEL="${1:-}"
OUTPUT_DIR="${2:-$REPO_ROOT/models/exports}"

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model_name> [output_dir]"
    echo "Example: $0 qwen2.5-coder:7b"
    echo "Available models:"
    docker exec ${PROJECT_NAME}-ollama ollama list 2>/dev/null || echo "Container not running"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Backing up $MODEL to $OUTPUT_DIR ==="

# Method 1: Ollama export (creates GGUF)
echo "Exporting via ollama..."
docker exec ${PROJECT_NAME}-ollama ollama cp "$MODEL" "/tmp/$MODEL" 2>/dev/null || true
docker cp "${PROJECT_NAME}-ollama:/tmp/$MODEL" "$OUTPUT_DIR/" 2>/dev/null || true

# Method 2: Direct model directory copy (if export fails)
echo "Copying model blobs..."
MODEL_SAFE="${MODEL//:/}"
docker cp "${PROJECT_NAME}-ollama:/root/.ollama/models/manifests/registry.ollama.ai/library/${MODEL_SAFE}" "$OUTPUT_DIR/${MODEL_SAFE}_manifest" 2>/dev/null || true
docker cp "${PROJECT_NAME}-ollama:/root/.ollama/models/blobs" "$OUTPUT_DIR/${MODEL_SAFE}_blobs" 2>/dev/null || true

# Verify
if [[ -d "$OUTPUT_DIR/$MODEL" ]] || [[ -d "$OUTPUT_DIR/${MODEL_SAFE}" ]]; then
    echo "✅ Backup complete: $OUTPUT_DIR"
    du -sh "$OUTPUT_DIR"/*
else
    echo "⚠️ Backup may be incomplete — check Docker logs"
    docker exec ${PROJECT_NAME}-ollama ollama list
fi
