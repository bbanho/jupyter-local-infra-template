#!/usr/bin/env bash
# =============================================================================
# PULL_MODELS.SH — Pre-pull Ollama models for NEXUS Studies
# =============================================================================
# Runs at container startup. Models persist in ollama-data volume.
# =============================================================================

set -euo pipefail

MODELS=(
    "nomic-embed-text:v1.5"      # 768d embeddings (ADR-0036)
    "qwen2.5-coder:7b"           # Code generation (fallback if GPU allows)
    "qwen2.5-coder:3b"           # Code generation (lighter, fits 8GB VRAM)
    "llama3.1:8b"                # General reasoning
)

echo "=== Waiting for Ollama to be ready ==="
until curl -sf http://ollama:11434/api/tags >/dev/null; do
    sleep 2
done

echo "=== Pre-pulling models ==="
for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model" || echo "Warning: failed to pull $model (may already exist or OOM)"
done

echo "=== Available models ==="
ollama list