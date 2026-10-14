#!/usr/bin/env bash
# =============================================================================
# REBUILD.SH — Rebuild Jupyter Image with Cache
# =============================================================================
# Usage: ./scripts/rebuild.sh [--no-cache]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$REPO_ROOT/infra/docker-compose.jupyter.yaml"

NO_CACHE=false
if [[ "${1:-}" == "--no-cache" ]]; then
    NO_CACHE=true
fi

cd "$REPO_ROOT"

echo "=== Rebuilding Jupyter Image ==="
echo "Compose: $COMPOSE_FILE"
echo "No cache: $NO_CACHE"

# Build with BuildKit for better caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

BUILD_ARGS=()
if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

docker compose -f "$COMPOSE_FILE" build "${BUILD_ARGS[@]}" jupyter

echo "=== Verifying Build ==="
docker run --rm --gpus all nexus-jupyter:local python -c "
import torch, transformers, peft, bitsandbytes, flash_attn, sentence_transformers
print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Bitsandbytes: {bitsandbytes.__version__}')
print(f'Flash-Attn: {flash_attn.__version__}')
print(f'Sentence-Transformers: {sentence_transformers.__version__}')
"

echo "=== Restarting Container ==="
docker compose -f "$COMPOSE_FILE" up -d jupyter

echo "=== Waiting for Health ==="
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" ps jupyter | grep -q "healthy"; then
        echo "✅ Jupyter healthy"
        exit 0
    fi
    sleep 2
done

echo "⚠️ Health check timeout — check logs:"
docker compose -f "$COMPOSE_FILE" logs --tail=50 jupyter
exit 1