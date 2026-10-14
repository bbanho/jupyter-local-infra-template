#!/usr/bin/env bash
# =============================================================================
# ENTRYPOINT.SH — NEXUS Jupyter Container Entrypoint
# =============================================================================
# - Pulls Ollama models in background
# - Starts JupyterLab
# - Handles signals gracefully
# =============================================================================

set -euo pipefail

echo "=== NEXUS Jupyter Environment Starting ==="
echo "User: $(whoami) (UID: $(id -u))"
echo "Workdir: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Not detected')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //;s/,.*//' || echo 'Not found')"

# -----------------------------------------------------------------------------
# Start Ollama model pull in background
# -----------------------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    echo "Starting Ollama model pull in background..."
    /usr/local/bin/pull_models.sh &
    PULL_PID=$!
    echo "Pull PID: $PULL_PID"
else
    echo "Ollama not found in container (running in separate container)"
fi

# -----------------------------------------------------------------------------
# Verify Python environment
# -----------------------------------------------------------------------------
echo "=== Python Environment ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'Bitsandbytes: {bitsandbytes.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention: not available"
python -c "import sentence_transformers; print(f'Sentence Transformers: {sentence_transformers.__version__}')"

# -----------------------------------------------------------------------------
# Setup NEXUS repo access (read-only)
# -----------------------------------------------------------------------------
if [ -d "/home/jovyan/work/NEXUS" ]; then
    echo "=== NEXUS Repository Available ==="
    cd /home/jovyan/work/NEXUS
    git log --oneline -1 2>/dev/null || echo "Not a git repo"
    cd /home/jovyan/work
fi

# -----------------------------------------------------------------------------
# Signal handling
# -----------------------------------------------------------------------------
shutdown() {
    echo "=== Shutdown signal received ==="
    if [ -n "${PULL_PID:-}" ]; then
        kill $PULL_PID 2>/dev/null || true
        wait $PULL_PID 2>/dev/null || true
    fi
    exit 0
}
trap shutdown SIGTERM SIGINT

# -----------------------------------------------------------------------------
# Execute command
# -----------------------------------------------------------------------------
echo "=== Starting JupyterLab ==="
exec "$@"