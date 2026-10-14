# Architecture — Jupyter Local Infra Template

> Technical deep-dive for the `jupyter-local-infra` stack.  
> **Customize this document** for your project's specific services and topology.

---

## 1. Network Topology

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              HOST NETWORK                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ MCP SERVER   │  │ POSTGRES     │  │ VECTOR DB    │  │ OLLAMA (opt) │     │
│  │ :7474        │  │ :5432        │  │ :6333        │  │ :11434       │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │                 │              │
│         ▼                 ▼                 ▼                 ▼              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MCP PROXY (socat, host network)                    │   │
│  │  TCP-LISTEN:7475,fork,reuseaddr → TCP:127.0.0.1:7474                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
         ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
         │  JUPYTER NET   │ │  OLLAMA NET  │ │ CODE-SERVER  │
         │  (bridge)      │ │  (bridge)    │ │  (bridge)    │
         │                │ │              │ │              │
         │  jupyter:8888  │ │ ollama:11434 │ │ code:8080    │
         │  mcp-proxy:7474│ │              │ │              │
         └────────────────┘ └──────────────┘ └──────────────┘
```

### Network Modes

| Service | Network Mode | Reason |
|---------|--------------|--------|
| `mcp-proxy` | `host` | Must reach `127.0.0.1:7474` on host; `host.docker.internal` unreliable on Linux |
| `jupyter` | `bridge` (default) | Isolated, accesses MCP via `mcp-proxy:7474` (service name resolves) |
| `ollama` | `bridge` | Internal only; Jupyter reaches via `ollama:11434` |
| `code-server` | `bridge` | Optional, isolated |

### DNS Resolution (Inside Containers)

| Target | Resolves To |
|--------|-------------|
| `mcp-proxy` | Container IP (bridge network) → forwards to host MCP port |
| `ollama` | Container IP → Ollama API |
| `host.docker.internal` | **Not used** — unreliable on Linux host |

---

## 2. Volume Architecture

```
HOST                          CONTAINER (jupyter)              PURPOSE
─────────────────────────────────────────────────────────────────────────────
./notebooks          →  /home/jovyan/work/notebooks        Git-tracked .ipynb
./data               →  /home/jovyan/work/data              Datasets, CSVs, exports
./models             →  /home/jovyan/work/models            LoRA adapters, checkpoints
./cache              →  /cache                             pip/uv/HF cache (persisted)
HOST_REPO (ro)       →  /home/jovyan/work/HOST_REPO        Read-only source access
~/.cache/huggingface →  /cache/huggingface                 Shared HF cache
ollama-data (vol)    →  /root/.ollama (ollama container)   Model blobs persisted
```

### Cache Strategy

| Cache | Location | Persistence |
|-------|----------|-------------|
| **pip** | `/cache/pip` | `./cache` bind mount |
| **uv** | `/cache/uv` | `./cache` bind mount |
| **HuggingFace** | `/cache/huggingface` | `./cache` + host `~/.cache/huggingface` |
| **Ollama** | `/root/.ollama` | Named volume `ollama-data` |

> **Why separate volumes?** Ollama uses custom format; HF cache is standard. Rebuilding Jupyter image shouldn't re-download Ollama models.

---

## 3. GPU Resource Allocation

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Behavior

| Scenario | Result |
|----------|--------|
| **Single GPU** | All containers share GPU (time-sliced by CUDA MPS or process scheduling) |
| **Multi-GPU** | All GPUs visible to all containers (`count: all`) |
| **Memory** | No hard limit — OOM kills process; use `CUDA_VISIBLE_DEVICES` to isolate |

### Isolation Options (if needed)

```yaml
# In service definition:
environment:
  - CUDA_VISIBLE_DEVICES=0        # Jupyter uses GPU 0
  - CUDA_VISIBLE_DEVICES=1        # Ollama uses GPU 1
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          device_ids: ['0']       # Hard assignment (requires specific GPU IDs)
```

---

## 4. Build Pipeline (Dockerfile Layers)

```
LAYER 0: BASE          nvidia/cuda:12.4.1-devel-ubuntu22.04
    │
    ├─ System deps (python3.11, build-essential, cmake, ninja, blas)
    │
LAYER 1: UV            ghcr.io/astral-sh/uv (fast installer)
    │
LAYER 2: PYTORCH       torch 2.4.1 + cu124 index
    │
LAYER 3: TRANSFORMERS  transformers, peft, accelerate, bitsandbytes, triton
    │
LAYER 4: FLASH-ATTN    flash-attn 2.6.3 (--no-build-isolation for CUDA 12.4)
    │
LAYER 5: EMBEDDINGS    sentence-transformers, faiss-gpu, hnswlib
    │
LAYER 6: GRAPH/DATA    networkx, pandas, numpy, scipy, sklearn, matplotlib
    │
LAYER 7: JUPYTER       jupyterlab, ipywidgets, jupyter-server-proxy
    │
LAYER 8: MCP/API       mcp, fastapi, uvicorn, httpx, websockets
    │
LAYER 9: DB            psycopg2, asyncpg, sqlalchemy, redis
    │
LAYER 10: UTILS        rich, typer, pyyaml, python-dotenv, tqdm, orjson
    │
LAYER 11: CONFIG       jupyter_notebook_config.py, entrypoint.sh, pull_models.sh
    │
LAYER 12: RUNTIME      USER 1000, WORKDIR /home/jovyan/work, ENTRYPOINT
```

### Layer Caching Strategy

- **Layers 0-4** (system + PyTorch + core ML) — rarely change, cached for weeks
- **Layers 5-7** (domain packages) — change when adding new libraries
- **Layers 8-10** (utils) — stable
- **Layer 11** (config) — changes frequently, small
- **Layer 12** (runtime) — minimal

> **Rebuild time**: ~3 min (cached) vs ~15 min (no cache)

---

## 5. MCP Proxy Design

### Why socat?

| Alternative | Problem |
|-------------|---------|
| `host.docker.internal:7474` | Doesn't resolve on Linux host (Docker Desktop only) |
| `nginx`/`traefik` | Overkill for TCP proxy; adds latency |
| `sshd -R` | Complex, requires SSH keys |
| **socat** | Single binary, TCP→TCP, `fork` handles multiple connections, `reuseaddr` avoids bind wait |

### Configuration

```yaml
mcp-proxy:
  image: alpine/socat:latest
  network_mode: host
  command: >
    TCP-LISTEN:7475,fork,reuseaddr
    TCP:127.0.0.1:7474
```

- **Port 7475** inside container network (Jupyter connects to `mcp-proxy:7474` — Compose maps service name to container IP)
- **Port 7474** on host (real MCP server)
- **fork** = new process per connection (stateless)
- **reuseaddr** = immediate restart without TIME_WAIT

### Jupyter → MCP Flow

```python
# Inside notebook
import httpx

# Resolves to mcp-proxy container IP (bridge network)
async with httpx.AsyncClient(base_url="http://mcp-proxy:7474") as client:
    resp = await client.post("/your_endpoint", json={"query": "..."})
```

---

## 6. Security Model

### Threat Model

| Asset | Threat | Mitigation |
|-------|--------|------------|
| **MCP Server** | Container compromise → data exfiltration | MCP Proxy only exposes read operations; no write tools |
| **Postgres/Vector DB** | Direct access from container | Not exposed to containers; only MCP can query |
| **Ollama** | Model extraction | Local only; no auth; bind to container network |
| **Jupyter** | Token brute-force | Simple token; not for public exposure |
| **Host FS** | Container escape | Host repo mounted `:ro`; no sensitive paths mounted |

### Hardening Checklist

- [ ] Jupyter token → strong random (if ever exposed via tunnel)
- [ ] MCP Proxy → restrict to localhost only (already `127.0.0.1`)
- [ ] Ollama → add auth proxy if multi-user
- [ ] Volumes → encrypt at rest (LUKS/ZFS) for sensitive data
- [ ] Network → `docker network create --internal` for stricter isolation

---

## 7. Failure Modes & Recovery

| Failure | Symptom | Recovery |
|---------|---------|----------|
| **GPU OOM** | Container killed, `nvidia-smi` shows XID error | Reduce batch size, enable gradient checkpointing, use `bitsandbytes` 4-bit |
| **MCP Proxy down** | `Connection refused` to `mcp-proxy:7474` | `docker restart PROJECT-mcp-proxy`; check host MCP port |
| **Ollama OOM** | Model pull fails, API returns 500 | Smaller model, increase swap |
| **Cache corruption** | `uv pip install` fails with hash mismatch | `docker compose down -v` (removes cache volumes) → rebuild |
| **Port conflict** | `address already in use` | Check `lsof -i :8888 :7475 :11434`; stop conflicting service |

---

## 8. Performance Baselines

| Operation | Expected Time (RTX 4090 24GB) |
|-----------|-------------------------------|
| **Image build (cached)** | ~3 min |
| **Image build (clean)** | ~15 min |
| **Ollama pull (nomic-embed)** | ~30 sec |
| **QLoRA 3B (r=16, 1k steps)** | ~8 min |
| **QLoRA 7B (r=16, 1k steps)** | ~22 min |

---

## 9. Extending the Stack

### Add TensorBoard
```yaml
# In docker-compose.jupyter.yaml
tensorboard:
  image: tensorflow/tensorflow:latest-gpu
  command: tensorboard --logdir /home/jovyan/work/logs --host 0.0.0.0
  volumes:
    - ./logs:/home/jovyan/work/logs
  ports:
    - "6006:6006"
```

### Add MLflow
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  command: mlflow server --host 0.0.0.0 --port 5000
  volumes:
    - ./mlruns:/mlruns
  ports:
    - "5000:5000"
```

### Add Custom Python Package
```dockerfile
# In Dockerfile.jupyter, after Layer 10:
RUN uv pip install --system --no-cache git+https://github.com/your/repo.git@main
```

---

## 10. Migration from Colab

| Colab Feature | Local Equivalent |
|---------------|------------------|
| **Free A100/T4** | Your GPU (RTX 3060/4090) |
| **Drive mount** | `./data` bind mount |
| **GitHub sync** | `./notebooks` git-tracked |
| **Secrets** | `.env` file (not committed) |
| **Pre-installed libs** | Dockerfile layers |
| **Session persistence** | Volumes survive `down/up` |
| **Colab Pro+** | No limits, your hardware |

---

*Generated from `infra/` configs. Update this doc when changing architecture.*
