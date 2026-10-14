# Usage Guide — Jupyter Local Infra

> Daily workflows for the local ML development stack.

---

## 1. First-Time Setup

```bash
# 1. Clone your repo (created from template)
git clone git@github.com:YOUR_ORG/YOUR_REPO.git
cd YOUR_REPO

# 2. Configure environment
cp .env.example .env
# Edit .env with your paths, tokens, model names

# 3. Start stack
docker compose -f infra/docker-compose.jupyter.yaml up -d --build

# 4. Verify
open http://localhost:8888  # Token from .env
```

### Verify Services
| Service | Check |
|---------|-------|
| Jupyter | `curl http://localhost:8888/api` |
| Ollama | `curl http://localhost:11434/api/tags` |
| MCP Proxy | `curl http://localhost:7475/health` (or `nc -zv localhost 7475`) |

---

## 2. Daily Workflow

### Start Work
```bash
docker compose -f infra/docker-compose.jupyter.yaml up -d
# JupyterLab at http://localhost:8888
```

### Stop Work (preserves volumes)
```bash
docker compose -f infra/docker-compose.jupyter.yaml down
```

### Full Reset (clears cache volumes)
```bash
docker compose -f infra/docker-compose.jupyter.yaml down -v
docker compose -f infra/docker-compose.jupyter.yaml up -d --build
```

---

## 3. Notebook Management

### Create New Notebook
1. JupyterLab UI → File → New → Notebook
2. Save in `./notebooks/your_name.ipynb`
3. Commit: `git add notebooks/your_name.ipynb && git commit -m "add: your_name notebook"`

### Colab ↔ Local Sync

#### Local → Colab (push)
```bash
./scripts/sync_notebooks.sh push notebooks/your_notebook.ipynb
# Outputs Gist URL + Colab link
```

#### Colab → Local (pull)
```bash
./scripts/sync_notebooks.sh pull <gist_id> [dest_path]
# Downloads from gist to ./notebooks/
```

### Colab Connection
1. Open notebook in Colab
2. **Conectar → Conectar ao ambiente de execução local**
3. URL: `http://localhost:8888/?token=YOUR_TOKEN`
4. Colab UI now runs on **your GPU**, with **your data**

---

## 4. Model Management (Ollama)

### Pull New Model
```bash
# From host
docker exec -it YOUR_PROJECT-ollama ollama pull qwen2.5-coder:14b

# Or from inside Jupyter notebook
!docker exec -it YOUR_PROJECT-ollama ollama pull qwen2.5-coder:14b
```

### List Models
```bash
curl http://localhost:11434/api/tags | jq '.models[] | {name, size}'
```

### Remove Model
```bash
docker exec -it YOUR_PROJECT-ollama ollama rm model-name
```

### Pre-pull on Startup
Edit `.env`:
```bash
OLLAMA_MODELS="nomic-ai/nomic-embed-text-v1.5 qwen2.5-coder:3b llama3.1:8b"
```
Models pulled automatically on `docker compose up`.

### Backup Models
```bash
./scripts/backup_models.sh
# Creates ./models/ollama-backup-YYYYMMDD.tar.gz
```

---

## 5. LoRA Training (MoLE Pipeline)

### Train Single Shard
```bash
python scripts/train_shard.py \
  --shard-id 0 \
  --model Qwen/Qwen2.5-Coder-3B \
  --data /home/jovyan/work/data/shards/shard_0.jsonl \
  --output /home/jovyan/work/models/lora-shard-00 \
  --r 16 --alpha 32 --steps 500 --bf16
```

### Train All Shards (Parallel)
```bash
# Option A: tmux (recommended)
for i in {0..7}; do
  tmux new-session -d -s shard$i \
    "python scripts/train_shard.py --shard-id $i --bf16"
done
# Attach: tmux attach -t shard0

# Option B: GNU Parallel
parallel -j 4 python scripts/train_shard.py --shard-id {} --bf16 ::: {0..7}

# Option C: Background (less control)
for i in {0..7}; do
  python scripts/train_shard.py --shard-id $i --bf16 > "logs/shard_$i.log" 2>&1 &
done
```

### Monitor Training
```bash
# TensorBoard (if enabled)
open http://localhost:6006

# Logs
docker compose -f infra/docker-compose.jupyter.yaml logs -f jupyter

# GPU usage
watch -n 1 nvidia-smi
```

### Extract Holographic Keys (after all shards trained)
```bash
# Run in notebook: notebooks/02_mole_holographic.ipynb → Phase 3
# Or standalone:
python -c "
from notebooks.02_mole_holographic import extract_keys
extract_keys()
"
```

---

## 6. Data Management

### Export Data from Your Graph/DB
```python
# In notebook
import httpx

async def export_your_data():
    async with httpx.AsyncClient(base_url="http://mcp-proxy:7474") as client:
        resp = await client.post("/your_export_endpoint", json={"limit": 10000})
        return resp.json()

# Save to ./data/
import json
with open("/home/jovyan/work/data/your_export.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

### Large Files (gitignored)
- `./data/` — CSVs, embeddings, exports >100MB
- `./models/` — LoRA adapters, checkpoints
- Use `git lfs` if you must version large files

---

## 7. Debugging & Troubleshooting

### Container Logs
```bash
# All services
docker compose -f infra/docker-compose.jupyter.yaml logs -f

# Specific service
docker compose -f infra/docker-compose.jupyter.yaml logs -f jupyter
docker compose -f infra/docker-compose.jupyter.yaml logs -f ollama
```

### Shell Into Container
```bash
# Jupyter (main dev env)
docker exec -it YOUR_PROJECT-jupyter bash

# Ollama
docker exec -it YOUR_PROJECT-ollama bash
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `nvidia-smi` fails in container | NVIDIA Container Toolkit not installed | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| GPU OOM | Batch too large, no gradient checkpointing | Reduce `batch_size`, increase `grad_accum`, enable `--bf16` |
| MCP connection refused | Host MCP not running or wrong port | Check `MCP_HOST_PORT` in `.env`; verify host service on `:7474` |
| Ollama 500 on pull | VRAM full | Stop Jupyter, pull model, restart: `docker compose stop jupyter && docker exec ollama pull ... && docker compose start jupyter` |
| Port 8888 in use | Another Jupyter running | `lsof -i :8888` → kill or change `JUPYTER_PORT` in `.env` |
| Permission denied on volumes | UID/GID mismatch | `sudo chown -R 1000:1000 ./notebooks ./data ./models ./cache` |

### Health Checks
```bash
# All services healthy?
docker compose -f infra/docker-compose.jupyter.yaml ps

# Expected output:
# NAME                    STATUS
# ml-project-jupyter      Up (healthy)
# ml-project-ollama       Up (healthy)
# ml-project-mcp-proxy    Up (healthy)
```

---

## 8. Rebuilding

### After Dockerfile Changes
```bash
./scripts/rebuild.sh
# Or manually:
docker compose -f infra/docker-compose.jupyter.yaml build --no-cache jupyter
docker compose -f infra/docker-compose.jupyter.yaml up -d jupyter
```

### After .env Changes
```bash
docker compose -f infra/docker-compose.jupyter.yaml up -d --force-recreate
```

### Clean Rebuild (nuclear)
```bash
docker compose -f infra/docker-compose.jupyter.yaml down -v
docker system prune -f
docker compose -f infra/docker-compose.jupyter.yaml up -d --build
```

---

## 9. GPU Isolation (Multi-User / Multi-Project)

### Option A: CUDA_VISIBLE_DEVICES
```yaml
# In docker-compose.jupyter.yaml
jupyter:
  environment:
    - CUDA_VISIBLE_DEVICES=0
ollama:
  environment:
    - CUDA_VISIBLE_DEVICES=1
```

### Option B: MPS (Multi-Process Service)
```bash
# On host (requires root)
nvidia-cuda-mps-control -d
# Then all containers share GPU with context switching
```

### Option C: Separate Stacks
```bash
# Project A on GPU 0
CUDA_VISIBLE_DEVICES=0 docker compose -f infra/docker-compose.jupyter.yaml -p proj-a up -d

# Project B on GPU 1
CUDA_VISIBLE_DEVICES=1 docker compose -f infra/docker-compose.jupyter.yaml -p proj-b up -d
```

---

## 10. Security Checklist

- [ ] `.env` not committed (in `.gitignore`)
- [ ] `JUPYTER_TOKEN` = strong random string
- [ ] `CODE_SERVER_PASSWORD` = strong random string
- [ ] No SSH keys in image or volumes
- [ ] Host repo mounted `:ro` (read-only)
- [ ] MCP Proxy only binds `127.0.0.1`
- [ ] No public port exposure without VPN/SSH tunnel

---

## 11. Updating the Template

If you improve the template and want to update your repo:

```bash
# Fetch template updates
git remote add template git@github.com:YOUR_ORG/jupyter-local-infra.git
git fetch template

# Merge (careful — may conflict with your customizations)
git merge template/main --allow-unrelated-histories
# Resolve conflicts, keep your .env, notebooks, custom configs
```

---

*Update this doc when adding new workflows.*
