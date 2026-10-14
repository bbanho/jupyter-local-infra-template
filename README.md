# Jupyter Local Infra — ML Development Stack Template

> **Local-first JupyterLab + Ollama + MCP Proxy** for ML experimentation, LoRA training, and graph-based workflows.  
> Template repository — clone, customize, and run on your own GPU.

[![Docker](https://img.shields.io/badge/docker-compose-blue?logo=docker)](infra/docker-compose.jupyter.yaml)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2012.4-green?logo=nvidia)](infra/Dockerfile.jupyter)
[![Template](https://img.shields.io/badge/this-a%20template-brightgreen?logo=github)](https://github.com/new?template_name=jupyter-local-infra&template_owner=YOUR_ORG)

---

## 🎯 Purpose

| Problem | Solution |
|---------|----------|
| Colab sessions timeout at 12h, no persistence | **Local Jupyter** — runs indefinitely, survives reboots |
| Colab can't access local services (MCP, Postgres, Qdrant) | **MCP Proxy** — exposes host services to containers |
| GPU memory fragmented across environments | **Single GPU pool** — your GPU runs everything |
| Notebooks lost when Colab recycles VM | **Git-tracked notebooks** — `./notebooks/` versioned |
| Models re-download every session | **Persistent volumes** — `ollama-data`, `cache` survive rebuilds |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COLAB UI (Browser)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ "Conectar ao ambiente de execução local" → http://localhost:8888    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ WebSocket (Jupyter protocol)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE (Host)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  JUPYTER    │  │   OLLAMA    │  │ MCP PROXY   │  │ CODE-SERVER │        │
│  │  (GPU)      │  │  (GPU)      │  │  (Host Net) │  │  (Optional) │        │
│  │  :8888      │  │  :11434     │  │  :7475→:7474│  │  :8080      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┼────────────────┼────────────────┘               │
│                          ▼                ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED VOLUMES (Persisted)                       │   │
│  │  ./notebooks  ./data  ./models  ./cache  YOUR_REPO(ro)  ollama-data │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOST SERVICES (Your Stack)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ MCP SERVER  │  │ POSTGRES    │  │ VECTOR DB   │                         │
│  │ :7474       │  │ :5432       │  │ :6333       │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with driver ≥ 550 (CUDA 12.4)
- **Docker** + **NVIDIA Container Toolkit** (`nvidia-smi` works in containers)
- **Git** + **GitHub CLI** (`gh auth login`)

### 1. Create Your Repo from This Template
```bash
# Option A: GitHub UI (click "Use this template")
# Option B: CLI
gh repo create YOUR_ORG/YOUR_REPO --template YOUR_ORG/jupyter-local-infra --private
git clone git@github.com:YOUR_ORG/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your paths, tokens, model names
```

### 3. Start Stack
```bash
docker compose -f infra/docker-compose.jupyter.yaml up -d --build
```

### 4. Access Services
| Service | URL | Default Credentials |
|---------|-----|---------------------|
| **JupyterLab** | http://localhost:8888 | Token: `change-me` |
| **Ollama API** | http://localhost:11434 | — |
| **MCP Proxy** | http://localhost:7475 | Proxies host `:7474` |
| **VS Code** (optional) | http://localhost:8080 | Password: `change-me` |

### 5. Connect Colab (Optional)
1. Open any notebook in Colab
2. **Conectar → Conectar ao ambiente de execução local**
3. URL: `http://localhost:8888/?token=YOUR_TOKEN`
4. ✅ Colab UI runs on **your GPU**, with **your data**, **your services**

---

## 📁 Repository Structure

```
jupyter-local-infra/
├── infra/                          # Docker & Config
│   ├── docker-compose.jupyter.yaml # Main compose (Jupyter + Ollama + MCP Proxy)
│   ├── Dockerfile.jupyter          # Custom image: PyTorch, PEFT, Flash-Attn, nomic-embed
│   ├── jupyter_notebook_config.py  # Jupyter config (Colab proxy, large outputs)
│   ├── entrypoint.sh               # Container startup (model pull, env verify)
│   └── pull_models.sh              # Pre-pulls embedding & coding models
├── docs/                           # Documentation Templates
│   ├── ARCHITECTURE.md             # Technical deep-dive (customize for your stack)
│   ├── USAGE.md                    # Workflows, troubleshooting
│   └── MOLE_DEVELOPMENT.md         # Mixture of LoRA Experts guide
├── notebooks/                      # Versioned notebooks (git-tracked)
│   ├── 00_setup_validation.ipynb   # Environment verification (run first)
│   └── 02_mole_holographic.ipynb   # MoLE pipeline template (adapt to your data)
├── data/                           # Datasets, embeddings, exports (gitignored large files)
├── models/                         # LoRA adapters, checkpoints (gitignored)
├── cache/                          # pip, uv, HF cache (persisted across rebuilds)
├── scripts/                        # Helper scripts
│   ├── rebuild.sh                  # Rebuild image with cache
│   ├── backup_models.sh            # Export Ollama models
│   ├── sync_notebooks.sh           # Bidirectional Colab ↔ Local sync
│   └── train_shard.py              # Single LoRA shard training
├── .gitignore
├── .env.example                    # Environment variables template
└── README.md
```

---

## 🔧 Key Features

### Pre-installed ML Stack (Dockerfile)
| Category | Packages |
|----------|----------|
| **PyTorch** | `torch==2.4.1` (CUDA 12.4), `torchvision`, `torchaudio` |
| **Transformers** | `transformers==4.44`, `peft==0.12`, `accelerate==0.33` |
| **Quantization** | `bitsandbytes==0.44` (4-bit/8-bit QLoRA) |
| **Flash Attention** | `flash-attn==2.6.3` (built from source for CUDA 12.4) |
| **Embeddings** | `sentence-transformers==3.1`, `faiss-gpu` |
| **Graph** | `networkx==3.3`, `pandas`, `scikit-learn` |
| **Jupyter** | `jupyterlab==4.2`, `ipywidgets`, `jupyter-server-proxy` |
| **MCP** | `mcp==1.2`, `fastapi`, `uvicorn` |

### Persisted Volumes
| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./notebooks` | `/home/jovyan/work/notebooks` | **Git-tracked** notebooks |
| `./data` | `/home/jovyan/work/data` | CSVs, embeddings, exports |
| `./models` | `/home/jovyan/work/models` | LoRA adapters, checkpoints |
| `./cache` | `/cache` | pip, uv, HuggingFace cache |
| `YOUR_REPO` | `/home/jovyan/work/YOUR_REPO:ro` | Read-only source access |
| `ollama-data` | `/root/.ollama` | Ollama model blobs |

---

## 🔗 Integration Pattern (MCP + Ollama + Your Repo)

### MCP Access from Notebooks
```python
# Inside Jupyter notebook — connects to host MCP via proxy
import httpx

async def mcp_query(query: str, top_k: int = 10):
    async with httpx.AsyncClient(base_url="http://mcp-proxy:7474") as client:
        resp = await client.post("/your_endpoint", json={"query": query, "top_k": top_k})
        return resp.json()
```

### Ollama Embeddings (Local, No API Key)
```python
from sentence_transformers import SentenceTransformer

# Uses local Ollama (no internet required after pull)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
embeddings = model.encode(["text to embed"], normalize_embeddings=True)
```

### Your Repo Access (Read-Only Mount)
```python
# Mount your repo at /home/jovyan/work/YOUR_REPO in docker-compose
import sys
sys.path.insert(0, "/home/jovyan/work/YOUR_REPO")

from your_module import your_function
```

---

## 📚 Documentation (Customize for Your Project)

| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical deep-dive — network, volumes, GPU, security |
| [USAGE.md](docs/USAGE.md) | Daily workflows — Colab sync, model mgmt, debugging, backup |
| [MOLE_DEVELOPMENT.md](docs/MOLE_DEVELOPMENT.md) | Mixture of LoRA Experts — sharding, LoRA training, router, aggregation |

---

## 🛠️ Common Operations

### Rebuild Image (after Dockerfile changes)
```bash
./scripts/rebuild.sh
```

### Pull New Ollama Model
```bash
docker exec -it YOUR_REPO-ollama ollama pull qwen2.5-coder:14b
```

### Backup Models
```bash
./scripts/backup_models.sh
```

### Sync Notebooks Colab ↔ Local
```bash
./scripts/sync_notebooks.sh pull   # Colab → Local
./scripts/sync_notebooks.sh push   # Local → Colab (via gist/Drive)
```

### View Logs
```bash
docker compose -f infra/docker-compose.jupyter.yaml logs -f jupyter
docker compose -f infra/docker-compose.jupyter.yaml logs -f ollama
```

### Shell Into Container
```bash
docker exec -it YOUR_REPO-jupyter bash
```

---

## 🔒 Security Notes

- **MCP Proxy** runs on host network (`network_mode: host`) — only exposes `:7475` locally
- **Jupyter Token** — **change in `.env` before exposing**
- **Your Repo** mounted **read-only** — container cannot modify source
- **No SSH keys** in image — use host Git for commits
- **Ollama** binds to `0.0.0.0` inside container network only

> ⚠️ **Never expose ports publicly** without VPN/SSH tunnel. This is a **local development** stack.

---

## 🤝 Customizing for Your Project

1. **Fork/Use Template** → your private repo
2. **Edit `.env`** — paths, tokens, model names, service ports
3. **Update `docker-compose.jupyter.yaml`** — mount your repo, adjust service names
4. **Customize notebooks** — replace data sources, adjust shard count, model names
5. **Update docs** — replace NEXUS-specific references with your architecture
6. **Commit & push** — your team clones and runs identically

---

## 📄 License

MIT License — use freely, customize, distribute.

---

## 🔗 Example Related Repositories (Replace with Yours)

| Repo | Purpose |
|------|---------|
| `YOUR_GRAPH_ENGINE` | Graph database, MCP server, vector search |
| `YOUR_GATEWAY` | LLM Gateway (OpenAI-compatible, key rotation) |
| `YOUR_AGENT_PLATFORM` | Agent platform (CLI, skills, sessions) |

---

**Template maintained by:** [Your Org] — adapt freely for your team.
