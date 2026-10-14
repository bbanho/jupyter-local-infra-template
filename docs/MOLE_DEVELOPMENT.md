# Holographic MoLE Development Guide

> **Mixture of LoRA Experts** with holographic context routing for graph-based context fragmentation.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUPERAGENT CONTEXT (128k-1M tokens)                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            ┌───────────────┐                   ┌───────────────┐
            │  SEMANTIC     │                   │  SEMANTIC     │
            │  FRAGMENTER   │                   │  FRAGMENTER   │
            │  (BGE clust)  │                   │  (BGE clust)  │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
         ┌──────────┼──────────┐               ┌────────┼────────┐
         ▼          ▼          ▼               ▼        ▼        ▼
    ┌────────┐ ┌────────┐ ┌────────┐       ┌────────┐ ┌────────┐
    │ Shard 0│ │ Shard 1│ │ Shard 2│  ...  │ Shard N│ │ Shard M│
    │ (LoRA) │ │ (LoRA) │ │ (LoRA) │       │ (LoRA) │ │ (LoRA) │
    └────┬───┘ └────┬───┘ └────┬───┘       └────┬───┘ └────┬───┘
         │          │          │                │          │
         └──────────┼──────────┼────────────────┼──────────┘
                    ▼          ▼                ▼
         ┌─────────────────────────────────────────────────┐
         │         HOLOGRAPHIC ROUTER (Learned)            │
         │  query → embed → top-k shard_keys (cosine sim)  │
         └─────────────────────┬───────────────────────────┘
                               ▼
         ┌─────────────────────────────────────────────────┐
         │        AGGREGATOR (Cross-Attention / Gated)     │
         │  weighted_sum(shard_outputs) → cross_attend(q)  │
         └─────────────────────────────────────────────────┘
```

---

## 2. Implementation Phases

### Phase 1: Data Preparation (Notebook: `02_mole_holographic.ipynb`)

```python
# 1. Export your graph/code chunks via MCP
chunks = await mcp_query("your query", top_k=10000)

# 2. Embed with nomic-embed-text (768d)
embeddings = nomic_encode([c["text"] for c in chunks])

# 3. Semantic clustering → shards
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
shard_labels = kmeans.fit_predict(embeddings)

# 4. Save shards
for shard_id in range(8):
    shard_chunks = [c for i, c in enumerate(chunks) if shard_labels[i] == shard_id]
    save_jsonl(f"data/shards/shard_{shard_id}.jsonl", shard_chunks)
```

### Phase 2: LoRA Training per Shard (Parallel)

```bash
# scripts/train_shard.py --shard-id 0 --model Qwen/Qwen2.5-Coder-3B
# Runs on local Jupyter (your GPU, no timeout)
```

**Training Config (per shard):**
```python
LoraConfig(
    r=16,                    # Rank (capacity)
    lora_alpha=32,           # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

TrainingArguments(
    output_dir=f"models/lora-shard-{shard_id}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    max_steps=500,           # Per shard
    warmup_steps=50,
    fp16=True,               # or bf16 on Ampere+
    logging_steps=10,
    save_steps=100,
)
```

### Phase 3: Holographic Keys Extraction

```python
# After training each LoRA, extract "holographic key"
def extract_holographic_key(model, shard_chunks, tokenizer):
    """Mean-pooled last hidden state of shard's representative samples."""
    model.eval()
    keys = []
    for chunk in shard_chunks[:32]:  # Sample 32 chunks
        inputs = tokenizer(chunk["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Last layer, mean pool over sequence
            key = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            keys.append(key)
    return np.mean(keys, axis=0)  # [768] — the shard's "holographic fingerprint"

# Save keys
shard_keys = np.array([extract_holographic_key(...) for shard_id in range(8)])
np.save("models/shard_keys.npy", shard_keys)  # [8, 768]
```

### Phase 4: Router Training

```python
class HolographicRouter(nn.Module):
    def __init__(self, n_shards=8, dim=256):
        super().__init__()
        self.shard_keys = nn.Parameter(torch.from_numpy(shard_keys).float())  # [8, 768]
        self.query_proj = nn.Linear(768, 768)  # Project query to key space
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, query_embedding, top_k=3):
        # query_embedding: [768] or [B, 768]
        q = self.query_proj(query_embedding)  # [768]
        scores = (q @ self.shard_keys.T) / self.temperature  # [8]
        topk_scores, topk_indices = scores.topk(top_k)
        return topk_indices, torch.softmax(topk_scores, dim=-1)

# Train end-to-end with reconstruction loss
def train_router(router, shards_data, epochs=10):
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for query, target_chunks in shards_data:
            q_emb = nomic_encode([query])[0]
            indices, weights = router(q_emb)
            # Load top-k LoRAs, forward, aggregate, compute loss vs target
            loss = reconstruction_loss(aggregate(shards[indices], weights), target_chunks)
            loss.backward()
            optimizer.step()
```

### Phase 5: Aggregator (Synthesis)

```python
class HolographicAggregator(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query, shard_outputs, weights):
        # query: [1, seq_len, 768]
        # shard_outputs: [n_active, seq_len, 768]
        # weights: [n_active]
        
        # Weighted combination
        weighted = sum(w * out for w, out in zip(weights, shard_outputs))  # [seq_len, 768]
        weighted = weighted.unsqueeze(0)  # [1, seq_len, 768]
        
        # Cross-attention: query attends to weighted shard knowledge
        attended, _ = self.cross_attn(query, weighted, weighted)
        
        # Gated fusion
        fused = self.gate(torch.cat([query, attended], dim=-1))
        return self.norm(fused + query)  # Residual
```

---

## 3. Notebook Structure (`notebooks/02_mole_holographic.ipynb`)

| Cell | Purpose |
|------|---------|
| 1 | Imports + config (paths, model names, hyperparams) |
| 2 | Load shard data + shard_keys.npy |
| 3 | Define `HolographicRouter`, `HolographicAggregator` |
| 4 | Load base model (Qwen 3B) + PEFT adapters (dynamic loading) |
| 5 | Router training loop (reconstruction + contrastive loss) |
| 6 | Aggregator training (cross-attention) |
| 7 | End-to-end evaluation: latency, recall@k, quality vs baseline |
| 8 | Ablation: router only / aggregator only / full pipeline |
| 9 | Export production artifacts (router.pt, aggregator.pt, shard_keys.npy) |

---

## 4. Dynamic LoRA Loading (Production Pattern)

```python
class MoLEInference:
    def __init__(self, base_model_path, shards_dir, shard_keys_path):
        self.base = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.shard_keys = np.load(shard_keys_path)
        self.shard_dirs = sorted(glob(f"{shards_dir}/lora-shard-*"))
        self.router = HolographicRouter(n_shards=len(self.shard_dirs))
        self.router.load_state_dict(torch.load("models/router.pt"))
        self.aggregator = HolographicAggregator()
        self.aggregator.load_state_dict(torch.load("models/aggregator.pt"))
        self.loaded_adapters = {}  # Cache

    def get_adapter(self, shard_id):
        if shard_id not in self.loaded_adapters:
            adapter_path = self.shard_dirs[shard_id]
            self.base.load_adapter(adapter_path, adapter_name=f"shard_{shard_id}")
            self.loaded_adapters[shard_id] = f"shard_{shard_id}"
        return self.loaded_adapters[shard_id]

    def __call__(self, query, max_new_tokens=256):
        # 1. Embed query
        q_emb = nomic_encode([query])[0]
        
        # 2. Route
        indices, weights = self.router(torch.from_numpy(q_emb).float())
        
        # 3. Load & forward top-k adapters
        outputs = []
        for idx, w in zip(indices, weights):
            adapter_name = self.get_adapter(idx.item())
            self.base.set_adapter(adapter_name)
            with torch.no_grad():
                out = self.base.generate(
                    **self.tokenizer(query, return_tensors="pt").to(self.base.device),
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                outputs.append((w.item(), out.hidden_states[-1][:, -1, :]))  # Last token hidden
        
        # 4. Aggregate
        query_hidden = self.base(**self.tokenizer(query, return_tensors="pt").to(self.base.device)).hidden_states[-1]
        aggregated = self.aggregator(query_hidden, [o[1] for o in outputs], [o[0] for o in outputs])
        
        # 5. Generate from aggregated representation
        return self.tokenizer.decode(aggregated.argmax(dim=-1)[0], skip_special_tokens=True)
```

---

## 5. Benchmarking Protocol

| Metric | Baseline (Monolithic 3B) | MoLE (8 shards) | Target |
|--------|-------------------------|-----------------|--------|
| **Latency (tokens/s)** | ~45 | ~35 (router+agg overhead) | >30 |
| **VRAM Peak** | 6.2 GB | 4.8 GB (1 adapter at a time) | <8 GB |
| **Recall@10** | 0.72 | 0.78 | >0.75 |
| **Context Reconstruction F1** | 0.68 | 0.81 | >0.80 |
| **Drift Gate ΔV** | 0.49 | 0.28 | <0.35 |

### Evaluation Script
```python
# scripts/evaluate_mole.py
def evaluate(mole, test_queries, ground_truth):
    results = {"latency": [], "recall": [], "drift": [], "quality": []}
    for query, gt in zip(test_queries, ground_truth):
        start = time.time()
        response = mole(query)
        results["latency"].append(time.time() - start)
        
        # Recall: does response contain expected entities?
        results["recall"].append(compute_recall(response, gt))
        
        # Drift: ΔV between query embedding and response embedding
        results["drift"].append(delta_V(nomic_encode([query])[0], nomic_encode([response])[0]))
        
        # Quality: human eval or LLM-as-judge
        results["quality"].append(llm_judge(query, response, gt))
    
    return {k: np.mean(v) for k, v in results.items()}
```

---

## 6. Production Deployment (Export Artifacts)

### Artifacts to Export
```
models/
├── router.pt              # TorchScript: torch.jit.trace(router, (dummy_q,))
├── aggregator.pt          # TorchScript
├── shard_keys.npy         # [n_shards, 768] float32
├── lora-shard-00/         # adapter_config.json + adapter_model.safetensors
├── lora-shard-01/
└── ...
```

### Integration Pattern (Your Language)
```python
# Python example
class YourMoLEInference:
    def __init__(self, artifacts_dir):
        self.shard_keys = np.load(f"{artifacts_dir}/shard_keys.npy")
        self.router = torch.jit.load(f"{artifacts_dir}/router.pt")
        self.aggregator = torch.jit.load(f"{artifacts_dir}/aggregator.pt")
        # Load adapters dynamically...
    
    def route(self, query_emb, top_k=3):
        scores = (query_emb @ self.shard_keys.T)
        return scores.topk(top_k)
```

---

## 7. Checklist for Implementation

- [ ] Notebook `02_mole_holographic.ipynb` complete with all phases
- [ ] Shard data exported (`data/shards/shard_*.jsonl`)
- [ ] N LoRAs trained + saved (`models/lora-shard-*/`)
- [ ] Router + Aggregator trained + exported (`models/router.pt`, `aggregator.pt`)
- [ ] Benchmarks passing (latency, recall, drift gate)
- [ ] Production artifacts exported
- [ ] Integration in your codebase
- [ ] Documentation updated

---

## 8. References

| Topic | Source |
|-------|--------|
| **LoRA MoE** | `LoraMoE` (ICLR 2024), `MoLE` (arXiv 2024) |
| **Holographic Embeddings** | Nickel et al., *Holographic Embeddings of Knowledge Graphs* |
| **Switch Transformers** | Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models* |
| **Hybrid Search Fusion** | Qdrant docs: sparse+dense fusion, RRF/DBSF |

---

*This guide evolves with the notebook. Update after each phase completion.*
