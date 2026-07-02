# Token Reuse Buffer (TRB)

A semantic caching layer for LLM inference. TRB embeds incoming queries with a
sentence-transformer model, searches a FAISS index for semantically similar
past queries, and reuses prior context when a close match is found -- cutting
redundant tokens sent to the LLM on repeated or similar questions.

## How it works

1. **Embed** the incoming query with `all-MiniLM-L6-v2` (384-dim, normalized for cosine similarity).
2. **Search** a FAISS `IndexFlatIP` (exact inner-product search) for the closest previously-seen query.
3. **Cache hit** (similarity ≥ threshold): reuse the matched query/response as compressed context instead of resending the full prior conversation.
4. **Cache miss**: send the query to the LLM (`Qwen2.5-7B-Instruct`) normally and store it in the index for future reuse.

```
Query --> Embed --> FAISS search --> similarity >= θ ?
                                        |                 |
                                       Yes                No
                                        |                 |
                                 Reuse context      Full LLM call
                                 (compressed)        + cache the query
```

## Results

Tested on a 10-query set across two domains (ML concepts, transformer architecture):

| Metric                  | Value           |
|--------------------------|-----------------|
| Total queries             | 10              |
| Cache hits                | 3 (30%)         |
| Similarity threshold (θ)  | 0.65            |
| Total tokens before TRB   | 550             |
| Total tokens after TRB    | 112             |
| Total tokens saved        | 438             |
| Token reduction (overall) | 79.6%           |
| Avg latency (cache hit)   | 9.84s           |
| Avg latency (cache miss)  | 9.96s           |

### Threshold sensitivity

Hit rate as the similarity threshold θ is varied (no new LLM calls -- reused stored embeddings):

| θ    | Hits | Misses | Hit Rate |
|------|------|--------|----------|
| 0.50 | 6    | 4      | 60%      |
| 0.55 | 5    | 5      | 50%      |
| 0.60 | 3    | 7      | 30%      |
| 0.65 | 3    | 7      | 30%      |
| 0.70 | 2    | 8      | 20%      |
| 0.75 | 0    | 10     | 0%       |
| 0.80 | 0    | 10     | 0%       |

Charts (from `results/`):

- `trb_analysis.png` -- token usage, per-query savings, hit/miss ratio, similarity scores
- `trb_overall.png` -- total token efficiency gain
- `trb_sensitivity.png` -- hit rate vs. similarity threshold

## Tech stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector search**: FAISS (`IndexFlatIP`)
- **LLM**: `Qwen/Qwen2.5-7B-Instruct` (Hugging Face Transformers)
- **Environment**: Kaggle GPU notebook

## Usage

```bash
pip install -r requirements.txt
```

```python
from trb import TokenReuseBuffer, load_models

embedder, tokenizer, model = load_models()
trb = TokenReuseBuffer(embedder, tokenizer, model, similarity_threshold=0.65)

response = trb.query("Explain what a neural network is")
response = trb.query("What is backpropagation in neural networks")  # likely a cache hit

print(trb.stats)  # per-query stats: cache_hit, similarity_score, tokens saved, latency
```

The original experiment and plots are in `notebook/token_reuse_buffer.ipynb`.

## Limitations / next steps

- Evaluated on a small 10-query, 2-domain test set -- a larger, more diverse query set (50-100+) would give more reliable hit-rate and savings estimates.
- Uses exact FAISS search (`IndexFlatIP`); for larger caches an approximate index (e.g. HNSW/IVF) would scale better.
- Threshold is static; an adaptive or query-type-aware threshold could improve the precision/recall tradeoff.
