# Token Reuse Buffer (TRB)

A semantic caching layer for LLM inference. TRB embeds incoming queries with a sentence-transformer model, searches a FAISS index for semantically similar past queries, and — on a close match — returns the previously generated answer directly, skipping the LLM call entirely.

## How it works

1. Embed the incoming query with `all-MiniLM-L6-v2` (384-dim, L2-normalized for cosine similarity).
2. Search a FAISS `IndexFlatIP` (exact inner-product search) for the closest previously-seen query.
3. **Cache hit** (similarity ≥ threshold θ): return the stored response instantly — no LLM call, no tokens generated.
4. **Cache miss**: send the query to the LLM (`Qwen2.5-7B-Instruct`) normally, then embed and store the query, response, and topic label for future lookups.

```
Query --> Embed --> FAISS search --> similarity >= θ ?
                                        |            |
                                       Yes           No
                                        |            |
                                 Return cached   Full LLM call
                                  response,      + embed & cache
                                 skip LLM call    the new query
```

Each stored query also carries a ground-truth topic label (`group_id`), used only for evaluation — it lets the experiment check whether a cache hit actually matched a query on the *same* topic (precision) and how many true duplicate topics were successfully caught (recall).

## Results

Tested on a 55-query set spanning 5 technical domains, plus a 20-query general-knowledge negative control set to check for false hits on unrelated topics:

- **Neural Networks** — basics, backpropagation, overfitting, CNNs
- **Transformers** — architecture, self-attention, positional encoding, beam search
- **Python** — list comprehensions, decorators, the GIL, generators
- **SQL / Databases** — primary keys, JOINs, normalization, indexes
- **Web Development** — REST APIs, CORS, GET vs POST, WebSockets
- **General knowledge (negative control)** — 20 unrelated trivia questions that should never cache-hit

| Metric | Value |
|---|---|
| Total queries | 55 |
| Cache hits | 13 (23.6%) |
| Cache misses | 42 |
| Similarity threshold (θ) | 0.65 |
| Precision (hits that matched the correct topic) | 92.3% |
| Recall (true duplicate topics successfully caught) | 85.7% |
| False-positive hits | 1 |
| Total tokens saved (skipped generations) | 1,950 |
| Avg latency — cache hit | ~0.00s (index lookup only) |
| Avg latency — cache miss | 9.67s |

Every general-knowledge query correctly missed — no false hits across domains, confirming the embedding space cleanly separates unrelated topics at θ=0.65.

### Threshold sensitivity

Precision and hit count as θ is varied (re-evaluated against the same stored similarity scores, no new LLM calls):

| θ | Hits | Precision |
|---|---|---|
| 0.50 | 13 | 92.3% |
| 0.55 | 13 | 92.3% |
| 0.60 | 13 | 92.3% |
| 0.65 | 13 | 92.3% |
| 0.70 | 11 | 100% |
| 0.75 | 10 | 100% |
| 0.80 | 10 | 100% |

θ=0.65 was chosen as the operating point: it captures all 13 hits found at looser thresholds while keeping precision above 92%. Tightening past 0.70 trades a few hits away in exchange for eliminating the one false positive.

Charts:
- `trb_results.png` — cache hits vs. misses, similarity score distribution, precision/recall, latency comparison
- `trb_threshold_sensitivity.png` — precision vs. similarity threshold

## Tech stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector search**: FAISS (`IndexFlatIP`)
- **LLM**: `Qwen/Qwen2.5-7B-Instruct` (Hugging Face Transformers, fp16)
- **Environment**: Kaggle GPU notebook

## Usage

```python
pip install faiss-cpu sentence-transformers transformers torch
```

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

trb = TokenReuseBuffer(similarity_threshold=0.65)

trb.query("Explain what a neural network is", group_id="nn_basic")
trb.query("What is a neural network in simple terms", group_id="nn_basic")  # cache hit

# Each call returns: cache_hit, similarity_score, is_correct_hit,
# missed_true_duplicate, tokens_saved, latency_sec
```

The original experiment, model loading, and evaluation plots are in `notebook/TokenReuseBuffer.ipynb`.

## Limitations / next steps

- Evaluation uses isolated Q&A pairs with known ground-truth topic labels rather than live, multi-turn conversational traffic — real usage would need to handle context-dependent phrasing (e.g., follow-up questions that only make sense given prior turns).
- Uses exact FAISS search (`IndexFlatIP`); for larger caches an approximate index (e.g. HNSW/IVF) would scale better.
- Threshold is static; an adaptive or query-type-aware threshold could improve the precision/recall tradeoff further, especially to close the gap between θ=0.65 (13 hits, 1 false positive) and θ=0.70 (11 hits, 0 false positives).
- The one false-positive hit suggests some semantically-close-but-distinct queries (e.g. paraphrased questions on adjacent subtopics) can still cross the similarity threshold; a secondary verification step could catch these.
