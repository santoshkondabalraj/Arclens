# Arclens — System Design

This document explains the architectural decisions behind the Arclens RAG pipeline: what was chosen, what was tried and discarded, and why each decision produces better results for the specific problem of answering questions about YouTube playlist transcripts.

---

## Table of Contents

1. [Problem framing](#1-problem-framing)
2. [Retrieval design](#2-retrieval-design)
3. [Cross-encoder reranking and threshold calibration](#3-cross-encoder-reranking-and-threshold-calibration)
4. [Generation and citation pipeline](#4-generation-and-citation-pipeline)
5. [Ingestion pipeline order](#5-ingestion-pipeline-order)
6. [Chunk type design](#6-chunk-type-design)
7. [Namespace design](#7-namespace-design)
8. [Why query rewriting was removed](#8-why-query-rewriting-was-removed)
9. [Evaluation architecture](#9-evaluation-architecture)
10. [Key configuration values and their rationale](#10-key-configuration-values-and-their-rationale)

---

## 1. Problem framing

The core challenge is **vocabulary gap**: users ask questions using natural language that often does not match the vocabulary in the source transcripts. A video about Ray Dalio's investing principles uses phrases like "rules of thumb", "have-nots", and "printing money". A user asks about "lessons", "inequality", and "issuing currency". The pipeline must bridge these vocabulary gaps reliably, across any playlist in any domain.

Secondary challenge: **grounded refusal**. When a question is genuinely outside the playlist's content, the system must refuse cleanly — not hallucinate an answer, not fail silently, and not over-refuse legitimate questions that happen to use different vocabulary than the source.

Everything in the design follows from these two requirements.

---

## 2. Retrieval design

### 2.1 Dense retrieval: Gemini `text-embedding-004`

The pipeline uses Gemini's `text-embedding-004` model for both ingestion and query embedding. This is the most important single design choice for vocabulary-gap handling.

A dense embedding model trained on large corpora learns that "lessons" and "rules of thumb" are semantically related — they map to nearby points in the embedding space. This means a query about "lessons" will retrieve a chunk about "rules of thumb" with high cosine similarity, even though neither word appears in the other's text.

**Why the same model for ingestion and query:** The embedding space only makes sense if both the stored vectors and the query vector use the same model. Using different models (or different versions of the same model) creates a distribution mismatch: the dot products that the vector store computes would measure similarity in two different spaces, producing meaningless scores. Gemini `text-embedding-004` is used consistently for both.

**Why not a larger embedding model:** `text-embedding-004` achieves strong semantic coverage for the vocabulary ranges found in YouTube transcripts (conversational speech, not highly technical jargon). Larger models would increase ingestion time, query latency, and cost without meaningful accuracy gains on this data distribution.

### 2.2 Sparse retrieval: BM25

BM25 complements dense retrieval in cases where exact-term matching matters more than semantic similarity:

- Proper nouns (names, places, organisations) that the embedding model may not assign strong semantic proximity
- Numbers and dates that appear literally in user queries
- Rare domain-specific terms that dense models underweight

BM25 is built from the same body chunks at ingestion time and persisted to disk (`data/bm25_indexes/`). At query time it runs locally with zero API calls.

### 2.3 Hybrid fusion: Reciprocal Rank Fusion (RRF)

Dense retrieval produces a ranked list. BM25 produces a ranked list. Neither ranking is calibrated on the same scale, so scores cannot be directly compared or averaged.

RRF fuses rankings by position, not score:

```
rrf_score(d) = sum over retriever r of  1 / (k + rank(d, r))
```

Where `k=60` is a constant that moderates the influence of top-ranked documents. A document ranked 1st by both retrievers gets a high combined score; a document ranked 1st by one and not retrieved by the other gets a moderate score. Documents not retrieved by either are excluded.

RRF was chosen over score-normalised fusion because it makes no assumptions about the score distributions of the two retrievers. BM25 scores are unbounded integers; dense cosine similarities are bounded floats — there is no meaningful way to combine them on a single scale.

### 2.4 Candidate pool size

Both retrievers return the top 20 candidates (`DENSE_TOP_K=20`, `SPARSE_TOP_K=20`). After RRF, the merged pool passes to the cross-encoder reranker.

The pool is intentionally larger than what the LLM will see. The cross-encoder is more accurate than either retriever but too slow to run on hundreds of candidates. The two-stage design (fast retriever → accurate reranker) is standard in modern RAG systems.

---

## 3. Cross-encoder reranking and threshold calibration

### 3.1 Why cross-encoder reranking

Dense bi-encoders embed the query and document independently and score them by dot product. This is efficient at scale but the representation is lossy — the model cannot attend to query–document interaction during encoding.

A cross-encoder takes the query and document concatenated as a single input and produces a joint relevance score. It attends to the full context of both simultaneously, producing much more accurate relevance scores at the cost of higher latency.

The ms-marco-MiniLM-L-6-v2 cross-encoder was trained on the MS MARCO passage ranking dataset, which contains web search query–passage pairs. It generalises well to the YouTube transcript domain because transcripts use conversational language similar to the questions in MARCO.

### 3.2 Raw logits, not probabilities

The ms-marco cross-encoder outputs **raw logits**, not calibrated probabilities. This is a critical implementation detail.

| Score range | Meaning |
|---|---|
| `> 3` | Direct answer in chunk |
| `0 – 3` | Topically related |
| `-5 – 0` | Same domain, indirect match |
| `-10 – -5` | Semantic synonym / paraphrase match |
| `< -13` | Off-topic |

Early versions of the pipeline set `MIN_RERANK_SCORE=0.3`, treating the output as a probability. This silently filtered out most valid results. The threshold was progressively calibrated against real queries across several orders of magnitude (0.3 → -2.0 → -8.0 → -10.0).

### 3.3 Why `MIN_RERANK_SCORE=-10.0`

The -10.0 threshold was chosen empirically to cover the full range of semantic synonym matches while still correctly refusing off-topic queries.

The critical test case: `"What lessons should I learn?"` against a chunk containing `"rules of thumb"`. Neither "lessons" nor "learn" appears in the chunk; the cross-encoder must score this pair based on semantic inference. The observed score is approximately -9.93 — just inside the -10.0 cutoff.

Off-topic queries (e.g. "What is the capital of France?" against an economics corpus) score below -13. There is a ~3-point gap between the lowest valid score (-10) and the highest off-topic score (-13), which is the working operating margin of the refusal gate.

### 3.4 `RERANK_TOP_N=10`

After reranking, only the top N chunks by cross-encoder score are passed to the threshold check and LLM. The initial value was 5. For semantic synonym queries, the relevant chunk tends to rank lower than same-domain chunks because of vocabulary mismatch — it might rank 8th or 10th. With `RERANK_TOP_N=5`, it was silently excluded before reaching the threshold check. Raising to 10 resolved this without meaningfully increasing LLM prompt size.

### 3.5 Meta-phrase stripping

The cross-encoder was trained on clean query–passage pairs. User queries often include meta-instructions like `"according to the video"`, `"in this podcast"`, `"based on the content"`. These phrases penalise the cross-encoder score because they never appear in transcript text — the model reads them as a vocabulary mismatch.

`_META_PHRASE_RE` strips these phrases from the query before building cross-encoder pairs. The same stripped query is used for dense retrieval — meta-phrases add no semantic signal to the embedding either, since they never appear in the corpus.

---

## 4. Generation and citation pipeline

### 4.1 Gemini 2.5 Flash with thinking disabled

Gemini 2.5 Flash is the generation model. RAG generation is document-grounded Q&A: the answer is constrained to the retrieved chunks. Extended chain-of-thought reasoning (thinking mode) adds no quality benefit for this task and has two significant costs:

1. **Token budget consumption**: Thinking tokens are allocated from the same `max_output_tokens` budget as the visible output. With thinking enabled, a 2048-token budget left only ~400 tokens for the visible answer — approximately 380 characters — causing answers to be truncated mid-citation.

2. **Non-determinism**: Thinking mode makes Gemini 2.5 Flash non-deterministic at `temperature=0`. The same query with the same context produced different refusal decisions across runs.

`thinking_budget=0` disables thinking completely. `max_output_tokens=8192` provides sufficient budget for long answers with multiple citations.

### 4.2 Grounded generation only

The LLM prompt instructs the model to answer only from the provided chunks. The threshold_check node is the sole refusal gate — if no chunk clears the score threshold, the system refuses before the LLM is called. The LLM is never asked to evaluate whether it has enough information; it always has the filtered, relevant chunks when it runs.

Early designs asked the LLM to also decide whether context was "sufficient" and emit a JSON refusal if not. This created compounding problems: the LLM was inconsistent about when context was "sufficient", and raw JSON bleed-through would render in the UI. The cleaner design separates the concerns: threshold check handles refusal, LLM handles generation.

### 4.3 Citation extraction and deduplication

The LLM outputs inline citation markers in the format `[Source: "Title", Channel, ~Xs]`. The `format_citations` node parses these markers and resolves each to the corresponding retrieved chunk's `video_id` and `timestamp_seconds`.

Deduplication: adjacent overlapping chunks from the same video segment often produce identical `(video_id, timestamp_seconds)` pairs in the citation list. `format_citations` deduplicates on this key, keeping the first occurrence. This prevents the UI from showing four identical timestamp cards for the same moment in the video.

---

## 5. Ingestion pipeline order

**Chunk before clean** — not clean before chunk.

Transcripts are delivered as raw text with embedded timestamp sentinels (`[T:Xs]`). The sentinels mark the approximate start time of each spoken segment.

If the pipeline cleans first (strips sentinels), then chunks, the chunker produces segments with no temporal anchoring. When citations are generated, there is no timestamp to surface — the pipeline falls back to character offsets in the raw string (e.g. character 13310 in a transcript file is not 13310 seconds into the video).

The correct order:
1. **Chunk the raw transcript** (sentinels intact)
2. **Clean each chunk**: extract the first sentinel into `metadata["timestamp_seconds"]`, strip remaining sentinels and noise from the text

Each chunk's `timestamp_seconds` is the time of the first spoken word in that chunk. Citations in the UI link to this timestamp, dropping users into the video at precisely the right moment.

---

## 6. Chunk type design

Each ingested video produces two types of chunks, distinguished by `chunk_type` metadata:

| Type | Content | Size | Purpose |
|---|---|---|---|
| `body` | Transcript text, cleaned | 600 tokens, 150 overlap | Retrieval and generation |
| `metadata` | Title, channel, date, description | 192 tokens | Exact-match lookups on video identity |

Both types are indexed in Pinecone but only `body` chunks are used for retrieval during generation. The separation exists because metadata chunks (which repeat the same title and channel on every video) would pollute the retrieval results for content queries.

Probe and golden dataset generators filter to `body` chunks explicitly — metadata chunks are too shallow to generate high-value semantic probes from.

---

## 7. Namespace design

Pinecone uses namespaces to partition vector spaces within a single index. Each `(user_id, playlist_id)` pair gets its own namespace: `{user_id}_{playlist_id}`.

This design allows:
- Multiple users to ingest the same playlist independently without interference
- Per-user query routing with zero cross-contamination
- Targeted deletion (re-ingestion clears the namespace for that user×playlist pair only, leaving other users' data intact)

BM25 indexes are also partitioned: `data/bm25_indexes/{user_id}_{playlist_id}.pkl`.

---

## 8. Why query rewriting was removed

An early version of the pipeline included a `rewrite_query` node that extracted keywords from the user's question before retrieval. The intent was to reduce noise for retrieval models. The effect was the opposite.

**Why keyword extraction hurt retrieval:**

Dense embedding models are trained on full natural-language sentences. The embedding for `"What lessons should I learn?"` is a rich semantic vector that maps the concept "lessons from experience" close to "rules of thumb" in the shared embedding space. Extracting keywords produces `"lessons learn"` — a sparse bag that no longer carries the relational context. The embedding for `"lessons learn"` is a much weaker semantic signal.

**Why keyword expansion was rejected:**

A proposed fix was to expand synonyms in the extracted query (e.g. `"lessons" → "lessons rules takeaway key"`). This would work for the Ray Dalio economics corpus but would fail for any other domain. A sports playlist has different vocabulary; a cooking playlist has different vocabulary. Hardcoding synonym expansions is not playlist-agnostic.

**The correct solution:**

The embedding model handles synonyms natively. The fix was to stop interfering with it. Removing `rewrite_query` and passing the original question to all retrieval stages improved synonym recall across the board without any domain-specific configuration.

---

## 9. Evaluation architecture

### 9.1 Why structural scope only

Content-correctness evaluation (checking that answers contain specific expected information) requires a gold-standard expected answer per question. Gold answers are expensive to write, rapidly go stale as playlists change, and only work for one specific playlist.

Structural correctness evaluation checks **how** the pipeline behaves, not **what** it says:

| Metric | What it checks |
|---|---|
| `refusal_accuracy` | System refused iff the question is off-topic |
| `citation_presence` | Every non-refusal answer has at least one citation |
| `citation_precision` | Citations reference retrieved documents, not hallucinated sources |
| `faithfulness` | Answer text is supported by the retrieved context |

These metrics work for any playlist and any domain without any knowledge of the content.

### 9.2 Two-tier architecture

```
Tier 1: Retrieval regression              Tier 2: Structural smoke test
─────────────────────────────             ──────────────────────────────
Deliberate vocabulary-gap probes          Auto-generated from same chunks
Generated by Claude Opus                  Generated by Gemini Flash
Per-playlist                              Per-playlist
Requires ANTHROPIC_API_KEY                Uses existing Gemini credentials
Run manually; regression gate             Run after ingestion; health check
Fails if synonym retrieval regresses      Cannot detect synonym regression
tests/golden_qa_{playlist_id}.json        data/eval_probes/{playlist_id}.json
```

The key distinction: Tier 2 probes are **circular**. A probe generated from a chunk asking "What does the speaker say about X?" retrieves the same chunk by its own vocabulary — this tests that the pipeline is running, not that it handles vocabulary gaps. Tier 1 probes are deliberately non-circular: the question vocabulary must not appear in the source chunk.

### 9.3 Why Claude Opus for golden dataset generation

Two properties are required for high-value vocabulary-gap probes:

1. **Semantic understanding deep enough to rephrase** — the model must understand the concept in the chunk well enough to express it in genuinely different vocabulary, not just word-swap synonyms.
2. **Independence from the production pipeline** — if the same model generates the tests and answers them, the test set reflects the model's biases, not the retrieval system's properties.

Claude Opus (claude-opus-4-8) satisfies both. It is a different provider (Anthropic) from the production pipeline (Google/Gemini), ensuring test set independence. It is capable enough to write questions that use "guidelines" to describe what the source calls "rules of thumb", or "issuing new currency" to describe what the source calls "printing money".

The prompt explicitly instructs Claude Opus to avoid vocabulary that appears in the source chunks for `semantic_synonym` examples, and requires a `vocabulary_note` documenting the specific vocabulary bridge for each synonym example.

### 9.4 LangSmith integration

Both eval tiers use `langsmith.evaluate()` and upload results under distinct experiment prefixes:
- `regression-{playlist_id}` — Tier 1 runs
- `smoke-{playlist_id}` — Tier 2 runs

The distinct prefixes prevent confusion in the LangSmith dashboard: a green smoke test does not signal retrieval quality.

---

## 10. Key configuration values and their rationale

| Variable | Value | Rationale |
|---|---|---|
| `MIN_RERANK_SCORE` | `-10.0` | Covers semantic synonym matches (-10 to -5 range); refuses off-topic queries which score < -13 |
| `RERANK_TOP_N` | `10` | Synonym-relevant chunks can rank 8–10 due to vocabulary mismatch; top-5 silently excluded them |
| `DENSE_TOP_K` | `20` | Provides sufficient recall for diverse multi-topic questions without excessive cross-encoder calls |
| `SPARSE_TOP_K` | `20` | Matches dense pool to allow RRF to operate over comparable candidate sets |
| `BODY_CHUNK_SIZE` | `600` | Large enough for a complete spoken thought (≈ 60–90 seconds of speech); small enough to stay topically tight |
| `BODY_CHUNK_OVERLAP` | `150` | 25% overlap ensures a concept spanning two chunks is fully captured by at least one |
| `thinking_budget` | `0` | Disabled for both generation LLM and eval judge; prevents token budget consumption and non-determinism |
| `STALENESS_DAYS` | `30` | Transcripts older than 30 days may have had content added or removed; warn the user |
