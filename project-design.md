# Project Design: YouTube Playlist RAG Chat

## One-Liner

> My RAG app helps listeners answer questions from their personally curated YouTube playlist transcripts via a chat interface with 90% faithfulness and answers in under 8 seconds.

---

## Design Framework

### Use Case

Podcast listeners ask natural-language questions from their personally curated YouTube playlist transcripts via a chat interface. Answers are grounded, cited, and returned in under 8 seconds.

---

### Corpus

- **Source:** User-uploaded YouTube playlists
- **Size:** ~30,000–120,000 words per user corpus (5–20 hours of spoken content)
- **Content type:** Auto-generated and creator-published transcript text in English
- **Source of truth:** YouTube transcript API per playlist at ingestion time

---

### Ingestion & Cleaning

- Transcripts pulled via `youtube-transcript-api` per playlist video
- Cleaning pipeline:
  - Strips timestamp markers
  - Removes filler tokens
  - Deduplicates repeated auto-caption artifacts
  - Normalizes speaker-turn boundaries before chunking

---

### Ingestion & Freshness

- **Trigger:** User-triggered on playlist upload or manual refresh
- **Live sync:** None for MVP
- **Staleness warning:** Surfaced if playlist last ingested > 30 days ago

---

### Chunking & Embedding

| Layer | Chunk Size | Overlap | Purpose |
|---|---|---|---|
| Transcript body | 512–768 tokens | 150 tokens | Semantic content |
| Metadata / titles / intros | 128–256 tokens | — | Structural context |

- **Embedding model:** Gemini 2 (2048-token capacity)
- **Sparse index:** BM25 maintained in parallel

---

### Retrieval

- **Index:** Pinecone hybrid index
- **Dense retrieval:** Gemini 2 embeddings, cosine similarity
- **Sparse retrieval:** BM25 (in parallel)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-encoder rerank to top-5
- **Metadata filters:** Scoped to playlist / channel / date
- **Refusal condition:** Triggered if no chunk clears minimum reranker score threshold

---

### Generation

- **Model:** Gemini 2.0 Flash
- **Grounding:** Strict prompt — answer only from retrieved chunks
- **Citations:** Inline (episode title, channel, timestamp)
- **Refusal condition:** Structured refusal returned if fewer than 2 chunks clear confidence threshold

---

### Evaluation

| Metric | Target |
|---|---|
| Faithfulness (LLM-as-a-judge) | ≥ 90% |
| Answer relevance | ≥ 85% |
| Citation precision | Tracked per answer |

- **Golden test set:** 50–100 Q&A pairs evaluated at each pipeline change

---

### Latency Budget

| Stage | Expected Latency |
|---|---|
| Retrieval | ~500ms |
| Reranking | ~600ms |
| Generation | ~1.5–2.5s |
| Formatting | ~200ms |
| **Total (expected)** | **~3–4s** |
| **Hard ceiling** | **8 seconds** |

- **Agentic / multi-step queries:** Scoped to v2 with a separate 20-second SLA
