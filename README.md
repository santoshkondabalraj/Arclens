# Arclens

**Video intelligence layer over YouTube playlists.**

Ask natural-language questions about any YouTube playlist and get grounded, cited answers with direct links to the exact timestamp in the video.

![Arclens UI](defects/UI.png)

---

## How it works

```
User question
     │
     ▼
rewrite_query  ←  Gemini Flash strips meta-framing → content keywords
     │
     ▼
retrieve       ←  Dense (Gemini embeddings) + Sparse (BM25) via Pinecone
     │               fused with Reciprocal Rank Fusion (RRF)
     ▼
rerank         ←  Cross-encoder (ms-marco-MiniLM-L-6-v2) scores top candidates
     │
     ▼
threshold_check ← Refuse if no chunk clears MIN_RERANK_SCORE
     │
     ▼
generate       ←  Gemini 2.5 Flash — grounded answer from retrieved chunks only
     │
     ▼
format_citations ← Inline [Source: title, channel, ~Xs] → video_id + timestamp
     │
     ▼
Answer + clickable timestamp cards → inline YouTube player
```

---

## Features

- **Hybrid retrieval** — dense semantic search (Gemini embeddings) + BM25 lexical search, fused via RRF
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 scores each candidate for precise relevance
- **Query rewriting** — keyword extraction strips meta-framing ("according to the video") before retrieval
- **Grounded generation** — Gemini 2.5 Flash answers only from retrieved transcript chunks
- **Timestamp citations** — every claim links back to the exact second in the source video
- **Inline video player** — click any citation to watch from the cited moment without leaving the app
- **Staleness detection** — warns when a playlist hasn't been re-ingested in > 30 days
- **Debug endpoint** — `/debug/retrieve` shows rewritten query, scores, and threshold in real time

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph (StateGraph) |
| LLM | Gemini 2.5 Flash (generation + query rewriting) |
| Embeddings | Gemini `text-embedding-004` |
| Vector store | Pinecone (dense namespace per user×playlist) |
| Sparse index | BM25 (rank-bm25, persisted per playlist) |
| Reranker | sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Transcripts | youtube-transcript-api + pytubefix |
| Frontend | Alpine.js + Tailwind CSS (CDN, no build step) |
| Observability | LangSmith tracing |

---

## Getting started

### Prerequisites

- Python 3.11+
- [Google AI API key](https://aistudio.google.com/) (Gemini)
- [Pinecone API key](https://www.pinecone.io/) with a serverless index
- [LangSmith API key](https://smith.langchain.com/) (optional, for tracing)

### Install

```bash
git clone https://github.com/santoshkondabalraj/Arclens.git
cd Arclens
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Key settings in `.env`:

```
GOOGLE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=yt-rag-hybrid
MIN_RERANK_SCORE=-8.0     # cross-encoder logit threshold
RERANK_TOP_N=5
DENSE_TOP_K=20
SPARSE_TOP_K=20
```

### Run

```bash
uvicorn yt_rag.api.main:app --reload
```

Open **http://localhost:8000** — the Alpine.js UI loads automatically.

---

## Usage

### 1. Ingest a playlist

Paste a YouTube playlist URL in the **Add Playlist** accordion in the sidebar and click **Ingest Playlist**. Ingestion runs in the background; the playlist appears in your Library when complete.

You can also trigger ingestion via the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"playlist_url": "https://youtube.com/playlist?list=...", "user_id": "you"}'
```

### 2. Ask questions

Select a playlist from the Library, type a question, and press **Ask**. Results appear as source clip cards with timestamps; click any card to watch the clip inline.

### 3. Calibrate retrieval

Use the debug endpoint to see how a question is rewritten and how each chunk scores:

```bash
curl -X POST http://localhost:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What drives productivity growth?", "playlist_id": "...", "user_id": "..."}'
```

Response includes `original_question`, `rewritten_query`, per-chunk scores, and the active threshold.

---

## Project structure

```
src/yt_rag/
├── api/          # FastAPI app, routes, request/response models
├── chunking/     # Token-aware text splitters
├── embeddings/   # Gemini embedding wrapper
├── generation/   # Prompt templates + LangChain generation chain
├── graph/        # LangGraph StateGraph (nodes, state, graph wiring)
│   ├── nodes.py  # rewrite_query, retrieve, rerank, generate, format_citations
│   ├── state.py  # RAGState TypedDict
│   └── rag_graph.py
├── ingestion/    # YouTube loader, transcript cleaner, freshness tracking
└── retrieval/    # Pinecone store, BM25 index, hybrid + cross-encoder

ui/
└── index.html    # Single-file Alpine.js + Tailwind frontend

tests/            # Unit tests + golden QA dataset
scripts/          # CLI ingestion + evaluation helpers
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_FLASH_MODEL` | `gemini-2.5-flash` | Generation + rewrite model |
| `GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | Dense embedding model |
| `PINECONE_INDEX_NAME` | `yt-rag-hybrid` | Pinecone index name |
| `DENSE_TOP_K` | `20` | Candidates from dense retrieval |
| `SPARSE_TOP_K` | `20` | Candidates from BM25 retrieval |
| `RERANK_TOP_N` | `5` | Top chunks passed to generation |
| `MIN_RERANK_SCORE` | `-8.0` | Cross-encoder logit cutoff (raw logits, not 0–1) |
| `STALENESS_DAYS` | `30` | Days before a stale-corpus warning is shown |
| `BODY_CHUNK_SIZE` | `600` | Tokens per transcript chunk |
| `BODY_CHUNK_OVERLAP` | `150` | Token overlap between chunks |

> **Note on `MIN_RERANK_SCORE`:** the ms-marco cross-encoder outputs raw logits, not probabilities. Scores above `-2` are topically relevant; scores above `0` are directly relevant. `-8.0` is intentionally lenient so the LLM handles final relevance filtering.

---

## License

MIT
