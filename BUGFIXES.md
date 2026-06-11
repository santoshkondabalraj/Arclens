# Arclens RAG App — Bug Fix Log

---

## 1. Prompt & LLM Configuration

### 1.1 LangChain template variable collision
**Symptom:** `500 Internal Server Error` — `KeyError: 'title'` / `'channel'` / `'timestamp'`  
**Root cause:** The system prompt used `{title}`, `{channel}`, `{timestamp}` inside citation examples. `ChatPromptTemplate` treats single-brace expressions as input variables and tried to fill them during chain invocation.  
**Fix:** Escaped all literal braces in the prompt with double braces — `{{title}}`, `{{channel}}`, `{{timestamp}}` — so LangChain passes them through unchanged.  
**File:** `src/yt_rag/generation/prompts.py`

---

### 1.2 LLM over-refusal via JSON instruction (Gemini 2.5 Flash)
**Symptom:** Valid questions consistently refused with `{"refusal": true, "reason": "The available podcast transcripts do not contain information about this topic."}` even when relevant chunks were retrieved and passed the threshold.  
**Root cause:** The prompt instructed the LLM to output a structured JSON refusal if context was "insufficient". Gemini 2.5 Flash has built-in chain-of-thought thinking enabled by default, making it non-deterministic at `temperature=0` and overly conservative — it applied the refusal even when context was present.  
**Fix:** Removed the JSON refusal instruction from the prompt entirely. The LLM now always produces prose. If context is partial, it answers what it can and notes any gaps in natural language. `threshold_check` is the sole refusal gate.  
**File:** `src/yt_rag/generation/prompts.py`

---

### 1.3 Gemini 2.5 Flash thinking tokens consumed `max_output_tokens` budget
**Symptom:** Generation answers were truncated at ~382 characters, always ending mid-citation (e.g. `[Source: "Title", Channel,` with no timestamp or closing bracket). The truncation point was consistent across runs.  
**Root cause:** `ChatGoogleGenerativeAI` with `max_output_tokens=2048` and Gemini 2.5 Flash's default thinking mode enabled. The thinking tokens are allocated from the same `max_output_tokens` budget. With ~1600 tokens consumed by internal reasoning, only ~400 tokens remained for the visible response — roughly 380 characters.  
**Fix:** Set `thinking_budget=0` in `build_llm()` to disable thinking for the generation chain. RAG generation is document-grounded Q&A — it does not benefit from extended reasoning, and thinking mode adds latency and token cost without quality gain. Also raised `max_output_tokens` to `8192` as a safety margin for longer answers.  
**File:** `src/yt_rag/generation/chain.py`

---

## 2. API & Model Errors

### 2.1 Deprecated model — 404 NOT_FOUND
**Symptom:** `404 NOT_FOUND: models/gemini-2.0-flash is not found`  
**Root cause:** `gemini-2.0-flash` was deprecated and removed from the Google AI API.  
**Fix:** Updated `gemini_flash_model` to `gemini-2.5-flash` in `config.py` and `.env`.  
**File:** `src/yt_rag/config.py`

---

### 2.2 Request timeout too short
**Symptom:** `Request exceeded the 8-second latency ceiling` for most queries.  
**Root cause:** `asyncio.wait_for` timeout was 8 seconds, which was adequate for `gemini-2.0-flash` but too short for `gemini-2.5-flash` (which runs an internal thinking pass).  
**Fix:** Increased timeout to 30 seconds.  
**File:** `src/yt_rag/api/main.py`

---

## 3. Ingestion Pipeline & Timestamps

### 3.1 Timestamps displayed as character offsets (~13310s)
**Symptom:** Citations showed absurd timestamps like `~13310s` (≈ 3.7 hours) for a short video.  
**Root cause:** The ingestion pipeline cleaned documents (stripping `[T:Xs]` sentinel markers) **before** chunking. By the time metadata was extracted, the sentinels were gone, so `start_index` (a character offset in the raw transcript string, e.g. `13310`) was used as the timestamp in seconds.  
**Fix:** Reversed the pipeline order — chunk **before** cleaning. `stamp_and_clean_document()` now extracts the first `[T:Xs]` sentinel from each raw chunk into `metadata["timestamp_seconds"]`, then removes it and other noise from the text. Character offsets are never exposed.  
**Files:** `src/yt_rag/ingestion/cleaner.py`, `src/yt_rag/api/main.py`

---

### 3.2 Timestamp formatted as MM:SS instead of HH:MM:SS
**Symptom:** `Jump to 221:50` displayed in the UI (221 minutes, 50 seconds) instead of `3:41:50`.  
**Root cause:** `formatTime()` in the frontend treated all values as `MM:SS`, with no handling for durations over an hour.  
**Fix:** Added an HH:MM:SS branch — values ≥ 3600 seconds now render as hours + zero-padded minutes and seconds.  
**File:** `ui/index.html`

---

## 4. Video Playback

### 4.1 Jump-to-timestamp link navigated to `index.html#` instead of playing video
**Symptom:** Clicking a timestamp link opened a new browser tab pointing to `http://localhost:8000/ui/index.html#` instead of playing the video inline.  
**Root cause:** Timestamp links used `<a :href="ytLink(...)">`. When `video_id` was absent, `ytLink()` returned `"#"`, causing normal anchor navigation and a page reload/new tab.  
**Fix:** Replaced all `<a href>` timestamp elements with `<button @click="openVideo(c)">` (with `x-show="c.video_id"`) and clip cards with `<div @click="c.video_id && openVideo(c)">`. No anchor navigation occurs.  
**File:** `ui/index.html`

---

### 4.2 `video_id` missing from citations — gray thumbnails, broken playback
**Symptom:** All clip cards showed gray thumbnails and clicking them did nothing. `video_id` was empty on every citation.  
**Root cause:** The `Citation()` constructor call in `main.py` listed `title`, `channel`, and `timestamp_seconds` but **omitted `video_id=`**. Pydantic defaulted it to `""`.  
**Fix:** Added `video_id=c.get("video_id", "")` to the `Citation()` constructor.  
**File:** `src/yt_rag/api/main.py`

---

## 5. LangGraph State Management

### 5.1 `format_citations` saw empty `reranked_docs` after `generate`
**Symptom:** `format_citations` could not look up `video_id` for citations — it saw `reranked_docs = []` even though `generate` had retrieved relevant documents.  
**Root cause:** The `generate` node returned only `{"answer": answer}`. LangGraph merges partial state dicts, leaving `reranked_docs` at whatever value it had in the graph's accumulated state (often the initial `[]`). Because `generate` didn't include `reranked_docs` in its return, the previous `threshold_check` update was effectively lost by the time `format_citations` ran.  
**Fix:** `generate` now explicitly returns `{"answer": answer, "reranked_docs": reranked}`, carrying the docs forward.  
**File:** `src/yt_rag/graph/nodes.py`

---

## 6. Retrieval & Reranking

### 6.1 Duplicate citations for the same timestamp
**Symptom:** Results panel showed 4 identical citations all pointing to the same video at the same timestamp (e.g. `13:56 × 4`).  
**Root cause:** Multiple overlapping chunks from the same video segment all survived reranking and were independently cited by the LLM. `format_citations` had no deduplication step.  
**Fix:** Added a deduplication pass in `format_citations` keyed on `(video_id, timestamp_seconds)` — first occurrence is kept, subsequent duplicates are dropped.  
**File:** `src/yt_rag/graph/nodes.py`

---

### 6.2 Over-refusal — threshold required ≥ 2 passing chunks
**Symptom:** Legitimate single-topic questions were refused even when one highly relevant chunk was retrieved.  
**Root cause:** `threshold_check` used `if len(passing) < 2: refuse`. Topics with only one clearly relevant segment (e.g. a focused explainer video) always failed this check regardless of relevance score.  
**Fix:** Changed condition to `if not passing` — a single passing chunk is sufficient to attempt an answer.  
**File:** `src/yt_rag/graph/nodes.py`

---

### 6.3 Meta-phrases in queries caused cross-encoder score collapse
**Symptom:** `"What drives productivity growth?"` → score `+1.019` → answered. `"What drives productivity growth according to the video?"` → score `-2.394` → refused. Semantically identical queries, wildly different scores.  
**Root cause:** The cross-encoder (`ms-marco-MiniLM-L-6-v2`) was trained on web QA pairs. The phrase `"according to the video"` is a meta-instruction to the RAG system, not a concept that appears in transcript text. The model penalised the token mismatch, collapsing the relevance score even though the underlying information need was identical.  
**Fix:** Added `_META_PHRASE_RE` in `nodes.py` to strip meta-phrases (`"according to the video"`, `"in this lecture"`, `"based on the content"`, etc.) from the query **before** building cross-encoder pairs. The stripped query is also used for dense retrieval — meta-phrases add no embedding signal since they never appear in transcript chunks.  
**File:** `src/yt_rag/graph/nodes.py`

---

### 6.4 Rerank threshold `0.3` treated logits as probabilities — updated progressively
**Symptom:** Topically relevant chunks were filtered out, causing unnecessary refusals.  
**Root cause:** `min_rerank_score` was initially set to `0.3` as if cross-encoder scores were calibrated probabilities. The ms-marco cross-encoder outputs **raw logits**. As the threshold was progressively calibrated against real queries, additional failure modes were discovered:

| Threshold | Problem discovered |
|---|---|
| `0.3` | Filtered out most valid results (logits, not probabilities) |
| `-2.0` | Still refused queries where relevant chunk opened with off-topic content (rules chunk scored `-7.35`) |
| `-8.0` | Still refused semantic synonym queries ("lessons" vs "rules of thumb") scoring around `-9 to -10` |
| **`-10.0`** | **Current value — allows semantic paraphrase queries while still refusing off-topic queries (score `< -13`)** |

Score reference for `ms-marco-MiniLM-L-6-v2` raw logits:

| Range | Meaning |
|---|---|
| `> 3` | Direct answer in chunk |
| `0 – 3` | Topically related |
| `−5 – 0` | Same domain, indirect |
| `−10 – −5` | Semantic synonym / partial match |
| `< −13` | Off-topic — correctly filtered |

**Files:** `src/yt_rag/config.py`, `.env`, `.env.example`

---

### 6.5 LLM JSON refusal output leaked as answer text
**Symptom:** UI displayed raw JSON `{"refusal": true, "reason": "..."}` as the answer body instead of a proper refusal banner.  
**Root cause:** When the LLM output the JSON refusal format, `format_citations` had no detection logic. It treated the JSON string as a normal answer, so `main.py` returned `ChatResponse(answer='{"refusal":true,...}', refusal=False)`. The UI rendered the JSON verbatim.  
**Fix:** Added a JSON detection block at the top of `format_citations`. If the answer starts with `{` and parses as `{"refusal": true, ...}`, the node returns a proper `{"refusal": True, "reason": "..."}` final response. The UI then shows the styled amber refusal banner instead of raw JSON.  
**File:** `src/yt_rag/graph/nodes.py`

---

### 6.6 Keyword query rewriting undermined dense retrieval for synonym queries
**Symptom:** `"What thumb rules should I learn?"` returned the three Ray Dalio rules. `"What lessons should I learn?"` returned no answer. These are semantically identical questions.  
**Root cause:** The pipeline included a `rewrite_query` node that extracted keywords from the user's question (e.g. `"lessons learn"`) and passed them to **all** retrieval stages — dense embedding, BM25, and cross-encoder. This caused two compounding failures:

1. **Dense retrieval**: Gemini `text-embedding-004` embeds full natural-language questions better than sparse keyword bags. `"What lessons should I learn?"` has a rich semantic vector that bridges to `"rules of thumb"` in embedding space. `"lessons learn"` does not. Keyword extraction actively undermined the model's semantic capability.

2. **BM25 lexical gap**: Neither `"lessons"` nor `"learn"` appears in the rules chunk (which uses `"rules of thumb"` and `"take away"`). BM25 could not match the vocabulary regardless of query form.

3. **Cross-encoder**: `ms-marco-MiniLM-L-6-v2` was trained on natural-language query–passage pairs. Scoring `"lessons learn"` (a keyword bag) against a passage produces worse results than scoring the original question.

The fix of prompting the rewrite model to expand structural terms (`"lessons" → "lessons rules takeaway key"`) was attempted but rejected because it was domain-specific: it would work for economics but fail for any other playlist (sports, cooking, etc.).

**Fix:** Removed `rewrite_query` entirely. All retrieval stages now use the original question with only `_META_PHRASE_RE` stripping applied:
- **Dense retrieval**: original question → embedding model bridges synonyms semantically
- **BM25**: original question → stop-word removal retains content terms naturally
- **Cross-encoder**: original question → NL query–passage scoring as the model was trained

**Files:** `src/yt_rag/graph/nodes.py`, `src/yt_rag/graph/rag_graph.py`, `src/yt_rag/graph/state.py`, `src/yt_rag/api/main.py`

---

### 6.7 `rerank_top_n=5` silently excluded the most relevant chunk
**Symptom:** Even after fixing retrieval recall, `"What lessons should I learn?"` was still refused. The rules chunk (chunk_index=21, score `-9.93`) was above the `-10.0` threshold but never reached the LLM.  
**Root cause:** `rerank` kept only the top 5 chunks by cross-encoder score. For semantic synonym queries, the relevant chunk scored lower than generic same-domain content because of vocabulary mismatch — the rules chunk ranked 10th, outside the cutoff. The `threshold_check` node never saw it.  
**Fix:** Raised `rerank_top_n` from `5` to `10`. This expands the candidate pool passed to `threshold_check` and the LLM. All 10 chunks for this query type passed the `-10.0` threshold; the LLM correctly identified chunk_index=21 as the source for the explicit rules.  
**Files:** `src/yt_rag/config.py`, `.env`, `.env.example`

---

## 7. User Identity & Namespace Routing

### 7.1 Wrong Pinecone namespace used when playlist URL was typed manually
**Symptom:** Asking questions via the UI returned LLM refusals even though the same query worked in PowerShell. The Pinecone dense retrieval returned no results.  
**Root cause:** Two code paths set `playlistId` but neither set `userId`:
1. `init()` `$watch('playlistUrl')` — extracted `playlist_id` from a pasted URL but did not look up the matching `user_id`.
2. `selectPlaylist()` — only set `userId` if it was currently empty (`if (!this.userId)`), so any prior value from the ingest form was preserved.

The Pinecone namespace is `{user_id}_{playlist_id}`. Sending the wrong `user_id` pointed to an empty namespace, producing no dense results and causing the LLM to refuse.  
**Fix:**
- `selectPlaylist()` now **always** overwrites `userId` from the stored library record.
- The `$watch` on `playlistUrl` now also looks up and sets `userId` from the `playlists` array when a matching record exists.  
**File:** `ui/index.html`

---

## 8. Evaluation Suite

### 8.1 `citation_precision_evaluator` crashed with `AttributeError` in LangSmith runs
**Symptom:** `citation_precision` evaluator raised `AttributeError: 'dict' object has no attribute 'metadata'` when run via `langsmith.evaluate()`, even though the same evaluator worked in unit tests.  
**Root cause:** LangSmith serialises and deserialises run outputs as plain dicts when passing them to evaluators. Unit tests passed `Document` objects directly; LangSmith passed the same data after JSON round-tripping, producing `dict` objects where evaluator code assumed `LangChain Document` instances with `.metadata` attributes.  
**Fix:** Added `hasattr(d, "metadata")` guard to handle both types:
```python
if hasattr(d, "metadata"):
    source_titles.add(d.metadata.get("title", ""))
else:
    source_titles.add((d.get("metadata") or {}).get("title", ""))
```
**File:** `src/yt_rag/evaluation/judges.py`

---

### 8.2 `_eval_llm()` thinking tokens consumed `max_output_tokens` budget in evaluators
**Symptom:** `faithfulness` evaluator responses were truncated mid-JSON, causing JSON parse errors and eval failures.  
**Root cause:** Same issue as Bug 1.3 — Gemini 2.5 Flash's default thinking mode allocated most of the `max_output_tokens` budget to internal reasoning. The faithfulness evaluator prompt asks for a structured JSON verdict; with only ~400 tokens of visible output, longer responses were cut off.  
**Fix:** Set `thinking_budget=0` and `max_output_tokens=4096` in `_eval_llm()` inside `judges.py`. Faithfulness scoring is a classification task, not multi-step reasoning — thinking mode adds no value and breaks structured output.  
**File:** `src/yt_rag/evaluation/judges.py`

---

### 8.3 `UnicodeEncodeError` in `view_eval_data.py` on Windows
**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '─' in position 0` when running `scripts/view_eval_data.py` on Windows PowerShell.  
**Root cause:** The `_bar()` helper used `"─"` (U+2500, BOX DRAWINGS LIGHT HORIZONTAL) as the default separator character. Windows PowerShell's default console encoding (cp1252) does not include U+2500.  
**Fix:** Replaced `"─"` with `"-"` (ASCII hyphen) in `_bar()`'s default parameter.  
**File:** `scripts/view_eval_data.py`

---

### 8.4 Auto-generated probes were circular — did not test vocabulary-gap retrieval
**Symptom:** Tier 2 smoke probes passed consistently but did not expose synonym retrieval regressions. When `"lessons"` stopped finding `"rules of thumb"` chunks, the smoke tests still passed.  
**Root cause:** Probes generated by reading a chunk and writing a question about it naturally reuse the chunk's vocabulary. A probe asking `"What rules of thumb does Dalio discuss?"` retrieves the same chunk by exact keyword match — the test never verifies that the query `"lessons"` maps to the chunk containing `"rules of thumb"`. This is a fundamental circularity in auto-generated retrieval tests.  
**Fix:** Designed a two-tier eval architecture. Tier 2 (Gemini Flash, generated at ingestion) remains intentionally circular — it only checks that the pipeline runs and citations appear (structural health). Tier 1 uses Claude Opus to generate deliberate vocabulary-gap probes where questions use semantically equivalent but lexically distinct vocabulary from the source chunks. The `vocabulary_note` field on each synonym example documents the specific bridge being tested (e.g. `"'guidelines' bridges to 'rules of thumb' in the source chunk"`).  
**Files:** `src/yt_rag/evaluation/golden_generator.py` (new), `src/yt_rag/evaluation/probe_generator.py` (new)
