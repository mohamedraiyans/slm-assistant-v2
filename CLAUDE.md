# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLM Assistant — a retrieval-augmented chat assistant built with **FastAPI**, **ChromaDB**, and **Groq**. Users upload documents (`.txt`/`.pdf`/`.docx`); the assistant answers questions grounded in them, retrieved by semantic similarity rather than keyword matching. Repo: `github.com/mohamedraiyans/slm-assistant-v2`.

## Running Locally

```bash
# Without Docker
pip install -r requirements.txt
uvicorn app.main:app --reload

# With Docker (recommended)
docker-compose up --build
```

Requires a `.env` file at the repo root with:
```
GROQ_API_KEY=your-key-here
GROQ_MODEL=llama-3.3-70b-versatile
```
Get a free key at https://console.groq.com. `.env` is gitignored — never commit it.

The first request downloads ChromaDB's local embedding model (~80MB ONNX build of all-MiniLM-L6-v2) and caches it; every request after that runs fully offline for embeddings. Only chat generation calls out to Groq's cloud API.

Open `http://localhost:8000` for the chat UI.

## Architecture

### RAG Pipeline

```
Upload (.txt/.pdf/.docx) ──▶ DocumentExtractor ──▶ DocumentChunker ──▶ VectorStore (ChromaDB)
                                                                              │
User question ──────────────────────────────────────────▶ embed ──▶ semantic search
                                                                              │
                                                                              ▼
                                                                  top-k relevant chunks
                                                                              │
                                                                              ▼
                                                                GroqClient ──▶ Answer + Sources
```

- **Chunking**: word-based sliding window, 150 words per chunk with 30-word overlap (`DocumentChunker` in `rag_service.py`).
- **Embeddings**: generated locally by ChromaDB's built-in ONNX model — no API key, no GPU.
- **Retrieval**: cosine similarity search over stored chunks (`VectorStore.query`), returns `{filename, text, score}` per match.
- **Generation**: matched chunks are joined into a context block and sent to Groq along with the question; the LLM is told to answer only from that context and say so if it can't.

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI entry point; loads `.env` before anything else reads env vars |
| `app/api/routes.py` | HTTP endpoints — chat, history, document upload/list/delete |
| `app/services/rag_service.py` | `DocumentChunker`, `GroqClient`, `RAGService` (orchestrates chunking → indexing → retrieval → generation) |
| `app/services/vector_store.py` | `VectorStore` — ChromaDB wrapper (embed, upsert, query, delete) |
| `app/services/document_extractor.py` | Reads raw text out of `.txt`/`.pdf`/`.docx` files |
| `app/services/memory_service.py` | In-memory conversation history (not persisted across restarts) |
| `app/services/chat_service.py` | Wires `RAGService` + `MemoryService` together for one user turn |
| `app/static/index.html` | Single-file chat UI — upload sidebar, chat panel, sources panel (vanilla JS/CSS, no build step) |
| `data/docs/` | Original uploaded files (source of truth) |
| `data/vectordb/` | ChromaDB's persisted index — auto-generated, gitignored |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | `{message}` → `{response, sources}` (sources = matched chunks with filename/text/score) |
| GET | `/api/history` | Full conversation history |
| POST | `/api/history/clear` | Wipe conversation history |
| GET | `/api/documents` | List indexed filenames |
| POST | `/api/documents` | Multipart upload (field `file`) → extracts, chunks, embeds immediately, no restart needed |
| DELETE | `/api/documents/{filename}` | Removes file from disk and from the vector index |

### Frontend Behavior (`app/static/index.html`)

- Drag-and-drop or click-to-upload sidebar; calls `POST /api/documents`, then refreshes the document list.
- Each assistant reply shows a "Best match" source (highlighted card) plus a "Similar results" list of the rest.
- Match-score badges and the assistant's reply bubble are color-coded by the top match score: `≥50%` green, `25–49%` orange, `<25%` red (see `scoreClass()` in the `<script>` block). Because chunks can mix multiple topics, correct answers often score in the orange/red range — this reflects raw cosine similarity, not correctness.
- Query words that literally appear in a matched chunk are wrapped in `<mark>` for visual scanning (`highlightMatches()`); purely semantic matches (no shared words) won't highlight anything — that's expected.

## Testing

```bash
pytest tests/ -v
```

Tests are fully isolated — no real Groq calls, no real ChromaDB. `RAGService` and `ChatService` take their dependencies (`vector_store`, `llm_client`, `rag`, `memory`) via constructor injection, so tests substitute fakes (`FakeVectorStore`, `FakeLLMClient`, `FakeRAG`, `FakeMemory`). Follow this pattern for new services — don't reach for mocking libraries when a small fake class does the job.

## Adding/Removing Documents

- **Via the UI** (preferred): upload/delete from the sidebar — indexed or removed immediately, no restart.
- **Manually**: drop a file into `data/docs/` and restart the app; `RAGService.load_documents()` skips files already in the vector store, so restarts are safe and idempotent.

## Changing the Model

Edit `GROQ_MODEL` in `.env` (or `docker-compose.yml`'s `env_file`). See https://console.groq.com/docs/models for available models.

## Deployment

No CI/CD pipeline. `docker-compose up --build` is the deploy step wherever this runs — `data/docs/` and `data/vectordb/` are mounted as volumes so uploaded documents and the vector index survive container restarts.
