# SLM Assistant

A retrieval-augmented AI assistant built with **FastAPI**, **ChromaDB**, and **Groq**.  
Answers questions grounded in your own documents, retrieved by *meaning* — not just keywords.

---

## What It Does

Upload `.txt`, `.pdf`, or `.docx` files from the web UI (or drop them into `data/docs/`).
Each file is split into overlapping chunks and embedded into a local vector
database (ChromaDB). When you ask a question, the question itself is embedded
and compared against every chunk by semantic similarity — so "remote work"
can match a chunk that says "working from home", even with zero shared
keywords. The most relevant chunks are sent to Groq as context to generate
the answer, and are also shown in the UI as "Similar results" underneath it.

```
Upload           .txt/.pdf/.docx ──▶ chunk ──▶ embed ──▶ ChromaDB
                                                              │
User question ──▶ embed ──▶ semantic search ─────────────────┘
                                   │
                                   ▼
                        top-k relevant chunks
                                   │
                                   ▼
                          GroqClient ──▶ Answer + Sources
```

Embeddings are generated locally via ChromaDB's built-in ONNX model
(all-MiniLM-L6-v2) — no API key needed, downloads once (~80MB) on first run.

---

## Quick Start (Docker — recommended)

```bash
# 1. Add your Groq API key to .env (get one free at https://console.groq.com)
echo "GROQ_API_KEY=your-key-here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env

# 2. Start the API
docker-compose up --build

# 3. Open the chat UI
open http://localhost:8000

# ...or chat via curl
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the office hours?"}'
```

---

## Adding Your Own Documents

**Via the web UI (recommended):** open the app, drag a `.txt`, `.pdf`, or
`.docx` file onto the "Upload documents" panel on the left. It's indexed
immediately — no restart needed. Remove a document any time with the ✕ next
to its name.

**Manually:** drop a file into `data/docs/` and restart the app — it will be
indexed automatically at startup (already-indexed files are skipped).

```
data/
├── docs/            ← original files (source of truth)
│   ├── office.txt      ← included as example
│   ├── faq.txt         ← included as example
│   └── your_file.pdf   ← add as many as you like
└── vectordb/        ← ChromaDB's persisted vector index (auto-generated)
```

The assistant answers based on what is in these files, retrieved by meaning
rather than exact wording. If nothing relevant is found, it tells you
honestly.

---

## Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload
```

> **Note:** You need a free Groq API key. Copy it into `.env`:
> ```
> GROQ_API_KEY=your-key-here
> GROQ_MODEL=llama-3.3-70b-versatile
> ```
> Get a key at https://console.groq.com. The first request also downloads the
> local embedding model (~80MB, one-time, cached afterwards).

---

## API Endpoints

| Method | Path                    | Description                              |
|--------|-------------------------|-------------------------------------------|
| GET    | `/api/health`           | Health check                             |
| POST   | `/api/chat`             | Send a message, receive a reply + sources |
| GET    | `/api/history`          | View full conversation history           |
| POST   | `/api/history/clear`    | Wipe conversation history                |
| GET    | `/api/documents`        | List indexed documents                   |
| POST   | `/api/documents`        | Upload + index a `.txt`/`.pdf`/`.docx` file |
| DELETE | `/api/documents/{name}` | Remove a document from disk + the index  |

**Example — POST /api/chat**
```json
// Request
{ "message": "Can I work from home?" }

// Response
{
  "response": "Yes — you can work remotely up to 2 days per week with manager approval.",
  "sources": [
    { "filename": "faq.txt", "text": "Q: What is the remote work policy? A: ...", "score": 0.81 }
  ]
}
```

**Example — POST /api/documents** (multipart form upload, field name `file`)
```json
{ "filename": "handbook.pdf", "chunks_indexed": 12 }
```

---

## Run Tests

```bash
pytest tests/ -v
```

All tests are fully isolated — Groq and ChromaDB are never called during the
test suite. Dependencies are injected as fakes so tests run instantly without
any external services.

```
tests/
├── test_chat_service.py     # ChatService with fake RAG + fake Memory
├── test_memory_service.py   # MemoryService: save, order, clear, copy-safety
└── test_rag_service.py      # DocumentChunker, RAGService with a fake VectorStore
```

---

## Project Structure

```
slm_assistant/
├── app/
│   ├── main.py                       # FastAPI app entry point
│   ├── api/
│   │   └── routes.py                 # HTTP endpoints (chat, history, documents)
│   ├── services/
│   │   ├── chat_service.py           # Wires RAG + memory per user turn
│   │   ├── memory_service.py         # In-memory conversation history
│   │   ├── rag_service.py            # Chunking, indexing, retrieval, LLM call
│   │   ├── vector_store.py           # ChromaDB wrapper (embed, query, delete)
│   │   └── document_extractor.py     # Reads text out of .txt/.pdf/.docx
│   └── static/
│       └── index.html                # Chat UI + upload/knowledge-base sidebar
├── data/
│   ├── docs/                         # Original uploaded files
│   │   ├── office.txt
│   │   └── faq.txt
│   └── vectordb/                     # ChromaDB's persisted index (auto-generated)
├── tests/
│   ├── test_chat_service.py
│   ├── test_memory_service.py
│   └── test_rag_service.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Design Principles

**Single Responsibility** — each class does one thing:
- `DocumentExtractor` reads text out of `.txt`/`.pdf`/`.docx` files
- `DocumentChunker` splits text into overlapping chunks for retrieval
- `VectorStore` embeds and searches chunks via ChromaDB
- `GroqClient` sends prompts to the model
- `RAGService` orchestrates the four above
- `MemoryService` stores conversation history
- `ChatService` ties RAG and memory together for one user turn

**Dependency Injection** — every service accepts its dependencies as constructor
arguments. This makes unit testing trivial: swap real services for fakes in one line.

**Test-Driven** — tests are written at the unit level. Each test follows
Arrange / Act / Assert. No test touches the network or a real vector database
(a `FakeVectorStore` stands in for ChromaDB).

---

## Changing the Model

Edit `.env` (or `docker-compose.yml`'s `env_file`):

```
GROQ_MODEL=llama-3.1-8b-instant   # any model available on Groq
```

See https://console.groq.com/docs/models for available models.

---

## Tech Stack

| Layer            | Technology                          |
|-------------------|--------------------------------------|
| API               | FastAPI + Uvicorn                   |
| LLM backend       | Groq (cloud, free tier)             |
| Vector database   | ChromaDB (embedded, persisted to disk) |
| Embeddings        | ChromaDB's built-in ONNX all-MiniLM-L6-v2 (local, no API key) |
| Document parsing  | pypdf, python-docx                  |
| Testing           | pytest                              |
| Container         | Docker Compose                      |
