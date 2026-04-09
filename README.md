# SLM Assistant

A local, privacy-first AI assistant built with **FastAPI** and **Ollama**.  
Answers questions grounded in your own documents — no cloud, no API keys, no cost.

---

## What It Does

Drop `.txt` files into `data/docs/` and the assistant will read them at startup.
When you ask a question, it retrieves the most relevant document(s), builds a
context-aware prompt, and sends it to a local language model via Ollama.

```
User question
     │
     ▼
DocumentRetriever  ──picks best docs──▶  OllamaClient  ──▶  Answer
     │                                        ▲
     └── ranks by keyword overlap        local model
```

---

## Quick Start (Docker — recommended)

```bash
# 1. Start Ollama + the API
docker-compose up --build

# 2. Pull a model (first run only — run this in a separate terminal)
# Note: container name = <folder_name>-ollama-1. Verify with: docker ps
docker exec -it slm_assistant_v2-ollama-1 ollama pull llama3.2

# 3. Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the office hours?"}'
```

---

## Adding Your Own Documents

Put any `.txt` file in `data/docs/` and restart the app.

```
data/
└── docs/
    ├── office.txt      ← included as example
    ├── faq.txt         ← included as example
    └── your_file.txt   ← add as many as you like
```

The assistant answers based on what is in these files.  
If no relevant document is found, it tells you honestly.

---

## Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload
```

> **Note:** Ollama must be running separately.  
> Install it from https://ollama.com and run `ollama pull llama3.2`.

---

## API Endpoints

| Method | Path            | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/`             | Health check                       |
| POST   | `/chat`         | Send a message, receive a reply    |
| GET    | `/history`      | View full conversation history     |
| POST   | `/history/clear`| Wipe conversation history          |

**Example — POST /chat**
```json
// Request
{ "message": "What is the WiFi password?" }

// Response
{ "response": "The WiFi password is Conversy2024." }
```

---

## Run Tests

```bash
pytest tests/ -v
```

All tests are fully isolated — Ollama is never called during the test suite.
Dependencies are injected as fakes so tests run instantly without any external services.

```
tests/
├── test_chat_service.py     # ChatService with fake RAG + fake Memory
├── test_memory_service.py   # MemoryService: save, order, clear, copy-safety
└── test_rag_service.py      # DocumentLoader, DocumentRetriever, RAGService
```

---

## Project Structure

```
slm_assistant/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   └── routes.py            # HTTP endpoints
│   └── services/
│       ├── chat_service.py      # Wires RAG + memory per user turn
│       ├── memory_service.py    # In-memory conversation history
│       └── rag_service.py       # Document loading, retrieval, LLM call
├── data/
│   └── docs/                    # Drop your .txt files here
│       ├── office.txt
│       └── faq.txt
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
- `DocumentLoader` reads files from disk
- `DocumentRetriever` ranks documents by relevance
- `OllamaClient` sends prompts to the model
- `RAGService` orchestrates the three above
- `MemoryService` stores conversation history
- `ChatService` ties RAG and memory together for one user turn

**Dependency Injection** — every service accepts its dependencies as constructor
arguments. This makes unit testing trivial: swap real services for fakes in one line.

**Test-Driven** — tests are written at the unit level. Each test follows
Arrange / Act / Assert. No test touches the network or the filesystem
(except `DocumentLoader` tests which use `tmp_path`).

---

## Changing the Model

Edit `docker-compose.yml` or pass environment variables:

```yaml
environment:
  - OLLAMA_MODEL=llama3        # any model available in Ollama
  - OLLAMA_BASE_URL=http://ollama:11434
```

Pull the new model: `docker exec -it slm_assistant_v2-ollama-1 ollama pull llama3`

---

## Tech Stack

| Layer       | Technology          |
|-------------|---------------------|
| API         | FastAPI + Uvicorn   |
| LLM backend | Ollama (local)      |
| HTTP client | httpx               |
| Testing     | pytest              |
| Container   | Docker Compose      |
