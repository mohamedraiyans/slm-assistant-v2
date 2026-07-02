"""
VectorStore — wraps ChromaDB for semantic (meaning-based) document search.

Uses ChromaDB's built-in local embedding model (a small ONNX build of
all-MiniLM-L6-v2). No API key, no GPU, no external service — the model
downloads once (~80MB) the first time it's used and then runs fully offline.
Data is persisted to disk under `persist_dir` so the index survives restarts.
"""

from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "data/vectordb", collection_name: str = "documents"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list) -> None:
        """Embed and store chunks. Re-uploading the same chunk id overwrites it."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"filename": c.filename, "index": c.index} for c in chunks],
        )

    def query(self, question: str, top_k: int = 5) -> list:
        """Return the top_k chunks whose meaning is closest to the question."""
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, count),
        )
        matches = []
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            matches.append({
                "filename": meta["filename"],
                "text": text,
                "score": round(max(0.0, 1 - distance), 4),
            })
        return matches

    def has_document(self, filename: str) -> bool:
        existing = self._collection.get(where={"filename": filename}, limit=1)
        return len(existing["ids"]) > 0

    def list_filenames(self) -> list:
        existing = self._collection.get()
        return sorted({meta["filename"] for meta in existing["metadatas"]})

    def delete_document(self, filename: str) -> None:
        self._collection.delete(where={"filename": filename})
