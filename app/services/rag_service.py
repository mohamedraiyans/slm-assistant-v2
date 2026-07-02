"""
RAGService — Retrieval-Augmented Generation backed by a vector database.

Documents are split into overlapping chunks, embedded, and stored in
ChromaDB (see VectorStore). At query time, the question itself is embedded
and compared against every chunk by meaning (cosine similarity) rather than
exact keyword overlap — so a question like "remote work" can match a chunk
that says "working from home".
"""

from pathlib import Path
from dataclasses import dataclass
import os

from groq import Groq

from app.services.vector_store import VectorStore
from app.services.document_extractor import extract_text, SUPPORTED_EXTENSIONS


@dataclass
class Chunk:
    id: str
    filename: str
    text: str
    index: int


class DocumentChunker:
    """
    Splits text into chunks along line boundaries, so distinct facts (e.g.
    one line per FAQ answer, one line per office fact) don't get blended
    into a single diluted embedding. A line longer than chunk_size is
    further split with a word-based sliding window.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, filename: str, text: str) -> list:
        chunks = []
        index = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) <= self.chunk_size:
                chunks.append(Chunk(id=f"{filename}::{index}", filename=filename, text=line, index=index))
                index += 1
                continue
            start = 0
            while start < len(words):
                end = start + self.chunk_size
                chunk_text = " ".join(words[start:end])
                chunks.append(Chunk(
                    id=f"{filename}::{index}",
                    filename=filename,
                    text=chunk_text,
                    index=index,
                ))
                index += 1
                if end >= len(words):
                    break
                start = end - self.overlap
        return chunks


class GroqClient:
    """Client for the Groq API — fast cloud inference, free tier available."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = Groq(api_key=self.api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()


class RAGService:
    """
    Retrieves the most semantically relevant chunks for a question via the
    vector store, then asks Groq to answer using only that retrieved context.
    """

    def __init__(self, docs_dir: str = "data/docs", vector_store=None, chunker=None, llm_client=None):
        self.docs_dir = Path(docs_dir)
        self.vector_store = vector_store or VectorStore()
        self.chunker = chunker or DocumentChunker()
        self.llm_client = llm_client or GroqClient()

    def load_documents(self) -> None:
        """Index any files already sitting in data/docs at startup (skips already-indexed files)."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.docs_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if self.vector_store.has_document(path.name):
                continue
            self.index_file(path)

    def index_file(self, path: Path) -> int:
        """Extract, chunk, and embed one file into the vector store. Returns chunk count."""
        text = extract_text(path)
        chunks = self.chunker.chunk(path.name, text)
        self.vector_store.add_chunks(chunks)
        return len(chunks)

    def remove_document(self, filename: str) -> None:
        self.vector_store.delete_document(filename)

    def list_documents(self) -> list:
        return self.vector_store.list_filenames()

    def generate_answer(self, question: str, top_k: int = 5) -> dict:
        matches = self.vector_store.query(question, top_k=top_k)
        if not matches:
            context = "No relevant information found in the knowledge base."
        else:
            context = "\n\n".join(f"[{m['filename']}] {m['text']}" for m in matches)
        system_prompt = (
            "You are a helpful assistant answering questions using the company's "
            "knowledge base. Use the provided context to answer naturally and "
            "confidently, handling synonyms and paraphrased questions. "
            "If the answer isn't in the context, say so clearly."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        answer = self.llm_client.generate(system_prompt, user_message)
        return {"answer": answer, "sources": matches}
