"""
RAGService — Retrieval-Augmented Generation using local .txt documents.

Documents are split into line-level chunks at load time.
Retrieval scores chunks by meaningful keyword overlap (stopwords excluded).
Top-scoring chunks plus their immediate neighbours are sent to Groq
for a fast, natural AI-generated answer.
"""

from pathlib import Path
from dataclasses import dataclass
import os

from groq import Groq


@dataclass
class Document:
    filename: str
    content: str


@dataclass
class Chunk:
    filename: str
    text: str
    index: int


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class DocumentLoader:
    """Loads plain-text documents from a directory."""

    def __init__(self, docs_dir: str = "data/docs"):
        self.docs_dir = Path(docs_dir)

    def load_all(self) -> list:
        if not self.docs_dir.exists():
            return []
        documents = []
        for path in sorted(self.docs_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                documents.append(Document(filename=path.name, content=content))
        return documents


class DocumentChunker:
    """Splits documents into line-level chunks for fine-grained retrieval."""

    def chunk(self, documents: list) -> list:
        chunks = []
        for doc in documents:
            for line in doc.content.splitlines():
                line = line.strip()
                if line:
                    chunks.append(Chunk(
                        filename=doc.filename,
                        text=line,
                        index=len(chunks),
                    ))
        return chunks


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "not",
}


def _normalize(text: str) -> set:
    """Lowercase, strip punctuation, remove stopwords for keyword scoring."""
    words = {word.strip(",:?!.()") for word in text.lower().split()}
    return words - _STOPWORDS


class DocumentRetriever:
    """
    Scores chunks by meaningful keyword overlap with the query.
    Each top match also pulls in its next line so key:value pairs
    like Company/Location are always retrieved together.
    """

    def retrieve(self, query: str, chunks: list, top_k: int = 2) -> list:
        if not chunks:
            return []
        query_words = _normalize(query)
        if not query_words:
            return []

        scored = []
        for chunk in chunks:
            overlap = len(query_words & _normalize(chunk.text))
            if overlap == 0:
                continue
            score = overlap / (1 + len(chunk.text.split()) * 0.1)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        top_chunks = [r.chunk for r in scored[:top_k]]

        result_indices = {c.index for c in top_chunks}
        for chunk in top_chunks:
            ni = chunk.index + 1
            if ni < len(chunks) and chunks[ni].filename == chunk.filename:
                result_indices.add(ni)

        return [chunks[i] for i in sorted(result_indices)]


class GroqClient:
    """
    Client for the Groq API — fast cloud inference, free tier available.
    Responses arrive in ~1 second regardless of model size.
    """

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
    Orchestrates chunking, retrieval, and AI answer generation via Groq.

    Given a question, it:
      1. Retrieves the most relevant document chunks.
      2. Sends them as context to Groq for a natural AI answer.
    """

    def __init__(self, loader=None, chunker=None, retriever=None, llm_client=None):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or DocumentRetriever()
        self.llm_client = llm_client or GroqClient()
        self._chunks: list = []

    def load_documents(self) -> None:
        """Load and chunk documents from disk. Call once at startup."""
        self._chunks = self.chunker.chunk(self.loader.load_all())

    def generate_answer(self, question: str) -> str:
        relevant_chunks = self.retriever.retrieve(question, self._chunks)
        context = self._build_context(relevant_chunks)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using "
            "ONLY the context provided. Be concise and direct. "
            "If the context does not contain the answer, say so clearly."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        return self.llm_client.generate(system_prompt, user_message)

    def _build_context(self, chunks: list) -> str:
        if not chunks:
            return "No relevant information found."
        return "\n".join(f"- {chunk.text}" for chunk in chunks)
