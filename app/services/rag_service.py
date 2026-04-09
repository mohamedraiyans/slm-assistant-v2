"""
RAGService — Retrieval-Augmented Generation using local .txt documents.

Since all documents are small, the full content is always sent to Groq.
This allows Groq to reason semantically — handling synonyms, paraphrasing,
and indirect questions that keyword matching cannot resolve.

DocumentRetriever is kept for future use when document sets grow large.
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
    "know", "tell", "say", "get", "make", "go", "see", "think",
}


def _normalize(text: str) -> set:
    """Lowercase, strip punctuation, remove stopwords for keyword scoring."""
    words = {word.strip(",:?!.()") for word in text.lower().split()}
    return words - _STOPWORDS


class DocumentRetriever:
    """
    Keyword-based retriever — kept for future use when document sets are large.
    Not used in the current RAGService which sends full context to Groq.
    """

    def retrieve(self, query: str, chunks: list, top_k: int = 5) -> list:
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
    Sends full document context to Groq for every question.

    This allows Groq to reason semantically — it can handle synonyms,
    paraphrasing, and indirect questions. For example:
    - "quality or state of being physically strong" → matches "strength"
    - "do you know about me?" → reads the full CV
    - "what is your WiFi?" → finds the answer in office.txt

    Documents are small so sending full context costs negligible tokens.
    """

    def __init__(self, loader=None, chunker=None, retriever=None, llm_client=None):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or DocumentRetriever()
        self.llm_client = llm_client or GroqClient()
        self._chunks: list = []
        self._documents: list = []

    def load_documents(self) -> None:
        """Load and chunk documents from disk. Call once at startup."""
        self._documents = self.loader.load_all()
        self._chunks = self.chunker.chunk(self._documents)

    def generate_answer(self, question: str) -> str:
        context = self._full_context()
        system_prompt = (
            "You are a helpful personal assistant. "
            "You have access to the user's personal documents including their CV, "
            "interview preparation notes, and office information. "
            "Answer questions naturally and confidently using the context. "
            "When asked about skills, strengths, experience, or background, "
            "refer to the documents directly. "
            "If something is not in the documents, say so clearly."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        return self.llm_client.generate(system_prompt, user_message)

    def _full_context(self) -> str:
        """Return complete content of all documents."""
        if not self._documents:
            return "No documents available."
        return "\n\n".join(
            f"=== {doc.filename} ===\n{doc.content}"
            for doc in self._documents
        )
