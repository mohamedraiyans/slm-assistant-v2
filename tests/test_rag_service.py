"""
Tests for RAGService components.

DocumentLoader, DocumentChunker, DocumentRetriever, and RAGService
are each tested in isolation. GroqClient is never called — a fake is injected.
"""

from app.services.rag_service import (
    Document,
    Chunk,
    DocumentLoader,
    DocumentChunker,
    DocumentRetriever,
    RAGService,
    _normalize,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class FakeLLMClient:
    def __init__(self, reply: str = "mocked answer"):
        self.reply = reply
        self.last_user_message = None

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.last_user_message = user_message
        return self.reply


def make_chunks(*texts) -> list:
    return [Chunk(filename="test.txt", text=t, index=i) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_strips_trailing_punctuation():
    assert "location" in _normalize("Location:")

def test_normalize_lowercases_words():
    assert "wifi" in _normalize("WiFi")

def test_normalize_removes_stopwords():
    result = _normalize("where is the company located")
    assert "where" not in result
    assert "the" not in result
    assert "company" in result
    assert "located" in result

def test_normalize_handles_mixed_punctuation():
    result = _normalize("Hello, world!")
    assert "hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

def test_loader_returns_empty_for_missing_directory(tmp_path):
    loader = DocumentLoader(docs_dir=str(tmp_path / "nonexistent"))
    assert loader.load_all() == []

def test_loader_reads_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo bar", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    contents = {d.content for d in docs}
    assert "hello world" in contents and "foo bar" in contents

def test_loader_ignores_non_txt_files(tmp_path):
    (tmp_path / "data.csv").write_text("col1,col2", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    assert len(docs) == 1

def test_loader_skips_blank_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "real.txt").write_text("content", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

def test_chunker_splits_multiline_document():
    docs = [Document(filename="f.txt", content="line one\nline two\nline three")]
    chunks = DocumentChunker().chunk(docs)
    assert len(chunks) == 3
    assert chunks[0].text == "line one"
    assert chunks[2].text == "line three"

def test_chunker_skips_blank_lines():
    docs = [Document(filename="f.txt", content="hello\n\n\nworld")]
    chunks = DocumentChunker().chunk(docs)
    assert len(chunks) == 2

def test_chunker_preserves_source_filename():
    docs = [Document(filename="office.txt", content="WiFi: secret123")]
    chunks = DocumentChunker().chunk(docs)
    assert chunks[0].filename == "office.txt"

def test_chunker_assigns_sequential_indices():
    docs = [Document(filename="f.txt", content="a\nb\nc")]
    chunks = DocumentChunker().chunk(docs)
    assert [c.index for c in chunks] == [0, 1, 2]

def test_chunker_returns_empty_for_no_documents():
    assert DocumentChunker().chunk([]) == []


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------

def test_retriever_returns_empty_for_no_chunks():
    assert DocumentRetriever().retrieve("anything", []) == []

def test_retriever_finds_company_and_neighbour():
    chunks = make_chunks(
        "Company: Conversy AI Office",
        "Location: Kungsgatan 12, Stockholm, Sweden",
        "WiFi Password: Conversy2024",
    )
    result = DocumentRetriever().retrieve("where is the company located", chunks)
    texts = [c.text for c in result]
    assert "Company: Conversy AI Office" in texts
    assert "Location: Kungsgatan 12, Stockholm, Sweden" in texts

def test_retriever_excludes_stopword_only_matches():
    chunks = make_chunks("Q: What is the remote work policy?", "Company: Conversy AI Office")
    result = DocumentRetriever().retrieve("where is the company located", chunks)
    texts = [c.text for c in result]
    assert "Q: What is the remote work policy?" not in texts
    assert "Company: Conversy AI Office" in texts

def test_retriever_returns_empty_for_zero_overlap():
    chunks = make_chunks("quantum physics", "black holes")
    result = DocumentRetriever().retrieve("coffee and tea", chunks)
    assert result == []


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

def test_rag_passes_context_to_llm():
    fake_llm = FakeLLMClient()
    rag = RAGService(llm_client=fake_llm)
    rag._chunks = make_chunks("WiFi Password: Conversy2024", "Meeting Rooms: Alfa")
    rag.generate_answer("WiFi password")
    assert "WiFi Password: Conversy2024" in fake_llm.last_user_message

def test_rag_returns_llm_reply():
    rag = RAGService(llm_client=FakeLLMClient(reply="Conversy2024"))
    rag._chunks = make_chunks("WiFi Password: Conversy2024")
    assert rag.generate_answer("WiFi") == "Conversy2024"

def test_rag_handles_no_chunks_gracefully():
    fake_llm = FakeLLMClient(reply="I don't have that information")
    rag = RAGService(llm_client=fake_llm)
    rag._chunks = []
    result = rag.generate_answer("anything")
    assert result == "I don't have that information"
    assert "No relevant information found" in fake_llm.last_user_message

def test_rag_load_documents_populates_chunks(tmp_path):
    (tmp_path / "facts.txt").write_text("line one\nline two", encoding="utf-8")
    rag = RAGService(
        loader=DocumentLoader(docs_dir=str(tmp_path)),
        llm_client=FakeLLMClient(),
    )
    rag.load_documents()
    assert len(rag._chunks) == 2
