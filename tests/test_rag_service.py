"""
Tests for RAGService components.

DocumentChunker and RAGService are tested in isolation. GroqClient and the
real VectorStore (ChromaDB) are never touched — fakes are injected instead.
"""

from app.services.rag_service import Chunk, DocumentChunker, RAGService


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


class FakeVectorStore:
    def __init__(self, matches=None):
        self.matches = matches or []
        self.added_chunks = []
        self.deleted = []

    def add_chunks(self, chunks):
        self.added_chunks.extend(chunks)

    def query(self, question, top_k=5):
        return self.matches

    def has_document(self, filename):
        return any(c.filename == filename for c in self.added_chunks)

    def list_filenames(self):
        return sorted({c.filename for c in self.added_chunks})

    def delete_document(self, filename):
        self.deleted.append(filename)


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

def test_chunker_splits_long_text_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = DocumentChunker(chunk_size=150, overlap=30).chunk("f.txt", text)
    assert len(chunks) > 1

def test_chunker_keeps_short_text_as_one_chunk():
    chunks = DocumentChunker().chunk("f.txt", "just a short sentence")
    assert len(chunks) == 1
    assert chunks[0].text == "just a short sentence"

def test_chunker_preserves_source_filename():
    chunks = DocumentChunker().chunk("office.txt", "WiFi: secret123")
    assert chunks[0].filename == "office.txt"

def test_chunker_assigns_sequential_indices():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = DocumentChunker(chunk_size=150, overlap=30).chunk("f.txt", text)
    assert [c.index for c in chunks] == list(range(len(chunks)))

def test_chunker_returns_empty_for_blank_text():
    assert DocumentChunker().chunk("f.txt", "   ") == []

def test_chunker_overlaps_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = DocumentChunker(chunk_size=150, overlap=30).chunk("f.txt", text)
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] == second_words[29]


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

def test_rag_passes_matched_chunks_to_llm():
    fake_llm = FakeLLMClient()
    matches = [{"filename": "office.txt", "text": "WiFi Password: Conversy2024", "score": 0.9}]
    rag = RAGService(vector_store=FakeVectorStore(matches), llm_client=fake_llm)
    rag.generate_answer("WiFi password")
    assert "WiFi Password: Conversy2024" in fake_llm.last_user_message

def test_rag_returns_answer_and_sources():
    matches = [{"filename": "office.txt", "text": "WiFi Password: Conversy2024", "score": 0.9}]
    rag = RAGService(vector_store=FakeVectorStore(matches), llm_client=FakeLLMClient(reply="Conversy2024"))
    result = rag.generate_answer("WiFi")
    assert result["answer"] == "Conversy2024"
    assert result["sources"] == matches

def test_rag_handles_no_matches_gracefully():
    fake_llm = FakeLLMClient(reply="I don't have that information")
    rag = RAGService(vector_store=FakeVectorStore([]), llm_client=fake_llm)
    result = rag.generate_answer("anything")
    assert result["answer"] == "I don't have that information"
    assert result["sources"] == []
    assert "No relevant information found" in fake_llm.last_user_message

def test_rag_index_file_chunks_and_stores(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text("line one\nline two", encoding="utf-8")
    store = FakeVectorStore()
    rag = RAGService(vector_store=store, llm_client=FakeLLMClient())
    count = rag.index_file(path)
    assert count == 1
    assert store.added_chunks[0].filename == "facts.txt"

def test_rag_load_documents_skips_already_indexed(tmp_path):
    (tmp_path / "facts.txt").write_text("hello world", encoding="utf-8")
    store = FakeVectorStore()
    store.added_chunks.append(Chunk(id="facts.txt::0", filename="facts.txt", text="hello world", index=0))
    rag = RAGService(docs_dir=str(tmp_path), vector_store=store, llm_client=FakeLLMClient())
    rag.load_documents()
    assert len(store.added_chunks) == 1

def test_rag_load_documents_indexes_new_files(tmp_path):
    (tmp_path / "new.txt").write_text("brand new content", encoding="utf-8")
    store = FakeVectorStore()
    rag = RAGService(docs_dir=str(tmp_path), vector_store=store, llm_client=FakeLLMClient())
    rag.load_documents()
    assert store.added_chunks[0].filename == "new.txt"

def test_rag_remove_document_delegates_to_vector_store():
    store = FakeVectorStore()
    rag = RAGService(vector_store=store, llm_client=FakeLLMClient())
    rag.remove_document("old.txt")
    assert store.deleted == ["old.txt"]

def test_rag_list_documents_delegates_to_vector_store():
    store = FakeVectorStore()
    store.added_chunks.append(Chunk(id="a.txt::0", filename="a.txt", text="x", index=0))
    rag = RAGService(vector_store=store, llm_client=FakeLLMClient())
    assert rag.list_documents() == ["a.txt"]
