"""
Tests for ChatService.

ChatService has two dependencies (RAGService, MemoryService).
Both are replaced with fakes so this test suite is fast and isolated.
"""

from app.services.chat_service import ChatService


class FakeRAG:
    def __init__(self, reply: str = "mocked answer"):
        self.reply = reply
        self.received_question = None

    def generate_answer(self, question: str) -> str:
        self.received_question = question
        return self.reply


class FakeMemory:
    def __init__(self):
        self.saved = []

    def save(self, role: str, message: str) -> None:
        self.saved.append((role, message))

    def get_all(self) -> list:
        return list(self.saved)

    def clear(self) -> None:
        self.saved.clear()


def test_handle_chat_returns_rag_response():
    # Arrange
    chat = ChatService(FakeRAG(reply="Paris"), FakeMemory())
    # Act
    result = chat.handle_chat("What is the capital of France?")
    # Assert
    assert result == "Paris"


def test_handle_chat_saves_user_message_before_assistant():
    # Arrange
    rag = FakeRAG()
    memory = FakeMemory()
    chat = ChatService(rag, memory)
    # Act
    chat.handle_chat("hello")
    # Assert
    history = memory.get_all()
    assert history[0] == ("user", "hello")
    assert history[1] == ("assistant", "mocked answer")


def test_handle_chat_saves_exactly_two_messages_per_turn():
    # Arrange
    chat = ChatService(FakeRAG(), FakeMemory())
    # Act
    chat.handle_chat("hi")
    # Assert
    assert len(chat.memory.get_all()) == 2


def test_handle_chat_passes_user_input_to_rag():
    # Arrange
    rag = FakeRAG()
    chat = ChatService(rag, FakeMemory())
    # Act
    chat.handle_chat("What is RAG?")
    # Assert
    assert rag.received_question == "What is RAG?"


def test_multiple_turns_accumulate_in_memory():
    # Arrange
    rag = FakeRAG()
    memory = FakeMemory()
    chat = ChatService(rag, memory)
    # Act
    chat.handle_chat("first question")
    chat.handle_chat("second question")
    # Assert
    assert len(memory.get_all()) == 4


def test_handle_chat_with_empty_string_still_calls_rag():
    # Arrange
    rag = FakeRAG(reply="fallback")
    chat = ChatService(rag, FakeMemory())
    # Act
    result = chat.handle_chat("")
    # Assert
    assert result == "fallback"
    assert rag.received_question == ""
 