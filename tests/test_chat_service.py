"""
Tests for ChatService.

ChatService has two dependencies (RAGService, MemoryService).
Both are replaced with fakes so this test suite is fast and isolated.
"""

import pytest
from app.services.chat_service import ChatService


class FakeRAG:
    def __init__(self, reply: str = "mocked answer"):
        self.reply = reply
        self.received_question = None

    async def generate_answer(self, question: str) -> str:
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


@pytest.mark.asyncio
async def test_handle_chat_returns_rag_response():
    chat = ChatService(FakeRAG(reply="Paris"), FakeMemory())
    result = await chat.handle_chat("What is the capital of France?")
    assert result == "Paris"


@pytest.mark.asyncio
async def test_handle_chat_saves_user_message_before_assistant():
    rag = FakeRAG()
    memory = FakeMemory()
    chat = ChatService(rag, memory)
    await chat.handle_chat("hello")
    history = memory.get_all()
    assert history[0] == ("user", "hello")
    assert history[1] == ("assistant", "mocked answer")


@pytest.mark.asyncio
async def test_handle_chat_saves_exactly_two_messages_per_turn():
    chat = ChatService(FakeRAG(), FakeMemory())
    await chat.handle_chat("hi")
    assert len(chat.memory.get_all()) == 2


@pytest.mark.asyncio
async def test_handle_chat_passes_user_input_to_rag():
    rag = FakeRAG()
    chat = ChatService(rag, FakeMemory())
    await chat.handle_chat("What is RAG?")
    assert rag.received_question == "What is RAG?"


@pytest.mark.asyncio
async def test_multiple_turns_accumulate_in_memory():
    rag = FakeRAG()
    memory = FakeMemory()
    chat = ChatService(rag, memory)
    await chat.handle_chat("first question")
    await chat.handle_chat("second question")
    assert len(memory.get_all()) == 4


@pytest.mark.asyncio
async def test_handle_chat_with_empty_string_still_calls_rag():
    rag = FakeRAG(reply="fallback")
    chat = ChatService(rag, FakeMemory())
    result = await chat.handle_chat("")
    assert result == "fallback"
    assert rag.received_question == ""
