"""
ChatService — coordinates RAG and memory for each user turn.

Follows single-responsibility: this class only wires RAGService
and MemoryService together. It does not contain retrieval or
storage logic itself.
"""

from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService


class ChatService:
    def __init__(self, rag_service: RAGService, memory_service: MemoryService):
        self.rag = rag_service
        self.memory = memory_service

    def handle_chat(self, user_input: str) -> str:
        """Process one user message and return the assistant reply."""
        self.memory.save("user", user_input)
        response = self.rag.generate_answer(user_input)
        self.memory.save("assistant", response)
        return response
