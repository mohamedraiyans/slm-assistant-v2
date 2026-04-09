"""
API routes for the SLM Assistant.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/api")

_rag_service = RAGService()
_rag_service.load_documents()
_memory_service = MemoryService()
_chat_service = ChatService(_rag_service, _memory_service)


def get_chat_service() -> ChatService:
    return _chat_service


def get_memory_service() -> MemoryService:
    return _memory_service


class ChatRequest(BaseModel):
    message: str


@router.get("/health")
def health():
    return {"message": "SLM Assistant is running"}


@router.post("/chat")
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    response = chat_service.handle_chat(request.message)
    return {"response": response}


@router.get("/history")
def history(
    memory_service: MemoryService = Depends(get_memory_service),
):
    messages = memory_service.get_all()
    return {"history": [{"role": m.role, "content": m.content} for m in messages]}


@router.post("/history/clear")
def clear_history(
    memory_service: MemoryService = Depends(get_memory_service),
):
    memory_service.clear()
    return {"message": "History cleared"}