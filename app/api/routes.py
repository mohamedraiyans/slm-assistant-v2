"""
API routes for the SLM Assistant.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.chat_service import ChatService
from app.services.document_extractor import SUPPORTED_EXTENSIONS
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api")

_rag_service = RAGService()
_rag_service.load_documents()
_memory_service = MemoryService()
_chat_service = ChatService(_rag_service, _memory_service)


def get_chat_service() -> ChatService:
    return _chat_service


def get_memory_service() -> MemoryService:
    return _memory_service


def get_rag_service() -> RAGService:
    return _rag_service


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
    result = chat_service.handle_chat(request.message)
    return {"response": result["answer"], "sources": result["sources"]}


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


@router.get("/documents")
def list_documents(rag_service: RAGService = Depends(get_rag_service)):
    return {"documents": rag_service.list_documents()}


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
):
    filename = Path(file.filename).name  # strip any path components
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    rag_service.docs_dir.mkdir(parents=True, exist_ok=True)
    dest = rag_service.docs_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        chunk_count = rag_service.index_file(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    return {"filename": filename, "chunks_indexed": chunk_count}


@router.delete("/documents/{filename}")
def delete_document(
    filename: str,
    rag_service: RAGService = Depends(get_rag_service),
):
    dest = rag_service.docs_dir / filename
    if dest.exists():
        dest.unlink()
    rag_service.remove_document(filename)
    return {"message": f"Deleted {filename}"}
