"""DocumentExtractor — reads raw text out of uploaded files (.txt, .pdf, .docx)."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    raise ValueError(f"Unsupported file type: {suffix}")
