"""
Peek inside the local ChromaDB vector database — the equivalent of
running `SELECT * FROM table` in phpMyAdmin, but for the vector store.

Usage:
    python scripts/inspect_vectordb.py                # list everything
    python scripts/inspect_vectordb.py --file office.txt   # filter by file
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app.services.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Only show chunks from this filename")
    args = parser.parse_args()

    store = VectorStore()
    where = {"filename": args.file} if args.file else None
    data = store._collection.get(where=where, include=["documents", "metadatas", "embeddings"])

    ids = data["ids"]
    if not ids:
        print("No chunks found.")
        return

    print(f"{len(ids)} chunk(s) in data/vectordb/\n")
    for i, chunk_id in enumerate(ids):
        meta = data["metadatas"][i]
        text = data["documents"][i]
        vector = data["embeddings"][i]
        print(f"[{chunk_id}] file={meta['filename']} index={meta['index']}")
        print(f"  text: {text}")
        print(f"  vector: {len(vector)} dimensions, first 5 values = {[round(v, 4) for v in vector[:5]]}")
        print()


if __name__ == "__main__":
    main()
