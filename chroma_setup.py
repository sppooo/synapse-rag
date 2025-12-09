# chroma_setup.py

import os
import chromadb
from typing import List, Dict, Optional

# ---------------------------------------------------------
# CHROMA PERSISTENT CLIENT
# ---------------------------------------------------------
# Chroma will store its data in ./db (persisted between runs)
chroma_client = chromadb.PersistentClient(path="db")

# We let Chroma use its default ONNX embedding (all-MiniLM-L6-v2)
collection = chroma_client.get_or_create_collection(
    name="synapse_rag_v2"
)


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple sliding-window chunker.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # move window with overlap
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ---------------------------------------------------------
# DOCUMENT INGESTION HELPERS
# ---------------------------------------------------------
def ingest_text(
    text: str,
    source_name: str = "unknown_source",
    user_id: str = "global",
    domain: str = "general",
) -> int:
    """
    Chunk a raw text string and add it to ChromaDB with metadata.
    Returns number of chunks ingested.
    """
    text = text or ""
    text = text.strip()
    if not text:
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids: List[str] = []
    metadatas: List[Dict] = []

    for idx, chunk in enumerate(chunks):
        ids.append(f"{user_id}_{source_name}_{idx}")
        metadatas.append(
            {
                "source": source_name,
                "user_id": user_id,
                "domain": domain,
            }
        )

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def load_documents(folder: str = "documents") -> List[Dict]:
    """
    Old helper for seeding local /documents folder on boot.
    Keeps things simple: treat everything as UTF-8 text file.
    """
    docs: List[Dict] = []

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Looking for folder:", folder)

    if not os.path.exists(folder):
        print(f"[WARN] Documents folder '{folder}' not found.")
        return docs

    files = os.listdir(folder)
    print("[DEBUG] Files in documents:", files)

    for file in files:
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            print(f"[DEBUG] Skipping non-file: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue

        docs.append(
            {
                "text": text,
                "source_name": file,
                "user_id": "seed_docs",
                "domain": "general",
            }
        )

    print(f"[DEBUG] Total seed docs prepared: {len(docs)}")
    return docs


def ingest_documents():
    """
    Seed /documents folder into Chroma.
    """
    docs = load_documents()

    if not docs:
        print("[INFO] No seed documents found to ingest.")
        return

    total_chunks = 0
    for d in docs:
        n = ingest_text(d["text"], d["source_name"], d["user_id"], d["domain"])
        total_chunks += n

    print(f"[INFO] Seed ingestion complete. Total chunks: {total_chunks}")


# ---------------------------------------------------------
# QUERY FUNCTION (USED BY /search)
# ---------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Simple, safe sliding-window chunker that never loops infinitely.
    """
    text = text or ""
    text_len = len(text)
    if text_len == 0:
        return []

    # step size ensures progress
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    for start in range(0, text_len, step):
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    count = collection.count()
    print(f"[INFO] Collection currently has {count} records.")

    if count == 0:
        print("[INFO] No records found. Ingesting seed documents...")
        ingest_documents()
    else:
        print("[INFO] ChromaDB already initialized with data. Skipping seed ingestion.")
