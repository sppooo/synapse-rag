import os
import chromadb
from typing import List, Dict

# -----------------------------------
# CHROMA PERSISTENT CLIENT
# -----------------------------------
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="synapse_rag_v2")

# ---------- Helpers ----------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple overlapped text chunker.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)

    # Safety: truncate extremely long text to avoid MemoryError
    max_chars = 200_000  # 200k characters ~ 40–50 pages of text
    if n > max_chars:
        text = text[:max_chars]
        n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if chunk_size - overlap <= 0:
            break

    return chunks


def ingest_text(text: str, source_name: str, user_id: str = "seed", domain: str = "general") -> int:
    """
    Add a single logical document (text) into Chroma as multiple chunks.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    base_id = f"{user_id}_{source_name}"
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "user_id": user_id,
            "domain": domain,
        }
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)

# ---------- SEED DOCUMENTS ----------

def load_seed_documents(folder: str = "documents") -> List[Dict]:
    docs: List[Dict] = []

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Looking for folder:", folder)

    if not os.path.exists(folder):
        print(f"[WARN] Seed documents folder '{folder}' not found.")
        return docs

    files = os.listdir(folder)
    print("[DEBUG] Files in documents:", files)

    for file in files:
        path = os.path.join(folder, file)

        # Only regular files
        if not os.path.isfile(path):
            continue

        # Only read *text* files here, skip PDFs and others
        lower = file.lower()
        if not lower.endswith((".txt", ".md")):
            print(f"[INFO] Skipping non-text seed file: {file}")
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
                "user_id": "seed",
                "domain": "general",
            }
        )

    print(f"[DEBUG] Total seed docs prepared: {len(docs)}")
    return docs


def ingest_seed_documents():
    docs = load_seed_documents()
    if not docs:
        print("[INFO] No seed documents to ingest.")
        return

    total_chunks = 0
    for d in docs:
        n = ingest_text(d["text"], d["source_name"], d["user_id"], d["domain"])
        print(f"[INFO] Seed doc '{d['source_name']}' -> {n} chunks.")
        total_chunks += n

    print(f"[INFO] Seed ingestion complete. Total chunks: {total_chunks}")


if __name__ == "__main__":
    count = collection.count()
    print(f"[INFO] Collection currently has {count} records.")

    if count == 0:
        print("[INFO] No records found. Ingesting seed documents...")
        ingest_seed_documents()
    else:
        print("[INFO] ChromaDB already initialized with data. Skipping ingestion.")
