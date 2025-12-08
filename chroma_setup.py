# chroma_setup.py
import os
from typing import List, Dict, Any, Optional

import chromadb

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# -----------------------------
# CLIENT + COLLECTION
# -----------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="synapse_rag_v2")


# -----------------------------
# UTIL: CHUNKING
# -----------------------------
def _split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Simple sliding-window splitter."""
    text = text or ""
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            break

        # move window with overlap
        start = end - overlap

    return chunks


# -----------------------------
# UTIL: FILE READERS
# -----------------------------
def _read_plain_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """Read PDF using pypdf if installed."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        print("[WARN] pypdf not installed; cannot read PDF:", path)
        return ""

    try:
        reader = PdfReader(path)
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception as e:
        print(f"[WARN] Could not read PDF {path}: {e}")
        return ""


def _read_docx(path: str) -> str:
    """Read DOCX using python-docx if installed."""
    try:
        import docx  # type: ignore
    except ImportError:
        print("[WARN] python-docx not installed; cannot read DOCX:", path)
        return ""

    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[WARN] Could not read DOCX {path}: {e}")
        return ""


# -----------------------------
# BULK LOAD FROM FOLDER
# -----------------------------
def load_documents_from_folder(
    folder: str = DOCUMENTS_FOLDER,
    user_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan a folder for txt/md/pdf/docx and return prepared chunk dicts.
    """
    docs: List[Dict[str, Any]] = []

    if not os.path.exists(folder):
        print(f"[WARN] Documents folder '{folder}' not found.")
        return docs

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext in {".txt", ".md"}:
            text = _read_plain_text(path)
        elif ext == ".pdf":
            text = _read_pdf(path)
        elif ext == ".docx":
            text = _read_docx(path)
        else:
            print("[DEBUG] Skipping unsupported file type:", path)
            continue

        if not text.strip():
            continue

        chunks = _split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{filename}_{idx}",
                    "text": chunk,
                    "source": filename,
                    "user_id": user_id or "global",
                    "domain": domain or "general",
                }
            )

    print(f"[DEBUG] Prepared {len(docs)} chunks from folder {folder}")
    return docs


# -----------------------------
# INGEST HELPERS
# -----------------------------
def ingest_text(
    text: str,
    source_name: str = "uploaded",
    user_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> int:
    """
    Ingest arbitrary text (e.g. from uploads, APIs) into Chroma.
    Returns number of chunks added.
    """
    text = (text or "").strip()
    if not text:
        return 0

    chunks = _split_text(text)
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    base = source_name.replace(" ", "_")

    for idx, chunk in enumerate(chunks):
        ids.append(f"{base}_{idx}")
        docs.append(chunk)
        metas.append(
            {
                "source": source_name,
                "user_id": user_id or "global",
                "domain": domain or "general",
            }
        )

    if not docs:
        return 0

    collection.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def ingest_documents_from_folder() -> int:
    """
    One-shot bulk ingest from the DOCUMENTS_FOLDER.
    """
    docs = load_documents_from_folder()
    if not docs:
        return 0

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [
        {"source": d["source"], "user_id": d["user_id"], "domain": d["domain"]}
        for d in docs
    ]

    collection.add(ids=ids, documents=texts, metadatas=metas)
    return len(texts)


# -----------------------------
# QUERY
# -----------------------------
def query_chunks(
    query_text: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector store, optionally filtered by user_id + domain.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return []

    where: Dict[str, Any] = {}
    if user_id:
        where["user_id"] = user_id
    if domain:
        where["domain"] = domain

    try:
        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
    except Exception as e:
        print("[ERROR] Chroma query failed:", e)
        return []

    docs_list = results.get("documents") or []
    metas_list = results.get("metadatas") or []

    if not docs_list or not docs_list[0]:
        return []

    chunks: List[Dict[str, Any]] = []
    for i, text in enumerate(docs_list[0]):
        meta = metas_list[0][i] if metas_list and metas_list[0] else {}
        chunks.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "user_id": meta.get("user_id"),
                "domain": meta.get("domain"),
            }
        )

    return chunks


# -----------------------------
# DEBUG ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    count = collection.count()
    print("[INFO] Current collection size:", count)

    if count == 0:
        print("[INFO] No records found, ingesting from folder...")
        added = ingest_documents_from_folder()
        print(f"[INFO] Ingested {added} chunks.")
