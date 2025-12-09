# main.py

import base64
import binascii
import traceback
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chroma_setup import (
    query_chunks,
    ingest_documents,
    ingest_text,
    collection,
)

# Optional text-extraction libs
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import textract  # very broad, but heavier
except Exception:
    textract = None


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(
    title="Synapse RAG Backend",
    description="RAG endpoint powering n8n Agentic Workflow with file ingestion",
    version="1.1.0",
)


# ---------------------------------------------------------
# BOOTSTRAP: ENSURE COLLECTION HAS DATA
# ---------------------------------------------------------
try:
    count = collection.count()
    print(f"[BOOT] Chroma collection has {count} records.")
    if count == 0:
        print("[BOOT] No records found. Ingesting seed /documents...")
        ingest_documents()
    else:
        print("[BOOT] Existing data found. Skipping seed ingestion.")
except Exception as e:
    print("[BOOT ERROR] Could not check or ingest Chroma collection:", e)
    traceback.print_exc()


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------
class SearchQuery(BaseModel):
    query: str
    subject: Optional[str] = ""
    user_id: Optional[str] = "global"
    domain: Optional[str] = "general"


class IngestTextPayload(BaseModel):
    text: str
    source_name: str = "student_upload"
    user_id: Optional[str] = "global"
    domain: Optional[str] = "general"


class IngestFilePayload(BaseModel):
    base64_data: str
    filename: str = "upload"
    mime_type: Optional[str] = None
    user_id: Optional[str] = "global"
    domain: Optional[str] = "general"


# ---------------------------------------------------------
# TEXT EXTRACTION HELPERS
# ---------------------------------------------------------
def _extract_pdf(data: bytes) -> str:
    if not PdfReader:
        raise RuntimeError("PyPDF2 is not installed in this environment.")

    reader = PdfReader(BytesIO(data))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages_text).strip()


def _extract_docx(data: bytes) -> str:
    if not docx:
        raise RuntimeError("python-docx is not installed in this environment.")
    document = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()


def _extract_generic(data: bytes, filename: str) -> str:
    # Last-resort extraction: try textract if available, otherwise treat as UTF-8 text
    if textract:
        try:
            txt = textract.process(filename, input_data=data)
            return txt.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            print("[WARN] textract failed:", e)

    # Fallback: assume text
    try:
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def extract_text_from_bytes(data: bytes, filename: str, mime_type: Optional[str]) -> str:
    """
    Decide how to extract text based on extension / mime type.
    """
    fname_lower = filename.lower()

    try:
        if fname_lower.endswith(".pdf") or (mime_type and "pdf" in mime_type):
            return _extract_pdf(data)

        if fname_lower.endswith(".docx") or (mime_type and "word" in mime_type):
            return _extract_docx(data)

        if fname_lower.endswith(".txt") or (mime_type and "text" in mime_type):
            return data.decode("utf-8", errors="ignore").strip()

        # Default catch-all
        return _extract_generic(data, filename)
    except Exception as e:
        print("[ERROR] Failed to extract text from file:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Could not extract text from file.")


# ---------------------------------------------------------
# RAG SEARCH ENDPOINT (USED BY POSTMAN + n8n)
# ---------------------------------------------------------
@app.post("/search")
async def search_rag(data: SearchQuery):
    """
    Takes a query string (+ optional user_id/domain) and returns
    top-k relevant chunks from ChromaDB.
    """
    try:
        print(f"[REQUEST] /search -> {data.query!r} | user_id={data.user_id} | domain={data.domain}")
        chunks = query_chunks(
            data.query,
            top_k=5,
            user_id=data.user_id,
            domain=data.domain,
        )

        return {
            "query": data.query,
            "subject": data.subject,
            "user_id": data.user_id,
            "domain": data.domain,
            "chunks": chunks,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR] Unhandled error in /search:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# PLAIN TEXT INGESTION ENDPOINT
# ---------------------------------------------------------
@app.post("/ingest-text")
async def ingest_text_endpoint(payload: IngestTextPayload):
    """
    Ingest plain text (already extracted) into Chroma.
    Used for seed data or if n8n already has the text.
    """
    try:
        n = ingest_text(
            payload.text,
            source_name=payload.source_name,
            user_id=payload.user_id or "global",
            domain=payload.domain or "general",
        )
        return {
            "status": "ok",
            "chunks_added": n,
            "source_name": payload.source_name,
            "user_id": payload.user_id,
            "domain": payload.domain,
        }
    except Exception as e:
        print("[ERROR] /ingest-text failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# BASE64 FILE INGESTION ENDPOINT
# ---------------------------------------------------------
@app.post("/ingest-file")
async def ingest_file_endpoint(payload: IngestFilePayload):
    """
    Accepts a file as Base64 from n8n, extracts text server-side,
    chunks it and stores into Chroma with user_id + domain metadata.
    """
    try:
        try:
            file_bytes = base64.b64decode(payload.base64_data)
        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid Base64 data.")

        text = extract_text_from_bytes(
            data=file_bytes,
            filename=payload.filename,
            mime_type=payload.mime_type,
        )

        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from file.")

        n = ingest_text(
            text,
            source_name=payload.filename,
            user_id=payload.user_id or "global",
            domain=payload.domain or "general",
        )

        return {
            "status": "ok",
            "filename": payload.filename,
            "user_id": payload.user_id,
            "domain": payload.domain,
            "chunks_added": n,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR] /ingest-file failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ROOT HEALTHCHECK
# ---------------------------------------------------------
@app.get("/")
async def home():
    return {
        "message": "Synapse RAG backend is running ✅",
        "hint_search": "POST to /search with { 'query': 'your question', 'user_id': 'student_01', 'domain': 'general' }",
        "hint_ingest_text": "POST to /ingest-text with { 'text': '...', 'source_name': 'doc.txt', 'user_id': 'student_01' }",
        "hint_ingest_file": "POST to /ingest-file with { 'base64_data': '...', 'filename': 'doc.pdf', 'user_id': 'student_01' }",
    }
