# main.py
from typing import Optional, List, Dict, Any

import os
import traceback

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chroma_setup import (
    query_chunks,
    ingest_documents_from_folder,
    ingest_text,
    collection,
)

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(
    title="Synapse RAG Backend",
    description="Hybrid RAG backend for Synapse (local vector DB + web search)",
    version="2.0.0",
)


# ---------------------------------------------------------
# BOOTSTRAP: ENSURE COLLECTION HAS DATA
# ---------------------------------------------------------
try:
    count = collection.count()
    print(f"[BOOT] Chroma collection has {count} records.")

    if count == 0:
        print("[BOOT] No records found. Ingesting from documents/ ...")
        added = ingest_documents_from_folder()
        print(f"[BOOT] Ingested {added} chunks from folder.")
    else:
        print("[BOOT] Existing data found. Skipping ingestion.")
except Exception as e:
    print("[BOOT ERROR] Could not check or ingest Chroma collection:", e)
    traceback.print_exc()


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------
class SearchQuery(BaseModel):
    query: str
    subject: Optional[str] = ""
    domain: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = 5
    use_web: bool = True  # allow n8n to turn web search on/off if needed


class IngestRequest(BaseModel):
    text: str
    source_name: str = "uploaded"
    user_id: Optional[str] = None
    domain: Optional[str] = None


# ---------------------------------------------------------
# WEB SEARCH HELPER (e.g. Tavily)
# ---------------------------------------------------------
def web_search_chunks(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call a web search API to get up-to-date articles / pages
    and normalize them into [ {text, source, ...}, ... ] chunks.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # If no key configured, quietly skip web search
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        chunks: List[Dict[str, Any]] = []
        for result in data.get("results", []):
            content = result.get("content") or ""
            url = result.get("url") or ""
            if not content.strip():
                continue

            chunks.append(
                {
                    "text": content,
                    "source": f"web:{url}",
                    "user_id": "web",
                    "domain": "general",
                }
            )

        return chunks

    except Exception as e:
        print("[WARN] Web search failed:", e)
        return []


# ---------------------------------------------------------
# RAG SEARCH ENDPOINT (USED BY POSTMAN + n8n)
# ---------------------------------------------------------
@app.post("/search")
async def search_rag(data: SearchQuery):
    """
    Takes a query string (+ optional domain, user_id) and returns
    top-k chunks from local Chroma, with optional web fallback.
    """
    try:
        print(f"[REQUEST] /search -> {data.query!r}")

        # 1) Local vector search
        local_chunks = query_chunks(
            data.query,
            top_k=data.top_k,
            user_id=data.user_id,
            domain=data.domain,
        )

        chunks: List[Dict[str, Any]] = list(local_chunks)

        # 2) Web search fallback if local context is weak
        if data.use_web and len(chunks) < max(2, data.top_k // 2):
            web_chunks = web_search_chunks(data.query, max_results=data.top_k)
            chunks.extend(web_chunks)

        return {
            "query": data.query,
            "subject": data.subject,
            "domain": data.domain,
            "user_id": data.user_id,
            "chunks": chunks,
        }

    except Exception as e:
        print("[ERROR] Unhandled error in /search:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# INGESTION ENDPOINT (FOR UPLOAD WORKFLOW)
# ---------------------------------------------------------
@app.post("/ingest-text")
async def ingest_text_endpoint(body: IngestRequest):
    """
    n8n (or any client) sends raw text from a document / API / OCR.
    We chunk it and store in Chroma with user_id + domain metadata.
    """
    try:
        added = ingest_text(
            text=body.text,
            source_name=body.source_name,
            user_id=body.user_id,
            domain=body.domain,
        )
        return {"added_chunks": added}
    except Exception as e:
        print("[ERROR] /ingest-text failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ROOT HEALTHCHECK
# ---------------------------------------------------------
@app.get("/")
async def home():
    return {
        "message": "Synapse RAG backend (v2) is running ✅",
        "hint": "POST to /search with {'query': 'your question', 'domain': 'optional', 'user_id': 'optional'}",
    }
