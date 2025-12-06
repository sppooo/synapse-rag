from fastapi import FastAPI
from pydantic import BaseModel
from chroma_setup import query_chunks, ingest_documents
import os

app = FastAPI(
    title="Synapse RAG Backend",
    description="RAG endpoint powering n8n Agentic Workflow",
    version="1.0.1"
)


# -----------------------------------
# INITIAL INGEST (FIRST RUN)
# -----------------------------------
# If db is empty or missing → ingest documents from /documents
if not os.path.exists("db") or len(os.listdir("db")) == 0:
    print("[BOOT] No Chroma DB found. Ingesting documents...")
    ingest_documents()
else:
    print("[BOOT] Existing Chroma DB found. Skipping ingestion.")


# -----------------------------------
# REQUEST MODEL
# -----------------------------------
class SearchQuery(BaseModel):
    query: str
    subject: str | None = ""  # subject not used yet but kept for future filters


# -----------------------------------
# RAG SEARCH ENDPOINT
# -----------------------------------
@app.post("/search")
def search_rag(data: SearchQuery):
    """
    Accepts a text query (and optional subject).
    Returns top-k relevant chunks from ChromaDB.
    """
    print(f"[REQUEST] /search → {data.query!r}")

    chunks = query_chunks(data.query, top_k=5)

    return {
        "query": data.query,
        "subject": data.subject,
        "chunks": chunks
    }


# -----------------------------------
# ROOT HEALTHCHECK
# -----------------------------------
@app.get("/")
def home():
    return {
        "message": "Synapse RAG backend is running ✅",
        "hint": "POST to /search with { 'query': 'your question', 'subject': 'optional' }"
    }
