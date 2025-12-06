from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chroma_setup import query_chunks, ingest_documents, collection
import traceback

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(
    title="Synapse RAG Backend",
    description="RAG endpoint powering n8n Agentic Workflow",
    version="1.0.2",
)


# ---------------------------------------------------------
# BOOTSTRAP: ENSURE COLLECTION HAS DATA
# ---------------------------------------------------------
# We check how many records are stored in the Chroma collection.
# If it's empty, we ingest from the /documents folder.
try:
    count = collection.count()
    print(f"[BOOT] Chroma collection has {count} records.")
    if count == 0:
        print("[BOOT] No records found. Ingesting documents...")
        ingest_documents()
    else:
        print("[BOOT] Existing data found. Skipping ingestion.")
except Exception as e:
    print("[BOOT ERROR] Could not check or ingest Chroma collection:", e)
    traceback.print_exc()


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------
class SearchQuery(BaseModel):
    query: str
    subject: str | None = ""  # kept for future filtering if you want it


# ---------------------------------------------------------
# RAG SEARCH ENDPOINT (USED BY POSTMAN + n8n)
# ---------------------------------------------------------
@app.post("/search")
async def search_rag(data: SearchQuery):
    """
    Takes a query string (and optional subject) and returns
    top-k relevant chunks from ChromaDB.
    """
    try:
        print(f"[REQUEST] /search -> {data.query!r}")
        chunks = query_chunks(data.query, top_k=5)

        return {
            "query": data.query,
            "subject": data.subject,
            "chunks": chunks,
        }

    except Exception as e:
        # Log full traceback to Railway logs
        print("[ERROR] Unhandled error in /search:", e)
        traceback.print_exc()
        # Return a readable error message to the client
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ROOT HEALTHCHECK
# ---------------------------------------------------------
@app.get("/")
async def home():
    return {
        "message": "Synapse RAG backend is running ✅",
        "hint": "POST to /search with { 'query': 'your question', 'subject': 'optional' }",
    }
