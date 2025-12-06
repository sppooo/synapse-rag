import chromadb
import os

# -----------------------------------
# CHROMA PERSISTENT CLIENT
# -----------------------------------
chroma_client = chromadb.PersistentClient(path="db")

# IMPORTANT:
# We do NOT pass an embedding_function here.
# Chroma will use its default (all-MiniLM-L6-v2 ONNX) which is lighter
# than installing sentence-transformers + torch ourselves.
collection = chroma_client.get_or_create_collection(
    name="synapse_rag"
)


# -----------------------------------
# DOCUMENT INGESTION
# -----------------------------------
def load_documents(folder: str = "documents"):
    docs = []

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

        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"{file}_{idx}",
                "text": chunk,
                "source": file
            })

    print(f"[DEBUG] Total chunks prepared: {len(docs)}")
    return docs


def ingest_documents():
    docs = load_documents()

    if not docs:
        print("[INFO] No documents found to ingest.")
        return

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [{"source": d["source"]} for d in docs]

    print(f"[INFO] Adding {len(texts)} chunks to Chroma (Chroma will embed them).")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[INFO] Ingested {len(texts)} chunks into ChromaDB.")


# -----------------------------------
# QUERY FUNCTION (USED BY /search)
# -----------------------------------
def query_chunks(query_text: str, top_k: int = 5):
    if not query_text.strip():
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )
    except Exception as e:
        print("[ERROR] Chroma query failed:", e)
        return []

    chunks = []

    # Defensive checks: Chroma returns lists-of-lists
    docs_list = results.get("documents") or []
    metas_list = results.get("metadatas") or []

    if not docs_list or not docs_list[0]:
        print("[INFO] No documents returned from Chroma.")
        return []

    for i in range(len(docs_list[0])):
        text = docs_list[0][i]
        meta = metas_list[0][i] if metas_list and metas_list[0] else {}
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown")
        })

    return chunks


if __name__ == "__main__":
    count = collection.count()
    print(f"[INFO] Collection currently has {count} records.")

    if count == 0:
        print("[INFO] No records found. Ingesting documents...")
        ingest_documents()
    else:
        print("[INFO] ChromaDB already initialized with data. Skipping ingestion.")
