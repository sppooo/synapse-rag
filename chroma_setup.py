import chromadb
import os
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import docx2txt
import docx

# -----------------------------------
# CHROMA PERSISTENT CLIENT
# -----------------------------------
chroma_client = chromadb.PersistentClient(path="db")

collection = chroma_client.get_or_create_collection(
    name="synapse_rag_v3"
)

SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_DOCX = {".docx"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}


# -----------------------------------
# OCR helper for images
# -----------------------------------
def ocr_image(path: str) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[WARN] OCR failed for image {path}: {e}")
        return ""


# -----------------------------------
# PDF Processing (text + OCR fallback)
# -----------------------------------
def extract_text_from_pdf(path: str) -> str:
    text_output = []

    try:
        reader = PdfReader(path)
        # Extract text normally
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text_output.append(extracted)
    except Exception as e:
        print(f"[PDF WARN] Could not read PDF pages normally: {e}")

    # If PDF is scanned → OCR its pages
    try:
        images = convert_from_path(path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text_output.append(ocr_text)
    except Exception as e:
        print(f"[PDF WARN] Could not OCR PDF pages: {e}")

    final_text = "\n\n".join(text_output)
    return final_text


# -----------------------------------
# DOCX Processing (text + images)
# -----------------------------------
def extract_text_from_docx(path: str) -> str:
    text_output = []

    try:
        # Extract regular text
        text_output.append(docx2txt.process(path))
    except Exception as e:
        print(f"[DOCX WARN] Could not extract regular text: {e}")

    # Extract text from embedded images
    try:
        document = docx.Document(path)
        media_dir = os.path.join("tmp_docx_images")
        os.makedirs(media_dir, exist_ok=True)

        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                img_bytes = rel.target_part.blob
                img_path = os.path.join(media_dir, rel.target_ref.replace("/", "_"))

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                ocr_text = ocr_image(img_path)
                text_output.append(ocr_text)

    except Exception as e:
        print(f"[DOCX WARN] Could not OCR docx images: {e}")

    return "\n\n".join(text_output)


# -----------------------------------
# DOCUMENT INGESTION
# -----------------------------------
def load_documents(folder="documents"):
    docs = []

    if not os.path.exists(folder):
        print(f"[WARN] Folder '{folder}' not found.")
        return docs

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(file)[1].lower()
        text = ""

        # TEXT FILES
        if ext in SUPPORTED_TEXT:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except:
                print(f"[WARN] Cannot read {file}")

        # PDF FILES
        elif ext in SUPPORTED_PDF:
            print(f"[INFO] Processing PDF: {file}")
            text = extract_text_from_pdf(path)

        # DOCX FILES
        elif ext in SUPPORTED_DOCX:
            print(f"[INFO] Processing DOCX: {file}")
            text = extract_text_from_docx(path)

        # IMAGES
        elif ext in SUPPORTED_IMAGES:
            print(f"[INFO] OCR on image: {file}")
            text = ocr_image(path)

        else:
            print(f"[SKIP] Unsupported: {file}")
            continue

        if not text.strip():
            print(f"[WARN] No text from {file}, skipping.")
            continue

        # chunking
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"{file}_{idx}",
                "text": chunk,
                "source": file
            })

    return docs


def ingest_documents():
    docs = load_documents()

    if not docs:
        print("[INFO] Nothing to ingest.")
        return

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs]
    )

    print(f"[INFO] Ingested {len(docs)} chunks.")


# -----------------------------------
# QUERY FUNCTION
# -----------------------------------
def query_chunks(query_text: str, top_k=5):
    if not query_text.strip():
        return []

    try:
        res = collection.query(query_texts=[query_text], n_results=top_k)
    except Exception as e:
        print("[ERROR] Query failed:", e)
        return []

    docs_list = res.get("documents", [[]])[0]
    metas_list = res.get("metadatas", [[]])[0]

    return [
        {
            "text": docs_list[i],
            "source": metas_list[i].get("source", "unknown")
        }
        for i in range(len(docs_list))
    ]


if __name__ == "__main__":
    if collection.count() == 0:
        print("[INFO] No data found, ingesting...")
        ingest_documents()
    else:
        print("[INFO] Data already exists.")
