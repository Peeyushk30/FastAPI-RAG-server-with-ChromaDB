from fastapi import FastAPI, UploadFile, HTTPException
from sentence_transformers import SentenceTransformer
from chromadb.utils import persistent_client
from typing import List

app = FastAPI()

chroma_client = persistent_client(
    client_type="sqlite",
    db_path="./chroma_db"
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/ingest/")
async def ingest_document(files: List[UploadFile]):
    document_texts = []
    document_names = []

    for file in files:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Invalid file format")
        document_texts.append(text)
        document_names.append(file.filename)

    embeddings = model.encode(document_texts, show_progress_bar=False)
    chroma_client.insert({"name": document_names, "embedding": embeddings})
    return {"message": "Documents successfully ingested"}

@app.post("/query/")
async def query_documents(query: str):
    query_embedding = model.encode([query], show_progress_bar=False)[0]
    results = chroma_client.similarity_search(query_embedding, top_k=5)
    response = [{"document_name": res["name"], "score": res["score"]} for res in results]
    return {"results": response}
