# FastAPI-RAG-server-with-ChromaDB

Instructions for Running the Application

Setting up your environment with required dependencies:

bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn chromadb sentence-transformers

Start the FastAPI server:

bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
