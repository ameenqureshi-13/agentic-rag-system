from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from .ingest import process_and_store
from .agent import run_agent
from .vector_store import clear_database, list_documents

app = FastAPI()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------ Upload Endpoint ------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_and_store(file_path)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ Query Endpoint ------------------

@app.post("/query")
async def query_agent(payload: dict):
    question = payload.get("question")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    result = run_agent(question)

    return result


# ------------------ Chat Endpoint ------------------

@app.post("/chat")
async def chat(payload: dict):
    question = payload.get("question")
    session_id = payload.get("session_id")

    if not question or not session_id:
        raise HTTPException(status_code=400, detail="Both question and session_id are required")

    result = run_agent(question, session_id)

    return result

# ------------------ List Documents ------------------

@app.get("/documents")
async def documents():
    return list_documents()


# ------------------ Clear Database ------------------

@app.delete("/clear")
async def clear():
    clear_database()
    return {"status": "database cleared"}