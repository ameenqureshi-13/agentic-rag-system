from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import os

from .vector_store import add_documents


def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    return loader.load()


def process_and_store(file_path: str):
    # Load document (LangChain Document objects)
    documents = load_document(file_path)

    # Semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Add standardized metadata
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(file_path)

        # Ensure page exists (PDF loader includes page automatically)
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1

        # Add upload timestamp (ISO format)
        chunk.metadata["timestamp"] = datetime.utcnow().isoformat()

    # Store in vector DB
    add_documents(chunks)

    return {
        "status": "success",
        "chunks_created": len(chunks),
        "file": os.path.basename(file_path),
        "timestamp": datetime.utcnow().isoformat()
    }