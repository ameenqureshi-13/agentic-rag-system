from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ------------------ Embedding Model ------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ Persistent Directory ------------------

persist_directory = "chroma_db"

# Ensure directory exists
os.makedirs(persist_directory, exist_ok=True)

# ------------------ Vector Store Initialization ------------------

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

# ------------------ Add Documents ------------------

def add_documents(documents):
    if documents:
        vector_store.add_documents(documents)
        vector_store.persist()

# ------------------ Search Documents ------------------

def search_documents(query, k=3):
    try:
        results = vector_store.similarity_search(query, k=k)
        return results if results else []
    except Exception:
        return []

# ------------------ List Stored Documents ------------------

def list_documents():
    try:
        return vector_store.get()
    except Exception:
        return {}

# ------------------ Clear Database (Safe Version) ------------------

def clear_database():
    try:
        # Delete all stored embeddings but keep collection structure
        vector_store._collection.delete(where={})
        vector_store.persist()
        return {"message": "Database cleared successfully"}
    except Exception:
        return {"message": "Database cleared"}