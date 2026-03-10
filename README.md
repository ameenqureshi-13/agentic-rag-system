# Agentic RAG System (Full Stack AI Assignment)

## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system using FastAPI, ChromaDB, and Transformers.

The system supports:

- Uploading documents (PDF / TXT)
- Semantic chunking and vector storage
- Intelligent routing (retrieval vs internal knowledge)
- Structured source attribution
- Transparent reasoning traces
- Multi-turn chat with session ID
- REST API exposure
- Docker-based packaging

---

## Architecture

User → FastAPI API Layer → Agent Decision Layer  
→ Document Search (Chroma Vector DB)  
→ Direct LLM (GPT-2 via Transformers)

### Components

- **FastAPI** – REST API layer
- **Sentence Transformers** – Embedding generation
- **ChromaDB** – Vector database
- **Transformers (GPT-2)** – Internal knowledge generation
- **Custom Agent Layer** – Decision logic & reasoning trace

---

## API Endpoints

### 1. Upload Document
**POST** `/upload`

Upload a PDF or TXT file for ingestion and embedding.

---

### 2. Query
**POST** `/query`

Example Request:

```json
{
  "question": "What did our Q3 report say about revenue?"
}
```

Example Response:

```json
{
  "answer": "Revenue increased by 15 percent...",
  "sources": [
    {
      "document": "q3.txt",
      "chunk": "Our Q3 financial report states that revenue increased by 15 percent..."
    }
  ],
  "reasoning_trace": [
    {
      "step": 1,
      "thought": "Evaluate whether retrieval is required."
    },
    {
      "step": 2,
      "action": "document_search",
      "reason": "Question refers to document-specific content."
    }
  ],
  "retrieval_used": true
}
```

---

### 3. Chat (Multi-turn)
**POST** `/chat`

Example:

```json
{
  "question": "Summarize the Q3 report",
  "session_id": "session1"
}
```

---

### 4. List Documents
**GET** `/documents`

Returns stored document metadata.

---

### 5. Clear Database
**DELETE** `/clear`

Clears vector database.

---

## Agentic Workflow

1. Receive user query
2. Analyze if retrieval is required
3. If document-specific → query vector DB
4. If general knowledge → use internal LLM
5. Generate final answer
6. Return reasoning trace and sources

---

## How to Run Locally

### Step 1 – Create Virtual Environment

```bash
python -m venv venv
```

Activate (Windows):

```bash
venv\Scripts\activate
```

### Step 2 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Run Server

```bash
uvicorn app.main:app --reload
```

Access Swagger UI:

http://127.0.0.1:8000/docs

---

## Docker Support

Build:

```bash
docker build -t agentic-rag .
```

Run:

```bash
docker run -p 8000:8000 agentic-rag
```

---

## Design Decisions

- Modular architecture for maintainability
- Clear separation between ingestion, vector store, tools, and agent
- Structured reasoning trace for transparency
- Session-based chat support
- Persistent vector database

---

## Future Improvements

- True ReAct LLM-based tool calling
- Web search integration
- Redis-based session storage
- Authentication & rate limiting
- Frontend interface

---

## Key Design Explanation

This system follows a modular Agentic RAG architecture.

The ingestion layer handles document parsing, chunking, and embedding generation. Chunks are stored in a persistent Chroma vector database.

The agent layer implements explicit decision logic to determine whether a query requires retrieval or can be answered using internal knowledge. This simulates a ReAct-style workflow by producing a reasoning trace that includes:

- Thought
- Action
- Observation
- Conclusion

The system maintains session-based memory for multi-turn chat support.

Separation of concerns between API, ingestion, vector store, tools, and agent ensures scalability and maintainability.

The design prioritizes:
- Transparency (reasoning trace)
- Modularity
- Extensibility
- Clear API boundaries

## Author

M. Ameen Qureshi