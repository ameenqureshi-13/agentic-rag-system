from .tools import direct_llm
from .vector_store import search_documents

# ------------------ In-Memory Session Storage ------------------
chat_memory = {}


def run_agent(query: str, session_id: str = None):
    reasoning_trace = []
    retrieval_used = False
    sources = []

    # ------------------ Handle Multi-Turn Memory ------------------
    if session_id:
        history = chat_memory.get(session_id, "")
        query_with_history = history + "\nUser: " + query
    else:
        query_with_history = query

    # ------------------ Step 1: Thought ------------------
    reasoning_trace.append({
        "step": 1,
        "thought": "Received user query. Determining whether document retrieval is required."
    })

    keywords = ["report", "document", "file", "uploaded", "pdf", "doc"]

    # ------------------ RETRIEVAL PATH ------------------
    if any(word in query_with_history.lower() for word in keywords):
        retrieval_used = True

        reasoning_trace.append({
            "step": 2,
            "action": "document_search",
            "reason": "Query appears document-specific."
        })

        retrieved_docs = search_documents(query_with_history)
        context = ""

        if retrieved_docs:
            seen_chunks = set()

            for doc in retrieved_docs:
                chunk_text = doc.page_content.strip()
                context += chunk_text + "\n"

                if chunk_text not in seen_chunks:
                    sources.append({
                        "document": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "chunk": chunk_text[:300]
                    })
                    seen_chunks.add(chunk_text)

            reasoning_trace.append({
                "step": 3,
                "observation": "Relevant chunks retrieved from vector database."
            })

            augmented_prompt = f"""
You are a factual assistant.

Use ONLY the information in the context below.
If the answer is found, respond with that exact sentence.
Do NOT invent information.

Context:
{context}

Question:
{query_with_history}

Answer:
"""

            answer = direct_llm(augmented_prompt)

        else:
            answer = "No relevant documents found."

    # ------------------ DIRECT LLM PATH ------------------
    else:
        reasoning_trace.append({
            "step": 2,
            "action": "direct_llm",
            "reason": "Query appears to be general knowledge."
        })

        answer = direct_llm(query_with_history)

        reasoning_trace.append({
            "step": 3,
            "observation": "Answer generated using internal LLM knowledge."
        })

    # ------------------ Store Memory ------------------
    if session_id:
        chat_memory[session_id] = query_with_history + "\nAssistant: " + str(answer)

    # ------------------ Final Step ------------------
    reasoning_trace.append({
        "step": 4,
        "conclusion": "Final answer generated and returned to user."
    })

    return {
        "answer": str(answer),
        "sources": sources,
        "reasoning_trace": reasoning_trace,
        "retrieval_used": retrieval_used
    }