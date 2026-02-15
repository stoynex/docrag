from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Optional
import uuid

app = FastAPI(title="DocRAG API")

# ---- Models ----
class SearchResponseItem(BaseModel):
    chunk_id: str
    document_id: str
    title: Optional[str] = None
    snippet: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

class SummarizeRequest(BaseModel):
    doc_id: Optional[str] = None
    query: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    scope_doc_ids: Optional[List[str]] = None

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# ---- Dependency placeholder ----
def get_user_id():
    # Replace with real auth
    return "demo-user-id"

# ---- NLP Abstractions (replace with your provider) ----
def embed_texts(texts: List[str]) -> List[List[float]]:
    # TODO: call your embedding model
    raise NotImplementedError

def generate_answer(question: str, context_chunks: List[str]) -> str:
    # TODO: call your LLM with RAG prompt
    raise NotImplementedError

# ---- Routes ----
@app.get("/search", response_model=SearchResponse)
def search(q: str, user_id: str = Depends(get_user_id)):
    # TODO:
    # 1) embed q
    # 2) query pgvector for top-k chunks for user_id
    # 3) return snippets
    return SearchResponse(results=[])

@app.post("/summarize")
def summarize(req: SummarizeRequest, user_id: str = Depends(get_user_id)):
    # TODO:
    # - if doc_id: retrieve doc chunks and summarize
    # - if query: retrieve relevant chunks then summarize
    return {"summary": "TODO"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user_id: str = Depends(get_user_id)):
    # TODO:
    # 1) retrieve chunks (scoped if scope_doc_ids provided)
    # 2) generate grounded answer with citations
    return ChatResponse(answer="TODO", citations=[])
