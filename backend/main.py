from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="DocRAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class DocumentInput(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ConnectRepositoryRequest(BaseModel):
    repository_name: str = Field(min_length=1)
    documents: List[DocumentInput] = Field(min_length=1)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    chunk_count: int


class ConnectRepositoryResponse(BaseModel):
    repository_id: str
    repository_name: str
    message: str
    documents: List[DocumentSummary]


class RepositoryItem(BaseModel):
    repository_id: str
    repository_name: str
    document_count: int


class RepositoriesResponse(BaseModel):
    repositories: List[RepositoryItem]


class SearchRequest(BaseModel):
    repository_id: str
    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResponseItem(BaseModel):
    chunk_id: str
    document_id: str
    title: Optional[str] = None
    snippet: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResponseItem]


class SummarizeRequest(BaseModel):
    repository_id: str
    query: Optional[str] = None
    document_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: Optional[str] = None


class SummarizeResponse(BaseModel):
    summary: str
    citations: List[Citation]


class ChatRequest(BaseModel):
    repository_id: str
    question: str = Field(min_length=2)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]


# ---------- In-memory datastore ----------
# User isolation remains as a single demo user for local development.
repositories_by_user: Dict[str, Dict[str, dict]] = {}


# ---------- Utility ----------
def get_user_id() -> str:
    return "demo-user-id"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_words: int = 90, overlap: int = 20) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    idx = 0
    step = max(max_words - overlap, 1)
    while idx < len(words):
        chunks.append(" ".join(words[idx : idx + max_words]))
        idx += step
    return chunks


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in set(a.keys()) & set(b.keys()))
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def rank_chunks(repository: dict, query: str, top_k: int) -> List[dict]:
    q_vector = Counter(tokenize(query))
    scored = []
    for chunk in repository["chunks"]:
        score = cosine_similarity(q_vector, chunk["vector"])
        if score > 0:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def summarize_chunks(chunks: List[dict], query: Optional[str]) -> str:
    if not chunks:
        return "No relevant content was found to summarize."

    sentence_scores = []
    query_vec = Counter(tokenize(query)) if query else None

    for chunk in chunks:
        for sentence in split_sentences(chunk["text"]):
            vec = Counter(tokenize(sentence))
            if query_vec:
                score = cosine_similarity(query_vec, vec)
            else:
                score = sum(vec.values())
            sentence_scores.append((score, sentence))

    sentence_scores.sort(key=lambda pair: pair[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in sentence_scores:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            selected.append(sentence)
        if len(selected) == 3:
            break

    return " ".join(selected) if selected else "Unable to generate a summary from this content."


def generate_grounded_answer(question: str, chunks: List[dict]) -> str:
    if not chunks:
        return "I could not find relevant information in this repository to answer your question."

    best_sentences = []
    q_vec = Counter(tokenize(question))
    for chunk in chunks:
        for sentence in split_sentences(chunk["text"]):
            score = cosine_similarity(q_vec, Counter(tokenize(sentence)))
            if score > 0:
                best_sentences.append((score, sentence))

    best_sentences.sort(key=lambda item: item[0], reverse=True)
    response_lines = [sentence for _, sentence in best_sentences[:4]]

    if not response_lines:
        response_lines = [
            f"I found related documents in '{chunks[0]['title']}', but there are no strongly matching sentences for: {question}"
        ]

    intro = "Based on your repository content, here is what I found:"
    return f"{intro} {' '.join(response_lines)}"


def get_repository_or_404(user_id: str, repository_id: str) -> dict:
    user_repos = repositories_by_user.get(user_id, {})
    repo = user_repos.get(repository_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


# ---------- API ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/repositories/connect", response_model=ConnectRepositoryResponse)
def connect_repository(req: ConnectRepositoryRequest, user_id: str = Depends(get_user_id)):
    repository_id = str(uuid.uuid4())
    docs = []
    chunks = []

    for doc in req.documents:
        document_id = str(uuid.uuid4())
        doc_chunks = chunk_text(doc.content)
        docs.append(
            {
                "document_id": document_id,
                "title": doc.title,
                "content": doc.content,
                "chunk_count": len(doc_chunks),
            }
        )
        for idx, chunk in enumerate(doc_chunks):
            chunks.append(
                {
                    "chunk_id": f"{document_id}-c{idx + 1}",
                    "document_id": document_id,
                    "title": doc.title,
                    "text": chunk,
                    "vector": Counter(tokenize(chunk)),
                }
            )

    repositories_by_user.setdefault(user_id, {})[repository_id] = {
        "repository_id": repository_id,
        "repository_name": req.repository_name,
        "documents": docs,
        "chunks": chunks,
    }

    return ConnectRepositoryResponse(
        repository_id=repository_id,
        repository_name=req.repository_name,
        message="Repository connected successfully",
        documents=[DocumentSummary(**doc) for doc in docs],
    )


@app.get("/repositories", response_model=RepositoriesResponse)
def list_repositories(user_id: str = Depends(get_user_id)):
    user_repos = repositories_by_user.get(user_id, {})
    repos = [
        RepositoryItem(
            repository_id=repo["repository_id"],
            repository_name=repo["repository_name"],
            document_count=len(repo["documents"]),
        )
        for repo in user_repos.values()
    ]
    return RepositoriesResponse(repositories=repos)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, user_id: str = Depends(get_user_id)):
    repo = get_repository_or_404(user_id, req.repository_id)
    ranked = rank_chunks(repo, req.query, req.top_k)
    return SearchResponse(
        results=[
            SearchResponseItem(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                title=item.get("title"),
                snippet=item["text"][:260],
                score=round(item["score"], 4),
            )
            for item in ranked
        ]
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest, user_id: str = Depends(get_user_id)):
    repo = get_repository_or_404(user_id, req.repository_id)

    if req.document_id:
        selected = [c for c in repo["chunks"] if c["document_id"] == req.document_id]
    elif req.query:
        selected = rank_chunks(repo, req.query, top_k=6)
    else:
        selected = repo["chunks"][:6]

    summary = summarize_chunks(selected, req.query)
    citations = [
        Citation(chunk_id=c["chunk_id"], document_id=c["document_id"], title=c.get("title"))
        for c in selected[:3]
    ]
    return SummarizeResponse(summary=summary, citations=citations)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user_id: str = Depends(get_user_id)):
    repo = get_repository_or_404(user_id, req.repository_id)
    context = rank_chunks(repo, req.question, top_k=6)
    answer = generate_grounded_answer(req.question, context)
    citations = [
        Citation(chunk_id=c["chunk_id"], document_id=c["document_id"], title=c.get("title"))
        for c in context[:3]
    ]
    return ChatResponse(answer=answer, citations=citations)


@app.get("/", response_class=HTMLResponse)
def app_home() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>DocRAG Mobile App</title>
  <style>
    :root {
      font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
      --bg: #0f172a;
      --card: #111827;
      --muted: #94a3b8;
      --text: #f8fafc;
      --accent: #22d3ee;
      --accent-dark: #0891b2;
    }
    body {
      margin: 0;
      background: linear-gradient(140deg, #020617, #0f172a 45%, #111827);
      color: var(--text);
      min-height: 100vh;
    }
    .container {
      max-width: 950px;
      margin: 0 auto;
      padding: 1rem;
      display: grid;
      gap: 1rem;
    }
    .card {
      background: rgba(17, 24, 39, 0.88);
      border: 1px solid #1f2937;
      border-radius: 14px;
      padding: 1rem;
      box-shadow: 0 12px 30px rgba(0,0,0,.25);
    }
    h1, h2 { margin: 0 0 .75rem 0; }
    p { color: var(--muted); }
    label { display: block; font-size: .9rem; margin: .5rem 0 .25rem; }
    input, textarea, select, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #334155;
      background: #0b1220;
      color: var(--text);
      padding: .7rem;
      font-size: .95rem;
      box-sizing: border-box;
    }
    textarea { min-height: 95px; }
    button {
      background: var(--accent);
      color: #042f3a;
      border: none;
      font-weight: 700;
      margin-top: .6rem;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); color: #ecfeff; }
    .row { display: grid; gap: .7rem; }
    .pill {
      display: inline-block;
      padding: .15rem .5rem;
      border: 1px solid #334155;
      border-radius: 999px;
      font-size: .75rem;
      color: #bae6fd;
      margin: .2rem .25rem .2rem 0;
    }
    .result { border-top: 1px solid #1f2937; padding-top: .65rem; margin-top: .65rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #e2e8f0; }
    @media (min-width: 760px) {
      .row.two { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"card\">
      <h1>📄 DocRAG Assistant</h1>
      <p>Connect a repository, search content, summarize key points, and ask grounded questions. Built mobile-first.</p>
    </div>

    <section class=\"card\">
      <h2>1) Connect Repository</h2>
      <label>Repository name</label>
      <input id=\"repoName\" placeholder=\"Q4 Product Docs\" />
      <label>Document title</label>
      <input id=\"docTitle\" placeholder=\"Roadmap.md\" />
      <label>Document content</label>
      <textarea id=\"docContent\" placeholder=\"Paste your repository document text here...\"></textarea>
      <button onclick=\"connectRepo()\">Connect Repository</button>
      <p id=\"connectStatus\"></p>
    </section>

    <section class=\"card row two\">
      <div>
        <h2>2) Search</h2>
        <label>Connected repository</label>
        <select id=\"repoSelectSearch\"></select>
        <label>Search query</label>
        <input id=\"searchQuery\" placeholder=\"authentication flow\" />
        <button onclick=\"runSearch()\">Search</button>
        <div id=\"searchResults\"></div>
      </div>

      <div>
        <h2>3) Summarize</h2>
        <label>Connected repository</label>
        <select id=\"repoSelectSummary\"></select>
        <label>Optional focus query</label>
        <input id=\"summaryQuery\" placeholder=\"release risks\" />
        <button onclick=\"runSummary()\">Summarize</button>
        <div id=\"summaryOutput\"></div>
      </div>
    </section>

    <section class=\"card\">
      <h2>4) Ask Questions</h2>
      <label>Connected repository</label>
      <select id=\"repoSelectChat\"></select>
      <label>Question</label>
      <input id=\"chatQuestion\" placeholder=\"What are the key security requirements?\" />
      <button onclick=\"runChat()\">Ask</button>
      <div id=\"chatOutput\"></div>
    </section>
  </div>

<script>
async function fetchRepos() {
  const r = await fetch('/repositories');
  const data = await r.json();
  for (const id of ['repoSelectSearch','repoSelectSummary','repoSelectChat']) {
    const select = document.getElementById(id);
    select.innerHTML = '';
    if (!data.repositories.length) {
      select.innerHTML = '<option value="">No repositories yet</option>';
      continue;
    }
    data.repositories.forEach(repo => {
      const opt = document.createElement('option');
      opt.value = repo.repository_id;
      opt.textContent = `${repo.repository_name} (${repo.document_count} docs)`;
      select.appendChild(opt);
    });
  }
}

async function connectRepo() {
  const repository_name = document.getElementById('repoName').value.trim();
  const title = document.getElementById('docTitle').value.trim();
  const content = document.getElementById('docContent').value.trim();
  if (!repository_name || !title || !content) {
    document.getElementById('connectStatus').textContent = 'Please fill all repository/document fields.';
    return;
  }
  const res = await fetch('/repositories/connect', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ repository_name, documents: [{ title, content }]})
  });
  const data = await res.json();
  document.getElementById('connectStatus').innerHTML = `<span class='mono'>${data.message}</span>`;
  await fetchRepos();
}

async function runSearch() {
  const repository_id = document.getElementById('repoSelectSearch').value;
  const query = document.getElementById('searchQuery').value.trim();
  const out = document.getElementById('searchResults');
  if (!repository_id || !query) {
    out.textContent = 'Choose a repository and add a query.';
    return;
  }
  const res = await fetch('/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ repository_id, query, top_k: 5 })
  });
  const data = await res.json();
  out.innerHTML = data.results.length ? data.results.map(r =>
    `<div class='result'><div><strong>${r.title || 'Untitled'}</strong> · score ${r.score}</div><p>${r.snippet}</p><span class='pill'>${r.chunk_id}</span></div>`
  ).join('') : '<p>No matches found.</p>';
}

async function runSummary() {
  const repository_id = document.getElementById('repoSelectSummary').value;
  const query = document.getElementById('summaryQuery').value.trim();
  const out = document.getElementById('summaryOutput');
  if (!repository_id) {
    out.textContent = 'Choose a repository first.';
    return;
  }
  const payload = { repository_id };
  if (query) payload.query = query;
  const res = await fetch('/summarize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  const citations = (data.citations || []).map(c => `<span class='pill'>${c.title || c.document_id}: ${c.chunk_id}</span>`).join('');
  out.innerHTML = `<p>${data.summary}</p><div>${citations}</div>`;
}

async function runChat() {
  const repository_id = document.getElementById('repoSelectChat').value;
  const question = document.getElementById('chatQuestion').value.trim();
  const out = document.getElementById('chatOutput');
  if (!repository_id || !question) {
    out.textContent = 'Choose a repository and ask a question.';
    return;
  }
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ repository_id, question })
  });
  const data = await res.json();
  const citations = (data.citations || []).map(c => `<span class='pill'>${c.title || c.document_id}: ${c.chunk_id}</span>`).join('');
  out.innerHTML = `<p>${data.answer}</p><div>${citations}</div>`;
}

fetchRepos();
</script>
</body>
</html>"""
