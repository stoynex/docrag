# DocRAG

A mobile-friendly web app for connecting a document repository and running NLP-powered workflows:

- Connect repository documents
- Semantic-style search over document chunks
- Extractive summaries (global or query-focused)
- Grounded Q&A with citations

## Stack

- **Backend/UI:** FastAPI (serves REST API + responsive HTML app)
- **NLP logic:** Lightweight in-process tokenization, chunking, cosine similarity retrieval, and extractive response generation

## Run locally

```bash
cd backend
pip install fastapi uvicorn pydantic
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000`

## Main API endpoints

- `POST /repositories/connect` — Connect a repository by sending one or more documents
- `GET /repositories` — List connected repositories
- `POST /search` — Search top matching chunks
- `POST /summarize` — Summarize repository content (or focused by query/document)
- `POST /chat` — Ask grounded questions over repository context
- `GET /` — Mobile-friendly web application

## Notes

This implementation uses in-memory storage and deterministic NLP heuristics suitable for demos/prototyping.
