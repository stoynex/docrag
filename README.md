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
pip install -r requirements.txt
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000`

## Deploy

### Option A: Render (quickest)

This repo includes `render.yaml`, so Render can auto-detect the service settings.

1. Push this repository to GitHub.
2. In Render, click **New +** → **Blueprint**.
3. Connect your GitHub repo and select this repository.
4. Render will use:
   - Build: `pip install -r backend/requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Deploy and open the Render URL.

### Option B: Any Docker host (Railway/Fly.io/Cloud Run/VM)

A root `Dockerfile` is included.

```bash
docker build -t docrag:latest .
docker run -p 8000:8000 -e PORT=8000 docrag:latest
```

Then open `http://localhost:8000`.

### Option C: Plain VM/server with systemd + Nginx

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Put Nginx in front for TLS and domain routing.


## How to test this app

### 1) Quick manual UI test

1. Start the app:
   ```bash
   pip install -r backend/requirements.txt
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Open `http://127.0.0.1:8000` on desktop and mobile width.
3. In **Connect Repository**, add a repo name + one document and click **Connect Repository**.
4. Run all three workflows:
   - Search with terms present in the document.
   - Summarize with and without a focus query.
   - Ask a question whose answer exists in the uploaded text.
5. Confirm results include snippets/citations and no browser console errors.

### 2) API smoke test script (recommended)

With the server running, execute:

```bash
python scripts/api_smoke_test.py
```

This script validates end-to-end behavior for:
- `GET /health`
- `POST /repositories/connect`
- `GET /repositories`
- `POST /search`
- `POST /summarize`
- `POST /chat`

Optional custom URL:

```bash
BASE_URL=http://127.0.0.1:8000 python scripts/api_smoke_test.py
```

### 3) Curl checks (if you prefer raw HTTP)

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/repositories/connect   -H 'Content-Type: application/json'   -d '{
    "repository_name": "Docs",
    "documents": [{"title": "Auth", "content": "MFA is required for all admins."}]
  }'
```

## Main API endpoints

- `POST /repositories/connect` — Connect a repository by sending one or more documents
- `GET /repositories` — List connected repositories
- `POST /search` — Search top matching chunks
- `POST /summarize` — Summarize repository content (or focused by query/document)
- `POST /chat` — Ask grounded questions over repository context
- `GET /` — Mobile-friendly web application

## Notes

- Storage is in-memory only; data resets when the service restarts.
- This is a deterministic prototype intended for demos and local validation.
