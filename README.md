# DocRAG SaaS

Multi-tenant document upload, search, summarize, and Q&A system built with:

- Next.js (Frontend)
- FastAPI (Backend)
- OpenAI Responses API + File Search

## Deployment

Frontend → Vercel  
Backend → Render  

## Setup (Local)

### Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
cd frontend
npm install
npm run dev
