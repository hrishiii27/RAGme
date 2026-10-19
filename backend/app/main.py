"""FastAPI Main Entrypoint for RAGme."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router
from app.routes.feedback import router as feedback_router
from app.routes.search import router as search_router
from app.routes.stats import router as stats_router
from app.services import init_rag_system
from rag.pipeline import find_project_root

# Load environment variables (.env)
root = find_project_root()
load_dotenv(root / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Starting RAGme Backend Service...")
    print("Warming up RAG Pipeline (BM25 + Chroma + MiniLM)...")
    init_rag_system()
    print("RAGme Backend is Ready to Serve Citizen Inquiries.")
    print("=" * 60)
    yield
    print("Shutting down RAGme Backend Service...")


app = FastAPI(
    title="RAGme — Bharat Government Information Assistant",
    description="Production RAG system over Indian public documents with human-calibrated precision, hybrid retrieval, and citizen guidance.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(search_router, prefix="/api", tags=["Search"])
app.include_router(feedback_router, prefix="/api", tags=["Feedback"])
app.include_router(stats_router, prefix="/api", tags=["Stats"])


@app.get("/")
async def root_index():
    return {
        "service": "RAGme API",
        "description": "Official Bharat Government Documents RAG Assistant",
        "documentation": "/docs",
        "health": "/api/health",
        "stats": "/api/stats",
        "benchmarks": "/api/benchmarks",
    }
