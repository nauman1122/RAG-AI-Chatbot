
"""
RAG-based AI Chatbot — FastAPI application entry point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import embedding_service

logger = get_logger(__name__)

# Path to static assets
STATIC_DIR = Path(__file__).parent / "static"


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources on startup, clean up on shutdown."""
    logger.info("Starting RAG Chatbot …")
    logger.info("Embedding model : %s", settings.embedding_model)
    logger.info("LLM provider    : %s", settings.llm_provider)
    logger.info(
        "LLM model       : %s",
        settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model,
    )
    if settings.llm_provider == "ollama":
        logger.info("Ollama URL      : %s", settings.ollama_base_url)
    else:
        logger.info("Gemini endpoint : %s", settings.gemini_api_url)

    # Pre-load embedding model so the first request isn't slow
    embedding_service.load_model()

    # Ensure reports directory exists
    Path(settings.reports_dir).mkdir(exist_ok=True)

    logger.info("Startup complete ✓")

    yield  # ← app is running

    logger.info("Shutting down RAG Chatbot …")


# ── Application ───────────────────────────────────────────────────────────


app = FastAPI(
    title="RAG-based AI Chatbot",
    description=(
        "A local, production-grade Retrieval-Augmented Generation chatbot. "
        "Upload documents or scrape websites, then ask questions answered by "
        "a local LLM with relevant context retrieved from a FAISS vector store."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Root ──────────────────────────────────────────────────────────────────


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Serve the frontend UI."""
    return FileResponse(STATIC_DIR / "index.html")


# ── Run Server ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
