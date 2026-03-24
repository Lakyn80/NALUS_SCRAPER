"""
FastAPI application entry point.

Run locally:
    uvicorn app.api.main:app --reload --port 8029

Run via Docker:
    docker-compose up
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.rag_router import router
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------
    # Startup: connect to Qdrant, ingest corpus, wire real orchestrator
    # ------------------------------------------------------------------
    try:
        from app.api.startup import build_live_orchestrator
        import app.api.rag_router as rtr

        orchestrator = build_live_orchestrator()
        rtr._live_orchestrator = orchestrator
        logger.info("[main] live orchestrator ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[main] startup failed (%s) — using stub orchestrator", exc)

    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="NALUS RAG API",
    description="RAG pipeline nad judikaturou Ústavního soudu ČR",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
