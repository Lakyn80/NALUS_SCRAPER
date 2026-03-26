"""
FastAPI application entry point.

Run locally:
    uvicorn app.api.main:app --reload --port 8029

Run via Docker:
    docker-compose up
"""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.rag_router import router
from app.core.logging import get_logger

logger = get_logger(__name__)

load_dotenv()

_STARTUP_RETRY_DELAY_SECONDS = float(os.getenv("RAG_STARTUP_RETRY_DELAY_SECONDS", "5"))
_deferred_ingest_task: asyncio.Task[None] | None = None

def _strict_real_mode_enabled() -> bool:
    raw_value = os.getenv("RAG_STRICT_REAL_MODE", "")
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


async def _initialize_orchestrator() -> None:
    from app.api.startup import build_live_orchestrator
    import app.api.rag_router as rtr
    global _deferred_ingest_task

    rtr._live_orchestrator_status = "initializing"
    rtr._live_orchestrator_error = None
    build = await asyncio.to_thread(build_live_orchestrator)
    rtr._live_orchestrator = build.orchestrator
    rtr._live_orchestrator_status = "ready"
    rtr._live_orchestrator_error = None
    rtr._background_ingest_status = build.ingest_status
    rtr._background_ingest_error = build.ingest_message
    logger.info("[main] live orchestrator ready")

    if build.deferred_ingest is not None:
        _deferred_ingest_task = asyncio.create_task(_run_deferred_ingest(build.deferred_ingest))


async def _run_deferred_ingest(deferred_ingest) -> None:
    import app.api.rag_router as rtr

    rtr._background_ingest_status = "running"
    rtr._background_ingest_error = None
    try:
        await asyncio.to_thread(deferred_ingest)
        rtr._background_ingest_status = "completed"
        rtr._background_ingest_error = None
        logger.info("[main] background append ingest completed")
    except asyncio.CancelledError:
        rtr._background_ingest_status = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001
        rtr._background_ingest_status = "error"
        rtr._background_ingest_error = str(exc)
        logger.warning("[main] background append ingest failed (%s)", exc)


async def _build_orchestrator_bg() -> None:
    """Keep trying until the live orchestrator is ready or shutdown cancels the task."""
    import app.api.rag_router as rtr

    while True:
        try:
            await _initialize_orchestrator()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            rtr._live_orchestrator = None
            rtr._live_orchestrator_status = "error"
            rtr._live_orchestrator_error = str(exc)
            logger.warning(
                "[main] startup failed (%s) — retrying in %.1fs",
                exc,
                _STARTUP_RETRY_DELAY_SECONDS,
            )
            await asyncio.sleep(_STARTUP_RETRY_DELAY_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import app.api.rag_router as rtr
    global _deferred_ingest_task

    rtr._live_orchestrator = None
    rtr._live_orchestrator_status = "pending"
    rtr._live_orchestrator_error = None
    rtr._background_ingest_status = "idle"
    rtr._background_ingest_error = None
    _deferred_ingest_task = None
    startup_task = asyncio.create_task(_build_orchestrator_bg())

    yield

    if _deferred_ingest_task is not None and not _deferred_ingest_task.done():
        _deferred_ingest_task.cancel()
        try:
            await _deferred_ingest_task
        except asyncio.CancelledError:
            pass

    if not startup_task.done():
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="NALUS RAG API",
    description="RAG pipeline nad judikaturou Ústavního soudu ČR",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    import app.api.rag_router as rtr

    return {
        "status": "ok",
        "orchestrator_ready": rtr._live_orchestrator is not None,
        "orchestrator_status": rtr._live_orchestrator_status,
        "orchestrator_error": rtr._live_orchestrator_error,
        "background_ingest_status": rtr._background_ingest_status,
        "background_ingest_error": rtr._background_ingest_error,
        "strict_real_mode": _strict_real_mode_enabled(),
    }
