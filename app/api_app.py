"""
FastAPI application entry point.

Run:
    uvicorn app.api_app:app --reload
"""

from fastapi import FastAPI

from app.api.middleware import install_observability_middleware
from app.api.legal_v2_guard_router import router as legal_v2_guard_router
from app.api.rag_router import router as rag_router

app = FastAPI(title="NALUS RAG API", version="0.1.0")

install_observability_middleware(app)
app.include_router(rag_router)
app.include_router(legal_v2_guard_router)
