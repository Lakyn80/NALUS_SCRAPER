"""
FastAPI application entry point.

Run:
    uvicorn app.api_app:app --reload
"""

from fastapi import FastAPI

from app.api.rag_router import router as rag_router

app = FastAPI(title="NALUS RAG API", version="0.1.0")

app.include_router(rag_router)
