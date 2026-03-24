"""
FastAPI application entry point.

Run locally:
    uvicorn app.api.main:app --reload --port 8000

Run via Docker:
    docker-compose up
"""

from fastapi import FastAPI

from app.api.rag_router import router

app = FastAPI(
    title="NALUS RAG API",
    description="RAG pipeline nad judikaturou Ústavního soudu ČR",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
