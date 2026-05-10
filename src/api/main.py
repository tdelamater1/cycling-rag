"""FastAPI application exposing the /query endpoint for Open WebUI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.db.chroma import get_collection
from src.rag.query_engine import run_query


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_collection()  # initialise ChromaDB singleton before accepting requests
    yield


app = FastAPI(
    title="Cycling RAG",
    description="Natural language queries over your intervals.icu training data.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness check for Open WebUI and Docker health checks."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return a natural language answer.

    Args:
        request: JSON body with a `question` field.

    Returns:
        JSON body with an `answer` field.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    answer = await run_query(request.question)
    return QueryResponse(answer=answer)
