"""FastAPI entry point: /health, /ingest and /ask."""

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.auth import get_current_role, require_hr
from app.comp_db import describe_schema, get_engine, run_select
from app.config import get_settings
from app.graph import build_graph
from app.ingest import ALLOWED_EXTENSIONS, ingest_document
from app.llm import ANSWER_MAX_TOKENS, SHORT_MAX_TOKENS, get_chat_model
from app.vector_store import get_vector_store

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="hr-assistant")


# --- Request / response models ----------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    filename: str
    chunk: int


class AskResponse(BaseModel):
    answer: str
    route: Literal["docs", "data"]
    sources: list[Source] | None = None  # only for the docs route
    sql: str | None = None  # only for the data route
    # Which model answered each LLM step that ran, e.g. {"classify": "claude-haiku-5-5"}.
    # Steps answered without an LLM (refusal, "I don't know") don't appear.
    models: dict[str, str] = {}


class IngestResponse(BaseModel):
    filename: str
    chunks: int


# --- Dependencies -----------------------------------------------------------------------


@lru_cache
def get_graph():
    """Build the graph once, on the first /ask call (so the app starts even if a DB is down)."""
    settings = get_settings()
    engine = get_engine()
    return build_graph(
        classify_llm=get_chat_model(settings.llm_fast_model, SHORT_MAX_TOKENS),
        sql_llm=get_chat_model(settings.llm_sql_model, SHORT_MAX_TOKENS),
        answer_llm=get_chat_model(settings.llm_fast_model, ANSWER_MAX_TOKENS),
        retriever=get_vector_store().as_retriever(search_kwargs={"k": 4}),
        run_sql=lambda sql: run_select(engine, sql),
        schema=describe_schema(engine, settings.table_list),
        hr_role=settings.hr_role_value,
    )


# --- Endpoints --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile, _role: str = Depends(require_hr)) -> IngestResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB")

    chunks = ingest_document(filename, data, get_vector_store())
    if chunks == 0:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")
    return IngestResponse(filename=filename, chunks=chunks)


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(
    body: AskRequest,
    role: str = Depends(get_current_role),
    graph=Depends(get_graph),
) -> AskResponse:
    result = graph.invoke({"question": body.question, "user_role": role})
    route = result["route"]
    return AskResponse(
        answer=result["answer"],
        route=route,
        sources=result.get("sources", []) if route == "docs" else None,
        sql=result.get("sql") if route == "data" else None,
        models=result.get("models", {}),
    )
