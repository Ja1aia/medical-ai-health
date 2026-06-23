import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag_pipeline
from app.pii import detect_and_redact
from app.guardrails import check_input, check_output
from dotenv import load_dotenv
from app.cache import get_cached, set_cache

load_dotenv()

app = FastAPI(title="Medical AI Health Assistant")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    warning: str | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Medical AI Health Assistant is running"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # Step 1: Redact PII dari input
    clean_query = detect_and_redact(request.query)

    # Step 2: Guardrails input check
    input_check = check_input(clean_query)
    if not input_check["allowed"]:
        return QueryResponse(
            answer=input_check["message"], sources=[], warning=input_check["type"]
        )

    # Step 3: Cek cache dulu
    cached_answer = get_cached(clean_query)
    if cached_answer:
        return QueryResponse(answer=cached_answer, sources=[], warning="cached")

    result = rag_pipeline(clean_query)

    output_check = check_output(result["answer"])

    set_cache(clean_query, output_check["text"])

    return QueryResponse(
        answer=output_check["text"],
        sources=result["sources"],
        warning=output_check["warning"],
        cost_usd=result.get("cost_usd"),
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )
