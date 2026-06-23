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


class Source(BaseModel):
    text: str
    url: str | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
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
    # Step 3: Cek cache
    cached = get_cached(clean_query)
    if cached:
        return QueryResponse(
            answer=cached["answer"],
            sources=[Source(text=s) for s in cached["sources"] if s],
            warning="cached",
        )

    # Step 4: RAG pipeline
    result = rag_pipeline(clean_query)
    output_check = check_output(result["answer"])

    sources = [
        Source(
            text=doc[:200] + "..." if len(doc) > 200 else doc,
            url=meta.get("url", ""),
            source=meta.get("source", ""),
        )
        for doc, meta in zip(result["sources"], result.get("source_meta", []))
    ]

    # Step 5: Simpan ke cache dengan sources
    set_cache(clean_query, output_check["text"], [s.text for s in sources])

    return QueryResponse(
        answer=output_check["text"],
        sources=sources,
        warning=output_check["warning"],
        cost_usd=result.get("cost_usd"),
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )


class MCQRequest(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str


from app.rag import rag_pipeline, hybrid_search

@app.post("/eval_mcq")
def eval_mcq(request: MCQRequest):
    from google import genai as genai_client
    import os

    client = genai_client.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Retrieve lebih banyak konteks
    docs, _ = hybrid_search(request.question, top_k=10)
    context = "\n\n".join(docs)

    prompt = (
        "You are a medical expert taking a multiple choice exam. "
        "Use the context below to answer the question. "
        "You MUST respond with ONLY a single letter: A, B, C, or D. "
        "Do not write anything else.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {request.question}\n\n"
        f"A) {request.option_a}\n"
        f"B) {request.option_b}\n"
        f"C) {request.option_c}\n"
        f"D) {request.option_d}\n\n"
        "Your answer (single letter only):"
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents=prompt
    )

    return {"answer": response.text.strip()[:5]}
