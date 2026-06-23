import os
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("medquad")

print("Loading BM25 index...")
all_docs = collection.get()["documents"]
if not all_docs:
    raise RuntimeError("ChromaDB kosong saat load rag.py — ingest gagal.")
tokenized_docs = [doc.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)
print("BM25 ready.")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker ready.")

def hybrid_search(query, top_k=5):
    # Dense retrieval
    query_embedding = embedder.encode(query).tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding], n_results=top_k
    )
    dense_docs = dense_results["documents"][0]

    # BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:top_k]
    bm25_docs = [all_docs[i] for i in top_bm25_indices]

    # Combine dan deduplicate
    combined = list(dict.fromkeys(dense_docs + bm25_docs))
    return combined[:top_k]


def generate_answer(query, context_docs):
    context = "\n\n".join(context_docs)
    prompt = (
        "You are a medical AI assistant. "
        "Answer the question based ONLY on the provided context.\n"
        "If the context does not contain enough information, say "
        "'I don't have enough information to answer this. Please consult a doctor.'\n"
        "Always recommend consulting a healthcare professional for personal medical advice.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = client_genai.models.generate_content(
        model="gemini-3.1-flash-lite", contents=prompt
    )

    # Hitung cost
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

    # Gemini 2.0 Flash pricing (per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * 0.30
    output_cost = (output_tokens / 1_000_000) * 1.50
    total_cost = input_cost + output_cost

    return {
        "answer": response.text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(total_cost, 6),
    }


def rerank(query, docs, top_k=5):
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def rag_pipeline(query):
    # Retrieve lebih banyak dulu untuk reranking
    context_docs = hybrid_search(query, top_k=10)
    # Rerank ke top 5
    reranked_docs = rerank(query, context_docs, top_k=5)
    result = generate_answer(query, reranked_docs)
    return {
        "answer": result["answer"],
        "sources": reranked_docs,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
    }

