import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
cache_client = chromadb.PersistentClient(path="cache_db")
cache_collection = cache_client.get_or_create_collection("query_cache")

SIMILARITY_THRESHOLD = 0.95


def get_cached(query):
    query_embedding = embedder.encode(query).tolist()
    results = cache_collection.query(query_embeddings=[query_embedding], n_results=1)
    if not results["documents"][0]:
        return None
    distance = results["distances"][0][0]
    if distance < (1 - SIMILARITY_THRESHOLD):
        metadata = results["metadatas"][0][0]
        return {
            "answer": metadata.get("answer"),
            "sources": (
                metadata.get("sources", "").split("|||")
                if metadata.get("sources")
                else []
            ),
        }
    return None


def set_cache(query, answer, sources=None):
    query_embedding = embedder.encode(query).tolist()
    sources_str = "|||".join(
        [s if isinstance(s, str) else str(s) for s in (sources or [])]
    )
    cache_collection.add(
        ids=[f"cache_{hash(query)}"],
        documents=[query],
        embeddings=[query_embedding],
        metadatas=[{"answer": answer, "sources": sources_str}],
    )
