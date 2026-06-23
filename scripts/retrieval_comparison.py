import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pickle

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("medquad")

with open("chroma_db/bm25.pkl", "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
all_docs = bm25_data["docs"]


def bm25_search(query, top_k=5):
    tokenized = query.split()
    scores = bm25.get_scores(tokenized)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]
    return [all_docs[i] for i in top_indices]


def dense_search(query, top_k=5):
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0]


def hybrid_search(query, top_k=5):
    bm25_docs = bm25_search(query, top_k)
    dense_docs = dense_search(query, top_k)
    combined = list(dict.fromkeys(bm25_docs + dense_docs))
    return combined[:top_k]


def eval_strategy(strategy_fn, queries, answers):
    correct = 0
    for question, answer in zip(queries, answers):
        docs = strategy_fn(question)
        retrieved = " ".join(docs).lower()
        answer_words = set(answer.lower().split()[:20])
        if len(answer_words & set(retrieved.split())) > 5:
            correct += 1
    return correct / len(queries)


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("lavita/MedQuAD", split="train")

    queries, answers = [], []
    for i, row in enumerate(dataset):
        if len(queries) >= 20:
            break
        q = row.get("question", "")
        a = row.get("answer", "")
        if q and a and len(a) > 50:
            queries.append(q)
            answers.append(a)

    print("Running retrieval comparison (20 queries)...")
    bm25_score = eval_strategy(bm25_search, queries, answers)
    dense_score = eval_strategy(dense_search, queries, answers)
    hybrid_score = eval_strategy(hybrid_search, queries, answers)

    print("\n" + "=" * 40)
    print("RETRIEVAL COMPARISON")
    print("=" * 40)
    print(f"BM25 only  : {bm25_score:.2f}")
    print(f"Dense only : {dense_score:.2f}")
    print(f"Hybrid     : {hybrid_score:.2f}")
    winner = max(
        [("BM25", bm25_score), ("Dense", dense_score), ("Hybrid", hybrid_score)],
        key=lambda x: x[1],
    )
    print(f"Winner     : {winner[0]}")
