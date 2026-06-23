import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import pickle
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("medquad")
print("Loading docs...")
all_docs = collection.get()["documents"]
print(f"Total docs: {len(all_docs)}")
tokenized = [doc.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized)
with open("chroma_db/bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "docs": all_docs}, f)
print("BM25 saved.")
