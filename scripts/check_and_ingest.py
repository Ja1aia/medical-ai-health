import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("medquad")
count = collection.count()

print(f"ChromaDB document count: {count}")

if count == 0:
    print("ChromaDB is empty. Running ingest...")
    from scripts.ingest import ingest

    ingest()
else:
    print(f"ChromaDB has {count} documents. Skipping ingest.")
