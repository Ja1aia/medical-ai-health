import os
from datasets import load_dataset
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest():
    print("Loading dataset...")
    dataset = load_dataset("lavita/MedQuAD", split="train")

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("medquad")

    print("Ingesting documents...")
    batch_ids = []
    batch_docs = []
    batch_embeddings = []
    batch_size = 100

    for i, row in enumerate(dataset):
        answer = row.get("answer", "")
        question = row.get("question", "")

        if not answer or len(answer.strip()) < 50:
            continue

        chunks = chunk_text(answer)
        for j, chunk in enumerate(chunks):
            doc_id = f"doc_{i}_{j}"
            embedding = embedder.encode(chunk).tolist()

            batch_ids.append(doc_id)
            batch_docs.append(chunk)
            batch_embeddings.append(embedding)

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings
                )
                batch_ids, batch_docs, batch_embeddings = [], [], []
                print(f"Ingested {i} rows...")

    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings
        )

    print(f"Done. Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    ingest()