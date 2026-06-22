#!/bin/bash

# Cek apakah chroma_db sudah ada isinya (bukan hanya folder kosong)
if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "ChromaDB empty, running ingest..."
    python scripts/ingest.py
else
    echo "ChromaDB already exists, skipping ingest..."
fi

uvicorn app.main:app --host 0.0.0.0 --port 8080