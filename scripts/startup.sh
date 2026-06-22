#!/bin/bash
if [ ! -d "chroma_db" ]; then
    echo "Running ingest..."
    python scripts/ingest.py
fi
uvicorn app.main:app --host 0.0.0.0 --port $PORT