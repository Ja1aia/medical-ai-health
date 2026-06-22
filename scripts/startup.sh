echo "Checking ChromaDB document count..."
python scripts/check_and_ingest.py
uvicorn app.main:app --host 0.0.0.0 --port 8080