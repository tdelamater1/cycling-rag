#!/bin/bash
# Daily sync worker: pulls new activities from intervals.icu into Postgres,
# then embeds any new ones into ChromaDB. Runs once at startup, then every 24h.
set -e

echo "[worker] Starting initial sync..."
python scripts/sync_now.py
echo "[worker] Embedding new activities..."
python -m src.ingest.embedder
echo "[worker] Initial sync complete. Sleeping 24h between runs."

while true; do
    sleep 86400
    echo "[worker] Running scheduled sync..."
    python scripts/sync_now.py
    echo "[worker] Embedding new activities..."
    python -m src.ingest.embedder
    echo "[worker] Done."
done
