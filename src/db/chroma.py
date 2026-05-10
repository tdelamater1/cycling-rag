"""ChromaDB connection and vector search helpers."""

from typing import Any

import threading

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import config

_COLLECTION_NAME = "ride_summaries"
_EMBED_MODEL = "all-MiniLM-L6-v2"

_embedding_fn = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)

_lock = threading.Lock()
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Return the ride_summaries collection, initialising it once on first call.

    Uses a module-level singleton so the ChromaDB client and its Rust bindings
    are created in the main thread and reused, rather than recreated per-request
    inside a thread executor.
    """
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=config.chroma_path)
                _collection = client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    embedding_function=_embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
    return _collection


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def upsert_summary(
    collection: chromadb.Collection,
    activity_id: int,
    text: str,
    metadata: dict[str, Any],
) -> None:
    """Add or overwrite the natural language summary for one activity.

    Args:
        collection: Active ChromaDB collection.
        activity_id: intervals.icu activity ID (stored as string in Chroma).
        text: Natural language summary to embed.
        metadata: Flat dict of scalar values (activity_id, start_date, sport_type,
                  tss, ctl, atl, tsb).
    """
    collection.upsert(
        ids=[str(activity_id)],
        documents=[text],
        metadatas=[metadata],
    )


def upsert_summaries(
    collection: chromadb.Collection,
    items: list[dict[str, Any]],
) -> int:
    """Bulk upsert activity summaries.

    Each item must have keys: activity_id (int), text (str), metadata (dict).
    Returns the number of items upserted.

    Args:
        collection: Active ChromaDB collection.
        items: List of dicts with activity_id, text, metadata.
    """
    if not items:
        return 0

    collection.upsert(
        ids=[str(item["activity_id"]) for item in items],
        documents=[item["text"] for item in items],
        metadatas=[item["metadata"] for item in items],
    )
    return len(items)


def delete_activity(collection: chromadb.Collection, activity_id: int) -> None:
    """Remove a single activity's embedding from the collection.

    Args:
        collection: Active ChromaDB collection.
        activity_id: intervals.icu activity ID to remove.
    """
    collection.delete(ids=[str(activity_id)])


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_embedded_ids(collection: chromadb.Collection) -> set[str]:
    """Return the set of activity IDs already stored in the collection.

    Used by the embedder to skip activities that are already embedded.

    Args:
        collection: Active ChromaDB collection.
    """
    result = collection.get(include=[])  # IDs only, no documents or embeddings
    return set(result["ids"])


def query_similar(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Semantic search — return the top-k most similar activity summaries.

    Args:
        collection: Active ChromaDB collection.
        query_text: Natural language query to embed and search against.
        n_results: Number of results to return.
        where: Optional ChromaDB metadata filter (e.g. {"sport_type": "Ride"}).

    Returns:
        List of dicts, each with keys: id, document, metadata, distance.
        Ordered by ascending cosine distance (most similar first).
    """
    # ChromaDB raises if n_results > collection size
    n_results = min(n_results, collection.count())
    if n_results == 0:
        return []

    kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    result = collection.query(**kwargs)

    hits = []
    for i, doc_id in enumerate(result["ids"][0]):
        hits.append({
            "id": doc_id,
            "document": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        })
    return hits
