"""Hybrid retriever: async wrapper around ChromaDB semantic search."""

import asyncio
from datetime import date, datetime, timezone
from functools import partial

from src.db.chroma import get_collection, query_similar


def _date_where(since: date | None, until: date | None) -> dict | None:
    """Build a ChromaDB where-filter for a date range using unix timestamps.

    ChromaDB $gte/$lte operators require numeric values, so we use start_ts
    (int unix timestamp) rather than the start_date string field.
    """
    def _to_ts(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    conditions = []
    if since:
        conditions.append({"start_ts": {"$gte": _to_ts(since)}})
    if until:
        conditions.append({"start_ts": {"$lte": _to_ts(until)}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


async def retrieve_similar_rides(
    query_text: str,
    n_results: int = 5,
    since: date | None = None,
    until: date | None = None,
) -> list[dict]:
    """Semantic search over ChromaDB ride summaries.

    Runs synchronous ChromaDB calls in a thread executor so they don't block
    the async event loop.

    Args:
        query_text: Natural language text to search against.
        n_results: Maximum number of similar rides to return.
        since: Only return rides on or after this date.
        until: Only return rides on or before this date.

    Returns:
        List of hit dicts with keys: id, document, metadata, distance.
    """
    where = _date_where(since, until)
    loop = asyncio.get_running_loop()
    collection = await loop.run_in_executor(None, get_collection)
    fn = partial(query_similar, collection, query_text, n_results, where)
    return await loop.run_in_executor(None, fn)


def format_context_block(hits: list[dict], label: str = "Similar rides from training history") -> str:
    """Format ChromaDB hits as a plain-text block for inclusion in an LLM prompt.

    Args:
        hits: List of hit dicts returned by retrieve_similar_rides.
        label: Section heading for the block.
    """
    if not hits:
        return ""
    lines = [f"{label}:"]
    for hit in hits:
        lines.append(f"- {hit['document']}")
    return "\n".join(lines)
