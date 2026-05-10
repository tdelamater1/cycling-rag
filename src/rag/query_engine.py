"""Main RAG pipeline: classify query, retrieve Postgres context, call LLM, return response."""

import asyncio
import re
import sys
from datetime import date, timedelta
from enum import Enum, auto

import httpx

from src.config import config
from src.db.postgres import create_pool, get_activities_in_range, get_period_aggregates
from src.rag.prompts import (
    format_general_prompt,
    format_period_comparison_prompt,
    format_recommendation_prompt,
)
from src.rag.retriever import format_context_block, retrieve_similar_rides

_CYCLING_SPORT_TYPES = ["Ride", "VirtualRide"]


class QueryType(Enum):
    PERIOD_COMPARISON = auto()
    RECOMMENDATION = auto()
    GENERAL = auto()


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_COMPARISON_KEYWORDS = {"compare", "vs", "versus", "same period", "last year", "year ago"}
_RECOMMENDATION_KEYWORDS = {"recommend", "suggest", "what should", "interval", "session", "workout", "training plan"}


def classify_query(question: str) -> QueryType:
    """Classify a natural-language question into a query type."""
    q = question.lower()
    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        return QueryType.PERIOD_COMPARISON
    if any(kw in q for kw in _RECOMMENDATION_KEYWORDS):
        return QueryType.RECOMMENDATION
    return QueryType.GENERAL


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def parse_days(question: str) -> int:
    """Extract the number of days implied by the question.

    Examples: "last 90 days" → 90, "last month" → 30, default → 90.
    """
    q = question.lower()

    match = re.search(r"(?:last|past)\s+(\d+)\s+days?", q)
    if match:
        return int(match.group(1))

    if "last week" in q or "past week" in q:
        return 7
    if "last month" in q or "past month" in q:
        return 30
    if "last year" in q or "past year" in q:
        return 365
    if "3 months" in q or "three months" in q:
        return 90
    if "6 months" in q or "six months" in q:
        return 180

    return 90  # sensible default


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Send a chat request to Ollama and return the response text."""
    url = f"{config.ollama_base_url}/api/chat"
    payload = {
        "model": config.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # suppress qwen3 <think> block
    }
    async with httpx.AsyncClient(timeout=config.ollama_timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_query(question: str) -> str:
    """Run the full RAG pipeline for a natural-language question.

    Args:
        question: The athlete's question in plain English.

    Returns:
        The LLM's response as a string.
    """
    query_type = classify_query(question)
    days = parse_days(question)
    today = date.today()

    pool = await create_pool()
    try:
        if query_type == QueryType.PERIOD_COMPARISON:
            recent_start = today - timedelta(days=days)
            prior_end = today - timedelta(days=365)
            prior_start = prior_end - timedelta(days=days)

            recent_stats, prior_stats, recent_hits, prior_hits = await asyncio.gather(
                get_period_aggregates(pool, recent_start, today, _CYCLING_SPORT_TYPES),
                get_period_aggregates(pool, prior_start, prior_end, _CYCLING_SPORT_TYPES),
                retrieve_similar_rides(question, n_results=4, since=recent_start, until=today),
                retrieve_similar_rides(question, n_results=4, since=prior_start, until=prior_end),
            )

            context_parts = [
                format_context_block(recent_hits, f"Similar rides — last {days} days"),
                format_context_block(prior_hits, f"Similar rides — same period last year"),
            ]
            context = "\n\n".join(p for p in context_parts if p)

            system_p, user_p = format_period_comparison_prompt(
                question=question,
                recent=recent_stats,
                recent_label=f"Last {days} days",
                prior=prior_stats,
                prior_label=f"Same {days} days one year ago",
                context=context,
            )

        elif query_type == QueryType.RECOMMENDATION:
            recent_start = today - timedelta(days=14)

            recent_stats, recent_rows, hits = await asyncio.gather(
                get_period_aggregates(pool, recent_start, today, _CYCLING_SPORT_TYPES),
                get_activities_in_range(pool, today - timedelta(days=7), today, _CYCLING_SPORT_TYPES),
                retrieve_similar_rides(question, n_results=5),
            )

            latest = recent_rows[-1] if recent_rows else None
            context = format_context_block(hits, "Similar training blocks from history")

            system_p, user_p = format_recommendation_prompt(
                question=question,
                recent=recent_stats,
                recent_label="Last 14 days",
                latest_ctl=latest["ctl"] if latest else None,
                latest_atl=latest["atl"] if latest else None,
                latest_tsb=latest["tsb"] if latest else None,
                context=context,
            )

        else:  # GENERAL
            recent_start = today - timedelta(days=days)

            recent_stats, hits = await asyncio.gather(
                get_period_aggregates(pool, recent_start, today, _CYCLING_SPORT_TYPES),
                retrieve_similar_rides(question, n_results=5, since=recent_start, until=today),
            )

            context = format_context_block(hits, "Similar rides from this period")

            system_p, user_p = format_general_prompt(
                question=question,
                recent=recent_stats,
                recent_label=f"Last {days} days",
                context=context,
            )
    finally:
        await pool.close()

    return await _call_ollama(system_p, user_p)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or "How has my training been lately?"
    print(asyncio.run(run_query(question)))
