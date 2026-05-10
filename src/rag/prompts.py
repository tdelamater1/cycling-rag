"""Prompt templates for the RAG pipeline."""

from typing import Any

_SYSTEM_PROMPT = (
    "You are an expert cycling coach with deep knowledge of training physiology, "
    "power-based training, and periodisation. You have access to the athlete's "
    "actual training data from their GPS computer and power meter. "
    "Be specific, concise, and grounded in the numbers provided. "
    "Do not invent data that is not in the context."
)


def _fmt_stats(stats: dict[str, Any]) -> str:
    """Render a period-aggregate dict as a readable block."""
    def v(key: str, decimals: int = 1) -> str:
        val = stats.get(key)
        return "n/a" if val is None else f"{val:.{decimals}f}"

    return (
        f"  Rides:             {stats.get('activity_count', 0)}\n"
        f"  Total hours:       {v('total_duration_hours')} h\n"
        f"  Total distance:    {v('total_distance_km')} km\n"
        f"  Total elevation:   {v('total_elevation_meters')} m\n"
        f"  Total TSS:         {v('total_tss', 0)}\n"
        f"  Avg power:         {v('avg_power_watts')} W\n"
        f"  Avg norm. power:   {v('avg_normalized_power')} W\n"
        f"  Avg heart rate:    {v('avg_hr')} bpm"
    )


def format_period_comparison_prompt(
    question: str,
    recent: dict[str, Any],
    recent_label: str,
    prior: dict[str, Any],
    prior_label: str,
    context: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a period-comparison question.

    Args:
        question: The athlete's original question.
        recent: Aggregate stats dict for the recent period.
        recent_label: Human-readable label e.g. "Last 90 days".
        prior: Aggregate stats dict for the comparison period.
        prior_label: Human-readable label e.g. "Same 90 days last year".
        context: Optional ChromaDB semantic search results as a text block.
    """
    user = (
        f"Here is the athlete's training data for two periods:\n\n"
        f"**{recent_label}**\n{_fmt_stats(recent)}\n\n"
        f"**{prior_label}**\n{_fmt_stats(prior)}\n\n"
        + (f"{context}\n\n" if context else "")
        + f"Athlete's question: {question}\n\n"
        "Please compare the two periods, highlight key differences in volume, "
        "intensity, and load, and comment on the trend."
    )
    return _SYSTEM_PROMPT, user


def format_recommendation_prompt(
    question: str,
    recent: dict[str, Any],
    recent_label: str,
    latest_ctl: float | None,
    latest_atl: float | None,
    latest_tsb: float | None,
    context: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a session-recommendation question.

    Args:
        question: The athlete's original question.
        recent: Aggregate stats for the recent training block.
        recent_label: Human-readable label e.g. "Last 14 days".
        latest_ctl: Most recent CTL (fitness).
        latest_atl: Most recent ATL (fatigue).
        latest_tsb: Most recent TSB (form).
        context: Optional ChromaDB semantic search results as a text block.
    """
    def fv(val: float | None) -> str:
        return "n/a" if val is None else f"{val:.1f}"

    form_block = (
        f"  CTL (fitness):  {fv(latest_ctl)}\n"
        f"  ATL (fatigue):  {fv(latest_atl)}\n"
        f"  TSB (form):     {fv(latest_tsb)}"
    )

    user = (
        f"Here is the athlete's current training state:\n\n"
        f"**Current form**\n{form_block}\n\n"
        f"**{recent_label}**\n{_fmt_stats(recent)}\n\n"
        + (f"{context}\n\n" if context else "")
        + f"Athlete's question: {question}\n\n"
        "Based on the athlete's current fitness, fatigue, and recent workload, "
        "recommend a specific session (type, duration, intensity, structure). "
        "Explain your reasoning in terms of the numbers."
    )
    return _SYSTEM_PROMPT, user


def format_general_prompt(
    question: str,
    recent: dict[str, Any],
    recent_label: str,
    context: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a general training question.

    Args:
        question: The athlete's original question.
        recent: Aggregate stats for the recent period.
        recent_label: Human-readable label.
        context: Optional ChromaDB semantic search results as a text block.
    """
    user = (
        f"Here is the athlete's recent training data:\n\n"
        f"**{recent_label}**\n{_fmt_stats(recent)}\n\n"
        + (f"{context}\n\n" if context else "")
        + f"Athlete's question: {question}"
    )
    return _SYSTEM_PROMPT, user
