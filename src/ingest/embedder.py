"""Build natural language summaries from Postgres activities and store in ChromaDB."""

import asyncio
from datetime import date, timezone

import asyncpg

from src.db.chroma import get_collection, get_embedded_ids, upsert_summaries
from src.db.postgres import create_pool, get_activities_in_range

_CYCLING_SPORTS = ["Ride", "VirtualRide"]
_EMBED_BATCH_SIZE = 100
_EPOCH = date(2020, 1, 1)


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: int | None) -> str | None:
    if not seconds:
        return None
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m" if h else f"{m}m"


def _intensity_label(tss: float | None) -> str | None:
    if tss is None:
        return None
    if tss >= 150:
        return "Very high intensity effort."
    if tss >= 100:
        return "High intensity effort."
    if tss >= 50:
        return "Moderate effort."
    return "Easy effort."


def build_summary(row: asyncpg.Record) -> str:
    """Build a ~100-word natural language summary of an activity for embedding.

    Degrades gracefully when power data is absent (HR-only ride).

    Args:
        row: asyncpg Record from the activities table.
    """
    date_str = row["start_date"].strftime("%Y-%m-%d")
    sport_label = "Virtual ride" if row["sport_type"] == "VirtualRide" else "Outdoor ride"

    parts = [f"{sport_label} on {date_str}."]

    duration = _fmt_duration(row["duration_seconds"])
    if duration:
        parts.append(f"Duration: {duration}.")

    if row["distance_meters"]:
        parts.append(f"Distance: {row['distance_meters'] / 1000:.1f}km.")

    if row["elevation_meters"]:
        parts.append(f"Elevation: {row['elevation_meters']:.0f}m.")

    if row["normalized_power"]:
        parts.append(f"Normalized power: {row['normalized_power']:.0f}W.")
    elif row["avg_power_watts"]:
        parts.append(f"Average power: {row['avg_power_watts']:.0f}W.")

    if row["avg_hr"]:
        parts.append(f"Average HR: {row['avg_hr']:.0f}bpm.")

    if row["tss"]:
        parts.append(f"TSS: {row['tss']:.0f}.")

    label = _intensity_label(row["tss"])
    if label:
        parts.append(label)

    ctl, atl, tsb = row["ctl"], row["atl"], row["tsb"]
    if ctl is not None and atl is not None and tsb is not None:
        parts.append(
            f"Fitness (CTL): {ctl:.0f}, Fatigue (ATL): {atl:.0f}, Form (TSB): {tsb:.0f}."
        )

    return " ".join(parts)


def build_metadata(row: asyncpg.Record) -> dict:
    """Build the ChromaDB metadata dict for an activity row.

    All values must be scalars (str, int, float, bool). None fields are omitted
    since ChromaDB where-filters only match keys that exist.

    Args:
        row: asyncpg Record from the activities table.
    """
    dt = row["start_date"]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    meta: dict = {
        "activity_id": row["id"],
        "start_date": dt.strftime("%Y-%m-%d"),      # human-readable, for display
        "start_ts": int(dt.timestamp()),             # unix timestamp, for $gte/$lte range filters
        "sport_type": row["sport_type"],
    }
    for key in ("tss", "ctl", "atl", "tsb"):
        if row[key] is not None:
            meta[key] = float(row[key])
    return meta


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------


async def embed_activities(
    pool: asyncpg.Pool,
    *,
    since: date | None = None,
    until: date | None = None,
    full: bool = False,
    batch_size: int = _EMBED_BATCH_SIZE,
) -> int:
    """Fetch cycling activities from Postgres and upsert embeddings into ChromaDB.

    Skips activities already embedded unless full=True.

    Args:
        pool: Active asyncpg connection pool.
        since: Earliest activity date to consider. Defaults to 2020-01-01.
        until: Latest activity date to consider. Defaults to today.
        full: If True, re-embed all activities regardless of existing embeddings.
        batch_size: Number of summaries to upsert per ChromaDB call.

    Returns:
        Number of activities newly embedded.
    """
    collection = get_collection()
    already_embedded = set() if full else get_embedded_ids(collection)

    rows = await get_activities_in_range(
        pool,
        start=since or _EPOCH,
        end=until or date.today(),
        sport_types=_CYCLING_SPORTS,
    )

    pending = [r for r in rows if str(r["id"]) not in already_embedded]
    print(f"{len(rows)} cycling activities found, {len(pending)} to embed.")

    total = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        items = [
            {
                "activity_id": row["id"],
                "text": build_summary(row),
                "metadata": build_metadata(row),
            }
            for row in batch
        ]
        upserted = upsert_summaries(collection, items)
        total += upserted
        print(f"  Embedded {total}/{len(pending)}")

    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Embed cycling activities into ChromaDB.")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="Embed activities from this date.")
    parser.add_argument("--until", metavar="YYYY-MM-DD", help="Embed activities up to this date.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-embed all activities, overwriting existing embeddings.",
    )
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    until = date.fromisoformat(args.until) if args.until else None

    pool = await create_pool()
    try:
        count = await embed_activities(pool, since=since, until=until, full=args.full)
        print(f"Done. {count} activities embedded.")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
