"""Fetch activities from intervals.icu and upsert them into Postgres."""

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta

import asyncpg

from src.db.postgres import (
    create_pool,
    get_latest_activity_date,
    init_schema,
    upsert_activities,
)
from src.ingest.intervals_client import Activity, IntervalsClient

# When re-syncing from an existing database, overlap by this many days to pick
# up any edits intervals.icu may have applied to recent activities.
_OVERLAP_DAYS = 14

# Default lookback when the database is empty (first sync).
_DEFAULT_LOOKBACK_DAYS = 365


@dataclass
class SyncResult:
    """Summary of a completed sync run."""

    oldest_fetched: date
    newest_fetched: date
    fetched: int
    upserted: int


async def sync_range(
    pool: asyncpg.Pool,
    oldest: date,
    newest: date,
) -> SyncResult:
    """Fetch all activities in [oldest, newest] and upsert into Postgres.

    Args:
        pool: Active asyncpg connection pool.
        oldest: First date to fetch (inclusive).
        newest: Last date to fetch (inclusive).

    Returns:
        SyncResult with counts of fetched and upserted activities.
    """
    async with IntervalsClient() as client:
        activities: list[Activity] = await client.get_activities(oldest, newest)

    upserted = await upsert_activities(pool, activities)

    return SyncResult(
        oldest_fetched=oldest,
        newest_fetched=newest,
        fetched=len(activities),
        upserted=upserted,
    )


async def sync_incremental(
    pool: asyncpg.Pool,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> SyncResult:
    """Sync new activities since the last stored activity date.

    If the database is empty, fetches the last `lookback_days` days.
    Otherwise, re-fetches from `_OVERLAP_DAYS` before the latest stored
    activity to pick up any edits, up to today.

    Args:
        pool: Active asyncpg connection pool.
        lookback_days: Days to fetch on a first-time (empty DB) sync.

    Returns:
        SyncResult with counts of fetched and upserted activities.
    """
    today = date.today()
    latest = await get_latest_activity_date(pool)

    if latest is None:
        oldest = today - timedelta(days=lookback_days)
    else:
        oldest = latest - timedelta(days=_OVERLAP_DAYS)

    return await sync_range(pool, oldest=oldest, newest=today)


async def run(
    oldest: date | None = None,
    newest: date | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    dsn: str | None = None,
) -> SyncResult:
    """Top-level entry point: set up the DB and run a sync.

    If oldest/newest are provided, syncs that exact range. Otherwise runs
    an incremental sync.

    Args:
        oldest: Explicit range start; triggers a full-range sync when set.
        newest: Explicit range end (defaults to today when oldest is set).
        lookback_days: Used only for the first incremental sync on an empty DB.
        dsn: Postgres DSN; defaults to POSTGRES_URL from config.

    Returns:
        SyncResult with counts of fetched and upserted activities.
    """
    pool = await create_pool(dsn)
    try:
        await init_schema(pool)

        if oldest is not None:
            result = await sync_range(
                pool,
                oldest=oldest,
                newest=newest or date.today(),
            )
        else:
            result = await sync_incremental(pool, lookback_days=lookback_days)
    finally:
        await pool.close()

    return result
