"""PostgreSQL connection pool, schema initialisation, and activity queries."""

from datetime import date, datetime
from typing import Any

import asyncpg

from src.config import config

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_ACTIVITIES_TABLE = """
CREATE TABLE IF NOT EXISTS activities (
    id               BIGINT PRIMARY KEY,
    name             TEXT,
    sport_type       TEXT,
    start_date       TIMESTAMPTZ,
    duration_seconds INTEGER,
    distance_meters  FLOAT,
    elevation_meters FLOAT,
    avg_power_watts  FLOAT,
    normalized_power FLOAT,
    avg_hr           FLOAT,
    max_hr           FLOAT,
    tss              FLOAT,
    ctl              FLOAT,
    atl              FLOAT,
    tsb              FLOAT,
    intensity_factor FLOAT,
    avg_cadence      FLOAT,
    kilojoules       FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_activities_start_date ON activities(start_date);",
    "CREATE INDEX IF NOT EXISTS idx_activities_sport_type ON activities(sport_type);",
]

# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


async def create_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Args:
        dsn: Postgres connection string. Defaults to POSTGRES_URL from config.
    """
    return await asyncpg.create_pool(dsn or config.postgres_url)


async def init_schema(pool: asyncpg.Pool) -> None:
    """Create tables and indexes if they do not already exist."""
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_ACTIVITIES_TABLE)
        for stmt in _CREATE_INDEXES:
            await conn.execute(stmt)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

_UPSERT_ACTIVITY = """
INSERT INTO activities (
    id, name, sport_type, start_date, duration_seconds,
    distance_meters, elevation_meters, avg_power_watts, normalized_power,
    avg_hr, max_hr, tss, ctl, atl, tsb,
    intensity_factor, avg_cadence, kilojoules
) VALUES (
    $1, $2, $3, $4, $5,
    $6, $7, $8, $9,
    $10, $11, $12, $13, $14, $15,
    $16, $17, $18
)
ON CONFLICT (id) DO UPDATE SET
    name             = EXCLUDED.name,
    sport_type       = EXCLUDED.sport_type,
    start_date       = EXCLUDED.start_date,
    duration_seconds = EXCLUDED.duration_seconds,
    distance_meters  = EXCLUDED.distance_meters,
    elevation_meters = EXCLUDED.elevation_meters,
    avg_power_watts  = EXCLUDED.avg_power_watts,
    normalized_power = EXCLUDED.normalized_power,
    avg_hr           = EXCLUDED.avg_hr,
    max_hr           = EXCLUDED.max_hr,
    tss              = EXCLUDED.tss,
    ctl              = EXCLUDED.ctl,
    atl              = EXCLUDED.atl,
    tsb              = EXCLUDED.tsb,
    intensity_factor = EXCLUDED.intensity_factor,
    avg_cadence      = EXCLUDED.avg_cadence,
    kilojoules       = EXCLUDED.kilojoules
"""


async def upsert_activity(pool: asyncpg.Pool, activity: Any) -> None:
    """Insert or update a single activity row.

    Accepts either an Activity dataclass (from intervals_client) or a plain
    dict with matching keys.

    Args:
        pool: Active asyncpg connection pool.
        activity: Activity dataclass or dict with activity fields.
    """
    if hasattr(activity, "__dataclass_fields__"):
        a = activity
        values = (
            a.id, a.name, a.sport_type, a.start_date, a.duration_seconds,
            a.distance_meters, a.elevation_meters, a.avg_power_watts, a.normalized_power,
            a.avg_hr, a.max_hr, a.tss, a.ctl, a.atl, a.tsb,
            a.intensity_factor, a.avg_cadence, a.kilojoules,
        )
    else:
        d = activity
        values = (
            d["id"], d["name"], d["sport_type"], d["start_date"], d.get("duration_seconds"),
            d.get("distance_meters"), d.get("elevation_meters"), d.get("avg_power_watts"),
            d.get("normalized_power"), d.get("avg_hr"), d.get("max_hr"),
            d.get("tss"), d.get("ctl"), d.get("atl"), d.get("tsb"),
            d.get("intensity_factor"), d.get("avg_cadence"), d.get("kilojoules"),
        )

    async with pool.acquire() as conn:
        await conn.execute(_UPSERT_ACTIVITY, *values)


async def upsert_activities(pool: asyncpg.Pool, activities: list[Any]) -> int:
    """Bulk upsert a list of activities. Returns the count inserted/updated."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            for activity in activities:
                if hasattr(activity, "__dataclass_fields__"):
                    a = activity
                    values = (
                        a.id, a.name, a.sport_type, a.start_date, a.duration_seconds,
                        a.distance_meters, a.elevation_meters, a.avg_power_watts, a.normalized_power,
                        a.avg_hr, a.max_hr, a.tss, a.ctl, a.atl, a.tsb,
                        a.intensity_factor, a.avg_cadence, a.kilojoules,
                    )
                else:
                    d = activity
                    values = (
                        d["id"], d["name"], d["sport_type"], d["start_date"], d.get("duration_seconds"),
                        d.get("distance_meters"), d.get("elevation_meters"), d.get("avg_power_watts"),
                        d.get("normalized_power"), d.get("avg_hr"), d.get("max_hr"),
                        d.get("tss"), d.get("ctl"), d.get("atl"), d.get("tsb"),
                        d.get("intensity_factor"), d.get("avg_cadence"), d.get("kilojoules"),
                    )
                await conn.execute(_UPSERT_ACTIVITY, *values)
    return len(activities)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


async def get_latest_activity_date(pool: asyncpg.Pool) -> date | None:
    """Return the start_date of the most recent stored activity, or None."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT MAX(start_date) AS latest FROM activities"
        )
    if row and row["latest"]:
        return row["latest"].date()
    return None


async def get_activities_in_range(
    pool: asyncpg.Pool,
    start: date,
    end: date,
    sport_types: list[str] | None = None,
) -> list[asyncpg.Record]:
    """Fetch activity rows for a date range, optionally filtered by sport type.

    Args:
        pool: Active asyncpg connection pool.
        start: Range start date (inclusive).
        end: Range end date (inclusive).
        sport_types: If provided, restrict to these sport types.

    Returns:
        List of asyncpg Record objects ordered by start_date ascending.
    """
    if sport_types:
        query = """
            SELECT * FROM activities
            WHERE start_date >= $1::date
              AND start_date <  $2::date + INTERVAL '1 day'
              AND sport_type = ANY($3::text[])
            ORDER BY start_date
        """
        params: tuple = (start, end, sport_types)
    else:
        query = """
            SELECT * FROM activities
            WHERE start_date >= $1::date
              AND start_date <  $2::date + INTERVAL '1 day'
            ORDER BY start_date
        """
        params = (start, end)

    async with pool.acquire() as conn:
        return await conn.fetch(query, *params)


async def get_period_aggregates(
    pool: asyncpg.Pool,
    start: date,
    end: date,
    sport_types: list[str] | None = None,
) -> dict:
    """Compute aggregate training metrics for a date range.

    Returns a dict with keys: activity_count, total_duration_hours,
    total_distance_km, total_tss, avg_power_watts, avg_normalized_power,
    avg_hr, total_elevation_meters.

    Args:
        pool: Active asyncpg connection pool.
        start: Range start date (inclusive).
        end: Range end date (inclusive).
        sport_types: If provided, restrict to these sport types.
    """
    base = """
        SELECT
            COUNT(*)                                       AS activity_count,
            COALESCE(SUM(duration_seconds) / 3600.0, 0)   AS total_duration_hours,
            COALESCE(SUM(distance_meters)  / 1000.0, 0)   AS total_distance_km,
            COALESCE(SUM(tss), 0)                         AS total_tss,
            AVG(avg_power_watts)                          AS avg_power_watts,
            AVG(normalized_power)                         AS avg_normalized_power,
            AVG(avg_hr)                                   AS avg_hr,
            COALESCE(SUM(elevation_meters), 0)            AS total_elevation_meters
        FROM activities
        WHERE start_date >= $1::date
          AND start_date <  $2::date + INTERVAL '1 day'
    """

    if sport_types:
        query = base + " AND sport_type = ANY($3::text[])"
        params: tuple = (start, end, sport_types)
    else:
        query = base
        params = (start, end)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)

    return dict(row) if row else {}
