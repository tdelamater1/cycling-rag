"""Manual one-off sync: pull activities from intervals.icu into Postgres.

Usage examples:
  # Incremental — fetches since last stored activity (or 365 days on first run)
  python scripts/sync_now.py

  # Fetch the last N days
  python scripts/sync_now.py --days 90

  # Fetch from a specific date to today
  python scripts/sync_now.py --since 2024-01-01

  # Fetch an explicit date range
  python scripts/sync_now.py --since 2024-01-01 --until 2024-06-30
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta

# Allow running as `python scripts/sync_now.py` from the project root.
sys.path.insert(0, ".")

from src.ingest.sync import run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync cycling activities from intervals.icu into Postgres.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Fetch all activities from this date to --until (default: today).",
    )
    group.add_argument(
        "--days",
        type=int,
        metavar="N",
        help="Fetch the last N days (overrides incremental logic).",
    )
    parser.add_argument(
        "--until",
        metavar="YYYY-MM-DD",
        default=None,
        help="End date for the range (inclusive). Only used with --since.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    oldest: date | None = None
    newest: date | None = None
    lookback_days: int = 365

    if args.since:
        oldest = date.fromisoformat(args.since)
        newest = date.fromisoformat(args.until) if args.until else date.today()
    elif args.days:
        oldest = date.today() - timedelta(days=args.days)
        newest = date.today()

    if oldest:
        print(f"Syncing {oldest} → {newest} …")
    else:
        print("Running incremental sync …")

    result = await run(oldest=oldest, newest=newest, lookback_days=lookback_days)

    print(
        f"Done. Fetched {result.fetched} activities "
        f"({result.oldest_fetched} → {result.newest_fetched}), "
        f"upserted {result.upserted}."
    )


if __name__ == "__main__":
    asyncio.run(main())
