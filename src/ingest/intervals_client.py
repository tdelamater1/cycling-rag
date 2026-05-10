"""Client for the intervals.icu REST API."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import AsyncIterator

import httpx

from src.config import config

_BASE_URL = "https://intervals.icu/api/v1"
_MIN_REQUEST_INTERVAL = 1.0  # seconds — respect rate limit


@dataclass
class Activity:
    """Structured representation of an intervals.icu activity."""

    id: int
    name: str
    sport_type: str
    start_date: datetime
    duration_seconds: int | None
    distance_meters: float | None
    elevation_meters: float | None
    avg_power_watts: float | None
    normalized_power: float | None
    avg_hr: float | None
    max_hr: float | None
    tss: float | None
    ctl: float | None
    atl: float | None
    tsb: float | None
    intensity_factor: float | None
    avg_cadence: float | None
    kilojoules: float | None


@dataclass
class WellnessEntry:
    """Daily wellness / fitness metrics from intervals.icu."""

    date: date
    ctl: float | None  # Chronic Training Load (fitness)
    atl: float | None  # Acute Training Load (fatigue)
    tsb: float | None  # Training Stress Balance (form)


def _parse_activity(raw: dict) -> Activity:
    """Map a raw intervals.icu activity dict to an Activity dataclass."""
    start_raw = raw.get("start_date_local") or raw.get("start_date", "")
    # intervals.icu returns ISO8601; strip trailing Z if present
    start_date = datetime.fromisoformat(start_raw.rstrip("Z"))

    ctl = raw.get("icu_ctl")
    atl = raw.get("icu_atl")
    icu_intensity = raw.get("icu_intensity")
    icu_joules = raw.get("icu_joules")

    return Activity(
        id=int(str(raw["id"]).lstrip("i")),
        name=raw.get("name", ""),
        sport_type=raw.get("type", ""),
        start_date=start_date,
        duration_seconds=raw.get("moving_time") or raw.get("elapsed_time"),
        distance_meters=raw.get("distance"),
        elevation_meters=raw.get("total_elevation_gain"),
        avg_power_watts=raw.get("icu_average_watts"),
        normalized_power=raw.get("icu_weighted_avg_watts"),
        avg_hr=raw.get("average_heartrate"),
        max_hr=raw.get("max_heartrate"),
        tss=raw.get("icu_training_load"),
        ctl=ctl,
        atl=atl,
        tsb=(ctl - atl) if (ctl is not None and atl is not None) else None,
        intensity_factor=icu_intensity / 100.0 if icu_intensity is not None else None,
        avg_cadence=raw.get("average_cadence"),
        kilojoules=icu_joules / 1000.0 if icu_joules is not None else None,
    )


def _parse_wellness(raw: dict) -> WellnessEntry:
    """Map a raw intervals.icu wellness dict to a WellnessEntry dataclass."""
    return WellnessEntry(
        date=date.fromisoformat(raw["id"]),  # wellness records use date as id
        ctl=raw.get("ctl"),
        atl=raw.get("atl"),
        tsb=raw.get("tsb"),
    )


class IntervalsClient:
    """Async client for the intervals.icu API.

    Use as an async context manager:

        async with IntervalsClient() as client:
            activities = await client.get_activities(oldest, newest)
    """

    def __init__(
        self,
        athlete_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._athlete_id = athlete_id or config.intervals_athlete_id
        self._api_key = api_key or config.intervals_api_key
        self._client: httpx.AsyncClient | None = None
        self._last_request_at: float = 0.0

    async def __aenter__(self) -> "IntervalsClient":
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            # intervals.icu Basic auth: username must literally be "API_KEY"
            auth=("API_KEY", self._api_key),
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> list | dict:
        """Rate-limited GET request."""
        assert self._client is not None, "Use IntervalsClient as an async context manager"

        elapsed = asyncio.get_event_loop().time() - self._last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        response = await self._client.get(path, params=params)
        self._last_request_at = asyncio.get_event_loop().time()
        response.raise_for_status()
        return response.json()

    async def get_activities(
        self,
        oldest: date,
        newest: date,
    ) -> list[Activity]:
        """Fetch all activities between oldest and newest (inclusive).

        Args:
            oldest: Start of the date range (inclusive).
            newest: End of the date range (inclusive).

        Returns:
            List of Activity objects sorted ascending by start_date.
        """
        data = await self._get(
            f"/athlete/{self._athlete_id}/activities",
            params={
                "oldest": oldest.isoformat(),
                "newest": newest.isoformat(),
            },
        )
        activities = [_parse_activity(item) for item in data]
        activities.sort(key=lambda a: a.start_date)
        return activities

    async def iter_activities(
        self,
        oldest: date,
        newest: date,
        sport_types: list[str] | None = None,
    ) -> AsyncIterator[Activity]:
        """Yield activities one at a time, optionally filtered by sport type.

        Args:
            oldest: Start of the date range (inclusive).
            newest: End of the date range (inclusive).
            sport_types: If given, only yield activities whose sport_type is in
                this list (e.g. ["Ride", "VirtualRide"]).
        """
        activities = await self.get_activities(oldest, newest)
        for activity in activities:
            if sport_types is None or activity.sport_type in sport_types:
                yield activity

    async def get_wellness(
        self,
        oldest: date,
        newest: date,
    ) -> list[WellnessEntry]:
        """Fetch daily wellness data (CTL/ATL/TSB) for a date range.

        Args:
            oldest: Start of the date range (inclusive).
            newest: End of the date range (inclusive).

        Returns:
            List of WellnessEntry objects sorted ascending by date.
        """
        data = await self._get(
            f"/athlete/{self._athlete_id}/wellness",
            params={
                "oldest": oldest.isoformat(),
                "newest": newest.isoformat(),
            },
        )
        entries = [_parse_wellness(item) for item in data]
        entries.sort(key=lambda e: e.date)
        return entries
