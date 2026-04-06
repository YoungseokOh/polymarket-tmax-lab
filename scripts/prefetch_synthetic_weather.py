"""Pre-fetch weather data (Open-Meteo + Wunderground) for synthetic snapshots.

This script pre-populates the HTTP disk cache with the exact same requests
that the build-dataset pipeline will make, using parallelism to speed up fetching.
Run this before build-dataset to dramatically reduce pipeline runtime.

Usage:
    uv run python scripts/prefetch_synthetic_weather.py [--workers N] [--start-year YEAR] [--end-year YEAR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmtmax.http import CachedHttpClient  # noqa: E402
from pmtmax.weather.openmeteo_client import OpenMeteoClient  # noqa: E402

CACHE_DIR = REPO_ROOT / "data/cache"
SYNTHETIC_SNAPSHOTS = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"

MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "kma_gdps", "gfs_seamless"]
HOURLY_VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]

ARCHIVE_BASE_URL = "https://historical-forecast-api.open-meteo.com"
WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_PAGE_URL_TEMPLATE = "https://www.wunderground.com/history/daily/{path}/date/{date}"


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_city_specs(snapshots_path: Path) -> dict[str, dict]:
    """Load one representative spec per city (all dates have same station coords)."""
    with snapshots_path.open() as f:
        entries = json.load(f)
    city_specs: dict[str, dict] = {}
    for entry in entries:
        spec = entry.get("spec")
        if spec and spec.get("city") and spec["city"] not in city_specs:
            city_specs[spec["city"]] = spec
    return city_specs


def prefetch_openmeteo_day(
    openmeteo: OpenMeteoClient,
    city: str,
    spec: dict,
    target_date: date,
    model: str,
) -> tuple[str, date, str, str]:
    """Fetch and cache one (city, date, model) open-meteo historical forecast.
    Returns (city, date, model, status).
    """
    lat = spec["station_lat"]
    lon = spec["station_lon"]
    tz = spec["timezone"]
    date_str = target_date.isoformat()
    try:
        openmeteo.historical_forecast(
            latitude=lat,
            longitude=lon,
            model=model,
            hourly=HOURLY_VARS,
            start_date=date_str,
            end_date=date_str,
            timezone=tz,
        )
        return city, target_date, model, "ok"
    except Exception as exc:  # noqa: BLE001
        return city, target_date, model, f"error:{exc!s:.80}"


def prefetch_wunderground_day(
    http: CachedHttpClient,
    spec: dict,
    target_date: date,
) -> tuple[str, date, str]:
    """Fetch and cache one (station, date) Wunderground history page.
    Returns (city, date, status).
    """
    station_id = spec["station_id"]
    source_url = spec.get("official_source_url", "")
    city = spec["city"]
    date_str = target_date.isoformat()

    # Build the Wunderground HTML page URL (same as TruthSource does)
    url = f"{source_url.rstrip('/')}/date/{date_str}"
    try:
        http.get_text(url, use_cache=True)
        return city, target_date, "ok"
    except Exception as exc:  # noqa: BLE001
        return city, target_date, f"error:{exc!s:.80}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch weather cache for synthetic snapshots")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--start-year", type=int, default=2016, help="Start year (default: 2016)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--skip-openmeteo", action="store_true", help="Skip Open-Meteo pre-fetch")
    parser.add_argument("--skip-wunderground", action="store_true", help="Skip Wunderground pre-fetch")
    parser.add_argument("--city", action="append", help="Only process specific cities (repeatable)")
    args = parser.parse_args()

    http = CachedHttpClient(cache_dir=CACHE_DIR, timeout_seconds=30, retries=3)
    openmeteo = OpenMeteoClient(
        http,
        base_url="https://api.open-meteo.com",
        archive_base_url=ARCHIVE_BASE_URL,
    )

    print(f"Loading city specs from {SYNTHETIC_SNAPSHOTS} ...", flush=True)
    city_specs = load_city_specs(SYNTHETIC_SNAPSHOTS)
    if args.city:
        city_specs = {c: s for c, s in city_specs.items() if c in args.city}
    print(f"  Cities: {sorted(city_specs)}", flush=True)

    start_date = date(args.start_year, 1, 1)
    end_date = date(args.end_year, 5, 29)
    all_dates = list(daterange(start_date, end_date))
    print(f"  Date range: {start_date} to {end_date} ({len(all_dates)} days)", flush=True)

    # -----------------------------------------------------------------------
    # Open-Meteo pre-fetch
    # -----------------------------------------------------------------------
    if not args.skip_openmeteo:
        total_om = len(city_specs) * len(all_dates) * len(MODELS)
        print(f"\n[Open-Meteo] Pre-fetching {total_om:,} (city×date×model) entries ...", flush=True)

        tasks_om = [
            (city, spec, d, model)
            for city, spec in sorted(city_specs.items())
            for d in all_dates
            for model in MODELS
        ]

        ok_count = 0
        error_count = 0
        skip_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(prefetch_openmeteo_day, openmeteo, city, spec, d, model): (city, d, model)
                for city, spec, d, model in tasks_om
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, target_date, model, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status.startswith("error"):
                    error_count += 1
                else:
                    skip_count += 1

                if i % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    remaining = (total_om - i) / rate if rate > 0 else 0
                    print(
                        f"  [OM] {i:,}/{total_om:,} done — ok={ok_count} err={error_count} "
                        f"({rate:.1f}/s, ETA {remaining/60:.0f}m)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    last error: {city} {target_date} {model}: {status}", flush=True)

        elapsed = time.time() - t0
        print(f"[Open-Meteo] Done: ok={ok_count} err={error_count} in {elapsed/60:.1f}m", flush=True)

    # -----------------------------------------------------------------------
    # Wunderground pre-fetch
    # -----------------------------------------------------------------------
    if not args.skip_wunderground:
        # Filter: only Wunderground-sourced cities
        wu_specs = {c: s for c, s in city_specs.items() if "wunderground" in s.get("official_source_name", "").lower()}
        total_wu = len(wu_specs) * len(all_dates)
        print(f"\n[Wunderground] Pre-fetching {total_wu:,} (station×date) HTML pages ...", flush=True)
        print(f"  Stations: {sorted(wu_specs)}", flush=True)

        tasks_wu = [
            (spec, d)
            for city, spec in sorted(wu_specs.items())
            for d in all_dates
        ]

        ok_count = 0
        error_count = 0
        t0 = time.time()

        # Wunderground is sensitive to rate limits - use fewer workers
        wu_workers = min(args.workers, 4)
        print(f"  Using {wu_workers} workers (Wunderground rate-limit safety)", flush=True)

        with ThreadPoolExecutor(max_workers=wu_workers) as executor:
            futures = {
                executor.submit(prefetch_wunderground_day, http, spec, d): (spec["city"], d)
                for spec, d in tasks_wu
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, target_date, status = future.result()
                if status == "ok":
                    ok_count += 1
                else:
                    error_count += 1

                if i % 500 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    remaining = (total_wu - i) / rate if rate > 0 else 0
                    print(
                        f"  [WU] {i:,}/{total_wu:,} done — ok={ok_count} err={error_count} "
                        f"({rate:.1f}/s, ETA {remaining/3600:.1f}h)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    last error: {city} {target_date}: {status}", flush=True)

        elapsed = time.time() - t0
        print(f"[Wunderground] Done: ok={ok_count} err={error_count} in {elapsed/60:.1f}m", flush=True)

    print("\nPre-fetch complete!", flush=True)


if __name__ == "__main__":
    main()
