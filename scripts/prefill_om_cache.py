"""Pre-fill HTTP cache with per-day Open-Meteo historical forecast responses.

Fetches yearly Open-Meteo data (1 request per city×model×year×variable_set) and
splits into per-day cache entries matching what the pipeline's backfill_forecasts
step expects.

The pipeline makes TWO types of per-day requests per (city, model):
1. Probe: hourly=["temperature_2m"] (PROBE_HOURLY)
2. Full: hourly=["temperature_2m,dew_point_2m,relative_humidity_2m,wind_speed_10m,cloud_cover"]

Both use start_date=X&end_date=X format.

This reduces ~824K network requests to ~1,200 yearly range requests.

Usage:
    uv run python scripts/prefill_om_cache.py [--workers N] [--city CITY]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmtmax.http import CachedHttpClient  # noqa: E402
from pmtmax.utils import stable_hash  # noqa: E402
from pmtmax.weather.openmeteo_client import OpenMeteoClient  # noqa: E402

CACHE_DIR = REPO_ROOT / "data/cache"
SYNTHETIC_SNAPSHOTS = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"

START_DATE = date(2016, 1, 1)
END_DATE = date(2025, 5, 29)

ARCHIVE_BASE_URL = "https://historical-forecast-api.open-meteo.com"
MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "kma_gdps", "gfs_seamless"]

# Both variable sets used by the pipeline
PROBE_HOURLY = ["temperature_2m"]
FULL_HOURLY = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]


def load_city_specs(snapshots_path: Path) -> dict[str, dict]:
    with snapshots_path.open() as f:
        entries = json.load(f)
    city_specs: dict[str, dict] = {}
    for entry in entries:
        spec = entry.get("spec")
        if spec and spec.get("city") and spec["city"] not in city_specs:
            city_specs[spec["city"]] = spec
    return city_specs


def _om_cache_path(params: dict) -> Path:
    url = f"{ARCHIVE_BASE_URL}/v1/forecast"
    key = json.dumps({"url": url, "params": params}, sort_keys=True, default=str)
    return CACHE_DIR / f"{stable_hash(key)}.json"


def process_city_year_model_vars(
    openmeteo: OpenMeteoClient,
    city: str,
    spec: dict,
    year: int,
    model: str,
    hourly_vars: list[str],
) -> tuple[str, int, str, str, int, int, str]:
    """Fetch one year of OM data and split into per-day cache entries.

    Returns (city, year, model, vars_key, days_fetched, days_cached, status).
    """
    lat = spec["station_lat"]
    lon = spec["station_lon"]
    tz = spec["timezone"]
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), END_DATE)
    if start > END_DATE:
        return city, year, model, "", 0, 0, "skipped_out_of_range"

    vars_key = "probe" if hourly_vars == PROBE_HOURLY else "full"

    try:
        # Fetch yearly range
        payload = openmeteo.historical_forecast(
            latitude=lat,
            longitude=lon,
            model=model,
            hourly=hourly_vars,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            timezone=tz,
        )
    except Exception as exc:  # noqa: BLE001
        return city, year, model, vars_key, 0, 0, f"error:{exc!s:.200}"

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return city, year, model, vars_key, 0, 0, "ok_empty"

    # Group by day
    day_data: dict[str, dict[str, list]] = {}
    for i, t in enumerate(times):
        day_str = t[:10]
        if day_str not in day_data:
            day_data[day_str] = {"time": [], **{v: [] for v in hourly_vars}}
        day_data[day_str]["time"].append(t)
        for v in hourly_vars:
            vals = hourly.get(v, [])
            day_data[day_str][v].append(vals[i] if i < len(vals) else None)

    # Write per-day cache entries
    days_written = 0
    url = f"{ARCHIVE_BASE_URL}/v1/forecast"
    for day_str, day_hourly in day_data.items():
        # Build the response payload that the pipeline would cache for per-day fetch
        per_day_payload = {
            "latitude": lat,
            "longitude": lon,
            "generationtime_ms": 0.1,
            "utc_offset_seconds": 0,
            "timezone": tz,
            "hourly_units": dict.fromkeys(hourly_vars, "°C"),
            "hourly": day_hourly,
            "model": model,
        }
        # Compute the per-day cache key params
        per_day_params = {
            "end_date": day_str,
            "hourly": ",".join(hourly_vars),
            "latitude": lat,
            "longitude": lon,
            "models": model,
            "start_date": day_str,
            "timezone": tz,
        }
        cache_path = _om_cache_path(per_day_params)
        if not cache_path.exists():
            cache_path.write_text(json.dumps(per_day_payload, indent=2, sort_keys=True))
        days_written += 1

    return city, year, model, vars_key, len(day_data), days_written, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fill Open-Meteo per-day cache from yearly fetches")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--city", action="append", help="Filter cities")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--probe-only", action="store_true", help="Only fill probe cache (temperature_2m)")
    parser.add_argument("--full-only", action="store_true", help="Only fill full cache (all variables)")
    args = parser.parse_args()

    http = CachedHttpClient(cache_dir=CACHE_DIR, timeout_seconds=60, retries=5)
    openmeteo = OpenMeteoClient(
        http,
        base_url="https://api.open-meteo.com",
        archive_base_url=ARCHIVE_BASE_URL,
    )

    print("Loading city specs ...", flush=True)
    city_specs = load_city_specs(SYNTHETIC_SNAPSHOTS)
    if args.city:
        city_specs = {c: s for c, s in city_specs.items() if c in args.city}
    print(f"  Cities ({len(city_specs)}): {sorted(city_specs)}", flush=True)

    years = list(range(args.start_year, min(args.end_year, 2025) + 1))

    # Determine which variable sets to process
    var_sets = []
    if not args.full_only:
        var_sets.append(("probe", PROBE_HOURLY))
    if not args.probe_only:
        var_sets.append(("full", FULL_HOURLY))

    tasks = [
        (city, city_specs[city], year, model, hourly_vars)
        for city in sorted(city_specs)
        for year in years
        for model in MODELS
        for var_name, hourly_vars in var_sets
    ]
    total = len(tasks)
    print(f"  Tasks: {total} (city×year×model×var_set)", flush=True)

    ok_count = 0
    error_count = 0
    total_days = 0
    total_cached = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_city_year_model_vars, openmeteo, city, spec, year, model, hourly_vars
            ): (city, year, model, "probe" if hourly_vars == PROBE_HOURLY else "full")
            for city, spec, year, model, hourly_vars in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            city, year, model, vars_key, days_fetched, days_cached, status = future.result()
            if status.startswith("ok"):
                ok_count += 1
                total_days += days_fetched
                total_cached += days_cached
            else:
                error_count += 1

            if i % 200 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{total} ok={ok_count} err={error_count} "
                    f"days={total_days:,} cached={total_cached:,} "
                    f"({rate:.1f}/s ETA {remaining:.0f}s)",
                    flush=True,
                )
                if status.startswith("error"):
                    print(f"    err: {city} {year} {model} {vars_key}: {status}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {ok_count} tasks OK, {error_count} errors in {elapsed:.0f}s", flush=True)
    print(f"  Days covered: {total_days:,}", flush=True)
    print(f"  Cache entries written: {total_cached:,}", flush=True)


if __name__ == "__main__":
    main()
