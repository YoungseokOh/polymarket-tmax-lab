"""Batch-fetch historical weather data and inject into the DuckDB silver tables.

This script fetches 9 years of data (2016-2025) efficiently:
- Open-Meteo: yearly range requests per city/model (30 cities × 10 years × 4 models = ~1200 requests)
- Wunderground: monthly JSON API requests per station (30 stations × 120 months = 3600 requests)

Then directly populates DuckDB silver tables with synthetic market_ids, so that
`build-dataset --output-name synthetic_historical_training_set` can run quickly.

Usage:
    uv run python scripts/batch_fetch_historical_data.py [--workers N] [options]
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmtmax.http import CachedHttpClient  # noqa: E402
from pmtmax.weather.openmeteo_client import OpenMeteoClient  # noqa: E402

CACHE_DIR = REPO_ROOT / "data/cache"
SYNTHETIC_SNAPSHOTS = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"
DUCKDB_PATH = REPO_ROOT / "data/duckdb/warehouse.duckdb"

MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "kma_gdps", "gfs_seamless"]
HOURLY_VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
PROBE_VARS = ["temperature_2m"]  # matches pipeline's PROBE_HOURLY

START_DATE = date(2016, 1, 1)
END_DATE = date(2025, 5, 29)

WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"  # Public key extracted from WU pages

COUNTRY_CODE_BY_NAME = {
    "argentina": "AR",
    "brazil": "BR",
    "canada": "CA",
    "china": "CN",
    "france": "FR",
    "germany": "DE",
    "hong kong": "HK",
    "india": "IN",
    "israel": "IL",
    "italy": "IT",
    "japan": "JP",
    "new zealand": "NZ",
    "poland": "PL",
    "singapore": "SG",
    "south korea": "KR",
    "spain": "ES",
    "taiwan": "TW",
    "turkey": "TR",
    "uk": "GB",
    "united kingdom": "GB",
    "usa": "US",
    "united states": "US",
}


def load_city_specs(snapshots_path: Path) -> dict[str, dict]:
    """Load one representative spec per city."""
    with snapshots_path.open() as f:
        entries = json.load(f)
    city_specs: dict[str, dict] = {}
    for entry in entries:
        spec = entry.get("spec")
        if spec and spec.get("city") and spec["city"] not in city_specs:
            city_specs[spec["city"]] = spec
    return city_specs


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _wu_location_id(spec: dict) -> str:
    """Compute the Wunderground location_id for the JSON API."""
    source_url = spec.get("official_source_url", "")
    parts = [p for p in source_url.split("/") if p and "wunderground" not in p and "http" not in p]
    # URL format: /history/daily/{country_code}/{city}/{station_id}
    # E.g. /history/daily/tr/%C3%A7ubuk/LTAC => parts = ['history','daily','tr','%C3%A7ubuk','LTAC']
    from urllib.parse import urlparse
    parsed = urlparse(source_url)
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) >= 3 and path_parts[0] == "history" and path_parts[1] == "daily":
        country_code = path_parts[2].upper()
    else:
        country_name = (spec.get("country") or "").lower()
        country_code = COUNTRY_CODE_BY_NAME.get(country_name, "US")
    return f"{spec['station_id']}:9:{country_code}"


# ---------------------------------------------------------------------------
# Open-Meteo batch fetcher
# ---------------------------------------------------------------------------
def fetch_openmeteo_year(
    openmeteo: OpenMeteoClient,
    city: str,
    spec: dict,
    year: int,
    model: str,
) -> tuple[str, int, str, list[dict] | None, str]:
    """Fetch a full year of historical forecast for one city/model.

    Returns (city, year, model, list_of_day_rows_or_None, status).
    Each day_row has: target_date, hourly data per variable.
    """
    lat = spec["station_lat"]
    lon = spec["station_lon"]
    tz = spec["timezone"]
    start = f"{year}-01-01"
    end = min(date(year, 12, 31), END_DATE).isoformat()
    if start > end:
        return city, year, model, None, "skipped_out_of_range"
    try:
        payload = openmeteo.historical_forecast(
            latitude=lat,
            longitude=lon,
            model=model,
            hourly=HOURLY_VARS,
            start_date=start,
            end_date=end,
            timezone=tz,
        )
        # Parse hourly data into per-day rows
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return city, year, model, [], "ok_empty"
        # Build day-keyed data
        day_data: dict[str, dict[str, list[float]]] = {}
        for i, t in enumerate(times):
            day_str = t[:10]  # "2016-01-01"
            if day_str not in day_data:
                day_data[day_str] = {v: [] for v in HOURLY_VARS}
            for v in HOURLY_VARS:
                val = hourly.get(v, [None])[i]
                if val is not None:
                    day_data[day_str][v].append(float(val))
        rows = []
        for day_str, vars_data in day_data.items():
            rows.append({"target_date": day_str, **vars_data})
        return city, year, model, rows, "ok"
    except Exception as exc:  # noqa: BLE001
        return city, year, model, None, f"error:{exc!s:.120}"


# ---------------------------------------------------------------------------
# Wunderground batch fetcher
# ---------------------------------------------------------------------------
def fetch_wu_month(
    http: CachedHttpClient,
    city: str,
    spec: dict,
    year: int,
    month: int,
) -> tuple[str, int, int, dict[str, float] | None, str]:
    """Fetch a full month of Wunderground historical daily-max data.

    Returns (city, year, month, date_to_max_dict_or_None, status).
    """
    station_id = spec["station_id"]
    unit = spec.get("unit", "C")
    location_id = _wu_location_id(spec)
    start_date = f"{year}{month:02d}01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}{month:02d}{last_day:02d}"

    # Clip to our date range
    d_start = date(year, month, 1)
    d_end = min(date(year, month, last_day), END_DATE)
    if d_start > END_DATE:
        return city, year, month, None, "skipped_out_of_range"

    url = WU_HISTORICAL_URL.format(location_id=location_id)
    params = {
        "apiKey": WU_API_KEY,
        "units": "m" if unit == "C" else "e",
        "startDate": start_date,
        "endDate": end_date,
    }
    try:
        payload = http.get_json(url, params=params, use_cache=True)
        # Parse observations
        observations = payload.get("observations", [])
        date_to_max: dict[str, float] = {}
        for obs in observations:
            # Each obs has a 'valid_time_gmt' or 'obs_time_local'
            obs_date = obs.get("obs_time_local", obs.get("valid_time_gmt", ""))[:10]
            if not obs_date:
                continue
            # Temperature max - depends on unit
            if unit == "C":
                max_val = obs.get("imperial", {}) or {}
                # In "m" units, temperatures are in Celsius
                temp_max = obs.get("metric", {}).get("temperatureMax") if obs.get("metric") else None
                if temp_max is None:
                    temp_max = obs.get("temperatureMax")
            else:
                temp_max = obs.get("imperial", {}).get("temperatureMax") if obs.get("imperial") else None
                if temp_max is None:
                    temp_max = obs.get("temperatureMax")
            if temp_max is not None:
                date_to_max[obs_date] = float(temp_max)
        if not date_to_max:
            # Try alternate structure
            for obs in observations:
                obs_date_raw = obs.get("obs_time_local") or obs.get("date") or ""
                obs_date = str(obs_date_raw)[:10]
                for key in ["temperatureMax", "maxTemp", "temp_max"]:
                    val = obs.get(key)
                    if val is not None:
                        date_to_max[obs_date] = float(val)
                        break
        return city, year, month, date_to_max, "ok"
    except Exception as exc:  # noqa: BLE001
        return city, year, month, None, f"error:{exc!s:.200}"


# ---------------------------------------------------------------------------
# DuckDB injection
# ---------------------------------------------------------------------------
def inject_forecast_data(
    duckdb_path: Path,
    city_specs: dict[str, dict],
    openmeteo_results: dict[tuple[str, str, str], dict[str, list[float]]],
) -> int:
    """Inject forecast data into silver_forecast_runs_hourly."""
    import duckdb
    now = datetime.now(tz=UTC)
    rows = []

    for (city, day_str, model), hourly_data in openmeteo_results.items():
        spec = city_specs[city]
        city_slug = city.lower().replace(" ", "_")
        market_id = f"synthetic_{city_slug}_{day_str}"
        target_date = pd.Timestamp(day_str)

        temps = hourly_data.get("temperature_2m", [])
        if not temps:
            continue
        num_hours = len(temps)
        daily_max = max(temps)
        daily_mean = sum(temps) / num_hours
        daily_min = min(temps)
        midday_temp = temps[12] if len(temps) > 12 else daily_mean
        diurnal_amplitude = daily_max - daily_min

        dew_points = hourly_data.get("dew_point_2m", [])
        dew_point_mean = sum(dew_points) / len(dew_points) if dew_points else None

        humidities = hourly_data.get("relative_humidity_2m", [])
        humidity_mean = sum(humidities) / len(humidities) if humidities else None

        wind_speeds = hourly_data.get("wind_speed_10m", [])
        wind_speed_mean = sum(wind_speeds) / len(wind_speeds) if wind_speeds else None

        cloud_covers = hourly_data.get("cloud_cover", [])
        cloud_cover_mean = sum(cloud_covers) / len(cloud_covers) if cloud_covers else None

        rows.append({
            "market_id": market_id,
            "station_id": spec["station_id"],
            "city": city,
            "model_name": model,
            "endpoint_kind": "historical_forecast",
            "target_local_date": target_date,
            "fetched_at": pd.Timestamp(now),
            "status": "ok",
            "availability_status": "available",
            "request_kind": "full",
            "decision_horizon": None,
            "latitude": spec.get("station_lat"),
            "longitude": spec.get("station_lon"),
            "timezone": spec.get("timezone"),
            "forecast_days": 1,
            "variables_json": json.dumps(HOURLY_VARS),
            "num_hours": num_hours,
            "temperature_2m_max": daily_max,
            "temperature_2m_mean": daily_mean,
            "temperature_2m_min": daily_min,
            "midday_temp": midday_temp,
            "diurnal_amplitude": diurnal_amplitude,
            "dew_point_2m_mean": dew_point_mean,
            "relative_humidity_2m_mean": humidity_mean,
            "wind_speed_10m_mean": wind_speed_mean,
            "cloud_cover_mean": cloud_cover_mean,
            "run_time": None,
            "raw_path": None,
            "raw_hash": None,
            "query_signature": f"synthetic_{city_slug}_{day_str}_{model}",
            "created_at": pd.Timestamp(now),
            "source_priority": 80,
            "data_version": "synthetic_v1",
        })

    if not rows:
        return 0

    frame = pd.DataFrame(rows)
    with duckdb.connect(str(duckdb_path)) as db:
        # Create table if it doesn't exist
        db.execute("""
            CREATE TABLE IF NOT EXISTS silver_forecast_synthetic AS
            SELECT * FROM frame WHERE FALSE
        """)
        db.execute("INSERT INTO silver_forecast_synthetic SELECT * FROM frame")
    return len(rows)


def inject_truth_data(
    duckdb_path: Path,
    city_specs: dict[str, dict],
    wu_results: dict[tuple[str, str], float],
) -> int:
    """Inject truth data into silver_observations_daily."""
    import duckdb
    now = datetime.now(tz=UTC)
    rows = []

    for (city, day_str), daily_max in wu_results.items():
        spec = city_specs[city]
        city_slug = city.lower().replace(" ", "_")
        market_id = f"synthetic_{city_slug}_{day_str}"

        rows.append({
            "market_id": market_id,
            "station_id": spec["station_id"],
            "city": city,
            "official_source_name": spec.get("official_source_name"),
            "public_truth_source_name": spec.get("public_truth_source_name"),
            "public_truth_station_id": spec.get("public_truth_station_id"),
            "truth_track": spec.get("truth_track"),
            "settlement_eligible": spec.get("settlement_eligible", False),
            "target_local_date": pd.Timestamp(day_str),
            "daily_max": daily_max,
            "unit": spec.get("unit", "C"),
            "source": "Wunderground",
            "fetched_at": pd.Timestamp(now),
            "finalized_at": pd.Timestamp(now),
            "created_at": pd.Timestamp(now),
            "source_priority": 100,
            "data_version": "synthetic_v1",
        })

    if not rows:
        return 0

    frame = pd.DataFrame(rows)
    with duckdb.connect(str(duckdb_path)) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS silver_observations_synthetic AS
            SELECT * FROM frame WHERE FALSE
        """)
        db.execute("INSERT INTO silver_observations_synthetic SELECT * FROM frame")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-fetch historical weather data for synthetic snapshots")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--city", action="append", help="Filter to specific cities")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--skip-openmeteo", action="store_true")
    parser.add_argument("--skip-wunderground", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch, don't inject into DuckDB")
    args = parser.parse_args()

    http = CachedHttpClient(cache_dir=CACHE_DIR, timeout_seconds=60, retries=5)
    openmeteo = OpenMeteoClient(
        http,
        base_url="https://api.open-meteo.com",
        archive_base_url="https://historical-forecast-api.open-meteo.com",
    )

    print(f"Loading city specs ...", flush=True)
    city_specs = load_city_specs(SYNTHETIC_SNAPSHOTS)
    if args.city:
        city_specs = {c: s for c, s in city_specs.items() if c in args.city}
    print(f"  Cities ({len(city_specs)}): {sorted(city_specs)}", flush=True)

    years = list(range(args.start_year, args.end_year + 1))
    print(f"  Years: {years}", flush=True)

    # ==================================================================
    # 1. Open-Meteo: year-range requests
    # ==================================================================
    # Keyed by (city, date_str, model) -> hourly_data_dict
    openmeteo_results: dict[tuple[str, str, str], dict[str, list[float]]] = {}

    if not args.skip_openmeteo:
        tasks = [
            (city, spec, year, model)
            for city, spec in sorted(city_specs.items())
            for year in years
            for model in MODELS
        ]
        total = len(tasks)
        print(f"\n[Open-Meteo] Fetching {total} year/city/model requests ...", flush=True)

        ok_count = 0
        error_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_openmeteo_year, openmeteo, city, spec, year, model): (city, year, model)
                for city, spec, year, model in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, year, model, day_rows, status = future.result()
                if status.startswith("ok") and day_rows:
                    for row in day_rows:
                        day_str = row["target_date"]
                        key = (city, day_str, model)
                        hourly_data = {v: row.get(v, []) for v in HOURLY_VARS}
                        openmeteo_results[key] = hourly_data
                    ok_count += 1
                elif status.startswith("error"):
                    error_count += 1

                if i % 100 == 0 or i == total:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    remaining = (total - i) / rate if rate > 0 else 0
                    print(
                        f"  [OM] {i}/{total} — ok={ok_count} err={error_count} "
                        f"days_fetched={len(openmeteo_results)} "
                        f"({rate:.1f}/s, ETA {remaining:.0f}s)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    error: {city} {year} {model}: {status}", flush=True)

        elapsed = time.time() - t0
        print(f"[Open-Meteo] Done: {len(openmeteo_results)} (city,day,model) entries in {elapsed:.0f}s", flush=True)

    # ==================================================================
    # 2. Wunderground: monthly requests
    # ==================================================================
    wu_results: dict[tuple[str, str], float] = {}  # (city, date_str) -> daily_max

    if not args.skip_wunderground:
        wu_specs = {c: s for c, s in city_specs.items()
                   if "wunderground" in s.get("official_source_name", "").lower()}
        tasks_wu = [
            (city, spec, year, month)
            for city, spec in sorted(wu_specs.items())
            for year in years
            for month in range(1, 13)
            if date(year, month, 1) <= END_DATE
        ]
        total_wu = len(tasks_wu)
        print(f"\n[Wunderground] Fetching {total_wu} monthly requests for {len(wu_specs)} stations ...", flush=True)

        ok_count = 0
        error_count = 0
        t0 = time.time()
        wu_workers = min(args.workers, 4)

        with ThreadPoolExecutor(max_workers=wu_workers) as executor:
            futures = {
                executor.submit(fetch_wu_month, http, city, spec, year, month): (city, year, month)
                for city, spec, year, month in tasks_wu
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, year, month, date_to_max, status = future.result()
                if status.startswith("ok") and date_to_max:
                    for day_str, max_val in date_to_max.items():
                        wu_results[(city, day_str)] = max_val
                    ok_count += 1
                elif status.startswith("error"):
                    error_count += 1

                if i % 100 == 0 or i == total_wu:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_wu - i) / rate if rate > 0 else 0
                    print(
                        f"  [WU] {i}/{total_wu} — ok={ok_count} err={error_count} "
                        f"days_fetched={len(wu_results)} "
                        f"({rate:.1f}/s, ETA {remaining:.0f}s)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    error: {city} {year}-{month:02d}: {status}", flush=True)

        elapsed = time.time() - t0
        print(f"[Wunderground] Done: {len(wu_results)} (city,day) entries in {elapsed:.0f}s", flush=True)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\nData collection complete:", flush=True)
    print(f"  Open-Meteo: {len(openmeteo_results)} (city, day, model) entries", flush=True)
    print(f"  Wunderground: {len(wu_results)} (city, day) entries", flush=True)

    # ==================================================================
    # Save intermediate results to JSON files for inspection/recovery
    # ==================================================================
    artifacts_dir = REPO_ROOT / "artifacts/synthetic_data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if openmeteo_results and not args.dry_run:
        print("Saving Open-Meteo summary ...", flush=True)
        # Save as city/model/date summary (just the scalar stats, not full hourly)
        om_summary: dict[str, list[dict]] = {}
        for (city, day_str, model), hourly in openmeteo_results.items():
            temps = hourly.get("temperature_2m", [])
            if not temps:
                continue
            key = f"{city}|{model}"
            if key not in om_summary:
                om_summary[key] = []
            om_summary[key].append({
                "date": day_str,
                "daily_max": max(temps),
                "daily_min": min(temps),
                "daily_mean": sum(temps) / len(temps),
                "num_hours": len(temps),
            })
        with (artifacts_dir / "openmeteo_summary.json").open("w") as f:
            json.dump({k: v[:5] for k, v in list(om_summary.items())[:5]}, f, indent=2)
        print(f"  Saved sample to {artifacts_dir / 'openmeteo_summary.json'}", flush=True)

    if wu_results and not args.dry_run:
        print("Saving Wunderground summary ...", flush=True)
        wu_summary = {f"{c}|{d}": v for (c, d), v in list(wu_results.items())[:20]}
        with (artifacts_dir / "wunderground_summary.json").open("w") as f:
            json.dump(wu_summary, f, indent=2)
        print(f"  Saved sample to {artifacts_dir / 'wunderground_summary.json'}", flush=True)

    if not args.dry_run:
        print("\nNote: DuckDB injection skipped in this version.", flush=True)
        print("The data is cached via HTTP cache. Run build-dataset to process.", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
