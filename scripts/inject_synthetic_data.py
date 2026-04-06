"""Inject synthetic historical weather data into DuckDB silver tables.

Strategy:
  1. Open-Meteo ERA5 archive: yearly range requests per city (1 model = era5_reanalysis)
     Each year-city request covers all 365/366 days. ~30 cities × 10 years = 300 requests.
  2. Wunderground: monthly JSON API requests per station.
     ~30 stations × 120 months = 3600 requests.
  3. Directly INSERT into silver_forecast_runs_hourly and silver_observations_daily
     with synthetic market_ids (synthetic_{city_slug}_{date}).
  4. Run build-dataset for materialization only.

ERA5 is used as the forecast proxy (it is the actual observed atmospheric state).
We store it under all 4 model names so the pipeline's feature computation works.

Usage:
    uv run python scripts/inject_synthetic_data.py [options]
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmtmax.http import CachedHttpClient  # noqa: E402
from pmtmax.weather.openmeteo_client import OpenMeteoClient  # noqa: E402

CACHE_DIR = REPO_ROOT / "data/cache"
SYNTHETIC_SNAPSHOTS = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"
DUCKDB_PATH = REPO_ROOT / "data/duckdb/warehouse.duckdb"

# Models to populate in silver table (use ERA5 data under all model names)
MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "kma_gdps", "gfs_seamless"]
HOURLY_VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]

START_DATE = date(2016, 1, 1)
END_DATE = date(2025, 5, 29)

ERA5_BASE_URL = "https://archive-api.open-meteo.com"
WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"


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


def _wu_location_id(spec: dict) -> str | None:
    source_url = spec.get("official_source_url", "")
    station_id = spec.get("station_id", "")
    if not source_url or not station_id:
        return None
    parsed = urlparse(source_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "history" and parts[1] == "daily":
        country_code = parts[2].upper()
        return f"{station_id}:9:{country_code}"
    return None


# ===========================================================================
# ERA5 yearly fetcher
# ===========================================================================
def fetch_era5_year(
    http: CachedHttpClient,
    city: str,
    spec: dict,
    year: int,
) -> tuple[str, int, list[dict] | None, str]:
    """Fetch a full year of ERA5 hourly data.

    Returns (city, year, list_of_hourly_rows, status).
    Each hourly_row: {forecast_time_local, forecast_time_utc, local_date, hour, temp_vars...}
    """
    lat = spec["station_lat"]
    lon = spec["station_lon"]
    tz = spec["timezone"]
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), END_DATE)
    if start > END_DATE:
        return city, year, None, "skipped_out_of_range"

    url = f"{ERA5_BASE_URL}/v1/archive"
    params = {
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "timezone": tz,
    }
    try:
        payload = http.get_json(url, params=params, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        return city, year, None, f"error:{exc!s:.200}"

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return city, year, [], "ok_empty"

    import zoneinfo
    try:
        station_tz = zoneinfo.ZoneInfo(tz)
    except Exception:  # noqa: BLE001
        station_tz = zoneinfo.ZoneInfo("UTC")

    rows = []
    for i, t_local in enumerate(times):
        # t_local is like "2016-01-01T00:00"
        hour = int(t_local[11:13])
        local_date = t_local[:10]

        try:
            local_dt = datetime.fromisoformat(t_local).replace(tzinfo=station_tz)
            utc_dt = local_dt.astimezone(UTC)
            t_local_fmt = local_dt.isoformat()
            t_utc = utc_dt.isoformat()
        except Exception:  # noqa: BLE001
            t_local_fmt = t_local + ":00"
            t_utc = t_local + ":00+00:00"

        row = {
            "forecast_time_local": t_local_fmt,
            "forecast_time_utc": t_utc,
            "local_date": local_date,
            "hour": hour,
        }
        for v in HOURLY_VARS:
            vals = hourly.get(v, [])
            val = vals[i] if i < len(vals) else None
            row[v] = float(val) if val is not None else None
        rows.append(row)

    return city, year, rows, "ok"


# ===========================================================================
# Wunderground monthly fetcher
# ===========================================================================
def fetch_wu_month(
    http: CachedHttpClient,
    city: str,
    spec: dict,
    year: int,
    month: int,
) -> tuple[str, int, int, dict[str, float] | None, str]:
    """Fetch a full month of Wunderground hourly observations and compute daily max."""
    location_id = _wu_location_id(spec)
    if not location_id:
        return city, year, month, None, "error:no_location_id"

    unit = spec.get("unit", "C")
    last_day = calendar.monthrange(year, month)[1]
    start_date_obj = date(year, month, 1)
    end_date_obj = min(date(year, month, last_day), END_DATE)

    if start_date_obj > END_DATE:
        return city, year, month, None, "skipped_out_of_range"

    url = WU_HISTORICAL_URL.format(location_id=location_id)
    params = {
        "apiKey": WU_API_KEY,
        "units": "m" if unit == "C" else "e",
        "startDate": start_date_obj.strftime("%Y%m%d"),
        "endDate": end_date_obj.strftime("%Y%m%d"),
    }
    try:
        payload = http.get_json(url, params=params, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        return city, year, month, None, f"error:{exc!s:.200}"

    observations = payload.get("observations", [])
    if not observations:
        return city, year, month, {}, "ok_empty"

    # Group hourly temps by station-local date
    import zoneinfo
    tz = spec.get("timezone", "UTC")
    try:
        station_tz = zoneinfo.ZoneInfo(tz)
    except Exception:  # noqa: BLE001
        station_tz = zoneinfo.ZoneInfo("UTC")

    daily_temps: dict[str, list[float]] = {}
    for obs in observations:
        ts_gmt = obs.get("valid_time_gmt")
        temp = obs.get("temp")
        if ts_gmt is None or temp is None:
            continue
        try:
            utc_dt = datetime.fromtimestamp(int(ts_gmt), tz=UTC)
            local_dt = utc_dt.astimezone(station_tz)
            day_str = local_dt.strftime("%Y-%m-%d")
            daily_temps.setdefault(day_str, []).append(float(temp))
        except Exception:  # noqa: BLE001
            continue

    if not daily_temps:
        return city, year, month, {}, f"ok_no_valid_temps"

    date_to_max = {day_str: max(temps) for day_str, temps in daily_temps.items()}
    return city, year, month, date_to_max, "ok"


# ===========================================================================
# CWA truth fetcher (for Taipei)
# ===========================================================================
def fetch_cwa_truth(
    http: CachedHttpClient,
    city: str,
    spec: dict,
    year: int,
) -> tuple[str, int, dict[str, float] | None, str]:
    """Fetch CWA daily max for Taipei for a full year."""
    # CWA CODIS API: monthly call
    station_id = spec.get("station_id", "466920")
    all_day_max: dict[str, float] = {}

    import calendar
    from datetime import date

    for month in range(1, 13):
        start_obj = date(year, month, 1)
        if start_obj > END_DATE:
            break
        last_day = calendar.monthrange(year, month)[1]
        end_obj = min(date(year, month, last_day), END_DATE)

        url = "https://codis.cwa.gov.tw/api/station"
        params = {
            "format": "JSON",
            "station_id": station_id,
            "start": start_obj.strftime("%Y-%m-%d"),
            "end": end_obj.strftime("%Y-%m-%d"),
            "items": "TX",  # Maximum temperature
        }
        try:
            payload = http.get_json(url, params=params, use_cache=True)
            data = payload.get("data", {})
            # Parse CWA response format
            if isinstance(data, list):
                for item in data:
                    day_str = str(item.get("DataTime", ""))[:10]
                    val = item.get("TX")
                    if day_str and val is not None:
                        try:
                            all_day_max[day_str] = float(val)
                        except (TypeError, ValueError):
                            pass
        except Exception:  # noqa: BLE001
            pass  # Skip failed months

    if not all_day_max:
        return city, year, None, "error:no_cwa_data"
    return city, year, all_day_max, "ok"


# ===========================================================================
# DuckDB injection
# ===========================================================================
def check_existing_synthetic_count(duckdb_path: Path) -> tuple[int, int]:
    """Return count of already-injected synthetic rows."""
    import duckdb
    with duckdb.connect(str(duckdb_path)) as db:
        try:
            f = db.execute(
                "SELECT COUNT(*) FROM silver_forecast_runs_hourly WHERE market_id LIKE 'synthetic_%'"
            ).fetchone()[0]
        except Exception:  # noqa: BLE001
            f = 0
        try:
            t = db.execute(
                "SELECT COUNT(*) FROM silver_observations_daily WHERE market_id LIKE 'synthetic_%'"
            ).fetchone()[0]
        except Exception:  # noqa: BLE001
            t = 0
    return f, t


def clear_synthetic_data(duckdb_path: Path) -> None:
    """Remove existing synthetic rows to avoid duplicates."""
    import duckdb
    with duckdb.connect(str(duckdb_path)) as db:
        db.execute("DELETE FROM silver_forecast_runs_hourly WHERE market_id LIKE 'synthetic_%'")
        db.execute("DELETE FROM silver_observations_daily WHERE market_id LIKE 'synthetic_%'")
    print("  Cleared existing synthetic rows.", flush=True)


def inject_forecast_rows(duckdb_path: Path, rows: list[dict], batch_size: int = 100000) -> int:
    """Inject forecast hourly rows into silver_forecast_runs_hourly."""
    import duckdb
    total = 0
    with duckdb.connect(str(duckdb_path)) as db:
        existing_cols = [r[0] for r in db.execute("DESCRIBE silver_forecast_runs_hourly").fetchall()]
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            df = pd.DataFrame(batch)
            for col in existing_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[existing_cols]
            db.execute("INSERT INTO silver_forecast_runs_hourly SELECT * FROM df")
            total += len(batch)
            if total % 500000 == 0:
                print(f"    ... {total:,} forecast rows injected", flush=True)
    return total


def inject_truth_rows(duckdb_path: Path, rows: list[dict], batch_size: int = 100000) -> int:
    """Inject truth rows into silver_observations_daily."""
    import duckdb
    total = 0
    with duckdb.connect(str(duckdb_path)) as db:
        existing_cols = [r[0] for r in db.execute("DESCRIBE silver_observations_daily").fetchall()]
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            df = pd.DataFrame(batch)
            for col in existing_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[existing_cols]
            db.execute("INSERT INTO silver_observations_daily SELECT * FROM df")
            total += len(batch)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject synthetic historical weather data into DuckDB")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--city", action="append", help="Filter to specific cities")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--skip-era5", action="store_true")
    parser.add_argument("--skip-wunderground", action="store_true")
    parser.add_argument("--skip-inject", action="store_true", help="Fetch only, don't inject into DuckDB")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing synthetic rows first")
    parser.add_argument("--wu-workers", type=int, default=3, help="Workers for WU fetches (default: 3)")
    args = parser.parse_args()

    http = CachedHttpClient(cache_dir=CACHE_DIR, timeout_seconds=60, retries=5)

    print(f"Loading city specs from {SYNTHETIC_SNAPSHOTS} ...", flush=True)
    city_specs = load_city_specs(SYNTHETIC_SNAPSHOTS)
    if args.city:
        city_specs = {c: s for c, s in city_specs.items() if c in args.city}
    print(f"  Cities ({len(city_specs)}): {sorted(city_specs)}", flush=True)

    years = list(range(args.start_year, min(args.end_year, 2025) + 1))
    now_utc = datetime.now(tz=UTC)
    run_id = f"synthetic_era5_{now_utc.strftime('%Y%m%dT%H%M%S')}"

    if not args.skip_inject and args.clear_existing:
        print("Clearing existing synthetic data ...", flush=True)
        clear_synthetic_data(DUCKDB_PATH)

    if not args.skip_inject:
        existing_f, existing_t = check_existing_synthetic_count(DUCKDB_PATH)
        print(f"Existing synthetic rows: {existing_f:,} forecast, {existing_t:,} truth", flush=True)

    # ==================================================================
    # 1. ERA5 batch fetch
    # ==================================================================
    all_forecast_rows: list[dict] = []

    if not args.skip_era5:
        tasks_era5 = [
            (city, city_specs[city], year)
            for city in sorted(city_specs)
            for year in years
        ]
        total_era5 = len(tasks_era5)
        print(f"\n[ERA5] {total_era5} year×city requests ...", flush=True)

        ok_count = 0
        error_count = 0
        total_hours = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_era5_year, http, city, spec, year): (city, year)
                for city, spec, year in tasks_era5
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, year, hourly_rows, status = future.result()
                if status.startswith("ok") and hourly_rows:
                    spec = city_specs[city]
                    city_slug = city.lower().replace(" ", "_")
                    # Build DuckDB rows: one per hour, per model (replicate ERA5 for all 4 models)
                    for hr in hourly_rows:
                        day_str = hr["local_date"]
                        market_id = f"synthetic_{city_slug}_{day_str}"
                        base_row = {
                            "market_id": market_id,
                            "station_id": spec["station_id"],
                            "city": city,
                            "target_local_date": pd.Timestamp(day_str),
                            "timezone": spec["timezone"],
                            "provider": "open-meteo-era5",
                            "endpoint_kind": "historical_forecast",
                            "decision_horizon": None,
                            "availability_status": "available",
                            "issue_time_utc": None,
                            "requested_run_time_utc": None,
                            "retrieved_at_utc": pd.Timestamp(now_utc),
                            "forecast_time_local": hr["forecast_time_local"],
                            "forecast_time_utc": hr["forecast_time_utc"],
                            "local_date": pd.Timestamp(day_str),
                            "hour": hr["hour"],
                            "source_variables_json": json.dumps(HOURLY_VARS),
                            "temperature_2m": hr.get("temperature_2m"),
                            "dew_point_2m": hr.get("dew_point_2m"),
                            "relative_humidity_2m": hr.get("relative_humidity_2m"),
                            "wind_speed_10m": hr.get("wind_speed_10m"),
                            "cloud_cover": hr.get("cloud_cover"),
                            "raw_path": None,
                            "raw_hash": None,
                            "run_id": run_id,
                            "data_version": "synthetic_era5_v1",
                            "created_at": pd.Timestamp(now_utc),
                            "source_priority": 80,
                        }
                        # Replicate for each expected model name
                        for model in MODELS:
                            row = dict(base_row)
                            row["model_name"] = model
                            all_forecast_rows.append(row)
                    ok_count += 1
                    total_hours += len(hourly_rows)
                elif status.startswith("error"):
                    error_count += 1

                if i % 30 == 0 or i == total_era5:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_era5 - i) / rate if rate > 0 else 0
                    print(
                        f"  [ERA5] {i}/{total_era5} ok={ok_count} err={error_count} "
                        f"forecast_rows={len(all_forecast_rows):,} ({rate:.1f}/s ETA {remaining:.0f}s)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    err: {city} {year}: {status}", flush=True)

        elapsed = time.time() - t0
        print(
            f"[ERA5] Done: {ok_count} ok, {error_count} errors, "
            f"{len(all_forecast_rows):,} hourly×model rows in {elapsed:.0f}s",
            flush=True,
        )

    # ==================================================================
    # 2. Wunderground monthly fetch
    # ==================================================================
    all_truth_rows: list[dict] = []

    if not args.skip_wunderground:
        wu_specs = {
            c: s for c, s in city_specs.items()
            if "wunderground" in s.get("official_source_name", "").lower()
        }
        tasks_wu = [
            (city, wu_specs[city], year, month)
            for city in sorted(wu_specs)
            for year in years
            for month in range(1, 13)
            if date(year, month, 1) <= END_DATE
        ]
        total_wu = len(tasks_wu)
        print(
            f"\n[Wunderground] {total_wu} month×city requests "
            f"for {len(wu_specs)} stations ...",
            flush=True,
        )

        ok_count = 0
        error_count = 0
        empty_count = 0
        total_days = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.wu_workers) as executor:
            futures = {
                executor.submit(fetch_wu_month, http, city, spec, year, month): (city, year, month)
                for city, spec, year, month in tasks_wu
            }
            for i, future in enumerate(as_completed(futures), 1):
                city, year, month, date_to_max, status = future.result()
                if status.startswith("ok") and date_to_max:
                    spec = city_specs[city]
                    city_slug = city.lower().replace(" ", "_")
                    for day_str, max_val in date_to_max.items():
                        # Only include dates in our range
                        try:
                            d = date.fromisoformat(day_str)
                            if d < START_DATE or d > END_DATE:
                                continue
                        except Exception:  # noqa: BLE001
                            continue
                        market_id = f"synthetic_{city_slug}_{day_str}"
                        all_truth_rows.append({
                            "market_id": market_id,
                            "station_id": spec["station_id"],
                            "city": city,
                            "source": "Wunderground",
                            "official_source_name": spec.get("official_source_name"),
                            "truth_track": spec.get("truth_track"),
                            "settlement_eligible": spec.get("settlement_eligible", False),
                            "target_local_date": pd.Timestamp(day_str),
                            "daily_max": max_val,
                            "unit": spec.get("unit", "C"),
                            "finalized_at": pd.Timestamp(now_utc),
                            "revision_status": "final",
                            "raw_path": None,
                            "raw_hash": None,
                            "run_id": run_id,
                            "data_version": "synthetic_wu_v1",
                            "created_at": pd.Timestamp(now_utc),
                            "source_priority": 100,
                        })
                        total_days += 1
                    ok_count += 1
                elif status.startswith("ok"):
                    empty_count += 1
                else:
                    error_count += 1

                if i % 100 == 0 or i == total_wu:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_wu - i) / rate if rate > 0 else 0
                    print(
                        f"  [WU] {i}/{total_wu} ok={ok_count} empty={empty_count} err={error_count} "
                        f"days={total_days:,} ({rate:.1f}/s ETA {remaining:.0f}s)",
                        flush=True,
                    )
                    if status.startswith("error"):
                        print(f"    err: {city} {year}-{month:02d}: {status}", flush=True)

        elapsed = time.time() - t0
        print(
            f"[Wunderground] Done: {ok_count} ok, {error_count} errors, "
            f"{len(all_truth_rows):,} truth rows in {elapsed:.0f}s",
            flush=True,
        )

    # ==================================================================
    # 3. Summary and inject
    # ==================================================================
    print(f"\nData collection summary:", flush=True)
    print(f"  ERA5 forecast rows: {len(all_forecast_rows):,}", flush=True)
    print(f"  WU truth rows: {len(all_truth_rows):,}", flush=True)

    if not args.skip_inject:
        if all_forecast_rows:
            print(f"\nInjecting {len(all_forecast_rows):,} forecast rows ...", flush=True)
            t0 = time.time()
            total_f = inject_forecast_rows(DUCKDB_PATH, all_forecast_rows)
            print(f"  Done: {total_f:,} rows in {time.time()-t0:.0f}s", flush=True)

        if all_truth_rows:
            print(f"\nInjecting {len(all_truth_rows):,} truth rows ...", flush=True)
            t0 = time.time()
            total_t = inject_truth_rows(DUCKDB_PATH, all_truth_rows)
            print(f"  Done: {total_t:,} rows in {time.time()-t0:.0f}s", flush=True)

        # Final count
        final_f, final_t = check_existing_synthetic_count(DUCKDB_PATH)
        print(f"\nFinal DuckDB synthetic counts: {final_f:,} forecast, {final_t:,} truth", flush=True)
    else:
        print("  Skipping DuckDB injection (--skip-inject)", flush=True)

    print(f"\nDone! Next steps:", flush=True)
    print(f"  uv run pmtmax build-dataset \\", flush=True)
    print(f"    --markets-path configs/market_inventory/synthetic_historical_snapshots.json \\", flush=True)
    print(f"    --output-name synthetic_historical_training_set", flush=True)


if __name__ == "__main__":
    main()
