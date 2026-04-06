"""Pre-fill HTTP cache with per-day Wunderground JSON responses.

Fetches monthly WU data (1 request per station per month) and splits the
response into per-day cache entries that match exactly what the pipeline's
backfill_truth step will look for.

This reduces 103,110 network requests to ~3,600 monthly requests.

Usage:
    uv run python scripts/prefill_wu_cache.py [--workers N] [--city CITY]
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmtmax.http import CachedHttpClient  # noqa: E402
from pmtmax.utils import stable_hash  # noqa: E402

CACHE_DIR = REPO_ROOT / "data/cache"
SYNTHETIC_SNAPSHOTS = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"

START_DATE = date(2016, 1, 1)
END_DATE = date(2025, 5, 29)

WU_HISTORICAL_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"
WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"


def load_city_specs(snapshots_path: Path) -> dict[str, dict]:
    with snapshots_path.open() as f:
        entries = json.load(f)
    city_specs: dict[str, dict] = {}
    for entry in entries:
        spec = entry.get("spec")
        if spec and spec.get("city") and spec["city"] not in city_specs:
            city_specs[spec["city"]] = spec
    return city_specs


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


def _daily_cache_key(url: str, params: dict) -> str:
    """Compute the exact cache key used by CachedHttpClient.get_json for a GET request."""
    return json.dumps({"url": url, "params": params}, sort_keys=True, default=str)


def _daily_cache_path(url: str, params: dict) -> Path:
    key = _daily_cache_key(url, params)
    return CACHE_DIR / f"{stable_hash(key)}.json"


def process_city_month(
    http: CachedHttpClient,
    city: str,
    spec: dict,
    year: int,
    month: int,
) -> tuple[str, int, int, int, int, str]:
    """Fetch one month of WU data and split into per-day cache entries.

    Returns (city, year, month, days_fetched, days_cached, status).
    """
    location_id = _wu_location_id(spec)
    if not location_id:
        return city, year, month, 0, 0, "error:no_location_id"

    unit = spec.get("unit", "C")
    last_day = calendar.monthrange(year, month)[1]
    start_date_obj = date(year, month, 1)
    end_date_obj = min(date(year, month, last_day), END_DATE)

    if start_date_obj > END_DATE:
        return city, year, month, 0, 0, "skipped_out_of_range"

    base_url = WU_HISTORICAL_URL.format(location_id=location_id)
    units_param = "m" if unit == "C" else "e"

    # Monthly request (1 network call)
    monthly_params = {
        "apiKey": WU_API_KEY,
        "units": units_param,
        "startDate": start_date_obj.strftime("%Y%m%d"),
        "endDate": end_date_obj.strftime("%Y%m%d"),
    }
    try:
        monthly_payload = http.get_json(base_url, params=monthly_params, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        return city, year, month, 0, 0, f"error:{exc!s:.200}"

    observations = monthly_payload.get("observations", [])
    if not observations:
        return city, year, month, 0, 0, "ok_empty"

    # Group observations by local date
    try:
        import zoneinfo
        station_tz = zoneinfo.ZoneInfo(spec.get("timezone", "UTC"))
    except Exception:  # noqa: BLE001
        import zoneinfo
        station_tz = zoneinfo.ZoneInfo("UTC")

    day_obs: dict[str, list[dict]] = {}
    for obs in observations:
        ts_gmt = obs.get("valid_time_gmt")
        if ts_gmt is None:
            continue
        try:
            utc_dt = datetime.fromtimestamp(int(ts_gmt), tz=UTC)
            local_dt = utc_dt.astimezone(station_tz)
            day_str = local_dt.strftime("%Y%m%d")
            day_obs.setdefault(day_str, []).append(obs)
        except Exception:  # noqa: BLE001
            continue

    # Write per-day cache entries
    days_written = 0
    for day_str, day_observations in day_obs.items():
        # Construct the payload that the pipeline would get from the per-day API call
        per_day_payload = {
            "metadata": monthly_payload.get("metadata", {}),
            "observations": day_observations,
        }
        # Compute the cache key for the per-day API call
        per_day_params = {
            "apiKey": WU_API_KEY,
            "units": units_param,
            "startDate": day_str,
            "endDate": day_str,
        }
        cache_path = _daily_cache_path(base_url, per_day_params)
        if not cache_path.exists():
            cache_path.write_text(json.dumps(per_day_payload, indent=2, sort_keys=True))
        days_written += 1

    return city, year, month, len(day_obs), days_written, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fill WU per-day cache from monthly fetches")
    parser.add_argument("--workers", type=int, default=4, help="Workers (keep low for WU rate limits)")
    parser.add_argument("--city", action="append", help="Filter cities")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    http = CachedHttpClient(cache_dir=CACHE_DIR, timeout_seconds=60, retries=5)

    print(f"Loading city specs ...", flush=True)
    city_specs = load_city_specs(SYNTHETIC_SNAPSHOTS)
    wu_specs = {
        c: s for c, s in city_specs.items()
        if "wunderground" in s.get("official_source_name", "").lower()
    }
    if args.city:
        wu_specs = {c: s for c, s in wu_specs.items() if c in args.city}
    print(f"  WU cities ({len(wu_specs)}): {sorted(wu_specs)}", flush=True)

    years = list(range(args.start_year, min(args.end_year, 2025) + 1))
    tasks = [
        (city, wu_specs[city], year, month)
        for city in sorted(wu_specs)
        for year in years
        for month in range(1, 13)
        if date(year, month, 1) <= END_DATE
    ]
    total = len(tasks)
    print(f"  Tasks: {total} (months × cities)", flush=True)

    ok_count = 0
    error_count = 0
    total_days = 0
    total_cached = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_city_month, http, city, spec, year, month): (city, year, month)
            for city, spec, year, month in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            city, year, month, days_fetched, days_cached, status = future.result()
            if status.startswith("ok"):
                ok_count += 1
                total_days += days_fetched
                total_cached += days_cached
            else:
                error_count += 1

            if i % 100 == 0 or i == total:
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
                    print(f"    err: {city} {year}-{month:02d}: {status}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {ok_count} months OK, {error_count} errors", flush=True)
    print(f"  Total days with data: {total_days:,}", flush=True)
    print(f"  Total per-day cache entries written: {total_cached:,}", flush=True)
    print(f"  Time: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
