"""Generate synthetic MarketSnapshot wrappers for historical weather data.

Creates one MarketSnapshot per city per date for dates 2016-01-01 through
2025-05-29 for all 30 cities in the station catalog.  The weather forecast
data (Open-Meteo) and truth observations (Wunderground / HKO / CWA) are 100%
real historical data — only the market_id wrapper is synthetic.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_SRC = REPO_ROOT / "configs/market_inventory/historical_temperature_snapshots.json"
CATALOG_PATH = REPO_ROOT / "configs/market_inventory/station_catalog.json"
OUTPUT_PATH = REPO_ROOT / "configs/market_inventory/synthetic_historical_snapshots.json"

START_DATE = date(2016, 1, 1)
END_DATE = date(2025, 5, 29)


def daterange(start: date, end: date):
    """Yield each date from start through end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_templates(snapshots_path: Path) -> dict[str, dict]:
    """Return the most-recent clean spec per city from the snapshot file."""
    with snapshots_path.open() as f:
        entries = json.load(f)

    templates: dict[str, dict] = {}
    for entry in entries:
        spec = entry.get("spec")
        if spec is None or entry.get("parse_error"):
            continue
        city = spec.get("city", "")
        if not city:
            continue
        existing = templates.get(city)
        if existing is None or entry["captured_at"] > existing["captured_at"]:
            templates[city] = entry
    return templates


def make_synthetic_snapshot(city: str, target_date: date, template_entry: dict) -> dict:
    """Build one synthetic MarketSnapshot dict."""
    spec = template_entry["spec"]
    date_str = target_date.isoformat()
    city_slug = city.lower().replace(" ", "_")
    market_id = f"synthetic_{city_slug}_{date_str}"

    # Build a minimal but valid spec by cloning the template spec and
    # overriding the date-specific fields.
    new_spec = dict(spec)
    new_spec["market_id"] = market_id
    new_spec["event_id"] = None
    new_spec["condition_id"] = None
    new_spec["token_ids"] = []
    new_spec["target_local_date"] = date_str

    # Adjust the question / slug to reflect the synthetic date
    month_day = target_date.strftime("%B %-d")
    new_spec["question"] = f"Highest temperature in {city} on {month_day}?"
    new_spec["slug"] = f"synthetic-highest-temperature-in-{city_slug}-on-{date_str}"
    new_spec["notes"] = f"Synthetic wrapper for real weather data on {date_str}"

    # Minimal market dict (only the fields pipeline reads)
    market_dict = {
        "id": market_id,
        "eventId": None,
        "slug": new_spec["slug"],
        "question": new_spec["question"],
    }

    return {
        "captured_at": f"{date_str}T00:00:00Z",
        "clob_token_ids": [],
        "market": market_dict,
        "outcome_prices": {},
        "parse_error": None,
        "spec": new_spec,
    }


def main() -> None:
    print(f"Loading templates from {SNAPSHOTS_SRC} ...", flush=True)
    templates = load_templates(SNAPSHOTS_SRC)
    print(f"  Found {len(templates)} city templates: {sorted(templates)}", flush=True)

    all_dates = list(daterange(START_DATE, END_DATE))
    print(f"  Date range: {START_DATE} to {END_DATE}  ({len(all_dates)} days)", flush=True)

    snapshots: list[dict] = []
    skipped_cities: list[str] = []

    for city, template in sorted(templates.items()):
        city_count = 0
        for d in all_dates:
            snap = make_synthetic_snapshot(city, d, template)
            snapshots.append(snap)
            city_count += 1
        print(f"  {city}: generated {city_count} snapshots", flush=True)

    if skipped_cities:
        print(f"WARNING: skipped cities with no template: {skipped_cities}", file=sys.stderr)

    total = len(snapshots)
    print(f"\nTotal synthetic snapshots: {total}", flush=True)
    print(f"Saving to {OUTPUT_PATH} ...", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"Done. Wrote {total} entries to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
