"""Generate synthetic MarketSnapshot wrappers for Oct 2025 - Apr 2026.

These dates fall within the Open-Meteo single_run archive window (~Oct 2025+),
so ecmwf_ifs025, gfs_seamless, kma_gdps will return genuinely different model
forecasts instead of identical ERA5 data.  This fixes the NWP spread=0 issue.

Output: configs/market_inventory/synthetic_2025oct_2026apr_snapshots.json
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
OUTPUT_PATH = REPO_ROOT / "configs/market_inventory/synthetic_2025oct_2026apr_snapshots.json"

# single_run archive starts ~2025-10-01; go through yesterday to avoid partial days
START_DATE = date(2025, 10, 1)
END_DATE = date(2026, 4, 14)  # update as needed


def daterange(start: date, end: date):
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

    new_spec = dict(spec)
    new_spec["market_id"] = market_id
    new_spec["event_id"] = None
    new_spec["condition_id"] = None
    new_spec["token_ids"] = []
    new_spec["target_local_date"] = date_str

    month_day = target_date.strftime("%B %-d")
    new_spec["question"] = f"Highest temperature in {city} on {month_day}?"
    new_spec["slug"] = f"synthetic-highest-temperature-in-{city_slug}-on-{date_str}"
    new_spec["notes"] = f"Synthetic wrapper for real weather data on {date_str} (single_run archive)"

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
    for city, template in sorted(templates.items()):
        city_count = 0
        for d in all_dates:
            snap = make_synthetic_snapshot(city, d, template)
            snapshots.append(snap)
            city_count += 1
        print(f"  {city}: generated {city_count} snapshots", flush=True)

    total = len(snapshots)
    print(f"\nTotal synthetic snapshots: {total}", flush=True)
    print(f"Saving to {OUTPUT_PATH} ...", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"Done. Wrote {total} entries to {OUTPUT_PATH}", flush=True)
    print()
    print("Next steps:")
    print("  uv run pmtmax build-dataset \\")
    print(f"    --markets-path {OUTPUT_PATH} \\")
    print("    --forecast-missing-only \\")
    print("    --output-name historical_training_set \\")
    print("    --allow-canonical-overwrite")


if __name__ == "__main__":
    main()
