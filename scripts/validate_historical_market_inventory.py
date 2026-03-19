"""Validate a curated historical market snapshot inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pmtmax.config.settings import load_settings
from pmtmax.markets.inventory import validate_historical_inventory
from pmtmax.markets.repository import load_market_snapshots
from pmtmax.utils import dump_json

DEFAULT_INPUT = Path("configs/market_inventory/historical_temperature_snapshots.json")
DEFAULT_REPORT = Path("data/manifests/historical_inventory_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Curated MarketSnapshot[] JSON to validate.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the validation report JSON.",
    )
    args = parser.parse_args()

    config, _ = load_settings()
    snapshots = load_market_snapshots(args.input)
    report = validate_historical_inventory(
        snapshots,
        supported_cities=config.app.supported_cities,
        source_manifest=str(args.input),
    )
    dump_json(args.report, report.model_dump(mode="json"))
    print(f"Validated {len(snapshots)} curated snapshots -> {args.report} (issues: {len(report.issues)})")


if __name__ == "__main__":
    main()
