"""Build the supported-city active grouped weather watchlist artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from pmtmax.config.settings import load_settings
from pmtmax.http import CachedHttpClient
from pmtmax.markets.gamma_client import GammaClient
from pmtmax.markets.inventory import (
    build_active_weather_watchlist,
    discover_temperature_event_refs_from_gamma,
    fetch_temperature_event_pages,
)
from pmtmax.utils import dump_json

DEFAULT_OUTPUT = Path("artifacts/active_weather_watchlist.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the active weather watchlist artifact JSON.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of Gamma event pages to scan. Defaults to config.polymarket.max_pages.",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Limit the watchlist to specific supported cities. Can be provided multiple times.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the shared HTTP cache while fetching active event pages.",
    )
    args = parser.parse_args()

    config, _ = load_settings()
    supported_cities = args.city or config.app.supported_cities
    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    try:
        gamma = GammaClient(http, config.polymarket.gamma_base_url)
        refs = discover_temperature_event_refs_from_gamma(
            gamma,
            supported_cities=supported_cities,
            active=True,
            closed=False,
            max_pages=args.max_pages or config.polymarket.max_pages,
        )
        fetches = fetch_temperature_event_pages(http, refs, use_cache=not args.no_cache)
    finally:
        http.close()

    report = build_active_weather_watchlist(fetches, supported_cities=supported_cities)
    dump_json(args.output, report.model_dump(mode="json"))
    print(
        f"Built active watchlist with {len(report.entries)} entries -> {args.output} "
        f"(ready: {report.status_counts.get('ready', 0)})"
    )


if __name__ == "__main__":
    main()
