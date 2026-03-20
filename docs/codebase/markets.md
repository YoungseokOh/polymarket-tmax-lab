# markets

## Responsibility
`src/pmtmax/markets` owns market discovery, market-family filtering, public
market-data clients, settlement rule parsing, and the `MarketSpec` abstraction.

## Key Modules
- `gamma_client.py`: public Gamma API fetches for events and markets
- `discovery.py`: discovery service that scans active markets and returns filtered specs
- `inventory.py`: Polymarket grouped-event discovery, event-page aggregation, watchlist artifacts, and curated historical inventory validation helpers
- `market_filter.py`: narrows the universe to recurring max-temperature contracts
- `rule_parser.py`: turns rule text and captured market metadata into `MarketSpec`
- `market_spec.py`: source/station/date/unit/finalization schema
- `outcome_schema.py`: Celsius/Fahrenheit bin parsing and label normalization
- `clob_read_client.py` and `ws_market.py`: public price, book, stream, and historical token-price reads
- `resolution_sources.py`: source-family lookup and adapter selection helpers

## Inputs And Outputs
- Input: raw Gamma/CLOB payloads, curated Polymarket event pages, rule HTML or plain text
- Output: validated `MarketSpec`, normalized outcome schemas, public price snapshots, public token-price history payloads, supported-city watchlists, and curated `MarketSnapshot[]` inventories

## What Matters Most
- The parser must preserve exact settlement source, station, local date, unit, and revision policy.
- Market filtering should reject non-temperature or non-recurring weather markets.
- Curated inventory tooling should keep the raw event payload alongside the parsed snapshot so historical runs can be revalidated.
- Public market-data clients should stay read-only in research and paper modes.
- Official price-history collection should preserve empty coverage explicitly instead of filling gaps with synthetic prices.

## Change Checklist
- Parser changes must keep `tests/test_rule_parser.py` and `tests/test_market_spec.py` green.
- If supported rule families expand, update `docs/market-rules.md` and this file.
- If discovery output changes, update downstream dataset and paper-trading workflows.
