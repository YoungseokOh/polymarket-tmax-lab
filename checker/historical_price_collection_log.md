# Historical Price Recovery Log

Append-only operational log for `historical_real` official price-history recovery.

| Run Date | Shard | Mode | Outcome | Markets | Ready Delta | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| 2026-04-24 | `0..24 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=179`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `25..49`. |
| 2026-04-25 | `25..49 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=91`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `50..74`. |
| 2026-04-25 | `50..74 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=91`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `75..99`. |
| 2026-04-25 | `75..99 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=189`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `100..124`. |
| 2026-04-25 | `100..124 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=164`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `125..149`. |
| 2026-04-25 | `125..149 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=147`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `150..174`. |
| 2026-04-25 | `150..174 / 1834` | daily price recovery agent | success | 25 | +0 | No missing token requests in shard; token `ok` delta `+0`, decision-ready delta `+0`; next shard `175..199`. |
| 2026-04-25 | `175..199 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=175`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `200..224`. |
| 2026-04-25 | `200..224 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=170`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `225..249`. |
| 2026-04-25 | `225..249 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=175`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `250..274`. |
| 2026-04-25 | `250..274 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=70`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `275..299`. |
| 2026-04-25 | `275..299 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=96`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `300..324`. |
| 2026-04-25 | `300..324 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=161`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `325..349`. |
| 2026-04-25 | `325..349 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=7`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `350..374`. |
| 2026-04-25 | `350..374 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=140`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `375..399`. |
| 2026-04-25 | `375..399 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=179`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `400..424`. |
| 2026-04-25 | `400..424 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=168`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `425..449`. |
| 2026-04-25 | `425..449 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=98`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `450..474`. |
| 2026-04-25 | `450..474 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=84`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `475..499`. |
| 2026-04-25 | `475..499 / 1834` | daily price recovery agent | success | 25 | +0 | `empty=175`; token `ok` delta `+0`, decision-ready delta `+0`; next shard `500..524`. |
| 2026-04-26 | `targeted Ankara / 20` | targeted price-history backfill + variant panels | success | 20 | n/a | `backfill-price-history --only-missing --price-no-cache` wrote `220` ok requests and `11036` points. Target panel `targeted_ankara_backtest_panel_20260426` has `660` rows with `ok=641`, `missing=14`, `stale=5`; full local backlog panel `historical_backtest_panel_curated_multisource_targeted_ankara_20260426` has `50070` rows with `ok=14011`, `missing=36036`, `stale=23`. Checked-in inventory shard queue remains `500..524 / 1834`. |
| 2026-04-26 | `targeted Dallas/Atlanta/Miami / 60` | targeted price-history backfill + variant panels | success | 60 | n/a | `backfill-price-history --only-missing --price-no-cache` wrote `660` ok requests and `35365` points. Target panel `targeted_dallas_atlanta_miami_backtest_panel_20260426` has `1980` rows with `ok=1941`, `missing=11`, `stale=28`; full local backlog panel `historical_backtest_panel_curated_multisource_targeted_south_20260426` has `52050` rows with `ok=15952`, `missing=36047`, `stale=51`. Checked-in inventory shard queue remains `500..524 / 1834`. |
