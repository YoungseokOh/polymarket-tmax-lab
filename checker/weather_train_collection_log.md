# Weather Train Collection Log

Append-only operational log for `weather_train`.

| Run Date | Range | Mode | Outcome | Rows Added | Notes |
| --- | --- | --- | --- | ---: | --- |
| 2026-04-23 | `2024-01-01..2024-01-02` | smoke, Seoul only | success | 2 | First Open-Meteo weather-real rows. |
| 2026-04-23 | `2024-01-03..2024-02-02` | daily chunks | success | 930 | 30 cities per day, no failures. |
| 2026-04-23 | `2024-02-03..2024-05-02` | 90 daily chunks | success | 2,700 | 30 cities per day, no 429 observed. |
| 2026-04-23 | `2024-05-03..2024-05-16` | 14-day chunk | success | 420 | Added 14 days of full coverage. |
| 2026-04-23 | `2024-05-17` | single day with progress | success | 30 | New shorter HTTP controls and stderr progress verified. |
| 2026-04-23 | `2024-05-18..2024-05-24` | 7-day chunk with progress | success | 210 | Full `210/210 available`. |
| 2026-04-24 | `2026-01-01..2026-01-14` | weekly chunks | success | 420 | Recent-date range still fully available here. |
| 2026-04-24 | `2026-01-15..2026-01-21` | weekly chunk | partial | 150 | `60 retryable_error`, but `150` rows committed. |
| 2026-04-24 | `2026-01-22..2026-01-28` | 1-day chunks + 20s cooldown | retry-only | 0 | Every day returned `30/30 retryable_error`. |
| 2026-04-24 | `2024-05-25..2024-05-31` | weekly chunk | partial | 130 | `2024-05-25..2024-05-30` partially filled; `2024-05-31` was `0/30 retryable_error`. |
| 2026-04-24 | `2024-06-01..2024-06-07` | weekly chunk | interrupted | 0 | Run was stopped after long retry/backoff cycle; no new committed rows. |
| 2026-04-24 | `weather_train` pretrain | `gaussian_emos` | success | 0 | Trained on `4,992` rows; artifact written under `artifacts/workspaces/weather_train/models/v2/`. |
