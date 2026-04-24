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
| 2026-04-24 | `2024-05-31` | single-day probe | retry-only | 0 | Reconfirmed `30/30 retryable_error`; every station returned explicit `429 Too Many Requests`. |
| 2026-04-24 | `2024-06-01..2024-06-03` | 3-day probe with progress | retry-only | 0 | All `90/90` requests ended `retryable_error` with explicit `429`; gold rows stayed at `4,992`. |
| 2026-04-24 | `2024-06-01` | single-day probe | success | 30 | Free path reopened later the same day; full `30/30 available`. |
| 2026-04-24 | `2024-06-02..2024-06-03` | 2-day chunk with progress | success | 60 | Full `60/60 available`; older-gap backfill resumed. |
| 2026-04-24 | `2024-06-04..2024-06-10` | 7-day chunk with progress | success | 210 | Full `210/210 available`; no 429 observed during the reopened window. |
| 2026-04-24 | `2024-06-11..2024-06-17` | 7-day chunk with progress | success | 210 | Full `210/210 available`; the reopened window stayed stable. |
| 2026-04-24 | `2024-06-18..2024-06-24` | 7-day chunk with progress | success | 210 | Full `210/210 available`; no retryable failures appeared in this range either. |
| 2026-04-24 | `weather_train` pretrain | `gaussian_emos` | success | 0 | Trained on `4,992` rows; artifact written under `artifacts/workspaces/weather_train/models/v2/`. |
| 2026-04-24 | `weather_train` pretrain refresh | `gaussian_emos` | success | 0 | Refreshed on `5,712` rows; dataset signature `a5653c7514d8f2f3b4cae4460766abf81178789f752c0bb2d5804718476949a9`. |
| 2026-04-24 | `2024-06-25..2024-07-01` | 7-day chunk with progress | success | 210 | Full `210/210 available`; full older-backfill coverage now extends through `2024-07-01`. |
| 2026-04-24 | `2024-07-02..2024-07-08` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-07-09`. |
| 2026-04-24 | `2024-07-09..2024-07-15` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-07-16`. |
| 2026-04-24 | `2024-07-16..2024-07-22` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-07-23`. |
| 2026-04-24 | `weather_train pretrain auto-refresh` | gaussian_emos | success | 0 | Triggered at dataset row gap `840`; refreshed on `6,552` rows with dataset signature `67fce2030d4c5ee65df0a2428a5d30404bf544f608906db5c8227c005056fdab`. |
| 2026-04-24 | `2024-07-23..2024-07-29` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-07-30`. |
| 2026-04-24 | `2024-07-30..2024-08-05` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-08-06`. |
| 2026-04-24 | `2024-08-06..2024-08-12` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-08-13`. |
| 2026-04-24 | `weather_train pretrain auto-refresh` | gaussian_emos | success | 0 | Triggered at dataset row gap `630`; refreshed on `7,182` rows with dataset signature `13699427c9f3fd4485a66b4ce3485596b1569c2ee5ae414d48b857c540a339dc`. |
| 2026-04-24 | `2024-08-13..2024-08-19` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-08-20`. |
| 2026-04-24 | `2024-08-20..2024-08-26` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-08-27`. |
| 2026-04-24 | `2024-08-27..2024-09-02` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-09-03`. |
| 2026-04-24 | `weather_train pretrain auto-refresh` | gaussian_emos | success | 0 | Triggered at dataset row gap `630`; refreshed on `7,812` rows with dataset signature `f476f099cc1cad1a8cfc5f554ee8aec5efcbddd40dc52dc2e46f3e6c8994901f`. |
| 2026-04-24 | `2024-09-03..2024-09-09` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-09-10`. |
| 2026-04-24 | `2024-09-10..2024-09-16` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-09-17`. |
| 2026-04-24 | `2024-09-17..2024-09-23` | 7-day queue agent | success | 210 | Full `210/210 available`; next older-backfill queue is `2024-09-24`. |
| 2026-04-24 | `weather_train pretrain auto-refresh` | gaussian_emos | success | 0 | Triggered at dataset row gap `630`; refreshed on `8,442` rows with dataset signature `f273242a8dd2f0427351e80de93861391d4a0aeea7c3855d45e752d43c1a182a`. |
