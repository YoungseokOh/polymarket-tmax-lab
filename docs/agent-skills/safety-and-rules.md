# Safety And Rules

## Operating Rules
- 사용자 응답은 기본적으로 한국어
- live trading은 기본 비활성
- station/source fidelity를 모델 편의보다 우선
- 폴더 책임이나 workflow가 바뀌면 문서도 같은 변경에 포함
- live/paper execution 경로에서는 missing CLOB book을 synthetic으로 숨기지 말고 명시적 skip/error로 처리
- paper execution 경로에서는 legacy `book_source=fixture`를 거부하고, CLOB 미수집은 `missing_book`으로만 남긴다
- signal path에서는 missing calibrator나 forecast contract mismatch도 fail-closed reason으로 남긴다
- observation station live path는 `live_pilot_queue.json` -> `approve-live-candidate` 수동 승인 순서를 기본으로 유지한다
- observation candidate가 `research_public`이면 live 후보에서 제외하지는 않더라도 tier/risk flag를 숨기지 말고 더 보수적인 sizing을 유지한다
- observation path는 target-day market에만 적용한다. 오늘 관측으로 내일 시장을 덮어쓰지 않는다
- observation source priority는 `exact_public intraday -> documented research intraday -> METAR fallback`이다. source blending은 금지한다
- 기본 consumer command는 `champion` alias를 사용하므로 benchmark publish가 선행돼야 한다
- `live-mm`는 기존 주문 cancel 실패 시 새 quote를 올리지 않고 그 cycle을 중단
- canonical `historical_training_set*` / `historical_backtest_panel`은 기본 immutable이다
- canonical overwrite는 `--allow-canonical-overwrite`가 없는 한 금지한다
- canonical overwrite를 허용할 때도 먼저 `artifacts/recovery/` backup이 생성되는지 확인한다
- truth lag 회수는 cached payload를 맹신하지 말고 `--truth-no-cache --truth-per-source-limit 1`을 우선 고려한다
- `full_training_set_snapshots.json`은 checked-in training inventory이고 daily collection으로 자동 갱신되지 않는다
- `ops_daily`는 active-market evidence와 forward diagnostics 전용이며 champion 재학습/공개 alias 갱신을 하지 않는다
- `weather_train`은 weather-real pretrain 전용이며 Polymarket market id/rule/price/CLOB 데이터를 포함하지 않는다
- `historical_real`과 champion publish path는 계속 `real_market` profile만 허용한다
- public champion alias는 `artifacts/public_models/champion.*`만 신뢰한다

## Commit Rules
- 형식: `type: subject`
- 허용 type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`
- breaking change는 `type!: subject`
- source of truth: `docs/development/commit-convention.md`
- local enforcement: `scripts/check_commit_message.py`

## Doc Sync
- workflow 변경: `README.md`, `.claude/commands/`, relevant skill docs
- live safety 변경: `docs/operations/live-trading.md`, `AGENTS.md`
- data/storage 변경: `docs/agent-skills/data-ops.md`, `docs/codebase/storage-and-configs.md`
