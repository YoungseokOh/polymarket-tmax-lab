# Safety And Rules

## Operating Rules
- 사용자 응답은 기본적으로 한국어
- live trading은 기본 비활성
- station/source fidelity를 모델 편의보다 우선
- 폴더 책임이나 workflow가 바뀌면 문서도 같은 변경에 포함
- live/paper execution 경로에서는 missing CLOB book을 synthetic으로 숨기지 말고 명시적 skip/error로 처리
- signal path에서는 missing calibrator나 forecast contract mismatch도 fail-closed reason으로 남긴다
- 기본 consumer command는 `champion` alias를 사용하므로 benchmark publish가 선행돼야 한다
- `live-mm`는 기존 주문 cancel 실패 시 새 quote를 올리지 않고 그 cycle을 중단
- canonical `historical_training_set*` / `historical_backtest_panel`은 기본 immutable이다
- canonical overwrite는 `--allow-canonical-overwrite`가 없는 한 금지한다
- canonical overwrite를 허용할 때도 먼저 `artifacts/recovery/` backup이 생성되는지 확인한다
- truth lag 회수는 cached payload를 맹신하지 말고 `--truth-no-cache --truth-per-source-limit 1`을 우선 고려한다

## Commit Rules
- 형식: `type: subject`
- 허용 type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`
- breaking change는 `type!: subject`
- source of truth: `docs/commit-convention.md`
- local enforcement: `scripts/check_commit_message.py`

## Doc Sync
- workflow 변경: `README.md`, `.claude/commands/`, relevant skill docs
- live safety 변경: `docs/live-trading.md`, `AGENTS.md`
- data/storage 변경: `data/README.md`, `docs/codebase/storage-and-configs.md`
