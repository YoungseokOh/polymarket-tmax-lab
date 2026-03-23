# Safety And Rules

## Operating Rules
- 사용자 응답은 기본적으로 한국어
- live trading은 기본 비활성
- station/source fidelity를 모델 편의보다 우선
- 폴더 책임이나 workflow가 바뀌면 문서도 같은 변경에 포함
- live/paper execution 경로에서는 missing CLOB book을 synthetic으로 숨기지 말고 명시적 skip/error로 처리
- `live-mm`는 기존 주문 cancel 실패 시 새 quote를 올리지 않고 그 cycle을 중단

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
