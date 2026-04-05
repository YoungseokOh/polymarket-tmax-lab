# Agent Skills

이 디렉터리는 Codex와 Claude가 함께 참조하는 공용 skill reference 문서 모음이다.

## Skills
- `pmtmax-repo`: 저장소 온보딩과 읽기 순서
- `pmtmax-data-ops`: `bootstrap-lab`, warehouse, seed, legacy run 정리
- `pmtmax-market-rules`: 시장 탐색, 규칙 파싱, station/source fidelity
- `pmtmax-research-loop`: 데이터셋, 학습, 백테스트, paper-trader 흐름
- `pmtmax-autoresearch`: `recency_neighbor_oof` 기준 autoresearch 루프와 candidate YAML gating
- `pmtmax-commit`: 커밋 메시지와 staging, push 전 점검
- `pmtmax-release-checklist`: `bootstrap-lab` 이후 연구 환경 점검

## Source Of Truth
- 공용 운영 계약: `AGENTS.md`
- 커밋 규칙: `docs/commit-convention.md`
- 프로젝트별 Claude 진입점: `.claude/commands/`
- Codex repo-local skills: `tools/skills/`
- Claude project skills: `.claude/skills/`
