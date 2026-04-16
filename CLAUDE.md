# Claude Project Notes

Primary repository guidance lives in `AGENTS.md`. If `CLAUDE.md` and `AGENTS.md` ever differ, follow `AGENTS.md`.

- Start with `AGENTS.md`, then `docs/codebase/index.md`.
- Claude skills: `.claude/skills/` — auto-invoked by description match.
- Codex skills: `.agents/skills/` — same content, Codex-discovered path.
- Shared skill docs: `docs/agent-skills/`.
- Slash commands: `.claude/commands/`.

## Key Safety Rules (repeat from AGENTS.md)
- `build-dataset` → always `--markets-path configs/market_inventory/full_training_set_snapshots.json`
- canonical `historical_training_set*` / `historical_backtest_panel` overwrite → only with `--allow-canonical-overwrite`
- canonical overwrite → auto-backup first under `artifacts/recovery/`
- lag recovery truth probes → prefer `--truth-no-cache --truth-per-source-limit 1`
- autoresearch → edit one YAML candidate at a time under `artifacts/autoresearch/<run_tag>/candidates/`
- autoresearch → do not publish `champion` implicitly
- `scan-edge` → always `--min-model-prob 0.05 --max-model-prob 0.95`
- `benchmark-models` → always `--retrain-stride 30`
