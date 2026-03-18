# Commit Convention

This repository uses English Conventional Commits without scopes.

## Format
```text
type: subject
```

Examples:
- `feat: add det2prob calibration features`
- `fix: parse HKO finalization clause`
- `docs: add folder ownership guide`
- `refactor: split replay metrics from pnl`

## Allowed Types
- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`
- `ci`
- `perf`

## Subject Rules
- Write the summary in English.
- Prefer an imperative, lowercase summary.
- Do not end the summary with a period.
- Keep the header short enough to scan easily in `git log`.

## Breaking Changes
Use `!` after the type when the change is intentionally incompatible.

Example:
- `feat!: rename artifact metadata fields`

## Good And Bad Examples
Good:
- `feat: add lagged previous-run features`
- `fix: cap paper fills by available bankroll`
- `docs: explain agent workflow links`

Bad:
- `update stuff`
- `feat add some changes`
- `fix(markets) added calibration`
- `docs: update parser docs.`

## Local Enforcement
This repo ships a lightweight `commit-msg` checker:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The hook uses `scripts/check_commit_message.py`.

## Optional Template
Use `.gitmessage.txt` as a starting template:

```bash
git config commit.template .gitmessage.txt
```
