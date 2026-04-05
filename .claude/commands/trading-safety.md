Read `AGENTS.md`, `docs/live-trading.md`, and `docs/agent-skills/safety-and-rules.md`.

Focus on `${ARGUMENTS}` when provided.

Summarize:
- why live trading is disabled by default
- which environment flags and credentials gate the live broker
- which guardrails apply to paper and live execution
- how missing live books are handled in paper/live paths
- how the observation-station queue and `approve-live-candidate` manual approval step gate the small live pilot
- that observation overrides are target-day only and use `exact_public intraday -> documented research intraday -> METAR fallback`
- why `research_public` observation candidates need explicit tier/risk disclosure and smaller sizing
- that `live-mm` must skip a refresh cycle if cancel fails
- which files and docs must be updated if trading safety changes

Do not recommend enabling live trading unless the user explicitly asks for the
gated path.
