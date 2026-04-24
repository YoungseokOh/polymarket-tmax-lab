"""Automate baseline training, autoresearch, promotion tracking, and checker updates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pmtmax.cli.main import (
    _trust_check_report,
)
from pmtmax.cli.main import (
    autoresearch_analyze_paper as cli_autoresearch_analyze_paper,
)
from pmtmax.cli.main import (
    autoresearch_gate as cli_autoresearch_gate,
)
from pmtmax.cli.main import (
    autoresearch_init as cli_autoresearch_init,
)
from pmtmax.cli.main import (
    autoresearch_promote as cli_autoresearch_promote,
)
from pmtmax.cli.main import (
    autoresearch_step as cli_autoresearch_step,
)
from pmtmax.cli.main import (
    publish_champion as cli_publish_champion,
)
from pmtmax.cli.main import (
    train_advanced as cli_train_advanced,
)
from pmtmax.config.settings import load_settings
from pmtmax.modeling.autoresearch import (
    LgbmAutoresearchParams,
    LgbmAutoresearchSpec,
    autoresearch_analysis_dir,
    autoresearch_candidates_dir,
    autoresearch_manifest_path,
    autoresearch_models_dir,
    autoresearch_results_path,
    autoresearch_run_dir,
    path_signature,
    promoted_lgbm_emos_spec_path,
    save_lgbm_autoresearch_spec,
)

SEOUL_TZ = ZoneInfo("Asia/Seoul")
ACTIVE_RUN_RE = re.compile(r"Active autoresearch run: `([^`]+)`")
DEFAULT_MODEL_NAME = "lgbm_emos"
DEFAULT_BASELINE_VARIANT = "high_neighbor_oof"
DEFAULT_DATASET_PATH = Path("data/workspaces/historical_real/parquet/gold/historical_training_set.parquet")
DEFAULT_PANEL_PATH = Path("data/workspaces/historical_real/parquet/gold/historical_backtest_panel.parquet")
DEFAULT_MODEL_ARTIFACTS_DIR = Path("artifacts/workspaces/historical_real/models/v2")
DEFAULT_AUTORESEARCH_ROOT = Path("artifacts/workspaces/historical_real/autoresearch")
DEFAULT_WEATHER_PRETRAIN_PATH = Path("artifacts/workspaces/weather_train/models/v2/gaussian_emos.pkl")
DEFAULT_CHAMPION_METADATA_PATH = Path("artifacts/public_models/champion.json")


@dataclass(frozen=True)
class AutoCandidatePreset:
    slug: str
    description: str
    params: dict[str, object]


AUTO_CANDIDATE_PRESETS: tuple[AutoCandidatePreset, ...] = (
    AutoCandidatePreset(
        slug="lr_down_leaves_up",
        description="slightly lower learning rate plus a slightly wider tree",
        params={"learning_rate": 0.02, "num_leaves": 111},
    ),
    AutoCandidatePreset(
        slug="trees_subsample",
        description="more trees with row subsampling enabled",
        params={"n_estimators": 750, "subsample_freq": 1, "subsample": 0.85},
    ),
    AutoCandidatePreset(
        slug="recency_45d",
        description="baseline plus short recency weighting",
        params={"use_recency_weights": True, "recency_half_life_days": 45.0},
    ),
    AutoCandidatePreset(
        slug="alpha_lambda",
        description="slightly stronger regularization on the baseline",
        params={"reg_alpha": 0.2, "reg_lambda": 1.5},
    ),
    AutoCandidatePreset(
        slug="bin_boundary",
        description="baseline plus bin-boundary distance feature",
        params={"use_bin_boundary_dist": True},
    ),
)


@dataclass(frozen=True)
class CandidateState:
    candidate_name: str
    spec_path: Path
    description: str
    auto_generated: bool
    status_label: str
    next_stage: Literal["step", "gate", "paper", "promote", "publish"] | None
    step_status: str | None
    gate_passed: bool | None
    paper_decision: str | None
    promoted: bool
    published: bool
    quick_eval_path: Path | None
    gate_summary_path: Path | None
    paper_summary_path: Path | None
    promotion_summary_path: Path | None
    model_path: Path


@dataclass(frozen=True)
class ModelResearchLogEntry:
    run_date: str
    run_tag: str
    actions: str
    outcome: str
    candidate: str
    notes: str

    def to_markdown_row(self) -> str:
        return (
            f"| {self.run_date} | `{self.run_tag}` | {self.actions} | {self.outcome} | "
            f"`{self.candidate}` | {self.notes} |"
        )


@dataclass(frozen=True)
class ModelResearchSummary:
    run_tag: str
    actions: tuple[str, ...]
    candidates_processed: int
    latest_candidate: str | None
    latest_outcome: str
    baseline_trained: bool
    run_initialized: bool
    auto_candidate_created: bool
    promoted_candidate: str | None
    published_candidate: str | None


@dataclass(frozen=True)
class ResearchOps:
    train_baseline: Callable[..., None]
    init_run: Callable[..., None]
    step: Callable[..., None]
    gate: Callable[..., None]
    analyze_paper: Callable[..., None]
    promote: Callable[..., None]
    publish: Callable[..., None]


def default_research_ops() -> ResearchOps:
    return ResearchOps(
        train_baseline=cli_train_advanced,
        init_run=cli_autoresearch_init,
        step=cli_autoresearch_step,
        gate=cli_autoresearch_gate,
        analyze_paper=cli_autoresearch_analyze_paper,
        promote=cli_autoresearch_promote,
        publish=cli_publish_champion,
    )


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _artifact_metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _baseline_artifact_path(*, artifacts_dir: Path, model_name: str, variant: str) -> Path:
    return artifacts_dir / f"{model_name}__{variant}.pkl"


def _candidate_model_path(*, run_tag: str, root_dir: Path, candidate_name: str) -> Path:
    return autoresearch_models_dir(run_tag, root_dir=root_dir) / f"{DEFAULT_MODEL_NAME}__{candidate_name}.pkl"


def _sanitize_token(text: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "_" for char in text]
    token = "".join(cleaned).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "item"


def default_model_research_run_tag(*, baseline_variant: str, now: datetime | None = None) -> str:
    observed = now or datetime.now(tz=UTC)
    return f"{observed.strftime('%Y%m%d')}-lgbm-{_sanitize_token(baseline_variant)}-agent"


def allocate_new_run_tag(*, root_dir: Path, baseline_variant: str) -> str:
    base = default_model_research_run_tag(baseline_variant=baseline_variant)
    candidate = base
    suffix = 2
    while autoresearch_run_dir(candidate, root_dir=root_dir).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _load_manifest(path: Path) -> dict[str, object] | None:
    return _load_json(path)


def infer_active_run_tag(
    *,
    status_path: Path,
    root_dir: Path,
    dataset_signature: str,
    panel_signature: str,
    baseline_variant: str,
) -> str | None:
    if status_path.exists():
        match = ACTIVE_RUN_RE.search(status_path.read_text(encoding="utf-8"))
        if match:
            run_tag = match.group(1)
            manifest = _load_manifest(autoresearch_manifest_path(run_tag, root_dir=root_dir))
            if (
                manifest is not None
                and str(manifest.get("dataset_signature", "")) == dataset_signature
                and str(manifest.get("panel_signature", "")) == panel_signature
                and str(manifest.get("baseline_variant", "")) == baseline_variant
            ):
                return run_tag
    candidates = sorted(root_dir.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        manifest = _load_manifest(path)
        if manifest is None:
            continue
        if (
            str(manifest.get("dataset_signature", "")) == dataset_signature
            and str(manifest.get("panel_signature", "")) == panel_signature
            and str(manifest.get("baseline_variant", "")) == baseline_variant
        ):
            return str(manifest.get("run_tag") or path.parent.name)
    return None


def ensure_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Model Research Log\n\n"
        "Append-only operational log for `historical_real` model training and autoresearch.\n\n"
        "| Run Date | Run Tag | Actions | Outcome | Candidate | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n",
        encoding="utf-8",
    )


def parse_log(path: Path) -> list[ModelResearchLogEntry]:
    if not path.exists():
        return []
    entries: list[ModelResearchLogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        stripped = line.strip()
        if stripped.startswith("| Run Date ") or stripped.startswith("| --- "):
            continue
        parts = [part.strip() for part in stripped.split("|")[1:-1]]
        if len(parts) != 6:
            continue
        entries.append(
            ModelResearchLogEntry(
                run_date=parts[0],
                run_tag=parts[1].strip("`"),
                actions=parts[2],
                outcome=parts[3],
                candidate=parts[4].strip("`"),
                notes=parts[5],
            )
        )
    return entries


def append_log_entry(path: Path, entry: ModelResearchLogEntry) -> None:
    ensure_log(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{entry.to_markdown_row()}\n")


def _latest_step_status_by_candidate(results_path: Path) -> dict[str, dict[str, object]]:
    rows = _load_jsonl(results_path)
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        candidate_name = str(row.get("candidate_name", "")).strip()
        if candidate_name:
            latest[candidate_name] = row
    return latest


def _candidate_status_label(
    *,
    step_status: str | None,
    gate_passed: bool | None,
    paper_decision: str | None,
    promoted: bool,
    published: bool,
    publish_enabled: bool,
    recent_core_summary_path: Path | None,
) -> tuple[str, Literal["step", "gate", "paper", "promote", "publish"] | None]:
    if step_status is None:
        return "queued", "step"
    if step_status != "keep":
        return step_status, None
    if gate_passed is None:
        return "keep", "gate"
    if not gate_passed:
        return "gate_fail", None
    if paper_decision is None:
        return "gate_pass", "paper"
    if paper_decision != "GO":
        return f"paper_{paper_decision.lower()}", None
    if not promoted:
        return "paper_go", "promote"
    if published:
        return "published", None
    if publish_enabled:
        if recent_core_summary_path is None or not recent_core_summary_path.exists():
            return "publish_blocked", None
        return "ready_for_publish", "publish"
    return "promoted", None


def load_candidate_states(
    *,
    run_tag: str,
    root_dir: Path,
    champion_variant: str | None,
    publish_enabled: bool,
    recent_core_summary_path: Path | None,
) -> list[CandidateState]:
    candidates_dir = autoresearch_candidates_dir(run_tag, root_dir=root_dir)
    results_path = autoresearch_results_path(run_tag, root_dir=root_dir)
    latest_steps = _latest_step_status_by_candidate(results_path)
    states: list[CandidateState] = []
    for spec_path in sorted(candidates_dir.glob("*.yaml")):
        if spec_path.name == "candidate_template.yaml":
            continue
        spec = LgbmAutoresearchSpec.model_validate(_load_yaml(spec_path))
        candidate_name = spec.candidate_name
        analysis_dir = autoresearch_analysis_dir(run_tag, root_dir=root_dir)
        quick_eval_path = analysis_dir / f"quick_eval__{candidate_name}.json"
        gate_summary_path = analysis_dir / f"gate_summary__{candidate_name}.json"
        paper_summary_path = analysis_dir / "paper" / f"paper_analysis_summary__{candidate_name}.json"
        promotion_summary_path = analysis_dir / f"promotion_summary__{candidate_name}.json"
        step = _load_json(quick_eval_path)
        if step is None:
            step = latest_steps.get(candidate_name)
        gate_summary = _load_json(gate_summary_path)
        paper_summary = _load_json(paper_summary_path)
        promotion_summary = _load_json(promotion_summary_path)
        step_status = None if step is None else str(step.get("status", "") or "")
        gate_passed = None if gate_summary is None else bool(gate_summary.get("benchmark_gate_passed", False))
        paper_decision = None if paper_summary is None else str(paper_summary.get("overall_gate_decision", "") or "")
        promoted = promotion_summary is not None and promoted_lgbm_emos_spec_path(candidate_name).exists()
        published = champion_variant == candidate_name
        status_label, next_stage = _candidate_status_label(
            step_status=step_status or None,
            gate_passed=gate_passed,
            paper_decision=paper_decision or None,
            promoted=promoted,
            published=published,
            publish_enabled=publish_enabled,
            recent_core_summary_path=recent_core_summary_path,
        )
        states.append(
            CandidateState(
                candidate_name=candidate_name,
                spec_path=spec_path,
                description=spec.description,
                auto_generated=candidate_name.startswith("mr_"),
                status_label=status_label,
                next_stage=next_stage,
                step_status=step_status or None,
                gate_passed=gate_passed,
                paper_decision=paper_decision or None,
                promoted=promoted,
                published=published,
                quick_eval_path=quick_eval_path if quick_eval_path.exists() else None,
                gate_summary_path=gate_summary_path if gate_summary_path.exists() else None,
                paper_summary_path=paper_summary_path if paper_summary_path.exists() else None,
                promotion_summary_path=promotion_summary_path if promotion_summary_path.exists() else None,
                model_path=_candidate_model_path(run_tag=run_tag, root_dir=root_dir, candidate_name=candidate_name),
            )
        )
    return states


def _load_yaml(path: Path) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload: {path}")
    return payload


def next_auto_candidate_spec_path(
    *,
    run_tag: str,
    root_dir: Path,
    baseline_variant: str,
    auto_candidate_limit: int,
) -> Path | None:
    candidates_dir = autoresearch_candidates_dir(run_tag, root_dir=root_dir)
    existing_names = set()
    for path in candidates_dir.glob("*.yaml"):
        if path.name == "candidate_template.yaml":
            continue
        payload = _load_yaml(path)
        existing_names.add(str(payload.get("candidate_name", "")))
    run_token = _sanitize_token(run_tag)[:16]
    for index in range(1, auto_candidate_limit + 1):
        preset = AUTO_CANDIDATE_PRESETS[(index - 1) % len(AUTO_CANDIDATE_PRESETS)]
        candidate_name = f"mr_{run_token}_{index:02d}_{preset.slug}"
        if candidate_name in existing_names:
            continue
        spec = LgbmAutoresearchSpec(
            run_tag=run_tag,
            candidate_name=candidate_name,
            base_variant=baseline_variant,
            description=f"auto candidate: {preset.description}",
            params=LgbmAutoresearchParams.model_validate(preset.params),
        )
        spec_path = candidates_dir / f"{candidate_name}.yaml"
        save_lgbm_autoresearch_spec(spec_path, spec)
        return spec_path
    return None


def _load_champion_metadata(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    return payload or {}


def needs_baseline_training(
    *,
    artifact_path: Path,
    dataset_signature: str,
    pretrained_weather_model: Path | None,
    force: bool,
) -> bool:
    if force or not artifact_path.exists():
        return True
    metadata = _load_json(_artifact_metadata_path(artifact_path))
    if metadata is None:
        return True
    if str(metadata.get("dataset_signature", "")) != dataset_signature:
        return True
    expected_pretrain = None if pretrained_weather_model is None else str(pretrained_weather_model)
    recorded_pretrain = metadata.get("pretrained_weather_model")
    return expected_pretrain is not None and str(recorded_pretrain or "") != expected_pretrain


def _status_counts(states: list[CandidateState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        counts[state.status_label] = counts.get(state.status_label, 0) + 1
    return counts


def _latest_log_entry(entries: list[ModelResearchLogEntry]) -> ModelResearchLogEntry | None:
    return entries[-1] if entries else None


def render_status_markdown(
    *,
    run_tag: str,
    dataset_path: Path,
    panel_path: Path,
    dataset_signature: str,
    panel_signature: str,
    baseline_variant: str,
    baseline_artifact_path: Path,
    pretrained_weather_model: Path | None,
    states: list[CandidateState],
    champion_metadata: dict[str, object],
    log_entries: list[ModelResearchLogEntry],
    recent_core_summary_path: Path | None,
    publish_enabled: bool,
) -> str:
    latest_entry = _latest_log_entry(log_entries)
    status_counts = _status_counts(states)
    champion_variant = str(champion_metadata.get("variant", "unknown"))
    next_candidate = next((state for state in states if state.next_stage is not None), None)
    latest_promoted = next((state for state in reversed(states) if state.promoted), None)
    latest_published = next((state for state in reversed(states) if state.published), None)
    lines: list[str] = [
        "# Model Research Status",
        "",
        f"Updated: {datetime.now(tz=SEOUL_TZ).date().isoformat()} KST",
        "",
        "## Current Snapshot",
        "- workspace: `historical_real`",
        "- dataset profile: `real_market`",
        f"- dataset path: `{dataset_path}`",
        f"- panel path: `{panel_path}`",
        f"- dataset signature: `{dataset_signature}`",
        f"- panel signature: `{panel_signature}`",
        f"- baseline variant: `{baseline_variant}`",
        f"- baseline artifact: `{baseline_artifact_path}`",
        (
            f"- weather pretrain lineage: `{pretrained_weather_model}`"
            if pretrained_weather_model is not None
            else "- weather pretrain lineage: `none`"
        ),
        f"- active autoresearch run: `{run_tag}`",
        f"- current public champion: `{champion_variant}`",
        "",
        "## Candidate Ledger",
        f"- total candidate specs: `{len(states)}`",
        f"- status counts: `{status_counts}`",
    ]
    if latest_promoted is not None:
        lines.append(f"- latest promoted candidate: `{latest_promoted.candidate_name}`")
    if latest_published is not None:
        lines.append(f"- latest published candidate: `{latest_published.candidate_name}`")
    if recent_core_summary_path is not None:
        lines.append(f"- publish summary path override: `{recent_core_summary_path}`")
    lines.extend(["", "## Recent Candidates"])
    if states:
        for state in states[-5:]:
            stage = "done" if state.next_stage is None else state.next_stage
            lines.append(
                f"- `{state.candidate_name}`: `{state.status_label}`"
                f" (next `{stage}`)"
            )
    else:
        lines.append("- none yet")

    lines.extend(["", "## Current Judgment"])
    if latest_entry is not None:
        lines.append(
            f"- Latest agent turn: `{latest_entry.actions}` -> `{latest_entry.outcome}` on `{latest_entry.candidate}`."
        )
    if publish_enabled:
        if recent_core_summary_path is None or not recent_core_summary_path.exists():
            lines.append("- Publish remains blocked until a candidate-specific recent-core GO summary is supplied.")
        else:
            lines.append("- Publish is enabled and will fail closed on a non-GO recent-core summary.")
    else:
        lines.append("- Public champion publish is disabled by default; promotion stops at promoted YAML unless explicitly enabled.")

    lines.extend(["", "## Next Queue"])
    if next_candidate is not None:
        lines.append(
            f"1. Continue `{next_candidate.candidate_name}` from stage `{next_candidate.next_stage}`."
        )
    else:
        lines.append("1. No pending candidate stage is open; the next run will auto-create the next candidate if capacity remains.")
    lines.append("2. Keep `historical_real` mutating jobs serialized around this agent.")
    lines.append("3. Re-run recent-core benchmark for a promoted candidate before any public publish.")
    lines.extend(
        [
            "",
            "## Daily Agent Command",
            "",
            "```bash",
            "scripts/pmtmax-workspace historical_real uv run python scripts/run_model_research_agent.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_model_research_workspace(markets_path: Path) -> None:
    config, _ = load_settings()
    report = _trust_check_report(
        config=config,
        markets_path=markets_path,
        workflow="real_market",
    )
    issues = report.get("issues", [])
    if issues:
        messages = [str(issue.get("message", issue)) for issue in issues if isinstance(issue, dict)]
        raise RuntimeError("; ".join(messages) if messages else "model research trust-check failed")


def run_model_research_agent(
    *,
    status_path: Path,
    log_path: Path,
    markets_path: Path,
    dataset_path: Path,
    panel_path: Path,
    model_name: str,
    baseline_variant: str,
    model_artifacts_dir: Path,
    autoresearch_root: Path,
    pretrained_weather_model: Path | None,
    auto_candidate_limit: int,
    max_candidates: int,
    enable_training: bool,
    enable_gate: bool,
    enable_paper: bool,
    enable_promote: bool,
    enable_publish: bool,
    recent_core_summary_path: Path | None,
    force_baseline_train: bool,
    run_tag_override: str | None,
    ops: ResearchOps | None = None,
) -> ModelResearchSummary:
    _verify_model_research_workspace(markets_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset missing: {dataset_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"panel missing: {panel_path}")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_log(log_path)
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    autoresearch_root.mkdir(parents=True, exist_ok=True)
    dataset_signature = path_signature(dataset_path)
    panel_signature = path_signature(panel_path)
    ops = ops or default_research_ops()

    actions: list[str] = []
    baseline_trained = False
    run_initialized = False
    auto_candidate_created = False
    promoted_candidate: str | None = None
    published_candidate: str | None = None
    latest_candidate: str | None = None
    latest_outcome = "idle"

    baseline_artifact_path = _baseline_artifact_path(
        artifacts_dir=model_artifacts_dir,
        model_name=model_name,
        variant=baseline_variant,
    )
    requested_pretrain = pretrained_weather_model if pretrained_weather_model is not None and pretrained_weather_model.exists() else None
    if enable_training and needs_baseline_training(
        artifact_path=baseline_artifact_path,
        dataset_signature=dataset_signature,
        pretrained_weather_model=requested_pretrain,
        force=force_baseline_train,
    ):
        ops.train_baseline(
            dataset_path=dataset_path,
            model_name=model_name,
            artifacts_dir=model_artifacts_dir,
            variant=baseline_variant,
            pretrained_weather_model=requested_pretrain,
        )
        actions.append("train_baseline")
        baseline_trained = True

    run_tag = run_tag_override or infer_active_run_tag(
        status_path=status_path,
        root_dir=autoresearch_root,
        dataset_signature=dataset_signature,
        panel_signature=panel_signature,
        baseline_variant=baseline_variant,
    )
    if run_tag is None:
        run_tag = allocate_new_run_tag(root_dir=autoresearch_root, baseline_variant=baseline_variant)
        ops.init_run(
            run_tag=run_tag,
            dataset_path=dataset_path,
            panel_path=panel_path,
            baseline_variant=baseline_variant,
            root_dir=autoresearch_root,
            force=False,
        )
        actions.append("init_run")
        run_initialized = True

    champion_metadata = _load_champion_metadata(DEFAULT_CHAMPION_METADATA_PATH)
    champion_variant = str(champion_metadata.get("variant", "")) or None
    states = load_candidate_states(
        run_tag=run_tag,
        root_dir=autoresearch_root,
        champion_variant=champion_variant,
        publish_enabled=enable_publish,
        recent_core_summary_path=recent_core_summary_path,
    )
    pending = [state for state in states if state.next_stage is not None]
    if not pending and auto_candidate_limit > 0:
        auto_spec_path = next_auto_candidate_spec_path(
            run_tag=run_tag,
            root_dir=autoresearch_root,
            baseline_variant=baseline_variant,
            auto_candidate_limit=auto_candidate_limit,
        )
        if auto_spec_path is not None:
            actions.append("create_candidate")
            auto_candidate_created = True
            states = load_candidate_states(
                run_tag=run_tag,
                root_dir=autoresearch_root,
                champion_variant=champion_variant,
                publish_enabled=enable_publish,
                recent_core_summary_path=recent_core_summary_path,
            )
            pending = [state for state in states if state.next_stage is not None]

    processed = 0
    for state in pending:
        if processed >= max_candidates:
            break
        latest_candidate = state.candidate_name
        while state.next_stage is not None:
            if state.next_stage == "step":
                ops.step(spec_path=state.spec_path, dataset_path=dataset_path, root_dir=autoresearch_root)
                actions.append(f"step:{state.candidate_name}")
            elif state.next_stage == "gate":
                if not enable_gate:
                    break
                ops.gate(spec_path=state.spec_path, dataset_path=dataset_path, panel_path=panel_path, root_dir=autoresearch_root)
                actions.append(f"gate:{state.candidate_name}")
            elif state.next_stage == "paper":
                if not enable_paper:
                    break
                ops.analyze_paper(spec_path=state.spec_path, dataset_path=dataset_path, root_dir=autoresearch_root)
                actions.append(f"paper:{state.candidate_name}")
            elif state.next_stage == "promote":
                if not enable_promote:
                    break
                ops.promote(spec_path=state.spec_path, root_dir=autoresearch_root, publish_champion=False, force=False)
                actions.append(f"promote:{state.candidate_name}")
                promoted_candidate = state.candidate_name
            elif state.next_stage == "publish":
                if not enable_publish or recent_core_summary_path is None:
                    break
                ops.publish(
                    model_path=state.model_path,
                    model_name=model_name,
                    variant=state.candidate_name,
                    recent_core_summary_path=recent_core_summary_path,
                )
                actions.append(f"publish:{state.candidate_name}")
                published_candidate = state.candidate_name
            champion_metadata = _load_champion_metadata(DEFAULT_CHAMPION_METADATA_PATH)
            champion_variant = str(champion_metadata.get("variant", "")) or None
            states = load_candidate_states(
                run_tag=run_tag,
                root_dir=autoresearch_root,
                champion_variant=champion_variant,
                publish_enabled=enable_publish,
                recent_core_summary_path=recent_core_summary_path,
            )
            state = next((candidate for candidate in states if candidate.candidate_name == state.candidate_name), state)
            latest_outcome = state.status_label
            if state.next_stage is None:
                break
            if state.next_stage == "gate" and not enable_gate:
                break
            if state.next_stage == "paper" and not enable_paper:
                break
            if state.next_stage == "promote" and not enable_promote:
                break
            if state.next_stage == "publish" and (not enable_publish or recent_core_summary_path is None):
                break
        processed += 1

    champion_metadata = _load_champion_metadata(DEFAULT_CHAMPION_METADATA_PATH)
    champion_variant = str(champion_metadata.get("variant", "")) or None
    states = load_candidate_states(
        run_tag=run_tag,
        root_dir=autoresearch_root,
        champion_variant=champion_variant,
        publish_enabled=enable_publish,
        recent_core_summary_path=recent_core_summary_path,
    )
    if latest_candidate is None and states:
        latest_candidate = states[-1].candidate_name
        latest_outcome = states[-1].status_label
    elif latest_candidate is None:
        latest_candidate = "-"
        latest_outcome = "idle"

    log_entries = parse_log(log_path)
    notes = (
        f"baseline `{baseline_variant}`; candidates `{len(states)}`; "
        f"promoted `{promoted_candidate or 'none'}`; published `{published_candidate or 'none'}`."
    )
    entry = ModelResearchLogEntry(
        run_date=datetime.now(tz=SEOUL_TZ).date().isoformat(),
        run_tag=run_tag,
        actions=", ".join(actions) if actions else "status_only",
        outcome=latest_outcome,
        candidate=latest_candidate or "-",
        notes=notes,
    )
    append_log_entry(log_path, entry)
    log_entries = parse_log(log_path)
    status_path.write_text(
        render_status_markdown(
            run_tag=run_tag,
            dataset_path=dataset_path,
            panel_path=panel_path,
            dataset_signature=dataset_signature,
            panel_signature=panel_signature,
            baseline_variant=baseline_variant,
            baseline_artifact_path=baseline_artifact_path,
            pretrained_weather_model=requested_pretrain,
            states=states,
            champion_metadata=champion_metadata,
            log_entries=log_entries,
            recent_core_summary_path=recent_core_summary_path,
            publish_enabled=enable_publish,
        ),
        encoding="utf-8",
    )

    return ModelResearchSummary(
        run_tag=run_tag,
        actions=tuple(actions),
        candidates_processed=processed,
        latest_candidate=None if latest_candidate == "-" else latest_candidate,
        latest_outcome=latest_outcome,
        baseline_trained=baseline_trained,
        run_initialized=run_initialized,
        auto_candidate_created=auto_candidate_created,
        promoted_candidate=promoted_candidate,
        published_candidate=published_candidate,
    )
