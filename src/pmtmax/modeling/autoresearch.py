"""Autoresearch specs and run metadata for LGBM experiments."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pmtmax.config.settings import EnvSettings
from pmtmax.modeling.advanced.lgbm_emos import (
    LgbmEMOSVariantConfig,
    resolve_lgbm_emos_variant,
)
from pmtmax.utils import load_yaml_with_extends, stable_hash

DEFAULT_AUTORESEARCH_ROOT = EnvSettings().artifacts_dir / "autoresearch"
PROMOTED_LGBM_EMOS_DIR = Path("configs/autoresearch/lgbm_emos/promoted")


class LgbmAutoresearchParams(BaseModel):
    """Editable LGBM-EMOS hyperparameters for autoresearch candidates."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int | None = None
    num_leaves: int | None = None
    max_depth: int | None = None
    learning_rate: float | None = None
    min_child_samples: int | None = None
    use_recency_weights: bool | None = None
    recency_half_life_days: float | None = None
    use_oof_scale: bool | None = None
    subsample_freq: int | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    use_quantile_loss: bool | None = None
    use_neighbor_delta: bool | None = None
    fixed_std: float | None = None
    drop_dead_features: bool | None = None
    use_city_lat: bool | None = None
    use_city_month: bool | None = None
    use_clim_anomaly: bool | None = None
    use_forecast_bias: bool | None = None
    use_bin_position: bool | None = None
    use_bin_boundary_dist: bool | None = None
    quantile_center_alpha: float | None = None

    def override_payload(self) -> dict[str, Any]:
        """Return only explicitly-set values."""

        return self.model_dump(exclude_none=True)


class LgbmAutoresearchSpec(BaseModel):
    """One YAML-backed LGBM candidate spec."""

    model_config = ConfigDict(extra="forbid")

    run_tag: str = Field(min_length=3, max_length=120)
    candidate_name: str = Field(min_length=3, max_length=120, pattern=r"^[a-z0-9_]+$")
    model_name: Literal["lgbm_emos"] = "lgbm_emos"
    base_variant: str = "recency_neighbor_oof"
    description: str = ""
    params: LgbmAutoresearchParams = Field(default_factory=LgbmAutoresearchParams)

    @field_validator("run_tag")
    @classmethod
    def _validate_run_tag(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("run_tag must not be empty")
        return normalized

    def build_variant_config(self) -> LgbmEMOSVariantConfig:
        """Resolve one concrete variant config by overriding the baseline."""

        base = asdict(resolve_lgbm_emos_variant(self.base_variant))
        base.update(self.params.override_payload())
        base["name"] = self.candidate_name
        return LgbmEMOSVariantConfig(**base)

    def normalized_payload(self) -> dict[str, Any]:
        """Return a stable payload for writing back to disk."""

        payload = self.model_dump(mode="json")
        payload["params"] = self.params.override_payload()
        return payload


class AutoresearchManifest(BaseModel):
    """Persistent metadata for one autoresearch run."""

    run_tag: str
    model_name: Literal["lgbm_emos"] = "lgbm_emos"
    baseline_variant: str
    root_dir: str
    dataset_path: str
    dataset_signature: str
    panel_path: str
    panel_signature: str
    created_at: datetime
    current_champion_variant: str | None = None


class AutoresearchStepResult(BaseModel):
    """One keep/discard/crash ledger row."""

    run_tag: str
    candidate_name: str
    baseline_variant: str
    evaluated_at: datetime
    status: Literal["keep", "discard", "crash"]
    dataset_signature: str
    baseline_artifact_path: str
    candidate_artifact_path: str
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    metric_deltas: dict[str, float]
    keep_metric: Literal["crps", "brier", "mae"] = "crps"
    notes: str = ""


def default_autoresearch_run_tag(now: datetime | None = None) -> str:
    """Return one date-stamped default run tag."""

    observed_at = now or datetime.now(tz=UTC)
    return observed_at.strftime("%Y%m%d-lgbm-recency-neighbor-oof")


def autoresearch_run_dir(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    """Return the directory for one autoresearch run."""

    return root_dir / run_tag


def autoresearch_manifest_path(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "manifest.json"


def autoresearch_program_path(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "program.md"


def autoresearch_results_path(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "results.jsonl"


def autoresearch_candidates_dir(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "candidates"


def autoresearch_models_dir(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "models"


def autoresearch_analysis_dir(run_tag: str, *, root_dir: Path = DEFAULT_AUTORESEARCH_ROOT) -> Path:
    return autoresearch_run_dir(run_tag, root_dir=root_dir) / "analysis"


def path_signature(path: Path) -> str:
    """Return a cheap filesystem signature for one artifact path."""

    stat = path.stat()
    payload = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return stable_hash(str(payload))


def load_lgbm_autoresearch_spec(path: Path) -> LgbmAutoresearchSpec:
    """Load one YAML candidate spec from disk."""

    return LgbmAutoresearchSpec.model_validate(load_yaml_with_extends(path.resolve()))


def save_lgbm_autoresearch_spec(path: Path, spec: LgbmAutoresearchSpec) -> None:
    """Write one normalized YAML candidate spec."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.normalized_payload(), sort_keys=False))


def promoted_lgbm_emos_spec_path(candidate_name: str, *, base_dir: Path = PROMOTED_LGBM_EMOS_DIR) -> Path:
    """Return the canonical promoted-spec path for one candidate."""

    return base_dir / f"{candidate_name}.yaml"


def load_promoted_lgbm_emos_specs(*, base_dir: Path = PROMOTED_LGBM_EMOS_DIR) -> dict[str, LgbmAutoresearchSpec]:
    """Load all promoted LGBM candidate specs that act like persistent variants."""

    specs: dict[str, LgbmAutoresearchSpec] = {}
    if not base_dir.exists():
        return specs
    for path in sorted(base_dir.glob("*.yaml")):
        spec = load_lgbm_autoresearch_spec(path)
        specs[spec.candidate_name] = spec
    return specs


def supported_promoted_lgbm_emos_variants(*, base_dir: Path = PROMOTED_LGBM_EMOS_DIR) -> tuple[str, ...]:
    """Return promoted YAML-backed variant names."""

    return tuple(sorted(load_promoted_lgbm_emos_specs(base_dir=base_dir)))


def load_promoted_lgbm_emos_variant(
    variant: str,
    *,
    base_dir: Path = PROMOTED_LGBM_EMOS_DIR,
) -> LgbmEMOSVariantConfig | None:
    """Resolve one promoted YAML-backed variant, if present."""

    specs = load_promoted_lgbm_emos_specs(base_dir=base_dir)
    spec = specs.get(variant)
    if spec is None:
        return None
    return spec.build_variant_config()


def render_autoresearch_program(manifest: AutoresearchManifest) -> str:
    """Render the shared `program.md` instructions for Claude/Codex."""

    return (
        f"# PM Tmax Autoresearch Program\n\n"
        f"- Run tag: `{manifest.run_tag}`\n"
        f"- Baseline variant: `{manifest.baseline_variant}`\n"
        f"- Model family: `{manifest.model_name}`\n"
        f"- Dataset: `{manifest.dataset_path}`\n"
        f"- Panel: `{manifest.panel_path}`\n\n"
        "## Mission\n"
        "Improve `lgbm_emos` over the current baseline using small, reviewable candidate specs.\n"
        "Do not edit canonical datasets, canonical panels, or public champion aliases inside the loop.\n\n"
        "## Rules\n"
        "1. Only create or edit one candidate YAML at a time under `candidates/`.\n"
        "2. Keep the baseline fixed unless the human explicitly changes the manifest.\n"
        "3. Use `uv run pmtmax autoresearch-step --spec-path ...` for quick keep/discard decisions.\n"
        "4. Use `uv run pmtmax autoresearch-gate --spec-path ...` before any promotion.\n"
        "5. Use `uv run pmtmax autoresearch-analyze-paper --spec-path ...` before any champion publish review.\n"
        "6. Promotion means copying the winning YAML into `configs/autoresearch/lgbm_emos/promoted/`.\n"
        "7. Public champion publish is always explicit and never automatic.\n\n"
        "## Candidate design hints\n"
        "- Prefer one or two meaningful hyperparameter changes per candidate.\n"
        "- Optimize CRPS first, then Brier, then MAE.\n"
        "- Treat sigma collapse, missing calibrator, and degenerate paper outputs as failures.\n"
    )


def render_candidate_template(run_tag: str, *, baseline_variant: str) -> str:
    """Render one starter YAML candidate."""

    payload = {
        "run_tag": run_tag,
        "candidate_name": "candidate_name_here",
        "model_name": "lgbm_emos",
        "base_variant": baseline_variant,
        "description": "one or two targeted parameter changes",
        "params": {
            "num_leaves": 79,
            "learning_rate": 0.025,
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)
