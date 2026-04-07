"""Typer CLI."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from typer.models import OptionInfo

from pmtmax.backfill import BackfillPipeline
from pmtmax.backtest.dataset_builder import DatasetBuilder
from pmtmax.backtest.metrics import summarize_trade_log
from pmtmax.backtest.pnl import Position, settle_position
from pmtmax.backtest.rolling_origin import rolling_origin_splits
from pmtmax.config.settings import EnvSettings, load_settings
from pmtmax.execution.edge import compute_edge
from pmtmax.execution.fees import estimate_fee
from pmtmax.execution.guardrails import exposure_ok, forecast_fresh, spread_ok
from pmtmax.execution.live_broker import LiveBroker
from pmtmax.execution.opportunity_shadow import OpportunityShadowRunner, select_shadow_horizon
from pmtmax.execution.paper_broker import PaperBroker
from pmtmax.execution.revenue_gate import DEFAULT_PILOT_CONSTRAINTS, build_revenue_gate_report
from pmtmax.execution.sizing import capped_kelly
from pmtmax.execution.slippage import estimate_book_slippage
from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import configure_logging
from pmtmax.markets.book_utils import (
    fetch_book as _fetch_book,
)
from pmtmax.markets.book_utils import (
    synthetic_book as _synthetic_book,
)
from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.markets.gamma_client import GammaClient
from pmtmax.markets.inventory import (
    discover_temperature_event_refs_from_gamma,
    fetch_temperature_event_pages,
    snapshots_from_temperature_event_fetches,
)
from pmtmax.markets.market_spec import MarketSpec
from pmtmax.markets.repository import (
    bundled_market_snapshots,
    load_market_snapshots,
    save_market_snapshots,
)
from pmtmax.markets.station_registry import lookup_city_stations
from pmtmax.markets.station_registry import lookup_station
from pmtmax.markets.station_registry import supported_cities as catalog_supported_cities
from pmtmax.modeling.advanced.lgbm_emos import LgbmEMOSVariantConfig
from pmtmax.modeling.autoresearch import (
    DEFAULT_AUTORESEARCH_ROOT,
    AutoresearchManifest,
    AutoresearchStepResult,
    autoresearch_analysis_dir,
    autoresearch_candidates_dir,
    autoresearch_manifest_path,
    autoresearch_models_dir,
    autoresearch_program_path,
    autoresearch_results_path,
    autoresearch_run_dir,
    default_autoresearch_run_tag,
    load_lgbm_autoresearch_spec,
    path_signature,
    promoted_lgbm_emos_spec_path,
    render_autoresearch_program,
    render_candidate_template,
    save_lgbm_autoresearch_spec,
)
from pmtmax.modeling.champion import (
    score_leaderboard,
    score_trading_leaderboard,
    select_champion,
    select_trading_champion,
)
from pmtmax.modeling.design_matrix import group_id_series
from pmtmax.modeling.evaluation import (
    brier_score,
    calibration_gap,
    crps_from_samples,
    gaussian_nll,
    mae,
    rmse,
)
from pmtmax.modeling.predict import predict_market
from pmtmax.modeling.quick_eval import evaluate_saved_model, quick_eval_holdout
from pmtmax.modeling.train import (
    require_supported_model_name,
    require_supported_variant,
    supported_ablation_variants,
    supported_model_names,
    train_model,
)
from pmtmax.monitoring.hope_hunt import HopeHuntRunner
from pmtmax.monitoring.observation_station import (
    ObservationShadowRunner,
)
from pmtmax.monitoring.open_phase import (
    OpenPhaseShadowRunner,
    extract_open_phase_metadata,
    open_phase_age_bucket,
    select_open_phase_candidates,
)
from pmtmax.monitoring.station_dashboard import StationDashboardRunner
from pmtmax.storage.duckdb_store import DuckDBStore
from pmtmax.storage.firebase_mirror import FirebaseMirror
from pmtmax.storage.lab_bootstrap import (
    archive_legacy_runs,
    build_bootstrap_manifest,
    export_seed_bundle,
    inventory_legacy_runs,
    restore_seed_bundle,
    restore_warehouse_from_seed,
)
from pmtmax.storage.parquet_store import ParquetStore
from pmtmax.storage.schemas import (
    BookSnapshot,
    HopeHuntObservation,
    LegacyRunInventory,
    LiveTemperatureObservation,
    MarketSnapshot,
    ObservationOpportunity,
    OpenPhaseObservation,
    OpportunityObservation,
    ProbForecast,
    RiskLimits,
    TradeSignal,
)
from pmtmax.storage.warehouse import DataWarehouse, backup_duckdb_file, ordered_legacy_paths
from pmtmax.utils import dump_json, load_json, load_yaml_with_extends, set_global_seed, stable_hash
from pmtmax.weather.intraday_observation import fetch_intraday_observations
from pmtmax.weather.metar_client import MetarClient
from pmtmax.weather.openmeteo_client import OpenMeteoClient
from pmtmax.weather.truth_sources.base import celsius_to_fahrenheit

app = typer.Typer(help="Polymarket maximum temperature research and trading lab.")
console = Console()
DEFAULT_RECENT_HORIZON_POLICY_PATH = Path("configs/recent-core-horizon-policy.yaml")
DEFAULT_PAPER_ALL_SUPPORTED_HORIZON_POLICY_PATH = Path("configs/paper-all-supported-horizon-policy.yaml")
DEFAULT_PAPER_EXPLORATION_CONFIG_PATH = Path("configs/paper-exploration.yaml")
DEFAULT_PAPER_MULTIMODEL_ROOT = Path("artifacts/signals/v2/paper_multimodel")
DEFAULT_EXECUTION_SENSITIVITY_ROOT = Path("artifacts/signals/v2/execution_sensitivity")
DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_PATH = Path("artifacts/signals/v2/execution_watchlist_playbook.json")
DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_MD_PATH = Path("artifacts/signals/v2/execution_watchlist_playbook.md")
DEFAULT_BENCHMARK_SUMMARY_PATH = Path("artifacts/benchmarks/v2/benchmark_summary.json")
DEFAULT_MODEL_NAME = "champion"
TRADING_MODEL_ALIAS = "trading_champion"
PUBLIC_MODEL_ALIASES = {DEFAULT_MODEL_NAME, TRADING_MODEL_ALIAS}
RECENT_CORE_CITIES = ("Seoul", "NYC", "London")
MARKET_SCOPE_CHOICES = ("default", "recent_core", "supported_wu_open_phase")


def _resolve_option_value(value: Any, fallback: Any = None) -> Any:
    """Return plain default values when Typer OptionInfo leaks into direct calls."""

    if isinstance(value, OptionInfo):
        default = value.default
        return fallback if default is ... else default
    return value


def _default_dataset_path(*, sequence: bool = False, panel: bool = False) -> Path:
    """Return the canonical v2 dataset or panel path."""

    if panel:
        return Path("data/parquet/gold/v2/historical_backtest_panel.parquet")
    suffix = "historical_training_set_sequence.parquet" if sequence else "historical_training_set.parquet"
    return Path("data/parquet/gold/v2") / suffix


def _default_artifacts_dir() -> Path:
    """Return the canonical v2 model artifact directory."""

    return Path("artifacts/models/v2")


def _default_champion_metadata_path() -> Path:
    """Return the canonical champion metadata path."""

    return _default_artifacts_dir() / "champion.json"


def _default_alias_metadata_path(alias_name: str) -> Path:
    """Return the canonical metadata path for one public model alias."""

    if alias_name == DEFAULT_MODEL_NAME:
        return _default_champion_metadata_path()
    return _default_artifacts_dir() / f"{alias_name}.json"


def _default_model_path(model_name: str) -> Path:
    """Return the canonical v2 model artifact path."""

    return _default_artifacts_dir() / f"{model_name}.pkl"


def _default_backtest_output(filename: str) -> Path:
    """Return the canonical v2 backtest output path."""

    return Path("artifacts/backtests/v2") / filename


def _default_benchmark_output(filename: str) -> Path:
    """Return the canonical v2 benchmark output path."""

    return Path("artifacts/benchmarks/v2") / filename


def _default_signal_output(filename: str) -> Path:
    """Return the canonical v2 signal/report output path."""

    return Path("artifacts/signals/v2") / filename


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    """Append one JSON object to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _artifact_calibration_path(model_path: Path) -> Path:
    """Return the sibling calibrator path for one saved model artifact."""

    return model_path.with_name(f"{model_path.stem}.calibrator.pkl")


def _current_alias_variant(alias_name: str) -> str | None:
    """Return the current variant recorded for one public alias."""

    metadata_path = _default_alias_metadata_path(alias_name)
    if not metadata_path.exists():
        return None
    payload = load_json(metadata_path)
    if not isinstance(payload, dict):
        return None
    variant = payload.get("variant")
    return str(variant) if variant else None


def _load_autoresearch_manifest(
    run_tag: str,
    *,
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
) -> AutoresearchManifest:
    """Load one autoresearch run manifest or fail with a clear CLI error."""

    manifest_path = autoresearch_manifest_path(run_tag, root_dir=root_dir)
    if not manifest_path.exists():
        msg = (
            f"Autoresearch manifest does not exist: {manifest_path}. "
            "Run `uv run pmtmax autoresearch-init` first."
        )
        raise typer.BadParameter(msg)
    return AutoresearchManifest.model_validate(load_json(manifest_path))


def _resolve_lgbm_variant_inputs(
    *,
    model_name: str,
    variant: str | None,
    variant_spec: Path | None,
) -> tuple[str | None, LgbmEMOSVariantConfig | None, object | None]:
    """Resolve one built-in variant or one YAML-backed external candidate."""

    resolved_variant = _resolve_option_value(variant)
    resolved_spec_path = _resolve_option_value(variant_spec)
    if resolved_spec_path is None:
        return resolved_variant, None, None
    if resolved_variant is not None:
        raise typer.BadParameter("Use either --variant or --variant-spec, not both.")
    if model_name != "lgbm_emos":
        raise typer.BadParameter("--variant-spec is only supported for lgbm_emos.")
    spec = load_lgbm_autoresearch_spec(Path(resolved_spec_path))
    return spec.candidate_name, spec.build_variant_config(), spec


def _publish_existing_model_alias(
    *,
    alias_name: str,
    model_name: str,
    variant: str | None,
    source_model_path: Path,
    source_calibration_path: Path | None = None,
    leaderboard_path: Path | None = None,
) -> dict[str, object]:
    """Publish one alias from an already-trained artifact instead of retraining."""

    alias_path = _default_model_path(alias_name)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_model_path, alias_path)
    alias_calibration_path = alias_path.with_name(f"{alias_path.stem}.calibrator.pkl")
    if source_calibration_path is not None and source_calibration_path.exists():
        shutil.copyfile(source_calibration_path, alias_calibration_path)
    metadata: dict[str, object] = {
        "alias_name": alias_name,
        "alias_path": str(alias_path),
        "alias_calibration_path": str(alias_calibration_path),
        "contract_version": "v2",
        "model_name": model_name,
        "published_at": datetime.now(tz=UTC).isoformat(),
        "source_model_path": str(source_model_path),
        "source_calibration_path": str(source_calibration_path) if source_calibration_path is not None else None,
        "variant": variant,
    }
    if leaderboard_path is not None:
        metadata["leaderboard_path"] = str(leaderboard_path)
    dump_json(_default_alias_metadata_path(alias_name), metadata)
    return metadata


def _summarize_reason_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate one signal/report payload by reason, city, and horizon."""

    reason_counts: dict[str, int] = {}
    by_city: dict[str, int] = {}
    by_horizon: dict[str, int] = {}
    book_source_counts: dict[str, int] = {}
    edges: list[float] = []
    tradable_count = 0
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        city = row.get("city")
        if city:
            by_city[str(city)] = by_city.get(str(city), 0) + 1
        horizon = row.get("decision_horizon")
        if horizon:
            by_horizon[str(horizon)] = by_horizon.get(str(horizon), 0) + 1
        if reason == "tradable":
            tradable_count += 1
        edge = row.get("edge")
        if edge is not None:
            with suppress(TypeError, ValueError):
                edges.append(float(edge))
        row_book_counts = row.get("book_source_counts")
        if isinstance(row_book_counts, dict):
            for key, value in row_book_counts.items():
                try:
                    increment = int(value)
                except (TypeError, ValueError):
                    continue
                book_source_counts[str(key)] = book_source_counts.get(str(key), 0) + increment
    return {
        "row_count": len(rows),
        "tradable_count": tradable_count,
        "reason_counts": reason_counts,
        "by_city": by_city,
        "by_horizon": by_horizon,
        "book_source_counts": book_source_counts,
        "edge_summary": {
            "count": len(edges),
            "positive_count": sum(1 for edge in edges if edge > 0),
            "max": max(edges) if edges else 0.0,
            "mean": float(sum(edges) / len(edges)) if edges else 0.0,
        },
    }


def _effective_split_policy(
    split_policy: Literal["market_day", "target_day"],
) -> Literal["market_day", "target_day"]:
    """Validate the grouped split policy used in v2."""

    return split_policy


def _load_alias_metadata(alias_name: str = DEFAULT_MODEL_NAME, path: Path | None = None) -> dict[str, object]:
    """Load public alias metadata or fail closed when unpublished."""

    metadata_path = path or _default_alias_metadata_path(alias_name)
    if not metadata_path.exists():
        msg = (
            f"Model alias metadata does not exist: {metadata_path}. "
            "Run `uv run pmtmax benchmark-models` first."
        )
        raise typer.BadParameter(msg)
    payload = load_json(metadata_path)
    if not isinstance(payload, dict) or "model_name" not in payload:
        msg = f"Champion metadata is malformed: {metadata_path}"
        raise typer.BadParameter(msg)
    return _repair_alias_metadata(alias_name, payload, metadata_path=metadata_path)


def _repair_alias_metadata(
    alias_name: str,
    metadata: dict[str, object],
    *,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    """Repair broken alias metadata when local canonical artifacts already exist."""

    alias_path = _default_model_path(alias_name)
    alias_calibration_path = alias_path.with_name(f"{alias_path.stem}.calibrator.pkl")
    model_name = require_supported_model_name(str(metadata["model_name"]))

    def _candidate_path(*keys: str, fallback: Path | None = None) -> Path | None:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value:
                path = Path(value)
                if path.exists():
                    return path
        if fallback is not None and fallback.exists():
            return fallback
        return None

    source_model_path = _candidate_path("alias_path", "source_model_path", fallback=_default_model_path(model_name))
    source_calibrator_path = _candidate_path(
        "alias_calibration_path",
        "source_calibration_path",
        fallback=_default_model_path(model_name).with_name(f"{model_name}.calibrator.pkl"),
    )
    repaired = dict(metadata)
    repaired_any = False

    if not alias_path.exists() and source_model_path is not None:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        if source_model_path.resolve() != alias_path.resolve():
            shutil.copyfile(source_model_path, alias_path)
        repaired_any = True
    if not alias_calibration_path.exists() and source_calibrator_path is not None:
        alias_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        if source_calibrator_path.resolve() != alias_calibration_path.resolve():
            shutil.copyfile(source_calibrator_path, alias_calibration_path)
        repaired_any = True

    if alias_path.exists() and repaired.get("alias_path") != str(alias_path):
        repaired["alias_path"] = str(alias_path)
        repaired_any = True
    if alias_calibration_path.exists() and repaired.get("alias_calibration_path") != str(alias_calibration_path):
        repaired["alias_calibration_path"] = str(alias_calibration_path)
        repaired_any = True
    if source_model_path is not None and repaired.get("source_model_path") != str(source_model_path):
        repaired["source_model_path"] = str(source_model_path)
        repaired_any = True
    if source_calibrator_path is not None and repaired.get("source_calibration_path") != str(source_calibrator_path):
        repaired["source_calibration_path"] = str(source_calibrator_path)
        repaired_any = True

    final_metadata_path = metadata_path or _default_alias_metadata_path(alias_name)
    if repaired_any:
        dump_json(final_metadata_path, repaired)
    return repaired


def _load_champion_metadata(path: Path | None = None) -> dict[str, object]:
    """Load the default champion metadata."""

    return _load_alias_metadata(DEFAULT_MODEL_NAME, path)


def _resolve_model_name_alias(model_name: str) -> str:
    """Resolve the public `champion` alias into a concrete trainable model name."""

    if model_name not in PUBLIC_MODEL_ALIASES:
        return require_supported_model_name(model_name)
    champion = _load_alias_metadata(model_name)
    return require_supported_model_name(str(champion["model_name"]))


def _resolve_model_path(model_path: Path, model_name: str) -> tuple[Path, str]:
    """Resolve a CLI model path and public alias into a concrete artifact path/name."""

    if model_name in PUBLIC_MODEL_ALIASES:
        champion = _load_alias_metadata(model_name)
        return Path(str(champion.get("alias_path", _default_model_path(model_name)))), str(champion["model_name"])
    resolved_name = require_supported_model_name(model_name)
    if model_path == _default_model_path(DEFAULT_MODEL_NAME):
        return _default_model_path(resolved_name), resolved_name
    return model_path, resolved_name


def _runtime(include_stores: bool = True) -> tuple:
    config, env = load_settings()
    configure_logging(env.log_level)
    set_global_seed(config.app.random_seed)
    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    duckdb_store = DuckDBStore(config.app.duckdb_path) if include_stores else None
    parquet_store = ParquetStore(config.app.parquet_dir) if include_stores else None
    openmeteo = OpenMeteoClient(http, config.weather.openmeteo_base_url, config.weather.archive_base_url)
    return config, env, http, duckdb_store, parquet_store, openmeteo


def _load_recent_horizon_policy(path: Path = DEFAULT_RECENT_HORIZON_POLICY_PATH) -> dict[str, list[str]]:
    """Load the repo's city-specific recent horizon policy."""

    if not path.exists():
        return {}
    payload = load_yaml_with_extends(path.resolve())
    cities = payload.get("cities", {})
    result: dict[str, list[str]] = {}
    if not isinstance(cities, dict):
        return result
    for city, city_payload in cities.items():
        if not isinstance(city_payload, dict):
            continue
        allowed_horizons = city_payload.get("allowed_horizons", [])
        result[str(city)] = [str(horizon) for horizon in allowed_horizons]
    return result


def _load_paper_exploration_preset(
    path: Path = DEFAULT_PAPER_EXPLORATION_CONFIG_PATH,
) -> dict[str, object]:
    """Load one paper-only sensitivity sweep preset from YAML."""

    if not path.exists():
        return {}
    payload = load_yaml_with_extends(path.resolve())
    return payload if isinstance(payload, dict) else {}


def _resolve_recent_core_cities(
    cities: list[str] | None,
    *,
    core_recent_only: bool,
) -> list[str] | None:
    """Apply the recent-core operating preset when requested."""

    if not core_recent_only:
        return cities
    if cities is None:
        return list(RECENT_CORE_CITIES)
    return [city for city in cities if city in RECENT_CORE_CITIES]


def _supported_wu_open_phase_cities() -> list[str]:
    """Return supported Wunderground-family cities eligible for hope-hunt scans."""

    cities: list[str] = []
    for city in catalog_supported_cities():
        definition = lookup_station(city)
        if definition is None:
            continue
        if "wunderground" not in definition.official_source_name.lower():
            continue
        if definition.truth_track != "research_public":
            continue
        cities.append(definition.city)
    return sorted(set(cities))


def _resolve_market_scope(
    market_scope: str | None,
    *,
    core_recent_only: bool,
) -> str:
    """Normalize supported market-scope presets."""

    if core_recent_only:
        return "recent_core"
    scope = "default" if market_scope in (None, "") else str(market_scope)
    if scope not in MARKET_SCOPE_CHOICES:
        msg = f"Unsupported market scope: {scope}. Choose from {', '.join(MARKET_SCOPE_CHOICES)}."
        raise typer.BadParameter(msg)
    return scope


def _resolve_scoped_cities(
    cities: list[str] | None,
    *,
    market_scope: str,
) -> list[str] | None:
    """Apply one market-scope preset to a city list."""

    if market_scope == "recent_core":
        return _resolve_recent_core_cities(cities, core_recent_only=True)
    if market_scope == "supported_wu_open_phase":
        supported = _supported_wu_open_phase_cities()
        if cities is None:
            return supported
        wanted = {city.lower() for city in cities}
        return [city for city in supported if city.lower() in wanted]
    return cities


def _snapshot_matches_market_scope(snapshot: MarketSnapshot, *, market_scope: str) -> bool:
    """Return whether one parsed market matches the requested operating scope."""

    if market_scope == "default":
        return True
    spec = snapshot.spec
    if spec is None:
        return False
    if market_scope == "recent_core":
        return spec.city in RECENT_CORE_CITIES
    if market_scope == "supported_wu_open_phase":
        definition = lookup_station(spec.city)
        if definition is None:
            return False
        return (
            spec.adapter_key() == "wunderground"
            and spec.truth_track == "research_public"
            and "wunderground" in definition.official_source_name.lower()
            and definition.truth_track == "research_public"
        )
    msg = f"Unsupported market scope: {market_scope}"
    raise ValueError(msg)


def _safe_float(value: object) -> float | None:
    """Parse a best-effort float from raw market payload values."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _market_volume_from_snapshot(snapshot: MarketSnapshot) -> float | None:
    """Return one aggregate volume estimate for ranking hope-hunt candidates."""

    market = snapshot.market
    if (top_level_volume := _safe_float(market.get("volumeNum"))) is not None:
        return top_level_volume
    if (top_level_volume := _safe_float(market.get("volume"))) is not None:
        return top_level_volume
    component_volumes = [
        value
        for component in market.get("componentMarkets", [])
        if isinstance(component, dict)
        for value in (_safe_float(component.get("volumeNum")) or _safe_float(component.get("volume")),)
        if value is not None
    ]
    if component_volumes:
        return float(sum(component_volumes))
    return None


def _target_day_distance(spec: MarketSpec, *, observed_at: datetime) -> int:
    """Return the local-day distance between observation time and target date."""

    local_today = observed_at.astimezone(ZoneInfo(spec.timezone)).date()
    return (spec.target_local_date - local_today).days


def _select_policy_candidate_horizon(spec: MarketSpec, *, now_utc: datetime) -> str | None:
    """Return the default policy horizon for an active market based on local day."""

    local_today = now_utc.astimezone(ZoneInfo(spec.timezone)).date()
    delta_days = (spec.target_local_date - local_today).days
    if delta_days < 0:
        return None
    if delta_days == 0:
        return "morning_of"
    if delta_days == 1:
        return "previous_evening"
    return "market_open"


def _policy_allows_horizon(
    spec: MarketSpec,
    *,
    decision_horizon: str,
    horizon_policy: dict[str, list[str]],
) -> bool:
    """Return whether the city-level horizon policy allows this horizon."""

    allowed_horizons = horizon_policy.get(spec.city)
    if not allowed_horizons:
        return True
    return decision_horizon in allowed_horizons


def _resolve_signal_horizon(
    spec: MarketSpec,
    *,
    now_utc: datetime,
    horizon: str,
    horizon_policy: dict[str, list[str]],
) -> str | None:
    """Resolve the execution horizon for active-market signal paths."""

    if horizon != "policy":
        return horizon
    candidate = _select_policy_candidate_horizon(spec, now_utc=now_utc)
    if candidate is None:
        return None
    if not _policy_allows_horizon(spec, decision_horizon=candidate, horizon_policy=horizon_policy):
        return None
    return candidate


def _resolve_signal_horizon_with_reason(
    spec: MarketSpec,
    *,
    now_utc: datetime,
    horizon: str,
    horizon_policy: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    """Resolve a live-signal horizon and explain policy skips."""

    if horizon != "policy":
        return horizon, None
    candidate = _select_policy_candidate_horizon(spec, now_utc=now_utc)
    if candidate is None:
        return None, "past_market"
    if not _policy_allows_horizon(spec, decision_horizon=candidate, horizon_policy=horizon_policy):
        return candidate, "policy_filtered"
    return candidate, None


def _backfill_pipeline(
    config: Any,
    http: CachedHttpClient,
    openmeteo: OpenMeteoClient,
    *,
    use_truth_cache: bool = True,
) -> BackfillPipeline:
    warehouse = DataWarehouse.from_paths(
        duckdb_path=config.app.duckdb_path,
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        archive_root=config.app.archive_dir,
        recovery_root=Path("artifacts/recovery"),
    )
    return BackfillPipeline(
        http=http,
        openmeteo=openmeteo,
        warehouse=warehouse,
        models=config.weather.models,
        truth_snapshot_dir=config.app.raw_dir / "bronze",
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
        use_truth_cache=use_truth_cache,
    )


def _config_hash(config: Any, command: str, **kwargs: object) -> str:
    payload = {
        "command": command,
        "config": config.model_dump(mode="json"),
        "kwargs": kwargs,
    }
    return stable_hash(json.dumps(payload, sort_keys=True, default=str))


def _filter_snapshots_by_city(snapshots: list[MarketSnapshot], cities: list[str] | None) -> list[MarketSnapshot]:
    if not cities:
        return snapshots
    wanted = {city.lower() for city in cities}
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.spec is not None and snapshot.spec.city.lower() in wanted
    ]


def _market_ids_from_snapshots(snapshots: list[MarketSnapshot]) -> set[str]:
    """Return parsed market ids for a target snapshot set."""

    return {snapshot.spec.market_id for snapshot in snapshots if snapshot.spec is not None}


def _load_snapshots(
    *,
    markets_path: Path | None,
    cities: list[str] | None = None,
    active: bool | None = None,
    closed: bool | None = None,
) -> list[MarketSnapshot]:
    config, _, http, _, _, _ = _runtime(include_stores=False)
    try:
        return _load_snapshots_with_runtime(
            config=config,
            http=http,
            markets_path=markets_path,
            cities=cities,
            active=active,
            closed=closed,
        )
    finally:
        http.close()


def _load_snapshots_with_runtime(
    *,
    config: Any,
    http: CachedHttpClient,
    markets_path: Path | None,
    cities: list[str] | None = None,
    active: bool | None = None,
    closed: bool | None = None,
) -> list[MarketSnapshot]:
    """Load snapshots using an already-open runtime instead of creating a new one."""

    if markets_path is not None:
        if not markets_path.exists():
            msg = f"Market snapshot file does not exist: {markets_path}"
            raise FileNotFoundError(msg)
        return _filter_snapshots_by_city(load_market_snapshots(markets_path), cities)

    if active is not None or closed is not None:
        gamma = GammaClient(http, config.polymarket.gamma_base_url)
        refs = discover_temperature_event_refs_from_gamma(
            gamma,
            supported_cities=cities or config.app.supported_cities,
            active=active,
            closed=closed,
            max_pages=config.polymarket.max_pages,
        )
        fetches = fetch_temperature_event_pages(http, refs, use_cache=False)
        snapshots = snapshots_from_temperature_event_fetches(fetches)
        return _filter_snapshots_by_city(snapshots, cities)

    return bundled_market_snapshots(cities)


def _load_scoped_snapshots_with_runtime(
    *,
    config: Any,
    http: CachedHttpClient,
    markets_path: Path | None,
    cities: list[str] | None = None,
    market_scope: str = "default",
    active: bool | None = None,
    closed: bool | None = None,
) -> list[MarketSnapshot]:
    """Load snapshots with a shared runtime and apply one market-scope preset."""

    scoped_cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    snapshots = _load_snapshots_with_runtime(
        config=config,
        http=http,
        markets_path=markets_path,
        cities=scoped_cities,
        active=active,
        closed=closed,
    )
    if market_scope == "default":
        return snapshots
    return [snapshot for snapshot in snapshots if _snapshot_matches_market_scope(snapshot, market_scope=market_scope)]


def _load_scoped_snapshots(
    *,
    markets_path: Path | None,
    cities: list[str] | None = None,
    market_scope: str = "default",
    active: bool | None = None,
    closed: bool | None = None,
) -> list[MarketSnapshot]:
    """Load snapshots and apply one market-scope preset after parsing."""

    config, _, http, _, _, _ = _runtime(include_stores=False)
    try:
        return _load_scoped_snapshots_with_runtime(
            config=config,
            http=http,
            markets_path=markets_path,
            cities=cities,
            market_scope=market_scope,
            active=active,
            closed=closed,
        )
    finally:
        http.close()


def _bootstrap_snapshots(*, markets_path: Path | None, cities: list[str] | None) -> list[MarketSnapshot]:
    if markets_path is not None:
        if not markets_path.exists():
            msg = f"Market snapshot file does not exist: {markets_path}"
            raise FileNotFoundError(msg)
        return _filter_snapshots_by_city(load_market_snapshots(markets_path), cities)
    return bundled_market_snapshots(cities)


def _collection_preflight_report(snapshots: list[MarketSnapshot], env: EnvSettings) -> dict[str, object]:
    """Summarize manual env requirements for a historical collection run."""

    source_counts: dict[str, int] = {}
    city_counts: dict[str, int] = {}
    truth_track_counts: dict[str, int] = {}
    research_priority_counts: dict[str, int] = {}
    parsed_snapshots = 0
    settlement_eligible_count = 0
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        parsed_snapshots += 1
        source_key = spec.adapter_key()
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        city_counts[spec.city] = city_counts.get(spec.city, 0) + 1
        truth_track_counts[spec.truth_track] = truth_track_counts.get(spec.truth_track, 0) + 1
        research_priority_counts[spec.research_priority] = research_priority_counts.get(spec.research_priority, 0) + 1
        if spec.settlement_eligible:
            settlement_eligible_count += 1

    required_env: list[str] = []
    optional_env: list[str] = []
    missing_env: list[str] = []
    messages: list[str] = []

    if source_counts.get("wunderground", 0) > 0:
        optional_env.append("PMTMAX_WU_API_KEY")
        if env.wu_api_key:
            messages.append("PMTMAX_WU_API_KEY is configured for optional Wunderground audit collection.")
        else:
            messages.append(
                "Wunderground-family markets default to public airport archive collection. "
                "Seoul/RKSI uses AMO AIR_CALP; London/EGLC and NYC/KLGA use the Wunderground public historical API. "
                "PMTMAX_WU_API_KEY is optional and only needed for same-source audit collection."
            )

    return {
        "ready": not missing_env,
        "markets_total": len(snapshots),
        "markets_with_spec": parsed_snapshots,
        "markets_without_spec": len(snapshots) - parsed_snapshots,
        "cities": sorted(city_counts),
        "city_counts": city_counts,
        "source_counts": source_counts,
        "truth_track_counts": truth_track_counts,
        "research_priority_counts": research_priority_counts,
        "settlement_eligible_count": settlement_eligible_count,
        "required_env": required_env,
        "optional_env": optional_env,
        "missing_env": missing_env,
        "messages": messages,
    }


def _best_signal_across_outcomes(
    snapshot: MarketSnapshot,
    forecast_probs: dict[str, float],
    books: dict[str, BookSnapshot],
    *,
    mode: Literal["paper", "live"],
    clob: ClobReadClient | None = None,
    default_fee_bps: float = 30.0,
    order_size: float = 1.0,
) -> TradeSignal | None:
    """Scan all outcomes and return the signal with the highest edge."""
    _, best_after_cost, _, _ = _rank_execution_candidates(
        forecast_probs,
        books,
        clob=clob,
        default_fee_bps=default_fee_bps,
        order_size=order_size,
    )
    if best_after_cost is None:
        return None
    after_cost_edge = best_after_cost.get("after_cost_edge")
    if after_cost_edge is None or float(after_cost_edge) <= 0:
        return None
    return _trade_signal_from_candidate(snapshot, best_after_cost, mode=mode)


def _load_books_for_forecast(
    clob: ClobReadClient,
    snapshot: MarketSnapshot,
    forecast_probs: dict[str, float],
    *,
    allow_synthetic_fallback: bool = False,
) -> dict[str, BookSnapshot]:
    """Fetch one book per modeled outcome for an active market."""

    spec = snapshot.spec
    if spec is None:
        return {}
    books: dict[str, BookSnapshot] = {}
    for outcome_label in forecast_probs:
        idx = spec.outcome_labels().index(outcome_label) if outcome_label in spec.outcome_labels() else -1
        if idx < 0 or idx >= len(spec.token_ids):
            continue
        token_id = spec.token_ids[idx]
        books[outcome_label] = _fetch_book(
            clob,
            snapshot,
            token_id,
            outcome_label,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
    return books


def _default_fee_bps(config: Any) -> float:
    """Return the configured fallback fee rate in basis points."""

    execution = getattr(config, "execution", None)
    if execution is None:
        return 30.0
    if hasattr(execution, "default_fee_bps"):
        return float(execution.default_fee_bps)
    if hasattr(execution, "fee_bps"):
        return float(execution.fee_bps)
    return 30.0


def _resolve_fee_bps(
    clob: ClobReadClient | None,
    token_id: str,
    *,
    default_fee_bps: float,
) -> float:
    """Resolve token fee rate, falling back to config when unavailable."""

    if clob is None:
        return default_fee_bps
    try:
        return float(clob.get_fee_rate(token_id))
    except Exception:  # noqa: BLE001
        return default_fee_bps


def _visible_liquidity(book: BookSnapshot) -> float:
    """Return the total visible two-sided liquidity from a book snapshot."""

    return sum(level.size for level in book.bids) + sum(level.size for level in book.asks)


def _rank_execution_candidates(
    forecast_probs: dict[str, float],
    books: dict[str, BookSnapshot],
    *,
    clob: ClobReadClient | None,
    default_fee_bps: float,
    order_size: float = 1.0,
) -> tuple[list[dict[str, object]], dict[str, object] | None, dict[str, object] | None, dict[str, object] | None]:
    """Build comparable execution candidates across all modeled outcomes."""

    candidates: list[dict[str, object]] = []
    for outcome_label, fair_prob in forecast_probs.items():
        book = books.get(outcome_label)
        if book is None or book.source not in {"clob", "fixture", "gamma"} or not book.asks:
            continue
        best_bid = book.best_bid()
        best_ask = book.best_ask()
        spread = max(best_ask - best_bid, 0.0)
        visible_liquidity = _visible_liquidity(book)
        fee_bps = _resolve_fee_bps(clob, book.token_id, default_fee_bps=default_fee_bps)
        fee_per_share = estimate_fee(best_ask, taker_bps=fee_bps)
        edge_after_fee = fair_prob - best_ask - fee_per_share
        slippage = estimate_book_slippage("buy", book.asks, order_size)
        candidates.append(
            {
                "token_id": book.token_id,
                "outcome_label": outcome_label,
                "fair_probability": fair_prob,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "visible_liquidity": visible_liquidity,
                "fee_bps": fee_bps,
                "fee_estimate": fee_per_share,
                "slippage_estimate": slippage,
                "raw_gap": fair_prob - best_ask,
                "edge_after_fee": edge_after_fee,
                "after_cost_edge": compute_edge(fair_prob, best_ask, fee_per_share, slippage)
                if slippage is not None
                else None,
                "insufficient_depth": slippage is None,
                "book_source": book.source,
            }
        )

    best_after_cost = max(
        [candidate for candidate in candidates if candidate["after_cost_edge"] is not None],
        key=lambda candidate: float(candidate["after_cost_edge"]),
        default=None,
    )
    best_after_fee = max(candidates, key=lambda candidate: float(candidate["edge_after_fee"]), default=None)
    best_raw = max(candidates, key=lambda candidate: float(candidate["raw_gap"]), default=None)
    best_reporting = best_after_cost or best_after_fee or best_raw
    return candidates, best_after_cost, best_after_fee, best_reporting


def _trade_signal_from_candidate(
    snapshot: MarketSnapshot,
    candidate: dict[str, object],
    *,
    mode: Literal["paper", "live"],
    forecast_contract_version: str = "v1",
    probability_source: Literal["raw", "calibrated"] = "raw",
    distribution_family: str = "gaussian",
    decision_horizon: str | None = None,
) -> TradeSignal:
    """Convert a ranked execution candidate into a trade signal."""

    market_id = snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id"))
    return TradeSignal(
        market_id=market_id,
        token_id=str(candidate["token_id"]),
        outcome_label=str(candidate["outcome_label"]),
        side="buy",
        fair_probability=float(candidate["fair_probability"]),
        executable_price=float(candidate["best_ask"]),
        fee_estimate=float(candidate["fee_estimate"]),
        slippage_estimate=float(candidate["slippage_estimate"] or 0.0),
        edge=float(candidate["after_cost_edge"]),
        confidence=float(candidate["fair_probability"]),
        rationale=f"Best edge outcome {candidate['outcome_label']} p={float(candidate['fair_probability']):.3f}",
        mode=mode,
        forecast_contract_version=forecast_contract_version,
        probability_source=probability_source,
        distribution_family=distribution_family,
        decision_horizon=decision_horizon,
    )


def _book_source_counts(books: dict[str, BookSnapshot]) -> dict[str, int]:
    """Count book origins for one market evaluation."""

    counts: dict[str, int] = {}
    for book in books.values():
        counts[book.source] = counts.get(book.source, 0) + 1
    return counts


def _all_books_missing(books: dict[str, BookSnapshot]) -> bool:
    """Return whether every fetched outcome book is missing."""

    return bool(books) and all(book.source == "missing" for book in books.values())


def _evaluate_market_signal(
    snapshot: MarketSnapshot,
    forecast_probs: dict[str, float],
    books: dict[str, BookSnapshot],
    *,
    mode: Literal["paper", "live"],
    clob: ClobReadClient | None,
    default_fee_bps: float,
    edge_threshold: float,
    max_spread_bps: int,
    min_liquidity: float,
    forecast_contract_version: str = "v1",
    probability_source: Literal["raw", "calibrated"] = "raw",
    distribution_family: str = "gaussian",
    decision_horizon: str | None = None,
) -> dict[str, object]:
    """Evaluate one market for trading and return a structured decision payload."""

    result: dict[str, object] = {
        "reason": "missing_token_mapping",
        "signal": None,
        "book": None,
        "spread": None,
        "liquidity": None,
        "book_source_counts": _book_source_counts(books),
        "candidate": None,
    }
    if not books:
        return result
    if _all_books_missing(books):
        result["reason"] = "missing_book"
        return result

    candidates, best_after_cost, _, best_reporting = _rank_execution_candidates(
        forecast_probs,
        books,
        clob=clob,
        default_fee_bps=default_fee_bps,
    )
    result["candidate"] = best_reporting
    if not candidates or best_reporting is None:
        result["reason"] = "missing_book"
        return result

    positive_after_cost = [
        candidate
        for candidate in candidates
        if candidate["after_cost_edge"] is not None and float(candidate["after_cost_edge"]) > 0
    ]
    if positive_after_cost and best_after_cost is not None:
        signal = _trade_signal_from_candidate(
            snapshot,
            best_after_cost,
            mode=mode,
            forecast_contract_version=forecast_contract_version,
            probability_source=probability_source,
            distribution_family=distribution_family,
            decision_horizon=decision_horizon,
        )
        book = books[signal.outcome_label]
        spread = float(best_after_cost["spread"])
        liquidity = float(best_after_cost["visible_liquidity"])
        result.update(
            {
                "signal": signal,
                "book": book,
                "spread": spread,
                "liquidity": liquidity,
            }
        )
        if not spread_ok(book.best_bid(), book.best_ask(), max_spread_bps):
            result["reason"] = "after_cost_positive_but_spread_too_wide"
        elif liquidity < min_liquidity:
            result["reason"] = "after_cost_positive_but_liquidity_too_low"
        elif signal.edge < edge_threshold:
            result["reason"] = "after_cost_positive_but_below_threshold"
        else:
            result["reason"] = "tradable"
        return result

    if any(
        float(candidate["edge_after_fee"]) > 0 and bool(candidate["insufficient_depth"])
        for candidate in candidates
    ):
        result["reason"] = "insufficient_depth"
    elif any(float(candidate["edge_after_fee"]) > 0 for candidate in candidates):
        result["reason"] = "slippage_killed_edge"
    elif any(float(candidate["raw_gap"]) > 0 for candidate in candidates):
        result["reason"] = "fee_killed_edge"
    else:
        result["reason"] = "raw_gap_non_positive"
    return result


def _market_url_for_spec(spec: MarketSpec) -> str:
    """Return the public Polymarket event URL when the slug is available."""

    return f"https://polymarket.com/event/{spec.slug}" if spec.slug else ""


def _resolve_opportunity_shadow_horizon(
    spec: MarketSpec,
    *,
    now_utc: datetime,
    near_term_days: int,
) -> str | None:
    """Resolve the dynamic near-term horizon for opportunity shadowing."""

    return select_shadow_horizon(spec, now_utc=now_utc, near_term_days=near_term_days)


def _empty_opportunity_observation(
    snapshot: MarketSnapshot,
    *,
    observed_at: datetime,
    decision_horizon: str,
    reason: str,
    book_source_counts: dict[str, int] | None = None,
    forecast_generated_at: datetime | None = None,
    forecast_contract_version: str = "v1",
    probability_source: Literal["raw", "calibrated"] = "raw",
    distribution_family: str = "gaussian",
    forecast_mean: float | None = None,
    forecast_std: float | None = None,
    error: str | None = None,
) -> OpportunityObservation:
    """Build a minimal opportunity observation for skipped/error cases."""

    spec = snapshot.spec
    if spec is None:
        msg = "Snapshot is missing spec"
        raise ValueError(msg)
    return OpportunityObservation(
        observed_at=observed_at,
        market_id=spec.market_id,
        city=spec.city,
        question=spec.question,
        target_local_date=spec.target_local_date,
        decision_horizon=decision_horizon,
        reason=reason,
        market_url=_market_url_for_spec(spec),
        book_source_counts=book_source_counts or {},
        forecast_generated_at=forecast_generated_at,
        forecast_contract_version=forecast_contract_version,
        probability_source=probability_source,
        distribution_family=distribution_family,
        forecast_mean=forecast_mean,
        forecast_std=forecast_std,
        error=error,
    )


def _best_opportunity_candidate(
    forecast_probs: dict[str, float],
    books: dict[str, BookSnapshot],
    *,
    clob: ClobReadClient | None,
    default_fee_bps: float,
) -> dict[str, object] | None:
    """Return the most favorable outcome candidate, even if it is not tradable."""

    _, _, _, best_reporting = _rank_execution_candidates(
        forecast_probs,
        books,
        clob=clob,
        default_fee_bps=default_fee_bps,
    )
    return best_reporting


def _forecast_contract_rejection_reason(
    forecast: ProbForecast,
) -> str | None:
    """Return a fail-closed reason when a forecast does not satisfy the v2 contract."""

    if getattr(forecast, "contract_version", "v1") != "v2":
        return "forecast_contract_mismatch"
    if getattr(forecast, "probability_source", "raw") != "calibrated":
        return "missing_calibrator"
    return None


def _as_utc_datetime(value: datetime) -> datetime:
    """Normalize one datetime into UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_icao_station_id(station_id: str | None) -> bool:
    """Return whether one station id looks like an ICAO code."""

    if not station_id:
        return False
    normalized = station_id.strip().upper()
    return len(normalized) == 4 and normalized.isalpha()


def _observation_station_candidates(spec: MarketSpec) -> list[tuple[str, str]]:
    """Return ordered live observation station candidates for one market."""

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _append(station_id: str | None, confidence: str) -> None:
        if not _is_icao_station_id(station_id):
            return
        normalized = str(station_id).strip().upper()
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append((normalized, confidence))

    _append(spec.station_id, "primary_station")
    _append(spec.public_truth_station_id, "public_truth_station")
    for definition in lookup_city_stations(spec.city):
        _append(definition.station_id, "same_city_station")
        _append(definition.public_truth_station_id, "same_city_public_truth_station")
    return candidates


def _is_target_day_observation(spec: MarketSpec, *, observed_at: datetime) -> bool:
    """Return whether one observation cycle lines up with the market's target local day."""

    local_now = _as_utc_datetime(observed_at).astimezone(ZoneInfo(spec.timezone))
    return local_now.date() == spec.target_local_date


def _market_unit_from_celsius(spec: MarketSpec, value_c: float) -> float:
    """Convert one Celsius observation into the market's settlement unit."""

    if spec.unit == "F":
        return celsius_to_fahrenheit(value_c)
    return value_c


def _observation_adjusted_probabilities(
    spec: MarketSpec,
    forecast_probs: dict[str, float],
    *,
    observed_market_unit: float,
) -> tuple[dict[str, float], list[str], float]:
    """Zero out bins already made impossible by the latest observed temperature."""

    schema_by_label = {item.label: item for item in spec.outcome_schema}
    feasible: dict[str, float] = {}
    impossible: list[str] = []
    for label, probability in forecast_probs.items():
        outcome = schema_by_label.get(label)
        if outcome is None:
            continue
        if outcome.upper is not None and observed_market_unit > outcome.upper:
            impossible.append(label)
            continue
        feasible[label] = max(float(probability), 0.0)

    if not feasible:
        return forecast_probs, impossible, 0.0

    feasible_mass = float(sum(feasible.values()))
    if feasible_mass <= 0:
        uniform = 1.0 / float(len(feasible))
        adjusted = dict.fromkeys(feasible, uniform)
    else:
        adjusted = {label: probability / feasible_mass for label, probability in feasible.items()}

    for label in impossible:
        adjusted[label] = 0.0

    removed_mass = max(0.0, 1.0 - feasible_mass)
    return adjusted, impossible, removed_mass


def _impossible_price_mass(impossible_labels: list[str], books: dict[str, BookSnapshot]) -> float | None:
    """Return the total best-ask mass still assigned to impossible outcomes."""

    prices = [
        books[label].best_ask()
        for label in impossible_labels
        if label in books and books[label].source == "clob" and books[label].asks
    ]
    if not prices:
        return None
    return float(sum(prices))


def _candidate_tier_for_spec(spec: MarketSpec) -> str:
    """Return the live-candidate tier for one market spec."""

    if spec.truth_track == "research_public":
        return "research_public_live"
    return "exact_public_live"


def _observation_risk_flags(
    *,
    spec: MarketSpec,
    reason: str,
    observation_freshness_minutes: float | None,
    stale_minutes: int,
    impossible_price_mass: float | None,
) -> list[str]:
    """Return explicit review flags for one observation-driven candidate."""

    flags: list[str] = []
    if spec.truth_track != "exact_public":
        flags.append("research_public")
    if not spec.settlement_eligible:
        flags.append("settlement_ineligible")
    if observation_freshness_minutes is None:
        flags.append("missing_observation")
    elif observation_freshness_minutes > stale_minutes:
        flags.append("stale_observation")
    if reason == "after_cost_positive_but_spread_too_wide":
        flags.append("wide_spread")
    elif reason == "after_cost_positive_but_liquidity_too_low":
        flags.append("low_liquidity")
    elif reason == "after_cost_positive_but_below_threshold":
        flags.append("below_edge_threshold")
    elif reason == "missing_book":
        flags.append("missing_book")
    if impossible_price_mass is None or impossible_price_mass <= 0:
        flags.append("no_observation_dislocation")
    return flags


def _observation_queue_state(
    *,
    reason: str,
    risk_flags: list[str],
) -> Literal["tradable", "manual_review", "blocked"]:
    """Return the approval queue state for one observation-driven opportunity."""

    blocking_flags = {"missing_observation", "missing_book"}
    if any(flag in blocking_flags for flag in risk_flags):
        return "blocked"
    if reason == "tradable" and not {"research_public", "stale_observation"} & set(risk_flags):
        return "tradable"
    if reason in {
        "tradable",
        "after_cost_positive_but_spread_too_wide",
        "after_cost_positive_but_liquidity_too_low",
        "after_cost_positive_but_below_threshold",
    }:
        return "manual_review"
    return "blocked"


def _manual_approval_token(
    *,
    market_id: str,
    outcome_label: str | None,
    observation_observed_at: datetime | None,
    source_station_id: str | None,
) -> str | None:
    """Return a stable approval token for one observation candidate."""

    if outcome_label is None or observation_observed_at is None:
        return None
    payload = "|".join(
        [
            market_id,
            outcome_label,
            _as_utc_datetime(observation_observed_at).isoformat(),
            source_station_id or "",
        ]
    )
    return stable_hash(payload)[:16]


def _empty_observation_opportunity(
    snapshot: MarketSnapshot,
    *,
    observed_at: datetime,
    decision_horizon: str,
    reason: str,
    source_family: str | None = None,
    observation_source: str | None = None,
    station_id: str | None = None,
    source_station_id: str | None = None,
    candidate_tier: str | None = None,
    source_confidence: str | None = None,
    book_source_counts: dict[str, int] | None = None,
    forecast_generated_at: datetime | None = None,
    forecast_contract_version: str = "v1",
    probability_source: Literal["raw", "calibrated"] = "raw",
    distribution_family: str = "gaussian",
    forecast_mean: float | None = None,
    forecast_std: float | None = None,
    observed_temp_c: float | None = None,
    observed_temp_market_unit: float | None = None,
    observation_observed_at: datetime | None = None,
    observation_freshness_minutes: float | None = None,
    observation_override_mass: float | None = None,
    impossible_outcome_count: int = 0,
    impossible_price_mass: float | None = None,
    price_vs_observation_gap: float | None = None,
    queue_state: Literal["tradable", "manual_review", "blocked"] = "blocked",
    approval_required: bool = False,
    live_eligible: bool = False,
    manual_approval_token: str | None = None,
    approval_expires_at: datetime | None = None,
    risk_flags: list[str] | None = None,
    error: str | None = None,
) -> ObservationOpportunity:
    """Build a minimal observation-driven opportunity for skipped/error cases."""

    spec = snapshot.spec
    if spec is None:
        msg = "Snapshot is missing spec"
        raise ValueError(msg)
    return ObservationOpportunity(
        observed_at=observed_at,
        market_id=spec.market_id,
        city=spec.city,
        question=spec.question,
        target_local_date=spec.target_local_date,
        decision_horizon=decision_horizon,
        reason=reason,
        queue_state=queue_state,
        market_url=_market_url_for_spec(spec),
        source_family=source_family,
        observation_source=observation_source,
        truth_track=spec.truth_track,
        station_id=station_id or spec.station_id,
        source_station_id=source_station_id,
        candidate_tier=candidate_tier or _candidate_tier_for_spec(spec),
        source_confidence=source_confidence,
        book_source_counts=book_source_counts or {},
        forecast_generated_at=forecast_generated_at,
        forecast_contract_version=forecast_contract_version,
        probability_source=probability_source,
        distribution_family=distribution_family,
        forecast_mean=forecast_mean,
        forecast_std=forecast_std,
        observed_temp_c=observed_temp_c,
        observed_temp_market_unit=observed_temp_market_unit,
        observation_observed_at=observation_observed_at,
        observation_freshness_minutes=observation_freshness_minutes,
        observation_override_mass=observation_override_mass,
        impossible_outcome_count=impossible_outcome_count,
        impossible_price_mass=impossible_price_mass,
        price_vs_observation_gap=price_vs_observation_gap,
        approval_required=approval_required,
        live_eligible=live_eligible,
        manual_approval_token=manual_approval_token,
        approval_expires_at=approval_expires_at,
        risk_flags=risk_flags or [],
        error=error,
    )


def _metar_live_observations(
    spec: MarketSpec,
    *,
    metar: MetarClient,
) -> list[LiveTemperatureObservation]:
    """Return METAR-backed lower-bound candidates for one market."""

    observations: list[LiveTemperatureObservation] = []
    for station_id, confidence in _observation_station_candidates(spec):
        observation = metar.fetch_latest(station_id)
        if observation is None:
            continue
        observations.append(
            LiveTemperatureObservation(
                source_family="aviation",
                observation_source="aviationweather_metar",
                station_id=station_id,
                observed_at=_as_utc_datetime(observation.observed_at),
                lower_bound_temp_c=float(observation.temp_c),
                current_temp_c=float(observation.temp_c),
                daily_high_so_far_c=None,
                source_confidence=confidence,
            )
        )
    return observations


def _live_observation_priority(source_family: str | None) -> int:
    """Return deterministic source-family precedence for equally strong candidates."""

    priorities = {
        "official_intraday": 3,
        "research_intraday": 2,
        "aviation": 1,
    }
    return priorities.get(source_family or "", 0)


def _select_best_live_observation(
    spec: MarketSpec,
    candidates: list[LiveTemperatureObservation],
) -> LiveTemperatureObservation | None:
    """Pick the strongest lower-bound candidate for one market."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            _market_unit_from_celsius(spec, candidate.lower_bound_temp_c),
            _as_utc_datetime(candidate.observed_at),
            _live_observation_priority(candidate.source_family),
        ),
    )


def _fetch_live_observation(
    spec: MarketSpec,
    *,
    http: CachedHttpClient,
    metar: MetarClient,
    observed_at: datetime,
    allow_metar: bool,
) -> LiveTemperatureObservation | None:
    """Return the best available live lower-bound candidate for one market."""

    candidates = fetch_intraday_observations(
        spec,
        http=http,
        observed_at=observed_at,
    )
    if allow_metar:
        candidates.extend(_metar_live_observations(spec, metar=metar))
    return _select_best_live_observation(spec, candidates)


def _evaluate_observation_snapshot(
    snapshot: MarketSnapshot,
    *,
    builder: DatasetBuilder,
    clob: ClobReadClient,
    http: CachedHttpClient,
    metar: MetarClient,
    model_path: Path,
    model_name: str,
    config: Any,
    observed_at: datetime,
    decision_horizon: str,
    edge_threshold: float,
) -> ObservationOpportunity | None:
    """Evaluate one active market using the latest live observation as a lower bound."""

    spec = snapshot.spec
    if spec is None:
        return None
    if not _is_target_day_observation(spec, observed_at=observed_at):
        return _empty_observation_opportunity(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="not_target_day",
            station_id=spec.station_id,
            candidate_tier=_candidate_tier_for_spec(spec),
            risk_flags=["off_target_day"],
        )
    candidate_tier = _candidate_tier_for_spec(spec)
    observation = _fetch_live_observation(
        spec,
        http=http,
        metar=metar,
        observed_at=observed_at,
        allow_metar=bool(getattr(config.metar, "enabled", True)),
    )
    if observation is None:
        return _empty_observation_opportunity(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="missing_observation",
            source_family=None,
            observation_source=None,
            station_id=spec.station_id,
            candidate_tier=candidate_tier,
            source_confidence=None,
            source_station_id=None,
            risk_flags=["missing_observation"],
        )

    observation_freshness_minutes = max(
        (_as_utc_datetime(observed_at) - _as_utc_datetime(observation.observed_at)).total_seconds() / 60.0,
        0.0,
    )
    observed_temp_c = float(observation.lower_bound_temp_c)
    observed_temp_market_unit = _market_unit_from_celsius(spec, observed_temp_c)
    try:
        feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
        forecast = predict_market(
            model_path,
            model_name,
            spec,
            feature_frame,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_observation_opportunity(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="forecast_failed",
            source_family=observation.source_family,
            observation_source=observation.observation_source,
            station_id=spec.station_id,
            source_station_id=observation.station_id,
            candidate_tier=candidate_tier,
            source_confidence=observation.source_confidence,
            observed_temp_c=observed_temp_c,
            observed_temp_market_unit=observed_temp_market_unit,
            observation_observed_at=observation.observed_at,
            observation_freshness_minutes=observation_freshness_minutes,
            error=str(exc),
        )

    if not forecast_fresh(forecast.generated_at, config.execution.stale_forecast_minutes):
        return _empty_observation_opportunity(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="stale_forecast",
            source_family=observation.source_family,
            observation_source=observation.observation_source,
            station_id=spec.station_id,
            source_station_id=observation.station_id,
            candidate_tier=candidate_tier,
            source_confidence=observation.source_confidence,
            forecast_generated_at=forecast.generated_at,
            forecast_contract_version=getattr(forecast, "contract_version", "v1"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            forecast_mean=forecast.mean,
            forecast_std=forecast.std,
            observed_temp_c=observed_temp_c,
            observed_temp_market_unit=observed_temp_market_unit,
            observation_observed_at=observation.observed_at,
            observation_freshness_minutes=observation_freshness_minutes,
        )

    forecast_rejection_reason = _forecast_contract_rejection_reason(forecast)
    if forecast_rejection_reason is not None:
        return _empty_observation_opportunity(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason=forecast_rejection_reason,
            source_family=observation.source_family,
            observation_source=observation.observation_source,
            station_id=spec.station_id,
            source_station_id=observation.station_id,
            candidate_tier=candidate_tier,
            source_confidence=observation.source_confidence,
            forecast_generated_at=forecast.generated_at,
            forecast_contract_version=getattr(forecast, "contract_version", "v1"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            forecast_mean=forecast.mean,
            forecast_std=forecast.std,
            observed_temp_c=observed_temp_c,
            observed_temp_market_unit=observed_temp_market_unit,
            observation_observed_at=observation.observed_at,
            observation_freshness_minutes=observation_freshness_minutes,
        )

    active_probs = forecast.active_outcome_probabilities()
    observation_probs, impossible_labels, removed_mass = _observation_adjusted_probabilities(
        spec,
        active_probs,
        observed_market_unit=observed_temp_market_unit,
    )
    books = _load_books_for_forecast(clob, snapshot, observation_probs)
    impossible_price_mass = _impossible_price_mass(impossible_labels, books)
    evaluation = _evaluate_market_signal(
        snapshot,
        observation_probs,
        books,
        mode="paper",
        clob=clob,
        default_fee_bps=_default_fee_bps(config),
        edge_threshold=edge_threshold,
        max_spread_bps=config.execution.max_spread_bps,
        min_liquidity=config.execution.min_liquidity,
        forecast_contract_version=getattr(forecast, "contract_version", "v1"),
        probability_source=getattr(forecast, "probability_source", "raw"),
        distribution_family=getattr(forecast, "distribution_family", "gaussian"),
        decision_horizon=decision_horizon,
    )
    reason = str(evaluation["reason"])
    risk_flags = _observation_risk_flags(
        spec=spec,
        reason=reason,
        observation_freshness_minutes=observation_freshness_minutes,
        stale_minutes=config.observation_station.observation_stale_minutes,
        impossible_price_mass=impossible_price_mass,
    )
    queue_state = _observation_queue_state(reason=reason, risk_flags=risk_flags)
    approval_required = queue_state in {"tradable", "manual_review"}
    approval_expires_at = None
    if approval_required:
        approval_expires_at = observed_at + timedelta(minutes=config.observation_station.approval_ttl_minutes)
    observation_row = _empty_observation_opportunity(
        snapshot,
        observed_at=observed_at,
        decision_horizon=decision_horizon,
        reason=reason,
        queue_state=queue_state,
        source_family=observation.source_family,
        observation_source=observation.observation_source,
        station_id=spec.station_id,
        source_station_id=observation.station_id,
        candidate_tier=candidate_tier,
        source_confidence=observation.source_confidence,
        book_source_counts=dict(evaluation["book_source_counts"]),
        forecast_generated_at=forecast.generated_at,
        forecast_contract_version=getattr(forecast, "contract_version", "v1"),
        probability_source=getattr(forecast, "probability_source", "raw"),
        distribution_family=getattr(forecast, "distribution_family", "gaussian"),
        forecast_mean=forecast.mean,
        forecast_std=forecast.std,
        observed_temp_c=observed_temp_c,
        observed_temp_market_unit=observed_temp_market_unit,
        observation_observed_at=observation.observed_at,
        observation_freshness_minutes=observation_freshness_minutes,
        observation_override_mass=removed_mass,
        impossible_outcome_count=len(impossible_labels),
        impossible_price_mass=impossible_price_mass,
        price_vs_observation_gap=impossible_price_mass,
        approval_required=approval_required,
        live_eligible=approval_required,
        approval_expires_at=approval_expires_at,
        manual_approval_token=_manual_approval_token(
            market_id=spec.market_id,
            outcome_label=str(evaluation["candidate"]["outcome_label"]) if evaluation.get("candidate") is not None else None,
            observation_observed_at=observation.observed_at,
            source_station_id=observation.station_id,
        ),
        risk_flags=risk_flags,
    )
    best_candidate = evaluation.get("candidate")
    if best_candidate is None:
        return observation_row
    return observation_row.model_copy(update=best_candidate)


def _serialize_observation_opportunity(observation: ObservationOpportunity) -> dict[str, object]:
    """Normalize one observation-driven opportunity for JSON outputs."""

    row = observation.model_dump(mode="json")
    row["target_local_date"] = observation.target_local_date.isoformat()
    if observation.after_cost_edge is not None:
        row["edge"] = round(observation.after_cost_edge, 6)
    if observation.visible_liquidity is not None:
        row["liquidity"] = round(observation.visible_liquidity, 6)
    if observation.best_ask is not None:
        row["executable_price"] = round(observation.best_ask, 6)
    if observation.observation_freshness_minutes is not None:
        row["observation_freshness_minutes"] = round(observation.observation_freshness_minutes, 3)
    if observation.observed_temp_market_unit is not None:
        row["observed_temp_market_unit"] = round(observation.observed_temp_market_unit, 3)
    if observation.price_vs_observation_gap is not None:
        row["price_vs_observation_gap"] = round(observation.price_vs_observation_gap, 6)
    return row


def _evaluate_opportunity_snapshot(
    snapshot: MarketSnapshot,
    *,
    builder: DatasetBuilder,
    clob: ClobReadClient,
    model_path: Path,
    model_name: str,
    config: Any,
    observed_at: datetime,
    decision_horizon: str,
    edge_threshold: float,
) -> OpportunityObservation | None:
    """Evaluate one market snapshot into a decomposed opportunity observation."""

    spec = snapshot.spec
    if spec is None:
        return None
    try:
        feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
        forecast = predict_market(
            model_path,
            model_name,
            spec,
            feature_frame,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_opportunity_observation(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="forecast_failed",
            error=str(exc),
        )

    if not forecast_fresh(forecast.generated_at, config.execution.stale_forecast_minutes):
        return _empty_opportunity_observation(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason="stale_forecast",
            forecast_generated_at=forecast.generated_at,
            forecast_contract_version=getattr(forecast, "contract_version", "v1"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            forecast_mean=forecast.mean,
            forecast_std=forecast.std,
        )

    forecast_rejection_reason = _forecast_contract_rejection_reason(forecast)
    if forecast_rejection_reason is not None:
        return _empty_opportunity_observation(
            snapshot,
            observed_at=observed_at,
            decision_horizon=decision_horizon,
            reason=forecast_rejection_reason,
            forecast_generated_at=forecast.generated_at,
            forecast_contract_version=getattr(forecast, "contract_version", "v1"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            forecast_mean=forecast.mean,
            forecast_std=forecast.std,
        )

    books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
    evaluation = _evaluate_market_signal(
        snapshot,
        forecast.outcome_probabilities,
        books,
        mode="paper",
        clob=clob,
        default_fee_bps=_default_fee_bps(config),
        edge_threshold=edge_threshold,
        max_spread_bps=config.execution.max_spread_bps,
        min_liquidity=config.execution.min_liquidity,
        forecast_contract_version=getattr(forecast, "contract_version", "v1"),
        probability_source=getattr(forecast, "probability_source", "raw"),
        distribution_family=getattr(forecast, "distribution_family", "gaussian"),
        decision_horizon=decision_horizon,
    )
    observation = _empty_opportunity_observation(
        snapshot,
        observed_at=observed_at,
        decision_horizon=decision_horizon,
        reason=str(evaluation["reason"]),
        book_source_counts=dict(evaluation["book_source_counts"]),
        forecast_generated_at=forecast.generated_at,
        forecast_contract_version=getattr(forecast, "contract_version", "v1"),
        probability_source=getattr(forecast, "probability_source", "raw"),
        distribution_family=getattr(forecast, "distribution_family", "gaussian"),
        forecast_mean=forecast.mean,
        forecast_std=forecast.std,
    )
    best_candidate = evaluation.get("candidate")
    if best_candidate is None:
        return observation
    return observation.model_copy(update=best_candidate)


def _serialize_opportunity_observation(observation: OpportunityObservation) -> dict[str, object]:
    """Normalize an opportunity observation for CLI JSON outputs."""

    row = observation.model_dump(mode="json")
    row["target_local_date"] = observation.target_local_date.isoformat()
    if observation.after_cost_edge is not None:
        row["edge"] = round(observation.after_cost_edge, 6)
    if observation.visible_liquidity is not None:
        row["liquidity"] = round(observation.visible_liquidity, 6)
    if observation.best_ask is not None:
        row["executable_price"] = round(observation.best_ask, 6)
    return row


def _safe_slug(value: str) -> str:
    """Return a filesystem-safe slug for one diagnostic label."""

    cleaned = [
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    ]
    slug = "".join(cleaned).strip("_")
    return slug or "item"


def _infer_model_name_from_artifact(path: Path) -> str:
    """Best-effort model-name inference for a raw artifact path."""

    stem = path.stem
    if stem.startswith("gaussian_emos"):
        return "gaussian_emos"
    return "lgbm_emos"


def _find_autoresearch_model_artifact(candidate_name: str) -> Path | None:
    """Resolve one autoresearch candidate artifact from the local artifact tree."""

    pattern = f"lgbm_emos__{candidate_name}.pkl"
    matches = sorted(
        DEFAULT_AUTORESEARCH_ROOT.glob(f"**/{pattern}"),
        key=lambda path: (len(path.parts), str(path)),
    )
    for path in matches:
        if "/models/gate/" not in str(path):
            return path
    return matches[0] if matches else None


def _default_paper_multimodel_specs() -> list[dict[str, object]]:
    """Return the default champion-vs-top-challenger comparison pool."""

    champion_path, champion_model_name = _resolve_model_path(
        Path("artifacts/models/v2/champion.pkl"),
        DEFAULT_MODEL_NAME,
    )
    candidate_path = _find_autoresearch_model_artifact("neighbor_oof_half_life20")
    if candidate_path is None:
        msg = "Missing default autoresearch artifact: neighbor_oof_half_life20"
        raise FileNotFoundError(msg)

    specs = [
        {
            "label": "champion_alias",
            "model_path": champion_path,
            "model_name": champion_model_name,
        },
        {
            "label": "neighbor_oof_half_life20",
            "model_path": candidate_path,
            "model_name": "lgbm_emos",
        },
        {
            "label": "high_neighbor_oof",
            "model_path": Path("artifacts/models/v2/lgbm_emos__high_neighbor_oof.pkl"),
            "model_name": "lgbm_emos",
        },
        {
            "label": "ultra_high_neighbor_oof",
            "model_path": Path("artifacts/models/v2/lgbm_emos__ultra_high_neighbor_oof.pkl"),
            "model_name": "lgbm_emos",
        },
        {
            "label": "mega_neighbor_oof",
            "model_path": Path("artifacts/models/v2/lgbm_emos__mega_neighbor_oof.pkl"),
            "model_name": "lgbm_emos",
        },
        {
            "label": "ultra_high_neighbor_fast",
            "model_path": Path("artifacts/models/v2/lgbm_emos__ultra_high_neighbor_fast.pkl"),
            "model_name": "lgbm_emos",
        },
    ]
    missing = [str(spec["model_path"]) for spec in specs if not Path(str(spec["model_path"])).exists()]
    if missing:
        msg = f"Missing default paper-multimodel artifacts: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    return specs


def _resolve_paper_multimodel_specs(
    *,
    model_labels: list[str] | None,
    model_paths: list[Path] | None,
    model_names: list[str] | None,
) -> list[dict[str, object]]:
    """Resolve either a custom comparison pool or the built-in default pool."""

    if not model_labels and not model_paths and not model_names:
        return _default_paper_multimodel_specs()
    labels = list(model_labels or [])
    paths = list(model_paths or [])
    names = list(model_names or [])
    if not labels or not paths or len(labels) != len(paths):
        msg = "--model-label and --model-path must be provided together with matching counts."
        raise typer.BadParameter(msg)
    if names and len(names) != len(labels):
        msg = "--model-name count must match --model-label when provided."
        raise typer.BadParameter(msg)
    specs: list[dict[str, object]] = []
    for index, (label, path) in enumerate(zip(labels, paths, strict=True)):
        specs.append(
            {
                "label": label,
                "model_path": path,
                "model_name": names[index] if names else _infer_model_name_from_artifact(path),
            }
        )
    return specs


def _normalize_report_row(row: dict[str, object]) -> dict[str, object]:
    """Normalize one paper/opportunity/observation row for generic diagnostics."""

    normalized = dict(row)
    if normalized.get("after_cost_edge") is None and normalized.get("edge") is not None:
        normalized["after_cost_edge"] = normalized.get("edge")
    if normalized.get("visible_liquidity") is None and normalized.get("liquidity") is not None:
        normalized["visible_liquidity"] = normalized.get("liquidity")
    return normalized


def _row_float(row: dict[str, object], key: str) -> float | None:
    """Best-effort float extraction for one report row."""

    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _row_fill_notional(row: dict[str, object]) -> float | None:
    """Return simulated fill notional when available."""

    fill = row.get("fill")
    if not isinstance(fill, dict):
        return None
    price = fill.get("price")
    size = fill.get("size")
    if not isinstance(price, (int, float)) or not isinstance(size, (int, float)):
        return None
    return float(price) * float(size)


def _top_report_rows(
    rows: list[dict[str, object]],
    *,
    limit: int,
    reasons: set[str] | None = None,
    sort_key: str = "after_cost_edge",
) -> list[dict[str, object]]:
    """Return the most diagnostic rows for one blocker bucket."""

    filtered = [
        _normalize_report_row(row)
        for row in rows
        if reasons is None or str(row.get("reason")) in reasons
    ]
    filtered = [
        row
        for row in filtered
        if row.get("market_id") is not None
    ]
    filtered.sort(
        key=lambda row: (
            -(_row_float(row, sort_key) or -999.0),
            -(_row_float(row, "raw_gap") or -999.0),
            str(row.get("city", "")),
            str(row.get("market_id", "")),
        )
    )
    top_rows: list[dict[str, object]] = []
    for row in filtered[:limit]:
        top_rows.append(
            {
                "market_id": row.get("market_id"),
                "city": row.get("city"),
                "target_local_date": row.get("target_local_date"),
                "decision_horizon": row.get("decision_horizon"),
                "reason": row.get("reason"),
                "outcome_label": row.get("outcome_label"),
                "raw_gap": _row_float(row, "raw_gap"),
                "after_cost_edge": _row_float(row, "after_cost_edge"),
                "spread": _row_float(row, "spread"),
                "visible_liquidity": _row_float(row, "visible_liquidity"),
                "price_vs_observation_gap": _row_float(row, "price_vs_observation_gap"),
            }
        )
    return top_rows


def _group_report_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
) -> dict[str, list[dict[str, object]]]:
    """Group normalized report rows by one string key."""

    groups: dict[str, list[dict[str, object]]] = {}
    for raw_row in rows:
        row = _normalize_report_row(raw_row)
        group = str(row.get(key) or "unknown")
        groups.setdefault(group, []).append(row)
    return groups


def _nested_reason_counts(
    rows: list[dict[str, object]],
    *,
    key: str,
) -> dict[str, dict[str, int]]:
    """Return per-group reason counts for one result set."""

    nested: dict[str, Counter[str]] = {}
    for raw_row in rows:
        row = _normalize_report_row(raw_row)
        group = str(row.get(key) or "unknown")
        reason = str(row.get("reason") or "unknown")
        nested.setdefault(group, Counter())[reason] += 1
    return {
        group: dict(sorted(counter.items()))
        for group, counter in sorted(nested.items())
    }


def _report_group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize one flat row set for diagnostic reporting."""

    normalized = [_normalize_report_row(row) for row in rows]
    reason_counts = Counter(str(row.get("reason") or "unknown") for row in normalized)
    raw_values = [value for row in normalized if (value := _row_float(row, "raw_gap")) is not None]
    edge_values = [value for row in normalized if (value := _row_float(row, "after_cost_edge")) is not None]
    spread_values = [value for row in normalized if (value := _row_float(row, "spread")) is not None]
    liquidity_values = [
        value
        for row in normalized
        if (value := _row_float(row, "visible_liquidity")) is not None
    ]
    fill_values = [value for row in normalized if (value := _row_fill_notional(row)) is not None]
    return {
        "evaluation_rows": len(normalized),
        "fills": len(fill_values),
        "tradable_rows": reason_counts.get("tradable", 0),
        "reason_counts": dict(sorted(reason_counts.items())),
        "raw_gap_positive_count": sum(1 for value in raw_values if value > 0),
        "after_cost_edge_positive_count": sum(1 for value in edge_values if value > 0),
        "best_raw_gap": max(raw_values) if raw_values else None,
        "best_after_cost_edge": max(edge_values) if edge_values else None,
        "median_spread": float(pd.Series(spread_values).median()) if spread_values else None,
        "median_visible_liquidity": float(pd.Series(liquidity_values).median()) if liquidity_values else None,
        "filled_notional": round(sum(fill_values), 6),
    }


def _watchlist_from_group_summaries(
    grouped: dict[str, dict[str, object]],
    *,
    reason_key: str,
    label: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Build one ranked city watchlist from grouped blocker summaries."""

    rows: list[dict[str, object]] = []
    for group, summary in grouped.items():
        reason_counts = summary.get("reason_counts", {})
        if not isinstance(reason_counts, dict):
            continue
        blocker_count = int(reason_counts.get(reason_key, 0) or 0)
        if blocker_count <= 0:
            continue
        total = int(summary.get("evaluation_rows", 0) or 0)
        rows.append(
            {
                label: group,
                reason_key: blocker_count,
                "evaluation_rows": total,
                "ratio": round(blocker_count / total, 6) if total > 0 else None,
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row.get(reason_key, 0) or 0),
            -(float(row.get("ratio")) if row.get("ratio") is not None else -1.0),
            str(row.get(label, "")),
        )
    )
    return rows[:limit]


def _summarize_result_rows(
    rows: list[dict[str, object]],
    *,
    bankroll_remaining: float | None = None,
) -> dict[str, object]:
    """Summarize one flat row-oriented diagnostic result set."""

    by_city_rows = _group_report_rows(rows, key="city")
    by_horizon_rows = _group_report_rows(rows, key="decision_horizon")
    by_city = {
        city: _report_group_summary(city_rows)
        for city, city_rows in sorted(by_city_rows.items())
    }
    by_horizon = {
        horizon: _report_group_summary(horizon_rows)
        for horizon, horizon_rows in sorted(by_horizon_rows.items())
    }
    summary = {
        **_report_group_summary(rows),
        "bankroll_remaining": round(bankroll_remaining, 6) if bankroll_remaining is not None else None,
        "by_city": by_city,
        "by_horizon": by_horizon,
        "by_reason": {
            reason: _report_group_summary(reason_rows)
            for reason, reason_rows in sorted(_group_report_rows(rows, key="reason").items())
        },
        "by_city_reason": _nested_reason_counts(rows, key="city"),
        "by_horizon_reason": _nested_reason_counts(rows, key="decision_horizon"),
        "top_near_miss_markets": _top_report_rows(
            rows,
            limit=10,
            reasons={
                "fee_killed_edge",
                "slippage_killed_edge",
                "after_cost_positive_but_spread_too_wide",
                "after_cost_positive_but_liquidity_too_low",
                "after_cost_positive_but_below_threshold",
                "insufficient_depth",
            },
        ),
        "top_fee_killed_markets": _top_report_rows(
            rows,
            limit=10,
            reasons={"fee_killed_edge"},
            sort_key="raw_gap",
        ),
        "top_spread_blocked_markets": _top_report_rows(
            rows,
            limit=10,
            reasons={"after_cost_positive_but_spread_too_wide", "slippage_killed_edge", "insufficient_depth"},
            sort_key="spread",
        ),
        "top_policy_filtered_markets": _top_report_rows(
            rows,
            limit=10,
            reasons={"policy_filtered"},
            sort_key="raw_gap",
        ),
        "fee_sensitive_watchlist": _watchlist_from_group_summaries(
            by_city,
            reason_key="fee_killed_edge",
            label="city",
        ),
        "raw_edge_desert_watchlist": _watchlist_from_group_summaries(
            by_city,
            reason_key="raw_gap_non_positive",
            label="city",
        ),
        "policy_blocked_watchlist": _watchlist_from_group_summaries(
            by_city,
            reason_key="policy_filtered",
            label="city",
        ),
    }
    return summary


def _load_full_books_for_snapshot(
    clob: ClobReadClient,
    snapshot: MarketSnapshot,
) -> dict[str, BookSnapshot]:
    """Fetch all visible outcome books for one parsed market snapshot."""

    spec = snapshot.spec
    if spec is None:
        return {}
    return _load_books_for_forecast(
        clob,
        snapshot,
        dict.fromkeys(spec.outcome_labels(), 0.0),
    )


def _build_gamma_books_for_forecast(
    snapshot: MarketSnapshot,
    forecast_probs: dict[str, float],
) -> dict[str, BookSnapshot]:
    """Build synthetic books from Gamma mid-prices instead of fetching from CLOB."""

    spec = snapshot.spec
    if spec is None:
        return {}
    books: dict[str, BookSnapshot] = {}
    for outcome_label in forecast_probs:
        idx = spec.outcome_labels().index(outcome_label) if outcome_label in spec.outcome_labels() else -1
        if idx < 0 or idx >= len(spec.token_ids):
            continue
        token_id = spec.token_ids[idx]
        books[outcome_label] = _synthetic_book(snapshot, outcome_label, token_id)
    return books


def _run_paper_model_evaluation(
    *,
    snapshots: list[MarketSnapshot],
    builder: DatasetBuilder,
    clob: ClobReadClient,
    config: Any,
    model_path: Path,
    model_name: str,
    horizon: str,
    horizon_policy: dict[str, list[str]],
    bankroll: float,
    edge_threshold: float,
    max_spread_bps: int,
    min_liquidity: float,
    book_cache: dict[str, dict[str, BookSnapshot]] | None = None,
    price_source: str = "gamma",
    min_market_price: float = 0.0,
) -> tuple[list[dict[str, object]], float]:
    """Evaluate one model over active markets with paper-broker accounting."""

    broker = PaperBroker(bankroll=bankroll)
    current_exposure_by_city: dict[str, float] = {}
    results: list[dict[str, object]] = []

    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        now_utc = datetime.now(tz=UTC)
        if spec.target_local_date < now_utc.date():
            continue
        decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
            spec,
            now_utc=now_utc,
            horizon=horizon,
            horizon_policy=horizon_policy,
        )
        base_row: dict[str, object] = {
            "market_id": spec.market_id,
            "city": spec.city,
            "question": spec.question,
            "target_local_date": spec.target_local_date.isoformat(),
            "market_url": _market_url_for_spec(spec),
            "decision_horizon": decision_horizon,
        }
        if horizon_reason == "policy_filtered":
            results.append({**base_row, "reason": "policy_filtered"})
            continue
        if decision_horizon is None:
            continue
        try:
            feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
            forecast = predict_market(model_path, model_name, spec, feature_frame)
        except Exception as exc:  # noqa: BLE001
            results.append({**base_row, "reason": "forecast_failed", "error": str(exc)})
            continue
        if not forecast_fresh(forecast.generated_at, config.execution.stale_forecast_minutes):
            results.append(
                {
                    **base_row,
                    "forecast_contract_version": getattr(forecast, "contract_version", "v2"),
                    "probability_source": getattr(forecast, "probability_source", "raw"),
                    "distribution_family": getattr(forecast, "distribution_family", "gaussian"),
                    "reason": "stale_forecast",
                }
            )
            continue
        forecast_rejection_reason = _forecast_contract_rejection_reason(forecast)
        if forecast_rejection_reason is not None:
            results.append(
                {
                    **base_row,
                    "forecast_contract_version": getattr(forecast, "contract_version", "v2"),
                    "probability_source": getattr(forecast, "probability_source", "raw"),
                    "distribution_family": getattr(forecast, "distribution_family", "gaussian"),
                    "reason": forecast_rejection_reason,
                }
            )
            continue

        if book_cache is not None:
            books = dict(book_cache.get(spec.market_id, {}))
        elif price_source == "gamma":
            books = _build_gamma_books_for_forecast(snapshot, forecast.outcome_probabilities)
        else:
            books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
        # Gamma-price mode: bypass spread/liquidity guards — they are based on a synthetic
        # book, not a live CLOB, so they carry no real execution risk signal.
        effective_max_spread_bps = 1_000_000 if price_source == "gamma" else max_spread_bps
        effective_min_liquidity = 0.0 if price_source == "gamma" else min_liquidity
        evaluation = _evaluate_market_signal(
            snapshot,
            forecast.outcome_probabilities,
            books,
            mode="paper",
            clob=clob,
            default_fee_bps=_default_fee_bps(config),
            edge_threshold=edge_threshold,
            max_spread_bps=effective_max_spread_bps,
            min_liquidity=effective_min_liquidity,
            forecast_contract_version=getattr(forecast, "contract_version", "v2"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            decision_horizon=decision_horizon,
        )
        signal = evaluation["signal"]
        book = evaluation["book"]
        reason = str(evaluation["reason"])
        candidate = evaluation.get("candidate")
        fill_payload: dict[str, object] | None = None
        size_notional: float | None = None
        if reason == "tradable" and isinstance(signal, TradeSignal) and isinstance(book, BookSnapshot) and min_market_price > 0 and signal.executable_price < min_market_price:
            reason = "market_price_below_threshold"
        if reason == "tradable" and isinstance(signal, TradeSignal) and isinstance(book, BookSnapshot):
            size_notional = capped_kelly(signal.edge, signal.fair_probability, broker.bankroll, signal.executable_price)
            size = size_notional / max(signal.executable_price, 1e-6)
            current_city_exposure = current_exposure_by_city.get(spec.city, 0.0)
            if not exposure_ok(current_city_exposure, size_notional, config.execution.max_city_exposure):
                reason = "city_exposure_limit"
            elif not exposure_ok(sum(current_exposure_by_city.values()), size_notional, config.execution.global_max_exposure):
                reason = "global_exposure_limit"
            else:
                fill = broker.simulate_fill(signal, book=book, size=size)
                if fill is None:
                    reason = "broker_rejected"
                else:
                    current_exposure_by_city[spec.city] = current_city_exposure + size_notional
                    fill_payload = fill.model_dump(mode="json")
        results.append(
            {
                **base_row,
                "forecast_contract_version": getattr(forecast, "contract_version", "v2"),
                "probability_source": getattr(forecast, "probability_source", "raw"),
                "distribution_family": getattr(forecast, "distribution_family", "gaussian"),
                "outcome_label": candidate.get("outcome_label") if isinstance(candidate, dict) else None,
                "token_id": candidate.get("token_id") if isinstance(candidate, dict) else None,
                "fair_probability": candidate.get("fair_probability") if isinstance(candidate, dict) else None,
                "best_bid": candidate.get("best_bid") if isinstance(candidate, dict) else None,
                "best_ask": candidate.get("best_ask") if isinstance(candidate, dict) else None,
                "executable_price": candidate.get("best_ask") if isinstance(candidate, dict) else None,
                "spread": candidate.get("spread") if isinstance(candidate, dict) else None,
                "visible_liquidity": candidate.get("visible_liquidity") if isinstance(candidate, dict) else None,
                "liquidity": candidate.get("visible_liquidity") if isinstance(candidate, dict) else None,
                "fee_estimate": candidate.get("fee_estimate") if isinstance(candidate, dict) else None,
                "slippage_estimate": candidate.get("slippage_estimate") if isinstance(candidate, dict) else None,
                "raw_gap": candidate.get("raw_gap") if isinstance(candidate, dict) else None,
                "after_cost_edge": candidate.get("after_cost_edge") if isinstance(candidate, dict) else None,
                "edge": candidate.get("after_cost_edge") if isinstance(candidate, dict) else None,
                "book_source": candidate.get("book_source") if isinstance(candidate, dict) else None,
                "book_source_counts": evaluation["book_source_counts"],
                "size_notional": size_notional,
                "reason": reason,
                "fill": fill_payload,
            }
        )
    # Normalize: if total Kelly notional exceeds bankroll, scale all positions down
    # proportionally so the portfolio fits within the bankroll (simultaneous Kelly).
    total_notional = sum(
        float(r["size_notional"])
        for r in results
        if r.get("reason") == "tradable" and r.get("size_notional") is not None
    )
    if total_notional > bankroll:
        scale = bankroll / total_notional
        for r in results:
            if r.get("reason") == "tradable" and r.get("size_notional") is not None:
                r["size_notional"] = float(r["size_notional"]) * scale
                if isinstance(r.get("fill"), dict) and r["fill"].get("size_notional") is not None:
                    r["fill"]["size_notional"] = float(r["fill"]["size_notional"]) * scale
                    r["fill"]["size"] = float(r["fill"].get("size", 0)) * scale
    return results, broker.bankroll


def _open_phase_observation_from_opportunity(
    observation: OpportunityObservation,
    *,
    market_created_at: datetime | None,
    market_deploying_at: datetime | None,
    market_accepting_orders_at: datetime | None,
    market_opened_at: datetime | None,
    open_phase_age_hours: float,
) -> OpenPhaseObservation:
    """Attach listing/open-phase metadata to a standard opportunity observation."""

    payload = observation.model_dump(mode="python")
    payload.update(
        {
            "market_created_at": market_created_at,
            "market_deploying_at": market_deploying_at,
            "market_accepting_orders_at": market_accepting_orders_at,
            "market_opened_at": market_opened_at,
            "open_phase_age_hours": open_phase_age_hours,
        }
    )
    return OpenPhaseObservation.model_validate(payload)


def _serialize_open_phase_observation(observation: OpenPhaseObservation) -> dict[str, object]:
    """Normalize an open-phase observation for CLI JSON outputs."""

    row = observation.model_dump(mode="json")
    row["target_local_date"] = observation.target_local_date.isoformat()
    if observation.after_cost_edge is not None:
        row["edge"] = round(observation.after_cost_edge, 6)
    if observation.visible_liquidity is not None:
        row["liquidity"] = round(observation.visible_liquidity, 6)
    if observation.best_ask is not None:
        row["executable_price"] = round(observation.best_ask, 6)
    if observation.open_phase_age_hours is not None:
        row["open_phase_age_hours"] = round(observation.open_phase_age_hours, 4)
    return row


def _hope_hunt_priority_bucket(
    *,
    spec: MarketSpec,
    open_phase_age_hours: float | None,
    reason: str,
) -> str:
    """Bucket a hope-hunt observation for reporting and ranking."""

    if open_phase_age_hours is None:
        return "unknown_open_time"
    if open_phase_age_hours <= 24:
        if reason == "after_cost_positive_but_spread_too_wide":
            return "fresh_listing_spread_blocked"
        return "fresh_listing"
    return "mature_core" if spec.research_priority == "core" else "mature_expansion"


def _hope_hunt_priority_score(
    *,
    open_phase_age_hours: float | None,
    target_day_distance: int,
    market_volume: float | None,
    reason: str,
    raw_gap: float | None,
    after_cost_edge: float | None,
) -> float:
    """Score one hope-hunt candidate using open-phase-first heuristics."""

    score = 0.0
    if open_phase_age_hours is None:
        score -= 8.0
    elif open_phase_age_hours <= 6:
        score += 40.0
    elif open_phase_age_hours <= 24:
        score += 32.0
    elif open_phase_age_hours <= 48:
        score += 20.0
    else:
        score += 8.0

    if target_day_distance >= 2:
        score += 20.0
    elif target_day_distance == 1:
        score += 10.0
    elif target_day_distance == 0:
        score += 4.0

    if market_volume is None:
        score += 2.0
    elif market_volume <= 500:
        score += 12.0
    elif market_volume <= 2_000:
        score += 9.0
    elif market_volume <= 10_000:
        score += 4.0
    else:
        score -= 2.0

    if reason == "tradable":
        score += 18.0
    elif reason == "after_cost_positive_but_spread_too_wide":
        score += 14.0
    elif reason != "raw_gap_non_positive":
        score += 6.0

    if raw_gap is not None:
        score += max(min(raw_gap * 200.0, 10.0), -10.0)
    if after_cost_edge is not None:
        score += max(min(after_cost_edge * 400.0, 20.0), -20.0)
    return round(score, 6)


def _hope_hunt_candidate_alert(
    *,
    reason: str,
    open_phase_age_hours: float | None,
    after_cost_edge: float | None,
) -> bool:
    """Return whether one observation deserves an explicit hope alert."""

    if after_cost_edge is not None and after_cost_edge > 0:
        return True
    return (
        reason == "after_cost_positive_but_spread_too_wide"
        and open_phase_age_hours is not None
        and open_phase_age_hours <= 24.0
    )


def _hope_hunt_observation_from_opportunity(
    snapshot: MarketSnapshot,
    observation: OpportunityObservation,
    *,
    market_created_at: datetime | None,
    market_deploying_at: datetime | None,
    market_accepting_orders_at: datetime | None,
    market_opened_at: datetime | None,
    open_phase_age_hours: float | None,
    observed_at: datetime,
) -> HopeHuntObservation:
    """Attach open-phase metadata and ranking features to an opportunity observation."""

    spec = snapshot.spec
    if spec is None:
        msg = "Snapshot is missing spec"
        raise ValueError(msg)
    target_day_distance = _target_day_distance(spec, observed_at=observed_at)
    market_volume = _market_volume_from_snapshot(snapshot)
    priority_bucket = _hope_hunt_priority_bucket(
        spec=spec,
        open_phase_age_hours=open_phase_age_hours,
        reason=observation.reason,
    )
    priority_score = _hope_hunt_priority_score(
        open_phase_age_hours=open_phase_age_hours,
        target_day_distance=target_day_distance,
        market_volume=market_volume,
        reason=observation.reason,
        raw_gap=observation.raw_gap,
        after_cost_edge=observation.after_cost_edge,
    )
    payload = observation.model_dump(mode="python")
    payload.update(
        {
            "market_created_at": market_created_at,
            "market_deploying_at": market_deploying_at,
            "market_accepting_orders_at": market_accepting_orders_at,
            "market_opened_at": market_opened_at,
            "open_phase_age_hours": open_phase_age_hours,
            "open_phase_age_bucket": open_phase_age_bucket(open_phase_age_hours),
            "target_day_distance": target_day_distance,
            "market_volume": market_volume,
            "priority_bucket": priority_bucket,
            "priority_score": priority_score,
            "candidate_alert": _hope_hunt_candidate_alert(
                reason=observation.reason,
                open_phase_age_hours=open_phase_age_hours,
                after_cost_edge=observation.after_cost_edge,
            ),
        }
    )
    return HopeHuntObservation.model_validate(payload)


def _serialize_hope_hunt_observation(observation: HopeHuntObservation) -> dict[str, object]:
    """Normalize one hope-hunt observation for CLI JSON outputs."""

    row = observation.model_dump(mode="json")
    row["target_local_date"] = observation.target_local_date.isoformat()
    if observation.after_cost_edge is not None:
        row["edge"] = round(observation.after_cost_edge, 6)
    if observation.visible_liquidity is not None:
        row["liquidity"] = round(observation.visible_liquidity, 6)
    if observation.best_ask is not None:
        row["executable_price"] = round(observation.best_ask, 6)
    if observation.open_phase_age_hours is not None:
        row["open_phase_age_hours"] = round(observation.open_phase_age_hours, 4)
    if observation.market_volume is not None:
        row["market_volume"] = round(observation.market_volume, 4)
    if observation.priority_score is not None:
        row["priority_score"] = round(observation.priority_score, 6)
    return row


def _build_hope_hunt_runner(
    *,
    model_path: Path,
    model_name: str,
    cities: list[str] | None,
    market_scope: str,
    markets_path: Path | None,
    interval_seconds: int,
    max_cycles: int,
    output: Path,
    latest_output: Path,
    summary_output: Path,
    state_path: Path,
    horizon: str,
    edge_threshold: float | None,
) -> tuple[HopeHuntRunner, CachedHttpClient]:
    """Construct a configured hope-hunt runner plus its shared HTTP client."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    scoped_cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    effective_edge_threshold = edge_threshold if edge_threshold is not None else config.backtest.default_edge_threshold

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_scoped_snapshots(
            markets_path=markets_path,
            cities=scoped_cities,
            market_scope=market_scope,
            active=True,
            closed=False,
        )

    def _evaluator(snapshots: list[MarketSnapshot], observed_at: datetime) -> list[HopeHuntObservation]:
        observations: list[HopeHuntObservation] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            if _target_day_distance(spec, observed_at=observed_at) < 0:
                continue
            metadata = extract_open_phase_metadata(snapshot, observed_at=observed_at)
            observation = _evaluate_opportunity_snapshot(
                snapshot,
                builder=builder,
                clob=clob,
                model_path=model_path,
                model_name=model_name,
                config=config,
                observed_at=observed_at,
                decision_horizon=horizon,
                edge_threshold=effective_edge_threshold,
            )
            if observation is None:
                continue
            observations.append(
                _hope_hunt_observation_from_opportunity(
                    snapshot,
                    observation,
                    market_created_at=(
                        metadata["market_created_at"] if isinstance(metadata["market_created_at"], datetime) else None
                    ),
                    market_deploying_at=(
                        metadata["market_deploying_at"]
                        if isinstance(metadata["market_deploying_at"], datetime)
                        else None
                    ),
                    market_accepting_orders_at=(
                        metadata["market_accepting_orders_at"]
                        if isinstance(metadata["market_accepting_orders_at"], datetime)
                        else None
                    ),
                    market_opened_at=(
                        metadata["market_opened_at"] if isinstance(metadata["market_opened_at"], datetime) else None
                    ),
                    open_phase_age_hours=(
                        float(metadata["open_phase_age_hours"])
                        if isinstance(metadata["open_phase_age_hours"], (int, float))
                        else None
                    ),
                    observed_at=observed_at,
                )
            )
        return observations

    runner = HopeHuntRunner(
        config=config,
        interval_seconds=interval_seconds,
        max_cycles=max_cycles,
        state_path=state_path,
        latest_output_path=latest_output,
        history_output_path=output,
        summary_output_path=summary_output,
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )
    return runner, http


def _summarize_backtest_metrics(
    prediction_rows: list[dict[str, object]],
    trade_rows: list[dict[str, object]],
    *,
    extra_metrics: dict[str, float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Build common predictive and trade metrics for a backtest run."""

    prediction_frame = pd.DataFrame(prediction_rows)
    trade_frame = pd.DataFrame(trade_rows)
    trade_summary = summarize_trade_log(trade_frame)
    calibration_probs = prediction_frame["top_probability"].to_numpy(dtype=float)
    calibration_outcomes = prediction_frame["top_is_correct"].to_numpy(dtype=float)
    metrics = {
        "mae": mae(prediction_frame["y_true"].to_numpy(), prediction_frame["y_pred"].to_numpy()),
        "rmse": rmse(prediction_frame["y_true"].to_numpy(), prediction_frame["y_pred"].to_numpy()),
        "nll": gaussian_nll(
            prediction_frame["y_true"].to_numpy(),
            prediction_frame["y_pred"].to_numpy(),
            prediction_frame["std"].to_numpy(),
        ),
        "avg_brier": float(prediction_frame["brier"].mean()),
        "avg_crps": float(prediction_frame["crps"].mean()),
        "calibration_gap": calibration_gap(calibration_probs, calibration_outcomes),
        **trade_summary,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics, prediction_frame, trade_frame


def _run_synthetic_backtest(
    frame: pd.DataFrame,
    *,
    model_name: str,
    variant: str | None = None,
    variant_config: LgbmEMOSVariantConfig | None = None,
    artifacts_dir: Path,
    bankroll: float,
    default_fee_bps: float,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    seed: int | None = None,
    min_train_size: int | None = None,
    retrain_stride: int = 1,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run the existing synthetic-book research backtest.

    retrain_stride: retrain the model only every N test steps; reuse the cached
    model for intermediate steps.  stride=1 (default) retrains every step.
    """

    broker = PaperBroker(bankroll=bankroll)
    prediction_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    effective_split_policy = _effective_split_policy(split_policy)
    effective_min_train = min_train_size if min_train_size is not None else 1
    cached_artifact = None
    step = 0
    for train, test in rolling_origin_splits(
        frame,
        min_train_size=effective_min_train,
        test_size=1,
        split_policy=effective_split_policy,
    ):
        train_kwargs: dict[str, object] = {
            "split_policy": effective_split_policy,
            "seed": seed,
        }
        if variant is not None:
            train_kwargs["variant"] = variant
        if variant_config is not None:
            train_kwargs["variant_config"] = variant_config
        if cached_artifact is None or step % retrain_stride == 0:
            cached_artifact = train_model(
                model_name,
                train,
                artifacts_dir,
                **train_kwargs,
            )
        artifact = cached_artifact
        step += 1
        for _, row in test.iterrows():
            spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
            snapshot = MarketSnapshot(
                captured_at=datetime.now(tz=UTC),
                market={"id": spec.market_id},
                spec=spec,
                outcome_prices=json.loads(str(row.get("market_prices_json", "{}"))),
                clob_token_ids=spec.token_ids,
            )
            forecast = predict_market(Path(artifact.path), model_name, spec, row.to_frame().T)
            winning_label = str(row["winning_outcome"])
            top_label, top_probability = max(
                forecast.outcome_probabilities.items(),
                key=lambda item: item[1],
            )
            prediction_rows.append(
                {
                    "target_date": row["target_date"],
                    "city": spec.city,
                    "y_true": row["realized_daily_max"],
                    "y_pred": forecast.mean,
                    "std": forecast.std,
                    "brier": brier_score(forecast.outcome_probabilities, winning_label),
                    "crps": crps_from_samples(pd.Series(forecast.samples).to_numpy(), float(row["realized_daily_max"])),
                    "top_probability": float(top_probability),
                    "top_is_correct": float(top_label == winning_label),
                }
            )

            outcome_labels = list(forecast.outcome_probabilities.keys())
            books: dict[str, BookSnapshot] = {}
            for ol in outcome_labels:
                tid = spec.token_ids[spec.outcome_labels().index(ol)] if spec.token_ids else ol
                books[ol] = _synthetic_book(snapshot, ol, tid)
            signal = _best_signal_across_outcomes(
                snapshot,
                forecast.outcome_probabilities,
                books,
                mode="paper",
                clob=None,
                default_fee_bps=default_fee_bps,
            )
            if signal is None:
                continue
            book = books[signal.outcome_label]
            size_notional = capped_kelly(signal.edge, signal.fair_probability, broker.bankroll, signal.executable_price)
            size = size_notional / max(signal.executable_price, 1e-6)
            if size <= 0:
                continue
            fill = broker.simulate_fill(signal, book=book, size=size)
            if fill is None:
                continue
            realized_pnl = settle_position(
                Position(
                    outcome_label=fill.outcome_label,
                    price=fill.price,
                    size=fill.size,
                    side=fill.side,
                ),
                winning_label,
                fee_paid=estimate_fee(fill.price * fill.size, taker_bps=default_fee_bps),
            )
            trade_rows.append(
                {
                    "market_id": fill.market_id,
                    "city": spec.city,
                    "decision_horizon": str(row["decision_horizon"]),
                    "outcome_label": fill.outcome_label,
                    "winning_outcome": winning_label,
                    "price": fill.price,
                    "size": fill.size,
                    "edge": signal.edge,
                    "realized_pnl": realized_pnl,
                    "pricing_source": "synthetic",
                }
            )

    metrics, _, _ = _summarize_backtest_metrics(prediction_rows, trade_rows)
    return metrics, trade_rows


def _run_real_history_backtest(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    model_name: str,
    variant: str | None = None,
    variant_config: LgbmEMOSVariantConfig | None = None,
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    seed: int | None = None,
    min_train_size: int | None = None,
    retrain_stride: int = 1,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using official historical market prices."""

    return _run_panel_pricing_backtest(
        frame,
        panel,
        model_name=model_name,
        variant=variant,
        variant_config=variant_config,
        artifacts_dir=artifacts_dir,
        flat_stake=flat_stake,
        default_fee_bps=default_fee_bps,
        pricing_source="real_history",
        split_policy=split_policy,
        seed=seed,
        min_train_size=min_train_size,
        retrain_stride=retrain_stride,
    )


def _quote_proxy_prices(
    market_price: float,
    *,
    half_spread: float,
    min_price: float = 0.0005,
    max_price: float = 0.9995,
) -> tuple[float, float]:
    """Build a conservative synthetic bid/ask proxy around a historical last trade."""

    bounded_price = min(max(float(market_price), min_price), max_price)
    bounded_half_spread = max(float(half_spread), 0.0)
    bid = max(min_price, bounded_price - bounded_half_spread)
    ask = min(max_price, bounded_price + bounded_half_spread)
    if ask < bid:
        ask = bid
    return bid, ask


def _run_quote_proxy_backtest(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    model_name: str,
    variant: str | None = None,
    variant_config: LgbmEMOSVariantConfig | None = None,
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    quote_proxy_half_spread: float,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    seed: int | None = None,
    min_train_size: int | None = None,
    retrain_stride: int = 1,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using last price plus an explicit quote proxy."""

    return _run_panel_pricing_backtest(
        frame,
        panel,
        model_name=model_name,
        variant=variant,
        variant_config=variant_config,
        artifacts_dir=artifacts_dir,
        flat_stake=flat_stake,
        default_fee_bps=default_fee_bps,
        pricing_source="quote_proxy",
        quote_proxy_half_spread=quote_proxy_half_spread,
        split_policy=split_policy,
        seed=seed,
        min_train_size=min_train_size,
        retrain_stride=retrain_stride,
    )


def _run_panel_pricing_backtest(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    model_name: str,
    variant: str | None = None,
    variant_config: LgbmEMOSVariantConfig | None = None,
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    pricing_source: Literal["real_history", "quote_proxy"],
    quote_proxy_half_spread: float = 0.0,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    seed: int | None = None,
    min_train_size: int | None = None,
    retrain_stride: int = 1,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using historical pricing or a quote proxy."""

    if flat_stake <= 0:
        raise typer.BadParameter("flat_stake must be positive.")
    if pricing_source == "quote_proxy" and quote_proxy_half_spread < 0:
        raise typer.BadParameter("quote_proxy_half_spread must be non-negative.")
    required_panel_columns = {
        "market_id",
        "decision_horizon",
        "outcome_label",
        "coverage_status",
        "market_price",
    }
    missing_panel = required_panel_columns.difference(panel.columns)
    if missing_panel:
        msg = f"Backtest panel is missing required columns {sorted(missing_panel)}."
        raise typer.BadParameter(msg)

    working_panel = panel.copy()
    working_panel["market_id"] = working_panel["market_id"].astype(str)
    working_panel["decision_horizon"] = working_panel["decision_horizon"].astype(str)
    working_panel["outcome_label"] = working_panel["outcome_label"].astype(str)
    working_panel["coverage_status"] = working_panel["coverage_status"].astype(str)
    working_panel["market_price"] = pd.to_numeric(working_panel["market_price"], errors="coerce")
    if "price_ts" in working_panel.columns:
        working_panel["price_ts"] = pd.to_datetime(working_panel["price_ts"], errors="coerce", utc=True)
    if "price_age_seconds" in working_panel.columns:
        working_panel["price_age_seconds"] = pd.to_numeric(working_panel["price_age_seconds"], errors="coerce")

    prediction_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    priced_decision_rows = 0
    skipped_missing_price = 0
    skipped_stale_price = 0
    skipped_non_positive_edge = 0
    price_ages: list[float] = []
    execution_price_premiums: list[float] = []

    effective_split_policy = _effective_split_policy(split_policy)
    effective_min_train = min_train_size if min_train_size is not None else 1
    cached_artifact = None
    step = 0
    for train, test in rolling_origin_splits(
        frame,
        min_train_size=effective_min_train,
        test_size=1,
        split_policy=effective_split_policy,
    ):
        train_kwargs: dict[str, object] = {
            "split_policy": effective_split_policy,
            "seed": seed,
        }
        if variant is not None:
            train_kwargs["variant"] = variant
        if variant_config is not None:
            train_kwargs["variant_config"] = variant_config
        if cached_artifact is None or step % retrain_stride == 0:
            cached_artifact = train_model(
                model_name,
                train,
                artifacts_dir,
                **train_kwargs,
            )
        artifact = cached_artifact
        step += 1
        for _, row in test.iterrows():
            spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
            forecast = predict_market(Path(artifact.path), model_name, spec, row.to_frame().T)
            winning_label = str(row["winning_outcome"])
            top_label, top_probability = max(
                forecast.outcome_probabilities.items(),
                key=lambda item: item[1],
            )
            prediction_rows.append(
                {
                    "target_date": row["target_date"],
                    "city": spec.city,
                    "y_true": row["realized_daily_max"],
                    "y_pred": forecast.mean,
                    "std": forecast.std,
                    "brier": brier_score(forecast.outcome_probabilities, winning_label),
                    "crps": crps_from_samples(pd.Series(forecast.samples).to_numpy(), float(row["realized_daily_max"])),
                    "top_probability": float(top_probability),
                    "top_is_correct": float(top_label == winning_label),
                }
            )

            best_edge = 0.0
            best_candidate: dict[str, object] | None = None
            has_covered_price = False
            has_stale_price = False
            for outcome_label, fair_probability in forecast.outcome_probabilities.items():
                panel_row = working_panel.loc[
                    (working_panel["market_id"] == spec.market_id)
                    & (working_panel["decision_horizon"] == str(row["decision_horizon"]))
                    & (working_panel["outcome_label"] == outcome_label)
                ].copy()
                if panel_row.empty:
                    continue
                selected = panel_row.iloc[-1]
                coverage_status = str(selected["coverage_status"])
                if coverage_status == "stale":
                    has_stale_price = True
                if coverage_status != "ok":
                    continue
                has_covered_price = True
                market_price = float(selected["market_price"])
                executable_price = market_price
                if pricing_source == "quote_proxy":
                    _, executable_price = _quote_proxy_prices(
                        market_price,
                        half_spread=quote_proxy_half_spread,
                    )
                fee_per_share = estimate_fee(executable_price, taker_bps=default_fee_bps)
                edge = compute_edge(fair_probability, executable_price, fee_per_share, 0.0)
                if edge > best_edge:
                    best_edge = edge
                    best_candidate = {
                        "outcome_label": outcome_label,
                        "fair_probability": fair_probability,
                        "market_price": market_price,
                        "executable_price": executable_price,
                        "edge": edge,
                        "selected": selected,
                    }
            if not has_covered_price:
                if has_stale_price:
                    skipped_stale_price += 1
                else:
                    skipped_missing_price += 1
                continue
            if best_candidate is None:
                skipped_non_positive_edge += 1
                continue
            outcome_label = str(best_candidate["outcome_label"])
            market_price = float(best_candidate["market_price"])  # type: ignore[arg-type]
            executable_price = float(best_candidate["executable_price"])  # type: ignore[arg-type]
            edge = float(best_candidate["edge"])  # type: ignore[arg-type]
            selected = best_candidate["selected"]
            size = flat_stake / max(executable_price, 1e-6)
            priced_decision_rows += 1
            age_seconds = selected.get("price_age_seconds")  # type: ignore[union-attr]
            if age_seconds is not None and pd.notna(age_seconds):
                price_ages.append(float(age_seconds))
            execution_price_premiums.append(executable_price - market_price)
            realized_pnl = settle_position(
                Position(
                    outcome_label=outcome_label,
                    price=executable_price,
                    size=size,
                    side="buy",
                ),
                winning_label,
                fee_paid=estimate_fee(flat_stake, taker_bps=default_fee_bps),
            )
            trade_rows.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "decision_horizon": str(row["decision_horizon"]),
                    "outcome_label": outcome_label,
                    "winning_outcome": winning_label,
                    "price": executable_price,
                    "reference_market_price": market_price,
                    "size": size,
                    "edge": edge,
                    "price_ts": selected.get("price_ts"),  # type: ignore[union-attr]
                    "price_age_seconds": age_seconds,
                    "realized_pnl": realized_pnl,
                    "pricing_source": pricing_source,
                }
            )

    metrics, _, _ = _summarize_backtest_metrics(
        prediction_rows,
        trade_rows,
        extra_metrics={
            "priced_decision_rows": float(priced_decision_rows),
            "skipped_missing_price": float(skipped_missing_price),
            "skipped_stale_price": float(skipped_stale_price),
            "skipped_non_positive_edge": float(skipped_non_positive_edge),
            "avg_price_age_seconds": float(sum(price_ages) / len(price_ages)) if price_ages else 0.0,
            "avg_execution_price_premium": (
                float(sum(execution_price_premiums) / len(execution_price_premiums))
                if execution_price_premiums
                else 0.0
            ),
        },
    )
    return metrics, trade_rows


@app.command("init-warehouse")
def init_warehouse() -> None:
    """Initialize the canonical warehouse directories and manifest."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="init-warehouse",
        config_hash=_config_hash(config, "init-warehouse"),
        notes="Initialize canonical warehouse layout.",
    )
    pipeline.run_id = run.run_id
    try:
        manifest_path = pipeline.warehouse.write_manifest()
        pipeline.warehouse.finish_run(run, status="completed", notes="Warehouse initialized.")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print_json(
        data={
            "duckdb_path": str(config.app.duckdb_path),
            "raw_root": str(config.app.raw_dir / "bronze"),
            "parquet_root": str(config.app.parquet_dir),
            "manifest_path": str(manifest_path),
        }
    )


@app.command("scan-markets")
def scan_markets(
    active_only: Annotated[bool, typer.Option("--active-only/--all", help="Restrict to currently active markets (default: include upcoming).")] = False,
    closed: bool = False,
    include_bundled: bool = False,
    output: Annotated[
        Path,
        typer.Option("--output", "--output-json", help="Where to persist discovered market snapshots."),
    ] = Path("artifacts/discovered_markets.json"),
) -> None:
    """Discover Polymarket maximum-temperature markets and persist parsed snapshots."""

    config, _, http, _, _, _ = _runtime()
    gamma = GammaClient(http, config.polymarket.gamma_base_url)
    refs = discover_temperature_event_refs_from_gamma(
        gamma,
        supported_cities=config.app.supported_cities,
        active=True if active_only else None,
        closed=closed,
        max_pages=config.polymarket.max_pages,
    )
    fetches = fetch_temperature_event_pages(http, refs, use_cache=False)
    snapshots = snapshots_from_temperature_event_fetches(fetches)
    if include_bundled:
        snapshots = snapshots + bundled_market_snapshots()
    save_market_snapshots(output, snapshots)

    table = Table(title="Discovered Temperature Markets")
    table.add_column("City")
    table.add_column("Question")
    table.add_column("Source")
    table.add_column("Parse")
    for snapshot in snapshots:
        if snapshot.spec is not None:
            table.add_row(
                snapshot.spec.city,
                snapshot.spec.question,
                snapshot.spec.official_source_name,
                "ok",
            )
        else:
            table.add_row(
                "unknown",
                str(snapshot.market.get("question", "")),
                "unknown",
                snapshot.parse_error or "failed",
            )
    console.print(table)
    console.print(f"Saved {len(snapshots)} snapshots -> {output}")


@app.command("scan-edge")
def scan_edge(
    model_path: Path = Path("artifacts/models/v2/champion.pkl"),
    model_name: str = DEFAULT_MODEL_NAME,
    markets_path: Annotated[
        Path,
        typer.Option("--markets-path", help="Discovered markets snapshot file (run scan-markets first)."),
    ] = Path("artifacts/discovered_markets.json"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    horizon: str = "policy",
    min_edge: float = 0.01,
    min_model_prob: float = 0.05,
    max_model_prob: float = 0.95,
    min_market_price: float = typer.Option(0.0, help="Skip outcome bins where Gamma mid-price is below this threshold (e.g. 0.10 filters out <10% bins where model edge is typically spurious)."),
    output: Annotated[
        Path,
        typer.Option("--output", "--output-json"),
    ] = Path("artifacts/edge_scan.json"),
) -> None:
    """Scan every outcome bin across discovered markets for YES and NO edge.

    Uses Gamma API outcomePrices (mid-prices) rather than CLOB bid/ask, since CLOB
    books for inactive/illiquid bins typically show phantom 0.001/0.999 spreads.

    For each bin, computes:
      YES edge = model_prob - gamma_price - fee
      NO edge  = (1 - model_prob) - (1 - gamma_price) - fee

    Reports all bins where abs(best_edge) >= min_edge, sorted by |edge| descending.
    Run scan-markets first to refresh the market snapshot file.
    """

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    if not markets_path.exists():
        console.print(f"[red]Markets file not found: {markets_path}[/red]")
        console.print("Run [bold]pmtmax scan-markets[/bold] first.")
        raise typer.Exit(1)

    snapshots = load_market_snapshots(markets_path)
    snapshots = _filter_snapshots_by_city(snapshots, cities)
    horizon_policy = _load_recent_horizon_policy()
    fee_bps = _default_fee_bps(config)

    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        now_utc = datetime.now(tz=UTC)
        if spec.target_local_date < now_utc.date():
            continue
        decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
            spec,
            now_utc=now_utc,
            horizon=horizon,
            horizon_policy=horizon_policy,
        )
        if decision_horizon is None or horizon_reason == "policy_filtered":
            continue
        feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
        forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)
        if not forecast_fresh(forecast.generated_at.replace(tzinfo=None), config.execution.stale_forecast_minutes):
            continue
        for outcome_label, model_prob in forecast.outcome_probabilities.items():
            # Skip bins where the model is near-certain — these are overconfident extrapolations
            if model_prob < min_model_prob or model_prob > max_model_prob:
                continue
            # Use Gamma mid-price: avoids the phantom 0.001/0.999 CLOB spread on illiquid bins
            gamma_price = snapshot.outcome_prices.get(outcome_label)
            if gamma_price is None or gamma_price <= 0.0 or gamma_price >= 1.0:
                continue
            if min_market_price > 0.0 and gamma_price < min_market_price:
                continue
            fee = estimate_fee(gamma_price, taker_bps=fee_bps)
            yes_edge = model_prob - gamma_price - fee
            no_edge = (1.0 - model_prob) - (1.0 - gamma_price) - fee
            best_edge = max(yes_edge, no_edge)
            if abs(best_edge) < min_edge:
                continue
            direction = "yes" if yes_edge >= no_edge else "no"
            rows.append(
                {
                    "city": spec.city,
                    "date": str(spec.target_local_date),
                    "question": spec.question,
                    "bin": outcome_label,
                    "model_prob": round(model_prob, 4),
                    "gamma_price": round(gamma_price, 4),
                    "yes_edge": round(yes_edge, 4),
                    "no_edge": round(no_edge, 4),
                    "best_edge": round(best_edge, 4),
                    "direction": direction,
                    "horizon": decision_horizon,
                }
            )

    rows.sort(key=lambda x: abs(float(x["best_edge"])), reverse=True)
    dump_json(output, rows)

    table = Table(title=f"Edge Scan — {len(rows)} signals (min_edge={min_edge:.0%})")
    table.add_column("City")
    table.add_column("Date")
    table.add_column("Bin")
    table.add_column("Dir")
    table.add_column("Model%")
    table.add_column("Mkt%")
    table.add_column("Edge")
    table.add_column("Horizon")
    for row in rows[:50]:
        table.add_row(
            str(row["city"]),
            str(row["date"]),
            str(row["bin"]),
            str(row["direction"]).upper(),
            f"{float(row['model_prob']):.1%}",
            f"{float(row['gamma_price']):.1%}",
            f"{float(row['best_edge']):+.1%}",
            str(row["horizon"]),
        )
    console.print(table)
    console.print(f"Saved {len(rows)} signals → {output}")


_DEFAULT_FULL_TRAINING_MARKETS = Path("configs/market_inventory/full_training_set_snapshots.json")


@app.command("build-dataset")
def build_dataset(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    decision_horizons: Annotated[list[str] | None, typer.Option("--decision-horizon")] = None,
    single_run_horizons: Annotated[list[str] | None, typer.Option("--single-run-horizon")] = None,
    contract: str = "both",
    strict_archive: Annotated[bool, typer.Option("--strict-archive/--no-strict-archive")] = True,
    allow_demo_fixture_fallback: bool = False,
    output_name: str = "historical_training_set",
    allow_canonical_overwrite: Annotated[bool, typer.Option("--allow-canonical-overwrite")] = False,
    truth_no_cache: Annotated[bool, typer.Option("--truth-no-cache")] = False,
) -> None:
    """Backfill bronze/silver tables, then materialize a gold training dataset."""

    if markets_path is None and _DEFAULT_FULL_TRAINING_MARKETS.exists():
        console.print(
            f"[yellow]No --markets-path given; defaulting to {_DEFAULT_FULL_TRAINING_MARKETS}[/yellow]"
        )
        markets_path = _DEFAULT_FULL_TRAINING_MARKETS

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo, use_truth_cache=not truth_no_cache)
    run = pipeline.warehouse.start_run(
        command="build-dataset",
        config_hash=_config_hash(
            config,
            "build-dataset",
            markets_path=markets_path,
            cities=cities,
            decision_horizons=decision_horizons,
            single_run_horizons=single_run_horizons,
            contract=contract,
            strict_archive=strict_archive,
            allow_demo_fixture_fallback=allow_demo_fixture_fallback,
            output_name=output_name,
            allow_canonical_overwrite=allow_canonical_overwrite,
            truth_no_cache=truth_no_cache,
        ),
        notes="Canonical backfill + materialization run.",
    )
    pipeline.run_id = run.run_id
    try:
        pipeline.backfill_markets(snapshots, source_name="build_dataset")
        pipeline.backfill_forecasts(
            snapshots,
            strict_archive=strict_archive,
            allow_fixture_fallback=allow_demo_fixture_fallback,
            single_run_horizons=single_run_horizons or decision_horizons or config.backtest.decision_horizons or None,
        )
        pipeline.backfill_truth(snapshots)
        frame = pipeline.materialize_training_set(
            snapshots,
            output_name=output_name,
            decision_horizons=decision_horizons or config.backtest.decision_horizons or None,
            contract=contract,
            allow_canonical_overwrite=allow_canonical_overwrite,
        )
        pipeline.warehouse.finish_run(run, status="completed", notes=f"Materialized {len(frame)} tabular rows.")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print(
        f"Built dataset with {len(frame)} rows at "
        f"{(config.app.parquet_dir / 'gold/v2' / f'{output_name}.parquet')}"
    )


@app.command("collection-preflight")
def collection_preflight(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
) -> None:
    """Report required manual env settings for historical collection inputs."""

    _, env = load_settings()
    snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
    console.print_json(data=_collection_preflight_report(snapshots, env))


@app.command("backfill-markets")
def backfill_markets(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    active: bool = False,
    closed: bool = False,
    include_bundled: bool = False,
) -> None:
    """Persist market raw snapshots and parsed specs into bronze/silver tables."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(
        markets_path=markets_path,
        cities=cities,
        active=active if active or closed else None,
        closed=closed if active or closed else None,
    )
    if include_bundled:
        snapshots = snapshots + bundled_market_snapshots(cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="backfill-markets",
        config_hash=_config_hash(
            config,
            "backfill-markets",
            markets_path=markets_path,
            cities=cities,
            active=active,
            closed=closed,
            include_bundled=include_bundled,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        result = pipeline.backfill_markets(snapshots, source_name="scan" if active or closed else "snapshot")
        pipeline.warehouse.finish_run(run, status="completed")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print_json(
        data={
            "bronze_market_snapshots": len(result["bronze_market_snapshots"]),
            "silver_market_specs": len(result["silver_market_specs"]),
        }
    )


@app.command("backfill-forecasts")
def backfill_forecasts(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    models: Annotated[list[str] | None, typer.Option("--model")] = None,
    single_run_horizons: Annotated[list[str] | None, typer.Option("--single-run-horizon")] = None,
    strict_archive: Annotated[bool, typer.Option("--strict-archive/--no-strict-archive")] = True,
    allow_demo_fixture_fallback: bool = False,
) -> None:
    """Backfill forecast payloads and normalized hourly forecast tables."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="backfill-forecasts",
        config_hash=_config_hash(
            config,
            "backfill-forecasts",
            markets_path=markets_path,
            cities=cities,
            models=models,
            single_run_horizons=single_run_horizons,
            strict_archive=strict_archive,
            allow_demo_fixture_fallback=allow_demo_fixture_fallback,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        result = pipeline.backfill_forecasts(
            snapshots,
            models=models,
            strict_archive=strict_archive,
            allow_fixture_fallback=allow_demo_fixture_fallback,
            single_run_horizons=single_run_horizons,
        )
        pipeline.warehouse.finish_run(run, status="completed")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print_json(
        data={
            "bronze_forecast_requests": len(result["bronze_forecast_requests"]),
            "silver_forecast_runs_hourly": len(result["silver_forecast_runs_hourly"]),
        }
    )


@app.command("backfill-truth")
def backfill_truth(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    truth_no_cache: Annotated[bool, typer.Option("--truth-no-cache")] = False,
) -> None:
    """Backfill official truth snapshots and normalized daily observations."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo, use_truth_cache=not truth_no_cache)
    run = pipeline.warehouse.start_run(
        command="backfill-truth",
        config_hash=_config_hash(
            config,
            "backfill-truth",
            markets_path=markets_path,
            cities=cities,
            truth_no_cache=truth_no_cache,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        result = pipeline.backfill_truth(snapshots)
        pipeline.warehouse.finish_run(run, status="completed")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    status_counts: dict[str, int] = {}
    if not result["bronze_truth_snapshots"].empty and "status" in result["bronze_truth_snapshots"].columns:
        status_counts = {
            str(key): int(value)
            for key, value in result["bronze_truth_snapshots"]["status"].astype(str).value_counts().to_dict().items()
        }
    console.print_json(
        data={
            "bronze_truth_snapshots": len(result["bronze_truth_snapshots"]),
            "silver_observations_daily": len(result["silver_observations_daily"]),
            "status_counts": status_counts,
        }
    )


@app.command("backfill-price-history")
def backfill_price_history(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    interval: str = "max",
    fidelity: int = 60,
) -> None:
    """Backfill Polymarket official outcome-token price history into bronze/silver tables."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    run = pipeline.warehouse.start_run(
        command="backfill-price-history",
        config_hash=_config_hash(
            config,
            "backfill-price-history",
            markets_path=markets_path,
            cities=cities,
            interval=interval,
            fidelity=fidelity,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        result = pipeline.backfill_price_history(
            snapshots,
            clob=clob,
            interval=interval,
            fidelity=fidelity,
            use_cache=True,
        )
        pipeline.warehouse.finish_run(run, status="completed")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
        http.close()
    status_counts: dict[str, int] = {}
    if not result["bronze_price_history_requests"].empty and "status" in result["bronze_price_history_requests"].columns:
        status_counts = {
            str(key): int(value)
            for key, value in result["bronze_price_history_requests"]["status"].astype(str).value_counts().to_dict().items()
        }
    console.print_json(
        data={
            "bronze_price_history_requests": len(result["bronze_price_history_requests"]),
            "silver_price_timeseries": len(result["silver_price_timeseries"]),
            "status_counts": status_counts,
        }
    )


@app.command("summarize-truth-coverage")
def summarize_truth_coverage(
    output: Path = Path("artifacts/truth_coverage.json"),
) -> None:
    """Summarize truth coverage, lagged markets, and archive-ready statuses."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    try:
        result = pipeline.summarize_truth_coverage()
    finally:
        pipeline.warehouse.close()
    payload = {
        "summary": result["summary"].to_dict(orient="records"),
        "details": result["details"].to_dict(orient="records"),
    }
    dump_json(output, payload)
    console.print_json(
        data={
            "summary_rows": len(result["summary"]),
            "detail_rows": len(result["details"]),
            "output_path": str(output),
        }
    )


@app.command("summarize-price-history-coverage")
def summarize_price_history_coverage(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    output: Path = Path("artifacts/price_history_coverage.json"),
) -> None:
    """Summarize request-level and decision-time official price-history coverage."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    try:
        result = pipeline.summarize_price_history_coverage(_market_ids_from_snapshots(snapshots) or None)
    finally:
        pipeline.warehouse.close()
        http.close()
    payload = {
        "request_summary": result["request_summary"].to_dict(orient="records"),
        "request_details": result["request_details"].to_dict(orient="records"),
        "panel_summary": result["panel_summary"].to_dict(orient="records"),
        "market_summary": result["market_summary"].to_dict(orient="records"),
        "details": result["details"].to_dict(orient="records"),
    }
    dump_json(output, payload)
    console.print_json(
        data={
            "request_summary_rows": len(result["request_summary"]),
            "panel_summary_rows": len(result["panel_summary"]),
            "detail_rows": len(result["details"]),
            "output_path": str(output),
        }
    )


@app.command("summarize-dataset-readiness")
def summarize_dataset_readiness(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    output: Path = Path("artifacts/dataset_readiness.json"),
) -> None:
    """Summarize snapshot, forecast, truth, and gold-row readiness for a target inventory."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    try:
        result = pipeline.summarize_dataset_readiness(snapshots)
    finally:
        pipeline.warehouse.close()
    payload = {
        "summary": result["summary"].to_dict(orient="records"),
        "details": result["details"].to_dict(orient="records"),
    }
    dump_json(output, payload)
    console.print_json(
        data={
            "summary_rows": len(result["summary"]),
            "detail_rows": len(result["details"]),
            "output_path": str(output),
        }
    )


@app.command("materialize-backtest-panel")
def materialize_backtest_panel(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    output_name: str = "historical_backtest_panel",
    max_price_age_minutes: int = 720,
    allow_canonical_overwrite: Annotated[bool, typer.Option("--allow-canonical-overwrite")] = False,
) -> None:
    """Materialize a decision-time official-price backtest panel from a gold training dataset."""

    frame = pd.read_parquet(dataset_path)
    if markets_path is not None or cities:
        snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
        market_ids = _market_ids_from_snapshots(snapshots)
        frame = frame.loc[frame["market_id"].astype(str).isin(market_ids)].copy()
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="materialize-backtest-panel",
        config_hash=_config_hash(
            config,
            "materialize-backtest-panel",
            dataset_path=dataset_path,
            markets_path=markets_path,
            cities=cities,
            output_name=output_name,
            max_price_age_minutes=max_price_age_minutes,
            allow_canonical_overwrite=allow_canonical_overwrite,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        panel = pipeline.materialize_backtest_panel(
            frame,
            output_name=output_name,
            max_price_age_minutes=max_price_age_minutes,
            allow_canonical_overwrite=allow_canonical_overwrite,
        )
        pipeline.warehouse.finish_run(run, status="completed", notes=f"Materialized {len(panel)} panel rows.")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
        http.close()
    console.print_json(
        data={
            "gold_backtest_panel": len(panel),
            "output_path": str(config.app.parquet_dir / "gold/v2" / f"{output_name}.parquet"),
            "max_price_age_minutes": max_price_age_minutes,
        }
    )


@app.command("materialize-training-set")
def materialize_training_set(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    decision_horizons: Annotated[list[str] | None, typer.Option("--decision-horizon")] = None,
    contract: str = "both",
    output_name: str = "historical_training_set",
    allow_canonical_overwrite: Annotated[bool, typer.Option("--allow-canonical-overwrite")] = False,
) -> None:
    """Materialize the gold training dataset from backfilled bronze/silver tables."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="materialize-training-set",
        config_hash=_config_hash(
            config,
            "materialize-training-set",
            markets_path=markets_path,
            cities=cities,
            decision_horizons=decision_horizons,
            contract=contract,
            output_name=output_name,
            allow_canonical_overwrite=allow_canonical_overwrite,
        ),
    )
    pipeline.run_id = run.run_id
    try:
        frame = pipeline.materialize_training_set(
            snapshots,
            output_name=output_name,
            decision_horizons=decision_horizons or config.backtest.decision_horizons or None,
            contract=contract,
            allow_canonical_overwrite=allow_canonical_overwrite,
        )
        pipeline.warehouse.finish_run(run, status="completed", notes=f"Materialized {len(frame)} rows.")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print_json(
        data={
            "gold_training_examples": len(frame),
            "output_path": str(config.app.parquet_dir / "gold/v2" / f"{output_name}.parquet"),
            "contract": contract,
        }
    )


@app.command("summarize-forecast-availability")
def summarize_forecast_availability(
    output: Path = Path("artifacts/forecast_availability.json"),
    top_k: int = 3,
) -> None:
    """Summarize archive coverage by city/model/horizon and recommend top available models."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    try:
        result = pipeline.summarize_forecast_availability(top_k=top_k)
    finally:
        pipeline.warehouse.close()
    payload = {
        "summary": result["summary"].to_dict(orient="records"),
        "recommended": result["recommended"].to_dict(orient="records"),
    }
    dump_json(output, payload)
    console.print_json(
        data={
            "summary_rows": len(result["summary"]),
            "recommended_rows": len(result["recommended"]),
            "output_path": str(output),
        }
    )


@app.command("migrate-legacy-warehouse")
def migrate_legacy_warehouse(
    legacy_paths: Annotated[list[Path] | None, typer.Option("--legacy-path")] = None,
    archive_legacy: bool = False,
) -> None:
    """Merge legacy DuckDB outputs into the canonical warehouse with validation."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    candidates = ordered_legacy_paths(
        legacy_paths or [path for path in config.app.duckdb_path.parent.glob("*.duckdb") if path != config.app.duckdb_path]
    )
    backup_path = backup_duckdb_file(config.app.duckdb_path, config.app.archive_dir)
    warehouse = DataWarehouse.from_paths(
        duckdb_path=config.app.duckdb_path,
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        archive_root=config.app.archive_dir,
    )
    run = warehouse.start_run(
        command="migrate-legacy-warehouse",
        config_hash=_config_hash(
            config,
            "migrate-legacy-warehouse",
            legacy_paths=candidates,
            archive_legacy=archive_legacy,
        ),
    )
    inventory = warehouse.inventory_legacy_databases(candidates)
    inventory_path = config.app.manifest_dir / "legacy_inventory.json"
    dump_json(inventory_path, inventory)
    try:
        result = warehouse.migrate_legacy_databases(candidates, archive_legacy=False)
        compact_counts = warehouse.compact()
        validation = warehouse.validate_migration(inventory)
        if not validation["ok"]:
            msg = f"Migration validation failed: {validation}"
            warehouse.finish_run(run, status="failed", notes=msg)
            raise RuntimeError(msg)
        archived: list[str] = []
        if archive_legacy:
            archived = warehouse.archive_legacy_databases(candidates)
        report = {
            "backup_path": str(backup_path) if backup_path is not None else None,
            "inventory_path": str(inventory_path),
            "legacy_paths": [str(path) for path in candidates],
            "migrated_tables": result,
            "compact_counts": compact_counts,
            "validation": validation,
            "archived_paths": archived,
        }
        dump_json(config.app.manifest_dir / "migration_report.json", report)
        warehouse.finish_run(run, status="completed", notes=f"Migrated {len(candidates)} legacy DBs.")
    except Exception as exc:  # noqa: BLE001
        warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        warehouse.close()
    console.print_json(data=report)


@app.command("compact-warehouse")
def compact_warehouse() -> None:
    """Rewrite parquet mirrors and refresh the canonical warehouse manifest."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    warehouse = DataWarehouse.from_paths(
        duckdb_path=config.app.duckdb_path,
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        archive_root=config.app.archive_dir,
    )
    run = warehouse.start_run(
        command="compact-warehouse",
        config_hash=_config_hash(config, "compact-warehouse"),
    )
    try:
        counts = warehouse.compact()
        warehouse.finish_run(run, status="completed")
    except Exception as exc:  # noqa: BLE001
        warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        warehouse.close()
    console.print_json(data=counts)


@app.command("inventory-legacy-runs")
def inventory_legacy_runs_command(
    output: Path = Path("data/manifests/legacy_runs_inventory.json"),
) -> None:
    """Inspect non-canonical raw/parquet run artifacts that can be archived."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    inventory = inventory_legacy_runs(
        raw_root=config.app.raw_dir,
        parquet_root=config.app.parquet_dir,
        manifest_root=config.app.manifest_dir,
    )
    dump_json(output, inventory.model_dump(mode="json"))
    console.print_json(
        data={
            "entries": len(inventory.entries),
            "manifest_path": inventory.manifest_path,
            "output_path": str(output),
        }
    )


@app.command("archive-legacy-runs")
def archive_legacy_runs_command(
    dry_run: Annotated[bool, typer.Option("--dry-run/--execute")] = True,
    inventory_path: Path = Path("data/manifests/legacy_runs_inventory.json"),
) -> None:
    """Archive inventoried legacy raw/parquet run artifacts."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    inventory = (
        inventory_legacy_runs(
            raw_root=config.app.raw_dir,
            parquet_root=config.app.parquet_dir,
            manifest_root=config.app.manifest_dir,
        )
        if not inventory_path.exists()
        else None
    )
    if inventory is None:
        inventory = LegacyRunInventory.model_validate(load_json(inventory_path))
    else:
        dump_json(inventory_path, inventory.model_dump(mode="json"))

    report = archive_legacy_runs(
        inventory=inventory,
        archive_root=config.app.archive_dir,
        dry_run=dry_run,
    )
    report_path = config.app.manifest_dir / "legacy_runs_archive_report.json"
    dump_json(report_path, report)
    console.print_json(
        data={
            **report,
            "inventory_path": str(inventory_path),
            "report_path": str(report_path),
        }
    )


@app.command("export-seed")
def export_seed(
    seed_path: Path = Path("artifacts/bootstrap/pmtmax_seed.tar.gz"),
) -> None:
    """Export canonical raw/parquet/manifests into a portable seed archive."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    manifest = export_seed_bundle(
        data_root=config.app.data_dir,
        raw_root=config.app.raw_dir,
        parquet_root=config.app.parquet_dir,
        manifest_root=config.app.manifest_dir,
        seed_path=seed_path,
    )
    console.print_json(data=manifest.model_dump(mode="json"))


@app.command("restore-seed")
def restore_seed(
    seed_path: Path = Path("artifacts/bootstrap/pmtmax_seed.tar.gz"),
) -> None:
    """Restore a portable seed archive and rebuild the canonical warehouse from parquet mirrors."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    manifest = restore_seed_bundle(seed_path=seed_path, data_root=config.app.data_dir)
    warehouse = DataWarehouse.from_paths(
        duckdb_path=config.app.duckdb_path,
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        archive_root=config.app.archive_dir,
    )
    try:
        counts = restore_warehouse_from_seed(
            warehouse=warehouse,
            parquet_root=config.app.parquet_dir,
            manifest_root=config.app.manifest_dir,
        )
    finally:
        warehouse.close()
    console.print_json(
        data={
            "seed_manifest": manifest.model_dump(mode="json"),
            "restored_tables": counts,
        }
    )


@app.command("bootstrap-lab")
def bootstrap_lab(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    decision_horizons: Annotated[list[str] | None, typer.Option("--decision-horizon")] = None,
    single_run_horizons: Annotated[list[str] | None, typer.Option("--single-run-horizon")] = None,
    contract: str = "both",
    output_name: str = "historical_training_set",
    strict_archive: Annotated[bool, typer.Option("--strict-archive/--no-strict-archive")] = True,
    allow_demo_fixture_fallback: bool = False,
    seed_path: Path = Path("artifacts/bootstrap/pmtmax_seed.tar.gz"),
    cleanup_legacy: bool = True,
) -> None:
    """Build a usable research dataset environment in one command."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _bootstrap_snapshots(markets_path=markets_path, cities=cities)
    archived_legacy_paths: list[str] = []
    steps: list[str] = []
    seed_restored = False

    if cleanup_legacy:
        legacy_inventory = inventory_legacy_runs(
            raw_root=config.app.raw_dir,
            parquet_root=config.app.parquet_dir,
            manifest_root=config.app.manifest_dir,
        )
        dump_json(
            config.app.manifest_dir / "legacy_runs_inventory.json",
            legacy_inventory.model_dump(mode="json"),
        )
        archive_report = archive_legacy_runs(
            inventory=legacy_inventory,
            archive_root=config.app.archive_dir,
            dry_run=False,
        )
        archived_legacy_paths = archive_report["archived_paths"]
        dump_json(config.app.manifest_dir / "legacy_runs_archive_report.json", archive_report)
        steps.append("archive_legacy_runs")

    should_restore_seed = (
        seed_path.exists()
        and (not config.app.duckdb_path.exists() or not (config.app.manifest_dir / "warehouse_manifest.json").exists())
    )
    if should_restore_seed:
        restore_seed_bundle(seed_path=seed_path, data_root=config.app.data_dir)
        warehouse = DataWarehouse.from_paths(
            duckdb_path=config.app.duckdb_path,
            parquet_root=config.app.parquet_dir,
            raw_root=config.app.raw_dir,
            manifest_root=config.app.manifest_dir,
            archive_root=config.app.archive_dir,
        )
        try:
            restore_warehouse_from_seed(
                warehouse=warehouse,
                parquet_root=config.app.parquet_dir,
                manifest_root=config.app.manifest_dir,
            )
        finally:
            warehouse.close()
        seed_restored = True
        steps.append("restore_seed")

    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="bootstrap-lab",
        config_hash=_config_hash(
            config,
            "bootstrap-lab",
            markets_path=markets_path,
            cities=cities,
            decision_horizons=decision_horizons,
            single_run_horizons=single_run_horizons,
            contract=contract,
            output_name=output_name,
            strict_archive=strict_archive,
            allow_demo_fixture_fallback=allow_demo_fixture_fallback,
            seed_path=seed_path,
            cleanup_legacy=cleanup_legacy,
        ),
        notes="One-shot research bootstrap.",
    )
    pipeline.run_id = run.run_id
    availability_output = Path("artifacts/bootstrap/forecast_availability.json")
    bootstrap_output = Path("artifacts/bootstrap/bootstrap_manifest.json")
    try:
        pipeline.warehouse.write_manifest()
        steps.append("init_warehouse")
        pipeline.backfill_markets(snapshots, source_name="bootstrap")
        steps.append("backfill_markets")
        pipeline.backfill_forecasts(
            snapshots,
            strict_archive=strict_archive,
            allow_fixture_fallback=allow_demo_fixture_fallback,
            single_run_horizons=single_run_horizons or decision_horizons or config.backtest.decision_horizons or None,
        )
        steps.append("backfill_forecasts")
        pipeline.backfill_truth(snapshots)
        steps.append("backfill_truth")
        frame = pipeline.materialize_training_set(
            snapshots,
            output_name=output_name,
            decision_horizons=decision_horizons or config.backtest.decision_horizons or None,
            contract=contract,
        )
        steps.append("materialize_training_set")
        availability = pipeline.summarize_forecast_availability()
        dump_json(
            availability_output,
            {
                "summary": availability["summary"].to_dict(orient="records"),
                "recommended": availability["recommended"].to_dict(orient="records"),
            },
        )
        steps.append("summarize_forecast_availability")
        warehouse_counts = pipeline.warehouse.compact()
        steps.append("compact_warehouse")
        bootstrap_manifest = build_bootstrap_manifest(
            seed_path=seed_path if seed_path.exists() else None,
            seed_restored=seed_restored,
            archived_legacy_paths=archived_legacy_paths,
            steps=steps,
            output_paths={
                "dataset_path": str(config.app.parquet_dir / "gold" / f"{output_name}.parquet"),
                "sequence_dataset_path": str(config.app.parquet_dir / "gold" / f"{output_name}_sequence.parquet"),
                "availability_path": str(availability_output),
                "warehouse_manifest_path": str(config.app.manifest_dir / "warehouse_manifest.json"),
            },
            warehouse_counts=warehouse_counts,
        )
        dump_json(bootstrap_output, bootstrap_manifest.model_dump(mode="json"))
        pipeline.warehouse.finish_run(
            run,
            status="completed",
            notes=f"Bootstrapped {len(frame)} tabular rows.",
        )
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print_json(data=bootstrap_manifest.model_dump(mode="json"))


@app.command("sync-firebase")
def sync_firebase(
    dry_run: bool = True,
    limit: int | None = None,
) -> None:
    """Mirror parquet/raw/manifests to Firebase Storage."""

    config, _, _, _, _, _ = _runtime(include_stores=False)
    mirror = FirebaseMirror(
        bucket_name=config.firebase.bucket_name,
        prefix=config.firebase.prefix,
        credentials_json=config.firebase.credentials_json or None,
    )
    payload = mirror.sync(
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        dry_run=dry_run,
        limit=limit,
    )
    dump_json(Path("artifacts/firebase_sync_manifest.json"), payload)
    console.print_json(data=payload)


@app.command("train-baseline")
def train_baseline(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    model_name: str = "gaussian_emos",
    artifacts_dir: Path = Path("artifacts/models/v2"),
) -> None:
    """Train a baseline probabilistic model."""

    config, _ = load_settings()
    frame = pd.read_parquet(dataset_path)
    artifact = train_model(
        require_supported_model_name(model_name),
        frame,
        artifacts_dir,
        split_policy="market_day",
        seed=config.app.random_seed,
    )
    console.print(f"Trained {model_name} -> {artifact.path}")


@app.command("train-advanced")
def train_advanced(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    model_name: str = "det2prob_nn",
    artifacts_dir: Path = Path("artifacts/models/v2"),
    variant: str | None = None,
    variant_spec: Path | None = None,
    publish_champion: bool = False,
) -> None:
    """Train an advanced probabilistic model, optionally publish the champion alias."""

    config, _ = load_settings()
    frame = pd.read_parquet(dataset_path)
    resolved_variant, resolved_variant_config, _ = _resolve_lgbm_variant_inputs(
        model_name=model_name,
        variant=variant,
        variant_spec=variant_spec,
    )
    artifact = train_model(
        require_supported_model_name(model_name),
        frame,
        artifacts_dir,
        split_policy="market_day",
        seed=config.app.random_seed,
        variant=resolved_variant,
        variant_config=resolved_variant_config,
    )
    console.print(f"Trained {model_name} -> {artifact.path}")
    if publish_champion:
        champion_path = _default_model_path(DEFAULT_MODEL_NAME)
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(artifact.path), champion_path)
        if artifact.calibration_path is not None:
            champion_cal_path = champion_path.with_name(f"{champion_path.stem}.calibrator.pkl")
            shutil.copyfile(Path(artifact.calibration_path), champion_cal_path)
        metadata: dict[str, object] = {
            "model_name": model_name,
            "alias_path": str(champion_path),
            "alias_calibration_path": str(champion_path.with_name(f"{champion_path.stem}.calibrator.pkl")),
            "source_model_path": artifact.path,
            "source_calibration_path": artifact.calibration_path,
            "variant": artifact.variant,
            "contract_version": "v2",
            "published_at": datetime.now(tz=UTC).isoformat(),
        }
        dump_json(_default_champion_metadata_path(), metadata)
        console.print(f"Champion alias published: {champion_path}")


@app.command("backtest")
def backtest(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    model_name: str = DEFAULT_MODEL_NAME,
    artifacts_dir: Path = Path("artifacts/models/v2"),
    bankroll: float = 10_000.0,
    pricing_source: Literal["synthetic", "real_history", "quote_proxy"] = "synthetic",
    panel_path: Path = Path("data/parquet/gold/v2/historical_backtest_panel.parquet"),
    flat_stake: float = 1.0,
    quote_proxy_half_spread: float = 0.02,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    variant: str | None = None,
    variant_spec: Path | None = None,
    last_n: int = 0,
    retrain_stride: int = 1,
) -> None:
    """Run a rolling-origin backtest with synthetic or official historical pricing.

    Use --last-n N to run only the final N rows as test points (fast-eval proxy).
    Use --retrain-stride N to retrain the model every N steps (e.g. 30 for ~47 min).
    """

    config, _ = load_settings()
    resolved_model_name = _resolve_model_name_alias(model_name)
    # If model_name is an alias and variant was not explicitly provided, pick up
    # the variant stored in the alias metadata (e.g. champion → high_capacity_fast).
    if variant_spec is None and variant is None and model_name in PUBLIC_MODEL_ALIASES:
        alias_meta = _load_alias_metadata(model_name)
        alias_variant = alias_meta.get("variant")
        if alias_variant:
            variant = str(alias_variant)
    variant, variant_config, _ = _resolve_lgbm_variant_inputs(
        model_name=resolved_model_name,
        variant=variant,
        variant_spec=variant_spec,
    )
    default_fee_bps = _default_fee_bps(config)
    frame = pd.read_parquet(dataset_path)
    # fast-eval: pass min_train_size so only the last `last_n` groups are used as
    # test points while training sets still use full history.
    # min_train_size for market_day policy is measured in groups, not rows.
    fast_eval_min_train: int | None = None
    if last_n > 0:
        key_frame = frame[["market_id", "target_date"]].astype({"market_id": str})
        num_groups = int(key_frame.astype(str).agg("|".join, axis=1).nunique())
        fast_eval_min_train = max(1, num_groups - last_n)
    required_columns = {"market_spec_json", "market_prices_json", "winning_outcome", "realized_daily_max"}
    missing = required_columns.difference(frame.columns)
    if missing:
        msg = f"Dataset is missing required columns {sorted(missing)}. Re-run `pmtmax build-dataset`."
        raise typer.BadParameter(msg)
    if len(frame) < 2:
        raise typer.BadParameter("Need at least two rows to backtest.")

    if pricing_source == "synthetic":
        metrics, trade_rows = _run_synthetic_backtest(
            frame,
            model_name=resolved_model_name,
            variant=variant,
            variant_config=variant_config,
            artifacts_dir=artifacts_dir,
            bankroll=bankroll,
            default_fee_bps=default_fee_bps,
            split_policy=split_policy,
            seed=config.app.random_seed,
            min_train_size=fast_eval_min_train,
            retrain_stride=retrain_stride,
        )
        metrics_output = _default_backtest_output("backtest_metrics.json")
        trades_output = _default_backtest_output("backtest_trades.json")
    else:
        if not panel_path.exists():
            msg = (
                f"Backtest panel does not exist: {panel_path}. "
                "Run `uv run pmtmax materialize-backtest-panel` first."
            )
            raise typer.BadParameter(msg)
        panel = pd.read_parquet(panel_path)
        panel_required = {"market_id", "decision_horizon", "outcome_label", "coverage_status", "market_price"}
        missing_panel = panel_required.difference(panel.columns)
        if missing_panel:
            msg = f"Backtest panel is missing required columns {sorted(missing_panel)}."
            raise typer.BadParameter(msg)
        if panel.empty or not (panel["coverage_status"].astype(str) == "ok").any():
            msg = (
                "Backtest panel has no coverage_status=ok rows. "
                "Run `uv run pmtmax summarize-price-history-coverage` to inspect gaps."
            )
            raise typer.BadParameter(msg)
        if pricing_source == "real_history":
            metrics, trade_rows = _run_real_history_backtest(
                frame,
                panel,
                model_name=resolved_model_name,
                variant=variant,
                variant_config=variant_config,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=default_fee_bps,
                split_policy=split_policy,
                seed=config.app.random_seed,
                min_train_size=fast_eval_min_train,
                retrain_stride=retrain_stride,
            )
            metrics_output = _default_backtest_output("backtest_metrics_real_history.json")
            trades_output = _default_backtest_output("backtest_trades_real_history.json")
        else:
            metrics, trade_rows = _run_quote_proxy_backtest(
                frame,
                panel,
                model_name=resolved_model_name,
                variant=variant,
                variant_config=variant_config,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=default_fee_bps,
                quote_proxy_half_spread=quote_proxy_half_spread,
                split_policy=split_policy,
                seed=config.app.random_seed,
                min_train_size=fast_eval_min_train,
                retrain_stride=retrain_stride,
            )
            metrics_output = _default_backtest_output("backtest_metrics_quote_proxy.json")
            trades_output = _default_backtest_output("backtest_trades_quote_proxy.json")

    metrics["contract_version"] = "v2"
    metrics["model_name"] = resolved_model_name
    metrics["split_policy"] = _effective_split_policy(split_policy)
    metrics["leakage_audit_passed"] = True
    dump_json(metrics_output, metrics)
    dump_json(trades_output, trade_rows)
    console.print_json(data=metrics)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std summary for a numeric list."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=0))


def _model_variant_label(model_name: str, variant: str | None) -> str:
    """Return a stable model label including an optional ablation variant."""

    return model_name if variant is None else f"{model_name}:{variant}"


def _grouped_holdout_split(
    frame: pd.DataFrame,
    *,
    split_policy: Literal["market_day", "target_day"],
    holdout_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Split a frame into chronological grouped train/test partitions."""

    ordered = frame.sort_values(
        [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
    ).reset_index(drop=True)
    group_ids = group_id_series(ordered, split_policy=split_policy)
    unique_groups = group_ids.drop_duplicates().tolist()
    if len(unique_groups) < 2:
        msg = f"Need at least two groups for split_policy={split_policy}."
        raise typer.BadParameter(msg)
    split_idx = max(1, int(len(unique_groups) * (1.0 - holdout_fraction)))
    split_idx = min(split_idx, len(unique_groups) - 1)
    train_groups = set(unique_groups[:split_idx])
    test_groups = set(unique_groups[split_idx:])
    train = ordered.loc[group_ids.isin(train_groups)].reset_index(drop=True).copy()
    test = ordered.loc[group_ids.isin(test_groups)].reset_index(drop=True).copy()
    if train.empty or test.empty:
        msg = f"Unable to build a non-empty grouped holdout for split_policy={split_policy}."
        raise typer.BadParameter(msg)
    return train, test, len(unique_groups), len(test_groups)


def _panel_lookup(panel: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    """Return the last observed panel row for each market/horizon/outcome."""

    working = panel.copy()
    working["market_id"] = working["market_id"].astype(str)
    working["decision_horizon"] = working["decision_horizon"].astype(str)
    working["outcome_label"] = working["outcome_label"].astype(str)
    working["coverage_status"] = working["coverage_status"].astype(str)
    working["market_price"] = pd.to_numeric(working["market_price"], errors="coerce")
    if "price_ts" in working.columns:
        working["price_ts"] = pd.to_datetime(working["price_ts"], errors="coerce", utc=True)
        working = working.sort_values(["market_id", "decision_horizon", "outcome_label", "price_ts"]).reset_index(drop=True)
    grouped = working.groupby(["market_id", "decision_horizon", "outcome_label"], sort=False).tail(1)
    return {
        (str(row.market_id), str(row.decision_horizon), str(row.outcome_label)): row._asdict()
        for row in grouped.itertuples(index=False)
    }


def _run_grouped_holdout_ablation(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    model_name: str,
    variant: str,
    variant_config: LgbmEMOSVariantConfig | None = None,
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    quote_proxy_half_spread: float,
    split_policy: Literal["market_day", "target_day"],
    seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Train one variant once and evaluate on a grouped chronological holdout."""

    train_frame, test_frame, total_groups, test_groups = _grouped_holdout_split(
        frame,
        split_policy=split_policy,
    )
    artifact = train_model(
        model_name,
        train_frame,
        artifacts_dir,
        split_policy=split_policy,
        seed=seed,
        variant=variant,
        variant_config=variant_config,
    )
    lookup = _panel_lookup(panel)
    prediction_rows: list[dict[str, object]] = []
    real_trade_rows: list[dict[str, object]] = []
    quote_trade_rows: list[dict[str, object]] = []
    priced_decision_rows = 0
    skipped_missing_price = 0
    skipped_stale_price = 0
    skipped_non_positive_edge = 0

    for _, row in test_frame.iterrows():
        spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
        forecast = predict_market(Path(artifact.path), model_name, spec, row.to_frame().T)
        winning_label = str(row["winning_outcome"])
        top_label, top_probability = max(
            forecast.outcome_probabilities.items(),
            key=lambda item: item[1],
        )
        prediction_rows.append(
            {
                "target_date": row["target_date"],
                "city": spec.city,
                "y_true": row["realized_daily_max"],
                "y_pred": forecast.mean,
                "std": forecast.std,
                "brier": brier_score(forecast.outcome_probabilities, winning_label),
                "crps": crps_from_samples(pd.Series(forecast.samples).to_numpy(), float(row["realized_daily_max"])),
                "top_probability": float(top_probability),
                "top_is_correct": float(top_label == winning_label),
            }
        )

        best_real_edge = 0.0
        best_real_candidate: dict[str, float | str] | None = None
        best_quote_edge = 0.0
        best_quote_candidate: dict[str, float | str] | None = None
        has_covered = False
        has_stale = False
        for outcome_label, fair_probability in forecast.outcome_probabilities.items():
            panel_key = (str(spec.market_id), str(row["decision_horizon"]), str(outcome_label))
            panel_row = lookup.get(panel_key)
            if panel_row is None:
                continue
            coverage = str(panel_row["coverage_status"])
            if coverage == "stale":
                has_stale = True
            if coverage != "ok":
                continue
            has_covered = True
            market_price = float(panel_row["market_price"])
            real_edge = compute_edge(
                float(fair_probability),
                market_price,
                estimate_fee(market_price, taker_bps=default_fee_bps),
                0.0,
            )
            if real_edge > best_real_edge:
                best_real_edge = real_edge
                best_real_candidate = {
                    "outcome_label": outcome_label,
                    "price": market_price,
                    "edge": real_edge,
                }
            _, quote_ask = _quote_proxy_prices(market_price, half_spread=quote_proxy_half_spread)
            quote_edge = compute_edge(
                float(fair_probability),
                quote_ask,
                estimate_fee(quote_ask, taker_bps=default_fee_bps),
                0.0,
            )
            if quote_edge > best_quote_edge:
                best_quote_edge = quote_edge
                best_quote_candidate = {
                    "outcome_label": outcome_label,
                    "price": quote_ask,
                    "edge": quote_edge,
                }

        if not has_covered:
            if has_stale:
                skipped_stale_price += 1
            else:
                skipped_missing_price += 1
            continue

        priced_decision_rows += 1
        if best_real_candidate is None:
            skipped_non_positive_edge += 1
        else:
            real_trade_rows.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "decision_horizon": str(row["decision_horizon"]),
                    "outcome_label": str(best_real_candidate["outcome_label"]),
                    "winning_outcome": winning_label,
                    "price": float(best_real_candidate["price"]),
                    "size": flat_stake / max(float(best_real_candidate["price"]), 1e-6),
                    "edge": float(best_real_candidate["edge"]),
                    "realized_pnl": settle_position(
                        Position(
                            outcome_label=str(best_real_candidate["outcome_label"]),
                            price=float(best_real_candidate["price"]),
                            size=flat_stake / max(float(best_real_candidate["price"]), 1e-6),
                            side="buy",
                        ),
                        winning_label,
                        fee_paid=estimate_fee(flat_stake, taker_bps=default_fee_bps),
                    ),
                    "pricing_source": "real_history",
                }
            )
        if best_quote_candidate is not None:
            quote_trade_rows.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "decision_horizon": str(row["decision_horizon"]),
                    "outcome_label": str(best_quote_candidate["outcome_label"]),
                    "winning_outcome": winning_label,
                    "price": float(best_quote_candidate["price"]),
                    "size": flat_stake / max(float(best_quote_candidate["price"]), 1e-6),
                    "edge": float(best_quote_candidate["edge"]),
                    "realized_pnl": settle_position(
                        Position(
                            outcome_label=str(best_quote_candidate["outcome_label"]),
                            price=float(best_quote_candidate["price"]),
                            size=flat_stake / max(float(best_quote_candidate["price"]), 1e-6),
                            side="buy",
                        ),
                        winning_label,
                        fee_paid=estimate_fee(flat_stake, taker_bps=default_fee_bps),
                    ),
                    "pricing_source": "quote_proxy",
                }
            )

    extra_metrics = {
        "dataset_rows": float(len(frame)),
        "train_rows": float(len(train_frame)),
        "test_rows": float(len(test_frame)),
        "total_groups": float(total_groups),
        "test_groups": float(test_groups),
        "priced_decision_rows": float(priced_decision_rows),
        "skipped_missing_price": float(skipped_missing_price),
        "skipped_stale_price": float(skipped_stale_price),
        "skipped_non_positive_edge": float(skipped_non_positive_edge),
    }
    real_metrics, _, _ = _summarize_backtest_metrics(
        prediction_rows,
        real_trade_rows,
        extra_metrics=extra_metrics,
    )
    quote_metrics, _, _ = _summarize_backtest_metrics(
        prediction_rows,
        quote_trade_rows,
        extra_metrics=extra_metrics,
    )
    metadata = {
        "artifact_path": artifact.path,
        "artifact_variant": artifact.variant,
        "artifact_status": artifact.status,
        "artifact_diagnostics": artifact.diagnostics,
        "calibration_path": artifact.calibration_path,
    }
    return real_metrics, quote_metrics, metadata


def _benchmark_row(
    *,
    model_name: str,
    seeds: list[int],
    real_history_runs: list[dict[str, float]],
    quote_proxy_runs: list[dict[str, float]],
) -> dict[str, object]:
    """Aggregate benchmark metrics for one candidate model."""

    row: dict[str, object] = {
        "model_name": model_name,
        "seed_count": len(seeds),
        "seeds": list(seeds),
    }
    for key in ("mae", "rmse", "nll", "avg_brier", "avg_crps", "calibration_gap"):
        mean_value, std_value = _mean_std([float(run.get(key, 0.0)) for run in real_history_runs])
        row[f"{key}_mean"] = mean_value
        row[f"{key}_std"] = std_value
    for prefix, runs in (("real_history", real_history_runs), ("quote_proxy", quote_proxy_runs)):
        for key in ("num_trades", "pnl", "hit_rate", "avg_edge", "priced_decision_rows"):
            mean_value, std_value = _mean_std([float(run.get(key, 0.0)) for run in runs])
            row[f"{prefix}_{key}_mean"] = mean_value
            row[f"{prefix}_{key}_std"] = std_value
    row["sample_adequacy_passed"] = bool(
        float(row.get("real_history_num_trades_mean", 0.0)) > 0
        and float(row.get("quote_proxy_num_trades_mean", 0.0)) > 0
        and float(row.get("real_history_priced_decision_rows_mean", 0.0)) > 0
    )
    return row


def _ablation_row(
    *,
    model_name: str,
    variant: str,
    split_policy: Literal["market_day", "target_day"],
    seeds: list[int],
    real_history_runs: list[dict[str, float]],
    quote_proxy_runs: list[dict[str, float]],
    metadata_runs: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate grouped-holdout ablation metrics for one model variant."""

    row = _benchmark_row(
        model_name=_model_variant_label(model_name, variant),
        seeds=seeds,
        real_history_runs=real_history_runs,
        quote_proxy_runs=quote_proxy_runs,
    )
    row["model_family"] = model_name
    row["variant"] = variant
    row["split_policy"] = split_policy

    diagnostics_keys = sorted(
        {
            key
            for metadata in metadata_runs
            for key, value in dict(metadata.get("artifact_diagnostics", {})).items()
            if isinstance(value, (int, float))
        }
    )
    for key in diagnostics_keys:
        mean_value, std_value = _mean_std(
            [
                float(dict(metadata.get("artifact_diagnostics", {})).get(key, 0.0))
                for metadata in metadata_runs
            ]
        )
        row[f"diag_{key}_mean"] = mean_value
        row[f"diag_{key}_std"] = std_value
    row["calibration_available"] = all(bool(metadata.get("calibration_path")) for metadata in metadata_runs)
    row["artifact_status"] = str(metadata_runs[0].get("artifact_status", "experimental")) if metadata_runs else "experimental"
    return row


def _write_leaderboard_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist leaderboard rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _publish_model_alias(
    *,
    alias_name: str,
    selected_model_name: str,
    frame: pd.DataFrame,
    artifacts_dir: Path,
    split_policy: Literal["market_day", "target_day"],
    seed: int,
    leaderboard_path: Path,
) -> dict[str, object]:
    """Train and publish one public model alias."""

    artifact = train_model(
        selected_model_name,
        frame,
        artifacts_dir,
        split_policy=split_policy,
        seed=seed,
    )
    source_model_path = Path(artifact.path)
    alias_path = _default_model_path(alias_name)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_model_path, alias_path)

    if artifact.calibration_path is None:
        msg = f"Alias {alias_name} for {selected_model_name} has no calibrator artifact and cannot be published."
        raise typer.BadParameter(msg)
    alias_calibrator_path = alias_path.with_name(f"{alias_path.stem}.calibrator.pkl")
    shutil.copyfile(Path(artifact.calibration_path), alias_calibrator_path)

    metadata = {
        "alias_name": alias_name,
        "model_name": selected_model_name,
        "alias_path": str(alias_path),
        "alias_calibration_path": str(alias_calibrator_path),
        "source_model_path": str(source_model_path),
        "source_calibration_path": artifact.calibration_path,
        "split_policy": split_policy,
        "seed": seed,
        "contract_version": "v2",
        "leaderboard_path": str(leaderboard_path),
        "published_at": datetime.now(tz=UTC).isoformat(),
    }
    dump_json(_default_alias_metadata_path(alias_name), metadata)
    return metadata


@app.command("benchmark-ablations")
def benchmark_ablations(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    panel_path: Path = Path("data/parquet/gold/v2/historical_backtest_panel.parquet"),
    model_name: str = "tuned_ensemble",
    variants: Annotated[list[str] | None, typer.Option("--variant")] = None,
    split_policies: Annotated[list[str] | None, typer.Option("--split-policy")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    artifacts_dir: Path = Path("artifacts/models/v2/ablations"),
    flat_stake: float = 1.0,
    quote_proxy_half_spread: float = 0.02,
    leaderboard_output: Path | None = None,
    leaderboard_csv_output: Path | None = None,
    summary_output: Path | None = None,
) -> None:
    """Benchmark internal ablation variants with a grouped one-shot holdout."""

    config, _ = load_settings()
    family = require_supported_model_name(model_name)
    available_variants = supported_ablation_variants(family)
    if not available_variants:
        msg = f"Model {family} has no registered ablation variants."
        raise typer.BadParameter(msg)

    selected_variants = list(dict.fromkeys(variants or list(available_variants)))
    for variant in selected_variants:
        require_supported_variant(family, variant)

    resolved_split_policies = [str(value) for value in (split_policies or ["market_day", "target_day"])]
    for split_policy in resolved_split_policies:
        if split_policy not in {"market_day", "target_day"}:
            raise typer.BadParameter(f"Unsupported split_policy: {split_policy}")
    benchmark_seeds = list(dict.fromkeys(seeds or [config.app.random_seed]))

    frame = pd.read_parquet(dataset_path)
    if not panel_path.exists():
        msg = (
            f"Backtest panel does not exist: {panel_path}. "
            "Run `uv run pmtmax materialize-backtest-panel` first."
        )
        raise typer.BadParameter(msg)
    panel = pd.read_parquet(panel_path)
    if panel.empty or not (panel["coverage_status"].astype(str) == "ok").any():
        raise typer.BadParameter("Backtest panel has no coverage_status=ok rows.")

    if leaderboard_output is None:
        leaderboard_output = _default_benchmark_output(f"{family}_ablation_leaderboard.json")
    if leaderboard_csv_output is None:
        leaderboard_csv_output = _default_benchmark_output(f"{family}_ablation_leaderboard.csv")
    if summary_output is None:
        summary_output = _default_benchmark_output(f"{family}_ablation_summary.json")

    rows: list[dict[str, object]] = []
    for split_policy in resolved_split_policies:
        for variant in selected_variants:
            real_history_runs: list[dict[str, float]] = []
            quote_proxy_runs: list[dict[str, float]] = []
            metadata_runs: list[dict[str, object]] = []
            for seed in benchmark_seeds:
                variant_artifacts_dir = artifacts_dir / family / split_policy / variant / f"seed_{seed}"
                real_metrics, quote_metrics, metadata = _run_grouped_holdout_ablation(
                    frame,
                    panel,
                    model_name=family,
                    variant=variant,
                    artifacts_dir=variant_artifacts_dir,
                    flat_stake=flat_stake,
                    default_fee_bps=_default_fee_bps(config),
                    quote_proxy_half_spread=quote_proxy_half_spread,
                    split_policy=split_policy,  # type: ignore[arg-type]
                    seed=seed,
                )
                real_history_runs.append(real_metrics)
                quote_proxy_runs.append(quote_metrics)
                metadata_runs.append(metadata)
            rows.append(
                _ablation_row(
                    model_name=family,
                    variant=variant,
                    split_policy=split_policy,  # type: ignore[arg-type]
                    seeds=benchmark_seeds,
                    real_history_runs=real_history_runs,
                    quote_proxy_runs=quote_proxy_runs,
                    metadata_runs=metadata_runs,
                )
            )

    leaderboard_frame = pd.DataFrame(rows).sort_values(
        [
            "split_policy",
            "avg_crps_mean",
            "avg_brier_mean",
            "real_history_pnl_mean",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    leaderboard_rows = leaderboard_frame.to_dict(orient="records")
    dump_json(leaderboard_output, leaderboard_rows)
    _write_leaderboard_csv(leaderboard_csv_output, leaderboard_rows)

    summary = {
        "model_family": family,
        "variants": selected_variants,
        "split_policies": resolved_split_policies,
        "seeds": benchmark_seeds,
        "dataset_path": str(dataset_path),
        "panel_path": str(panel_path),
        "leaderboard_path": str(leaderboard_output),
        "leaderboard_csv_path": str(leaderboard_csv_output),
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    dump_json(summary_output, summary)
    console.print_json(data=summary)


@app.command("benchmark-models")
def benchmark_models(
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    panel_path: Path = Path("data/parquet/gold/v2/historical_backtest_panel.parquet"),
    models: Annotated[list[str] | None, typer.Option("--model")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    artifacts_dir: Path = Path("artifacts/models/v2"),
    flat_stake: float = 1.0,
    quote_proxy_half_spread: float = 0.02,
    split_policy: Literal["market_day", "target_day"] = "market_day",
    publish_champion: bool = True,
    publish_trading_champion: bool = True,
    leaderboard_output: Path = Path("artifacts/benchmarks/v2/leaderboard.json"),
    leaderboard_csv_output: Path = Path("artifacts/benchmarks/v2/leaderboard.csv"),
    summary_output: Path = Path("artifacts/benchmarks/v2/benchmark_summary.json"),
    retrain_stride: int = 30,
) -> None:
    """Benchmark the canonical v2 models and optionally publish the champion alias."""

    config, _ = load_settings()
    frame = pd.read_parquet(dataset_path)
    if not panel_path.exists():
        msg = (
            f"Backtest panel does not exist: {panel_path}. "
            "Run `uv run pmtmax materialize-backtest-panel` first."
        )
        raise typer.BadParameter(msg)
    panel = pd.read_parquet(panel_path)
    if panel.empty or not (panel["coverage_status"].astype(str) == "ok").any():
        raise typer.BadParameter("Backtest panel has no coverage_status=ok rows.")

    candidate_models = models or list(config.models.benchmark_ladder)
    ordered_models = list(dict.fromkeys(require_supported_model_name(name) for name in candidate_models))
    benchmark_seeds = list(dict.fromkeys(seeds or [config.app.random_seed]))

    rows: list[dict[str, object]] = []
    for model_name in ordered_models:
        real_history_runs: list[dict[str, float]] = []
        quote_proxy_runs: list[dict[str, float]] = []
        for seed in benchmark_seeds:
            real_history_metrics, _ = _run_real_history_backtest(
                frame,
                panel,
                model_name=model_name,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=_default_fee_bps(config),
                split_policy=split_policy,
                seed=seed,
                retrain_stride=retrain_stride,
            )
            quote_proxy_metrics, _ = _run_quote_proxy_backtest(
                frame,
                panel,
                model_name=model_name,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=_default_fee_bps(config),
                quote_proxy_half_spread=quote_proxy_half_spread,
                split_policy=split_policy,
                seed=seed,
                retrain_stride=retrain_stride,
            )
            real_history_runs.append(real_history_metrics)
            quote_proxy_runs.append(quote_proxy_metrics)
        rows.append(
            _benchmark_row(
                model_name=model_name,
                seeds=benchmark_seeds,
                real_history_runs=real_history_runs,
                quote_proxy_runs=quote_proxy_runs,
            )
        )

    results_frame = pd.DataFrame(rows)
    leaderboard_frame = score_leaderboard(results_frame)
    trading_scores = score_trading_leaderboard(results_frame)[["model_name", "trading_champion_score"]]
    leaderboard_frame = leaderboard_frame.merge(trading_scores, on="model_name", how="left")
    champion_name = select_champion(results_frame)
    trading_champion_name = select_trading_champion(results_frame)
    leaderboard_rows = leaderboard_frame.to_dict(orient="records")
    dump_json(leaderboard_output, leaderboard_rows)
    _write_leaderboard_csv(leaderboard_csv_output, leaderboard_rows)

    champion_metadata: dict[str, object] | None = None
    if publish_champion:
        champion_metadata = _publish_model_alias(
            alias_name=DEFAULT_MODEL_NAME,
            selected_model_name=champion_name,
            frame=frame,
            artifacts_dir=artifacts_dir,
            split_policy=split_policy,
            seed=benchmark_seeds[0],
            leaderboard_path=leaderboard_output,
        )
    trading_champion_metadata: dict[str, object] | None = None
    if publish_trading_champion:
        trading_champion_metadata = _publish_model_alias(
            alias_name=TRADING_MODEL_ALIAS,
            selected_model_name=trading_champion_name,
            frame=frame,
            artifacts_dir=artifacts_dir,
            split_policy=split_policy,
            seed=benchmark_seeds[0],
            leaderboard_path=leaderboard_output,
        )

    summary = {
        "contract_version": "v2",
        "split_policy": split_policy,
        "candidate_models": ordered_models,
        "supported_models": list(supported_model_names()),
        "seeds": benchmark_seeds,
        "leaderboard_path": str(leaderboard_output),
        "leaderboard_csv_path": str(leaderboard_csv_output),
        "champion_model_name": champion_name,
        "champion_published": publish_champion,
        "champion_metadata": champion_metadata,
        "trading_champion_model_name": trading_champion_name,
        "trading_champion_published": publish_trading_champion,
        "trading_champion_metadata": trading_champion_metadata,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    dump_json(summary_output, summary)
    console.print_json(data=summary)


def _quick_eval_deltas(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict[str, float]:
    """Return candidate-minus-baseline deltas for shared quick-eval metrics."""

    keys = sorted(set(baseline_metrics).intersection(candidate_metrics))
    return {
        f"{key}_delta": float(candidate_metrics[key] - baseline_metrics[key])
        for key in keys
        if key != "n"
    }


def _quick_eval_candidate_beats_baseline(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> bool:
    """Return whether one candidate wins the quick-eval lexicographic comparison."""

    baseline_key = (
        float(baseline_metrics.get("crps", float("inf"))),
        float(baseline_metrics.get("brier", float("inf"))),
        float(baseline_metrics.get("mae", float("inf"))),
    )
    candidate_key = (
        float(candidate_metrics.get("crps", float("inf"))),
        float(candidate_metrics.get("brier", float("inf"))),
        float(candidate_metrics.get("mae", float("inf"))),
    )
    return candidate_key < baseline_key


def _paper_overall_gate_decision(
    *,
    paper_summary: dict[str, object],
    opportunity_summary: dict[str, object],
    shadow_summary: dict[str, object],
    open_phase_summary: dict[str, object],
    hope_hunt_summary: dict[str, object],
) -> str:
    """Collapse paper-analysis outputs into GO / INCONCLUSIVE / NO_GO."""

    tradable_total = int(paper_summary.get("tradable_count", 0)) + int(opportunity_summary.get("tradable_count", 0))
    shadow_tradable = int(dict(shadow_summary.get("reason_counts", {})).get("tradable", 0))
    open_phase_candidates = int(open_phase_summary.get("candidate_count", 0))
    hope_candidates = int(hope_hunt_summary.get("candidate_count", 0))
    if tradable_total > 0 or shadow_tradable > 0 or open_phase_candidates > 0 or hope_candidates > 0:
        return "GO"

    hard_reasons = {"missing_calibrator", "forecast_failed", "model_error"}
    hard_failures = 0
    for summary in (paper_summary, opportunity_summary, shadow_summary, open_phase_summary, hope_hunt_summary):
        reason_counts = summary.get("reason_counts", {})
        if isinstance(reason_counts, dict):
            hard_failures += sum(int(reason_counts.get(reason, 0)) for reason in hard_reasons)
    if hard_failures > 0:
        return "NO_GO"
    return "INCONCLUSIVE"


@app.command("autoresearch-init")
def autoresearch_init(
    run_tag: str | None = None,
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    panel_path: Path = Path("data/parquet/gold/v2/historical_backtest_panel.parquet"),
    baseline_variant: str = "recency_neighbor_oof",
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
    force: bool = False,
) -> None:
    """Create one autoresearch run scaffold plus the shared program.md."""

    run_tag = str(_resolve_option_value(run_tag, default_autoresearch_run_tag()))
    dataset_path = _resolve_option_value(dataset_path, Path("data/parquet/gold/v2/historical_training_set.parquet"))
    panel_path = _resolve_option_value(panel_path, Path("data/parquet/gold/v2/historical_backtest_panel.parquet"))
    baseline_variant = str(_resolve_option_value(baseline_variant, "recency_neighbor_oof"))
    root_dir = _resolve_option_value(root_dir, DEFAULT_AUTORESEARCH_ROOT)
    force = bool(_resolve_option_value(force, False))
    require_supported_variant("lgbm_emos", baseline_variant)

    run_dir = autoresearch_run_dir(run_tag, root_dir=root_dir)
    if run_dir.exists() and not force:
        raise typer.BadParameter(f"Autoresearch run already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    autoresearch_candidates_dir(run_tag, root_dir=root_dir).mkdir(parents=True, exist_ok=True)
    autoresearch_models_dir(run_tag, root_dir=root_dir).mkdir(parents=True, exist_ok=True)
    autoresearch_analysis_dir(run_tag, root_dir=root_dir).mkdir(parents=True, exist_ok=True)

    manifest = AutoresearchManifest(
        run_tag=run_tag,
        baseline_variant=baseline_variant,
        root_dir=str(run_dir),
        dataset_path=str(dataset_path),
        dataset_signature=path_signature(dataset_path),
        panel_path=str(panel_path),
        panel_signature=path_signature(panel_path),
        created_at=datetime.now(tz=UTC),
        current_champion_variant=_current_alias_variant(DEFAULT_MODEL_NAME),
        current_trading_variant=_current_alias_variant(TRADING_MODEL_ALIAS),
    )
    dump_json(autoresearch_manifest_path(run_tag, root_dir=root_dir), manifest.model_dump(mode="json"))
    autoresearch_program_path(run_tag, root_dir=root_dir).write_text(render_autoresearch_program(manifest))
    (autoresearch_candidates_dir(run_tag, root_dir=root_dir) / "candidate_template.yaml").write_text(
        render_candidate_template(run_tag, baseline_variant=baseline_variant)
    )
    autoresearch_results_path(run_tag, root_dir=root_dir).parent.mkdir(parents=True, exist_ok=True)
    autoresearch_results_path(run_tag, root_dir=root_dir).touch(exist_ok=True)
    console.print_json(data=manifest.model_dump(mode="json"))


@app.command("autoresearch-step")
def autoresearch_step(
    spec_path: Path,
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
) -> None:
    """Train one candidate spec, run quick eval, and append keep/discard/crash to the ledger."""

    spec_path = _resolve_option_value(spec_path)
    dataset_path = _resolve_option_value(dataset_path, Path("data/parquet/gold/v2/historical_training_set.parquet"))
    root_dir = _resolve_option_value(root_dir, DEFAULT_AUTORESEARCH_ROOT)
    spec = load_lgbm_autoresearch_spec(spec_path)
    manifest = _load_autoresearch_manifest(spec.run_tag, root_dir=root_dir)
    if manifest.dataset_signature != path_signature(dataset_path):
        raise typer.BadParameter("Dataset signature does not match the autoresearch manifest.")

    run_tag = spec.run_tag
    normalized_spec_path = autoresearch_candidates_dir(run_tag, root_dir=root_dir) / f"{spec.candidate_name}.yaml"
    save_lgbm_autoresearch_spec(normalized_spec_path, spec)

    config, _ = load_settings()
    frame = pd.read_parquet(dataset_path)
    models_dir = autoresearch_models_dir(run_tag, root_dir=root_dir)
    baseline_path = models_dir / f"lgbm_emos__{manifest.baseline_variant}.pkl"
    if not baseline_path.exists():
        baseline_artifact = train_model(
            "lgbm_emos",
            frame,
            models_dir,
            split_policy="market_day",
            seed=config.app.random_seed,
            variant=manifest.baseline_variant,
        )
        baseline_path = Path(baseline_artifact.path)

    _, holdout = quick_eval_holdout(frame)
    baseline_metrics = evaluate_saved_model(baseline_path, holdout)
    if baseline_metrics is None:
        raise typer.BadParameter(f"Quick eval failed for baseline artifact: {baseline_path}")

    candidate_variant = spec.build_variant_config()
    candidate_path = models_dir / f"lgbm_emos__{spec.candidate_name}.pkl"
    candidate_metrics: dict[str, float] = {}
    status: Literal["keep", "discard", "crash"]
    notes = ""
    try:
        candidate_artifact = train_model(
            "lgbm_emos",
            frame,
            models_dir,
            split_policy="market_day",
            seed=config.app.random_seed,
            variant=spec.candidate_name,
            variant_config=candidate_variant,
        )
        candidate_path = Path(candidate_artifact.path)
        loaded_candidate_metrics = evaluate_saved_model(candidate_path, holdout)
        if loaded_candidate_metrics is None:
            raise RuntimeError(f"Quick eval failed for candidate artifact: {candidate_path}")
        candidate_metrics = loaded_candidate_metrics
        status = "keep" if _quick_eval_candidate_beats_baseline(baseline_metrics, candidate_metrics) else "discard"
    except Exception as exc:
        status = "crash"
        notes = str(exc)

    result = AutoresearchStepResult(
        run_tag=run_tag,
        candidate_name=spec.candidate_name,
        baseline_variant=manifest.baseline_variant,
        evaluated_at=datetime.now(tz=UTC),
        status=status,
        dataset_signature=manifest.dataset_signature,
        baseline_artifact_path=str(baseline_path),
        candidate_artifact_path=str(candidate_path),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        metric_deltas=_quick_eval_deltas(baseline_metrics, candidate_metrics) if candidate_metrics else {},
        notes=notes,
    )
    output_path = autoresearch_analysis_dir(run_tag, root_dir=root_dir) / f"quick_eval__{spec.candidate_name}.json"
    dump_json(output_path, result.model_dump(mode="json"))
    _append_jsonl(autoresearch_results_path(run_tag, root_dir=root_dir), result.model_dump(mode="json"))
    console.print_json(data=result.model_dump(mode="json"))


@app.command("autoresearch-gate")
def autoresearch_gate(
    spec_path: Path,
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    panel_path: Path = Path("data/parquet/gold/v2/historical_backtest_panel.parquet"),
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
    split_policies: Annotated[list[str] | None, typer.Option("--split-policy")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    flat_stake: float = 1.0,
    quote_proxy_half_spread: float = 0.02,
) -> None:
    """Run grouped-holdout benchmark gates for one baseline/candidate pair."""

    spec_path = _resolve_option_value(spec_path)
    dataset_path = _resolve_option_value(dataset_path, Path("data/parquet/gold/v2/historical_training_set.parquet"))
    panel_path = _resolve_option_value(panel_path, Path("data/parquet/gold/v2/historical_backtest_panel.parquet"))
    root_dir = _resolve_option_value(root_dir, DEFAULT_AUTORESEARCH_ROOT)
    split_policies = _resolve_option_value(split_policies)
    seeds = _resolve_option_value(seeds)
    spec = load_lgbm_autoresearch_spec(spec_path)
    manifest = _load_autoresearch_manifest(spec.run_tag, root_dir=root_dir)
    if manifest.dataset_signature != path_signature(dataset_path):
        raise typer.BadParameter("Dataset signature does not match the autoresearch manifest.")
    if manifest.panel_signature != path_signature(panel_path):
        raise typer.BadParameter("Panel signature does not match the autoresearch manifest.")

    config, _ = load_settings()
    benchmark_seeds = list(dict.fromkeys(seeds or [config.app.random_seed]))
    resolved_split_policies = [str(value) for value in (split_policies or ["market_day", "target_day"])]
    for split_policy in resolved_split_policies:
        if split_policy not in {"market_day", "target_day"}:
            raise typer.BadParameter(f"Unsupported split_policy: {split_policy}")

    frame = pd.read_parquet(dataset_path)
    panel = pd.read_parquet(panel_path)
    artifacts_dir = autoresearch_models_dir(spec.run_tag, root_dir=root_dir) / "gate"
    rows: list[dict[str, object]] = []
    candidate_config = spec.build_variant_config()
    for split_policy in resolved_split_policies:
        for variant_name, variant_config in (
            (manifest.baseline_variant, None),
            (spec.candidate_name, candidate_config),
        ):
            real_runs: list[dict[str, float]] = []
            quote_runs: list[dict[str, float]] = []
            metadata_runs: list[dict[str, object]] = []
            for seed in benchmark_seeds:
                real_metrics, quote_metrics, metadata = _run_grouped_holdout_ablation(
                    frame,
                    panel,
                    model_name="lgbm_emos",
                    variant=variant_name,
                    variant_config=variant_config,
                    artifacts_dir=artifacts_dir / split_policy / variant_name / f"seed_{seed}",
                    flat_stake=flat_stake,
                    default_fee_bps=_default_fee_bps(config),
                    quote_proxy_half_spread=quote_proxy_half_spread,
                    split_policy=split_policy,  # type: ignore[arg-type]
                    seed=seed,
                )
                real_runs.append(real_metrics)
                quote_runs.append(quote_metrics)
                metadata_runs.append(metadata)
            rows.append(
                _ablation_row(
                    model_name="lgbm_emos",
                    variant=variant_name,
                    split_policy=split_policy,  # type: ignore[arg-type]
                    seeds=benchmark_seeds,
                    real_history_runs=real_runs,
                    quote_proxy_runs=quote_runs,
                    metadata_runs=metadata_runs,
                )
            )

    leaderboard_rows = sorted(
        rows,
        key=lambda row: (
            str(row["split_policy"]),
            float(row["avg_crps_mean"]),
            float(row["avg_brier_mean"]),
        ),
    )
    gate_leaderboard_path = autoresearch_analysis_dir(spec.run_tag, root_dir=root_dir) / f"gate_leaderboard__{spec.candidate_name}.json"
    gate_csv_path = gate_leaderboard_path.with_suffix(".csv")
    dump_json(gate_leaderboard_path, leaderboard_rows)
    _write_leaderboard_csv(gate_csv_path, leaderboard_rows)

    benchmark_gate_details: dict[str, dict[str, float | bool]] = {}
    benchmark_gate_passed = True
    for split_policy in resolved_split_policies:
        baseline_row = next(row for row in leaderboard_rows if row["split_policy"] == split_policy and row["variant"] == manifest.baseline_variant)
        candidate_row = next(row for row in leaderboard_rows if row["split_policy"] == split_policy and row["variant"] == spec.candidate_name)
        passed = (
            float(candidate_row["avg_crps_mean"]) < float(baseline_row["avg_crps_mean"])
            and bool(candidate_row.get("calibration_available", False))
        )
        benchmark_gate_passed = benchmark_gate_passed and passed
        benchmark_gate_details[split_policy] = {
            "passed": passed,
            "baseline_avg_crps_mean": float(baseline_row["avg_crps_mean"]),
            "candidate_avg_crps_mean": float(candidate_row["avg_crps_mean"]),
            "baseline_real_history_pnl_mean": float(baseline_row["real_history_pnl_mean"]),
            "candidate_real_history_pnl_mean": float(candidate_row["real_history_pnl_mean"]),
        }

    summary = {
        "run_tag": spec.run_tag,
        "candidate_name": spec.candidate_name,
        "baseline_variant": manifest.baseline_variant,
        "dataset_signature": manifest.dataset_signature,
        "panel_signature": manifest.panel_signature,
        "seeds": benchmark_seeds,
        "split_policies": resolved_split_policies,
        "leaderboard_path": str(gate_leaderboard_path),
        "leaderboard_csv_path": str(gate_csv_path),
        "benchmark_gate_passed": benchmark_gate_passed,
        "benchmark_gate_details": benchmark_gate_details,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    dump_json(
        autoresearch_analysis_dir(spec.run_tag, root_dir=root_dir) / f"gate_summary__{spec.candidate_name}.json",
        summary,
    )
    console.print_json(data=summary)


@app.command("autoresearch-analyze-paper")
def autoresearch_analyze_paper(
    spec_path: Path,
    dataset_path: Path = Path("data/parquet/gold/v2/historical_training_set.parquet"),
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
) -> None:
    """Run paper and live-shadow diagnostics for one autoresearch candidate artifact."""

    spec_path = _resolve_option_value(spec_path)
    dataset_path = _resolve_option_value(dataset_path, Path("data/parquet/gold/v2/historical_training_set.parquet"))
    root_dir = _resolve_option_value(root_dir, DEFAULT_AUTORESEARCH_ROOT)
    spec = load_lgbm_autoresearch_spec(spec_path)
    manifest = _load_autoresearch_manifest(spec.run_tag, root_dir=root_dir)
    if manifest.dataset_signature != path_signature(dataset_path):
        raise typer.BadParameter("Dataset signature does not match the autoresearch manifest.")

    config, _ = load_settings()
    run_tag = spec.run_tag
    models_dir = autoresearch_models_dir(run_tag, root_dir=root_dir)
    candidate_artifact_path = models_dir / f"lgbm_emos__{spec.candidate_name}.pkl"
    if not candidate_artifact_path.exists():
        frame = pd.read_parquet(dataset_path)
        candidate_artifact = train_model(
            "lgbm_emos",
            frame,
            models_dir,
            split_policy="market_day",
            seed=config.app.random_seed,
            variant=spec.candidate_name,
            variant_config=spec.build_variant_config(),
        )
        candidate_artifact_path = Path(candidate_artifact.path)

    paper_dir = autoresearch_analysis_dir(run_tag, root_dir=root_dir) / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    paper_signals_path = paper_dir / "paper_signals_recent_core.json"
    paper_trader(
        model_path=candidate_artifact_path,
        model_name="lgbm_emos",
        core_recent_only=True,
        output=paper_signals_path,
    )

    opportunity_report_path = paper_dir / "opportunity_report_recent_core.json"
    opportunity_report(
        model_path=candidate_artifact_path,
        model_name="lgbm_emos",
        market_scope="recent_core",
        output=opportunity_report_path,
    )

    shadow_history_path = paper_dir / "opportunity_shadow_recent_core.jsonl"
    shadow_latest_path = paper_dir / "opportunity_shadow_recent_core_latest.json"
    shadow_summary_path = paper_dir / "opportunity_shadow_recent_core_summary.json"
    shadow_state_path = paper_dir / "opportunity_shadow_recent_core_state.json"
    opportunity_shadow(
        model_path=candidate_artifact_path,
        model_name="lgbm_emos",
        market_scope="recent_core",
        max_cycles=1,
        output=shadow_history_path,
        latest_output=shadow_latest_path,
        summary_output=shadow_summary_path,
        state_path=shadow_state_path,
    )

    open_phase_history_path = paper_dir / "open_phase_shadow_supported_wu_open_phase.jsonl"
    open_phase_latest_path = paper_dir / "open_phase_shadow_supported_wu_open_phase_latest.json"
    open_phase_summary_path = paper_dir / "open_phase_shadow_supported_wu_open_phase_summary.json"
    open_phase_state_path = paper_dir / "open_phase_shadow_supported_wu_open_phase_state.json"
    open_phase_shadow(
        model_path=candidate_artifact_path,
        model_name="lgbm_emos",
        market_scope="supported_wu_open_phase",
        max_cycles=1,
        output=open_phase_history_path,
        latest_output=open_phase_latest_path,
        summary_output=open_phase_summary_path,
        state_path=open_phase_state_path,
    )

    hope_hunt_latest_path = paper_dir / "hope_hunt_supported_wu_open_phase_latest.json"
    hope_hunt_history_path = paper_dir / "hope_hunt_supported_wu_open_phase_history.jsonl"
    hope_hunt_summary_path = paper_dir / "hope_hunt_supported_wu_open_phase_summary.json"
    hope_hunt_report(
        model_path=candidate_artifact_path,
        model_name="lgbm_emos",
        market_scope="supported_wu_open_phase",
        output=hope_hunt_latest_path,
        history_output=hope_hunt_history_path,
        summary_output=hope_hunt_summary_path,
    )

    paper_rows = load_json(paper_signals_path)
    opportunity_rows = load_json(opportunity_report_path)
    paper_summary = _summarize_reason_rows(paper_rows if isinstance(paper_rows, list) else [])
    opportunity_summary = _summarize_reason_rows(opportunity_rows if isinstance(opportunity_rows, list) else [])
    shadow_summary = _load_optional_json(shadow_summary_path) or {}
    open_phase_summary = _load_optional_json(open_phase_summary_path) or {}
    hope_hunt_summary = _load_optional_json(hope_hunt_summary_path) or {}

    summary = {
        "run_tag": run_tag,
        "candidate_name": spec.candidate_name,
        "model_path": str(candidate_artifact_path),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "analysis_completed": True,
        "paper_recent_core": paper_summary,
        "opportunity_recent_core": opportunity_summary,
        "opportunity_shadow_recent_core": shadow_summary,
        "open_phase_supported_wu_open_phase": open_phase_summary,
        "hope_hunt_supported_wu_open_phase": hope_hunt_summary,
    }
    summary["overall_gate_decision"] = _paper_overall_gate_decision(
        paper_summary=paper_summary,
        opportunity_summary=opportunity_summary,
        shadow_summary=shadow_summary,
        open_phase_summary=open_phase_summary,
        hope_hunt_summary=hope_hunt_summary,
    )
    dump_json(paper_dir / f"paper_analysis_summary__{spec.candidate_name}.json", summary)
    console.print_json(data=summary)


@app.command("autoresearch-promote")
def autoresearch_promote(
    spec_path: Path,
    root_dir: Path = DEFAULT_AUTORESEARCH_ROOT,
    publish_champion: bool = False,
    publish_trading_champion: bool = False,
    force: bool = False,
) -> None:
    """Promote one gated candidate into the persistent promoted-spec registry."""

    spec_path = _resolve_option_value(spec_path)
    root_dir = _resolve_option_value(root_dir, DEFAULT_AUTORESEARCH_ROOT)
    publish_champion = bool(_resolve_option_value(publish_champion, False))
    publish_trading_champion = bool(_resolve_option_value(publish_trading_champion, False))
    force = bool(_resolve_option_value(force, False))
    spec = load_lgbm_autoresearch_spec(spec_path)

    gate_summary_path = autoresearch_analysis_dir(spec.run_tag, root_dir=root_dir) / f"gate_summary__{spec.candidate_name}.json"
    paper_summary_path = (
        autoresearch_analysis_dir(spec.run_tag, root_dir=root_dir)
        / "paper"
        / f"paper_analysis_summary__{spec.candidate_name}.json"
    )
    if not gate_summary_path.exists():
        raise typer.BadParameter(f"Gate summary does not exist: {gate_summary_path}")
    if not paper_summary_path.exists():
        raise typer.BadParameter(f"Paper analysis summary does not exist: {paper_summary_path}")

    gate_summary = load_json(gate_summary_path)
    paper_summary = load_json(paper_summary_path)
    if not bool(gate_summary.get("benchmark_gate_passed", False)):
        raise typer.BadParameter("Candidate did not pass the benchmark gate.")
    if str(paper_summary.get("overall_gate_decision", "INCONCLUSIVE")) == "NO_GO":
        raise typer.BadParameter("Paper analysis marked this candidate as NO_GO.")

    target_path = promoted_lgbm_emos_spec_path(spec.candidate_name)
    if target_path.exists() and not force:
        raise typer.BadParameter(f"Promoted spec already exists: {target_path}")
    save_lgbm_autoresearch_spec(target_path, spec)

    model_path = autoresearch_models_dir(spec.run_tag, root_dir=root_dir) / f"lgbm_emos__{spec.candidate_name}.pkl"
    if not model_path.exists():
        raise typer.BadParameter(f"Candidate artifact does not exist: {model_path}")
    calibration_path = _artifact_calibration_path(model_path)
    published: dict[str, object] = {}
    if publish_champion:
        published["champion"] = _publish_existing_model_alias(
            alias_name=DEFAULT_MODEL_NAME,
            model_name="lgbm_emos",
            variant=spec.candidate_name,
            source_model_path=model_path,
            source_calibration_path=calibration_path if calibration_path.exists() else None,
            leaderboard_path=gate_summary_path,
        )
    if publish_trading_champion:
        published["trading_champion"] = _publish_existing_model_alias(
            alias_name=TRADING_MODEL_ALIAS,
            model_name="lgbm_emos",
            variant=spec.candidate_name,
            source_model_path=model_path,
            source_calibration_path=calibration_path if calibration_path.exists() else None,
            leaderboard_path=gate_summary_path,
        )

    summary = {
        "run_tag": spec.run_tag,
        "candidate_name": spec.candidate_name,
        "promoted_spec_path": str(target_path),
        "published_aliases": published,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    dump_json(
        autoresearch_analysis_dir(spec.run_tag, root_dir=root_dir) / f"promotion_summary__{spec.candidate_name}.json",
        summary,
    )
    console.print_json(data=summary)


def _load_optional_json(path: Path) -> dict[str, object] | None:
    """Load a JSON object when present, otherwise return None."""

    if not path.exists():
        return None
    payload = load_json(path)
    if isinstance(payload, dict):
        return payload
    return None


def _load_optional_rows(path: Path) -> list[dict[str, object]]:
    """Load a JSON row array when present, otherwise return an empty list."""

    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _path_or_default(value: Path | None, default: Path) -> Path:
    """Resolve optional CLI path arguments against a concrete default path."""

    resolved = _resolve_option_value(value)
    if resolved is None:
        return default
    return resolved


@app.command("revenue-gate-report")
def revenue_gate_report(
    benchmark_summary_path: Path = typer.Option(
        DEFAULT_BENCHMARK_SUMMARY_PATH,
        help="Recent-core benchmark summary JSON",
    ),
    opportunity_summary_path: Path = typer.Option(
        Path("artifacts/signals/v2/shadow_summary.json"),
        help="Opportunity shadow summary JSON",
    ),
    open_phase_summary_path: Path = typer.Option(
        Path("artifacts/signals/v2/open_phase_shadow_summary.json"),
        help="Open-phase shadow summary JSON",
    ),
    observation_summary_path: Path = typer.Option(
        Path("artifacts/signals/v2/observation_shadow_summary.json"),
        help="Observation-station shadow summary JSON",
    ),
    output: Path = typer.Option(
        Path("artifacts/signals/v2/revenue_gate_summary.json"),
        help="Combined revenue gate JSON output",
    ),
    trading_alias_name: str = typer.Option(
        TRADING_MODEL_ALIAS,
        help="Required model alias for live pilot promotion",
    ),
    market_scope: str = typer.Option(
        "recent_core",
        help="Market scope: recent_core or supported_wu_open_phase",
    ),
) -> None:
    """Combine recent-core benchmark and shadow validations into one revenue gate report."""

    benchmark_summary_path = _resolve_option_value(benchmark_summary_path, DEFAULT_BENCHMARK_SUMMARY_PATH)
    opportunity_summary_path = _resolve_option_value(
        opportunity_summary_path,
        Path("artifacts/signals/v2/shadow_summary.json"),
    )
    open_phase_summary_path = _resolve_option_value(
        open_phase_summary_path,
        Path("artifacts/signals/v2/open_phase_shadow_summary.json"),
    )
    observation_summary_path = _resolve_option_value(
        observation_summary_path,
        Path("artifacts/signals/v2/observation_shadow_summary.json"),
    )
    output = _resolve_option_value(output, Path("artifacts/signals/v2/revenue_gate_summary.json"))
    trading_alias_name = _resolve_option_value(trading_alias_name, TRADING_MODEL_ALIAS)
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "recent_core"), core_recent_only=False)
    if market_scope == "supported_wu_open_phase" and open_phase_summary_path == Path(
        "artifacts/signals/v2/open_phase_shadow_summary.json"
    ):
        open_phase_summary_path = _default_signal_output("hope_hunt_summary.json")
    benchmark_summary = _load_optional_json(benchmark_summary_path)
    opportunity_summary = _load_optional_json(opportunity_summary_path)
    open_phase_summary = _load_optional_json(open_phase_summary_path)
    observation_summary = _load_optional_json(observation_summary_path)
    report = build_revenue_gate_report(
        benchmark_summary=benchmark_summary,
        opportunity_summary=opportunity_summary,
        open_phase_summary=open_phase_summary,
        observation_summary=observation_summary,
        trading_alias_name=trading_alias_name,
        pilot_constraints=DEFAULT_PILOT_CONSTRAINTS,
        market_scope=market_scope,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output, report)
    console.print_json(data=report)


def _build_station_dashboard_runner(
    *,
    opportunity_report_path: Path,
    observation_latest_path: Path,
    observation_summary_path: Path,
    queue_path: Path,
    open_phase_latest_path: Path,
    open_phase_summary_path: Path,
    revenue_gate_summary_path: Path,
    watchlist_playbook_path: Path,
    interval_seconds: int,
    max_cycles: int,
    json_output_path: Path,
    html_output_path: Path,
    state_path: Path,
) -> StationDashboardRunner:
    """Build one reusable station-dashboard renderer."""

    config, _env = load_settings()

    def _data_loader() -> dict[str, Any]:
        return {
            "opportunity_rows": _load_optional_rows(opportunity_report_path),
            "observation_rows": _load_optional_rows(observation_latest_path),
            "queue_rows": _load_optional_rows(queue_path),
            "open_phase_rows": _load_optional_rows(open_phase_latest_path),
            "revenue_gate_summary": _load_optional_json(revenue_gate_summary_path),
            "observation_summary": _load_optional_json(observation_summary_path),
            "open_phase_summary": _load_optional_json(open_phase_summary_path),
            "watchlist_playbook": _load_optional_json(watchlist_playbook_path),
        }

    return StationDashboardRunner(
        config=config,
        interval_seconds=interval_seconds,
        max_cycles=max_cycles,
        state_path=state_path,
        json_output_path=json_output_path,
        html_output_path=html_output_path,
        data_loader=_data_loader,
    )


def _run_station_cycle(
    *,
    model_path: Path,
    model_name: str,
    cities: list[str] | None,
    core_recent_only: bool,
    market_scope: str,
    markets_path: Path | None,
    horizon: str,
    min_edge: float | None,
    open_window_hours: float,
    benchmark_summary_path: Path,
    opportunity_report_path: Path,
    opportunity_shadow_summary_path: Path,
    observation_latest_path: Path,
    observation_summary_path: Path,
    observation_alerts_path: Path,
    queue_path: Path,
    open_phase_latest_path: Path,
    open_phase_summary_path: Path,
    revenue_gate_summary_path: Path,
    watchlist_playbook_path: Path,
    dashboard_json_output: Path,
    dashboard_html_output: Path,
    dashboard_state_path: Path,
) -> dict[str, Any]:
    """Run one end-to-end station cycle and return a compact summary."""
    config, _env, http, _duckdb, _parquet, openmeteo = _runtime(include_stores=False)
    shadow_config = config.opportunity_shadow
    observation_config = config.observation_station
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    metar = MetarClient(http, config.metar.base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    scoped_cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    horizon_policy = _load_recent_horizon_policy(DEFAULT_RECENT_HORIZON_POLICY_PATH)
    opportunity_shadow_history_path = _default_signal_output(shadow_config.history_output_path.name)
    opportunity_shadow_latest_path = _default_signal_output(shadow_config.latest_output_path.name)
    opportunity_shadow_state_path = _default_signal_output(shadow_config.state_path.name)
    observation_history_path = _default_signal_output(observation_config.history_output_path.name)
    observation_state_path = _default_signal_output(observation_config.state_path.name)
    open_phase_history_path = _default_signal_output("open_phase_shadow.jsonl")
    open_phase_state_path = _default_signal_output("open_phase_shadow_state.json")

    try:
        snapshots = _load_scoped_snapshots_with_runtime(
            config=config,
            http=http,
            markets_path=markets_path,
            cities=scoped_cities,
            market_scope=market_scope,
            active=True,
            closed=False,
        )

        def _opportunity_report_rows(*, observed_at: datetime) -> list[dict[str, object]]:
            observations: list[OpportunityObservation] = []
            for snapshot in snapshots:
                spec = snapshot.spec
                if spec is None or spec.target_local_date < observed_at.date():
                    continue
                decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                    spec,
                    now_utc=observed_at,
                    horizon=horizon,
                    horizon_policy=horizon_policy,
                )
                if horizon_reason == "policy_filtered":
                    observations.append(
                        _empty_opportunity_observation(
                            snapshot,
                            observed_at=observed_at,
                            decision_horizon=decision_horizon or "policy",
                            reason="policy_filtered",
                        )
                    )
                    continue
                if decision_horizon is None:
                    continue
                observation = _evaluate_opportunity_snapshot(
                    snapshot,
                    builder=builder,
                    clob=clob,
                    model_path=model_path,
                    model_name=resolved_model_name,
                    config=config,
                    observed_at=observed_at,
                    decision_horizon=decision_horizon,
                    edge_threshold=edge_threshold,
                )
                if observation is not None:
                    observations.append(observation)
            rows = [_serialize_opportunity_observation(observation) for observation in observations]
            rows.sort(
                key=lambda row: (
                    row.get("reason") != "tradable",
                    -(float(row.get("edge", -999.0)) if row.get("edge") is not None else -999.0),
                )
            )
            dump_json(opportunity_report_path, rows)
            return rows

        def _opportunity_shadow_evaluator(
            _cycle_snapshots: list[MarketSnapshot],
            observed_at: datetime,
        ) -> list[OpportunityObservation]:
            observations: list[OpportunityObservation] = []
            for snapshot in _cycle_snapshots:
                spec = snapshot.spec
                if spec is None:
                    continue
                decision_horizon = _resolve_opportunity_shadow_horizon(
                    spec,
                    now_utc=observed_at,
                    near_term_days=shadow_config.near_term_days,
                )
                if decision_horizon is None:
                    continue
                if not _policy_allows_horizon(spec, decision_horizon=decision_horizon, horizon_policy=horizon_policy):
                    observations.append(
                        _empty_opportunity_observation(
                            snapshot,
                            observed_at=observed_at,
                            decision_horizon=decision_horizon,
                            reason="policy_filtered",
                        )
                    )
                    continue
                observation = _evaluate_opportunity_snapshot(
                    snapshot,
                    builder=builder,
                    clob=clob,
                    model_path=model_path,
                    model_name=resolved_model_name,
                    config=config,
                    observed_at=observed_at,
                    decision_horizon=decision_horizon,
                    edge_threshold=config.backtest.default_edge_threshold,
                )
                if observation is not None:
                    observations.append(observation)
            return observations

        def _observation_evaluator(
            _cycle_snapshots: list[MarketSnapshot],
            observed_at: datetime,
        ) -> list[ObservationOpportunity]:
            observations: list[ObservationOpportunity] = []
            for snapshot in _cycle_snapshots:
                spec = snapshot.spec
                if spec is None:
                    continue
                decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                    spec,
                    now_utc=observed_at,
                    horizon=horizon,
                    horizon_policy=horizon_policy,
                )
                if horizon_reason == "policy_filtered":
                    observations.append(
                        _empty_observation_opportunity(
                            snapshot,
                            observed_at=observed_at,
                            decision_horizon=decision_horizon or "policy",
                            reason="policy_filtered",
                            source_family="aviation",
                            observation_source="aviationweather_metar",
                            station_id=spec.station_id,
                            risk_flags=["policy_filtered"],
                        )
                    )
                    continue
                if decision_horizon is None:
                    continue
                observation = _evaluate_observation_snapshot(
                    snapshot,
                    builder=builder,
                    clob=clob,
                    http=http,
                    metar=metar,
                    model_path=model_path,
                    model_name=resolved_model_name,
                    config=config,
                    observed_at=observed_at,
                    decision_horizon=decision_horizon,
                    edge_threshold=edge_threshold,
                )
                if observation is not None:
                    observations.append(observation)
            return observations

        def _open_phase_evaluator(
            _cycle_snapshots: list[MarketSnapshot],
            observed_at: datetime,
        ) -> list[OpenPhaseObservation]:
            observations: list[OpenPhaseObservation] = []
            candidates = select_open_phase_candidates(
                _cycle_snapshots,
                observed_at=observed_at,
                open_window_hours=open_window_hours,
            )
            for candidate in candidates:
                observation = _evaluate_opportunity_snapshot(
                    candidate.snapshot,
                    builder=builder,
                    clob=clob,
                    model_path=model_path,
                    model_name=resolved_model_name,
                    config=config,
                    observed_at=observed_at,
                    decision_horizon="market_open",
                    edge_threshold=edge_threshold,
                )
                if observation is None:
                    continue
                observations.append(
                    _open_phase_observation_from_opportunity(
                        observation,
                        market_created_at=candidate.market_created_at,
                        market_deploying_at=candidate.market_deploying_at,
                        market_accepting_orders_at=candidate.market_accepting_orders_at,
                        market_opened_at=candidate.market_opened_at,
                        open_phase_age_hours=candidate.open_phase_age_hours,
                    )
                )
            return observations

        _opportunity_report_rows(observed_at=datetime.now(tz=UTC))

        opportunity_shadow_runner = OpportunityShadowRunner(
            config=config,
            interval_seconds=shadow_config.interval_seconds,
            max_cycles=1,
            state_path=opportunity_shadow_state_path,
            latest_output_path=opportunity_shadow_latest_path,
            history_output_path=opportunity_shadow_history_path,
            summary_output_path=opportunity_shadow_summary_path,
            snapshot_fetcher=lambda: snapshots,
            evaluator=_opportunity_shadow_evaluator,
        )
        opportunity_shadow_runner.run_once()

        observation_runner = ObservationShadowRunner(
            config=config,
            interval_seconds=observation_config.interval_seconds,
            max_cycles=1,
            state_path=observation_state_path,
            latest_output_path=observation_latest_path,
            history_output_path=observation_history_path,
            summary_output_path=observation_summary_path,
            alerts_output_path=observation_alerts_path,
            queue_output_path=queue_path,
            snapshot_fetcher=lambda: snapshots,
            evaluator=_observation_evaluator,
        )
        observation_runner.run_once()

        open_phase_runner = OpenPhaseShadowRunner(
            config=config,
            interval_seconds=shadow_config.interval_seconds,
            max_cycles=1,
            state_path=open_phase_state_path,
            latest_output_path=open_phase_latest_path,
            history_output_path=open_phase_history_path,
            summary_output_path=open_phase_summary_path,
            snapshot_fetcher=lambda: snapshots,
            evaluator=_open_phase_evaluator,
        )
        open_phase_runner.run_once()

        report = build_revenue_gate_report(
            benchmark_summary=_load_optional_json(benchmark_summary_path),
            opportunity_summary=_load_optional_json(opportunity_shadow_summary_path),
            open_phase_summary=_load_optional_json(open_phase_summary_path),
            observation_summary=_load_optional_json(observation_summary_path),
            trading_alias_name=TRADING_MODEL_ALIAS,
            pilot_constraints=DEFAULT_PILOT_CONSTRAINTS,
            market_scope=market_scope,
        )
        dump_json(revenue_gate_summary_path, report)

        dashboard_runner = StationDashboardRunner(
            config=config,
            interval_seconds=config.station_dashboard.interval_seconds,
            max_cycles=1,
            state_path=dashboard_state_path,
            json_output_path=dashboard_json_output,
            html_output_path=dashboard_html_output,
            data_loader=lambda: {
                "opportunity_rows": _load_optional_rows(opportunity_report_path),
                "observation_rows": _load_optional_rows(observation_latest_path),
                "queue_rows": _load_optional_rows(queue_path),
                "open_phase_rows": _load_optional_rows(open_phase_latest_path),
                "revenue_gate_summary": report,
                "observation_summary": _load_optional_json(observation_summary_path),
                "open_phase_summary": _load_optional_json(open_phase_summary_path),
                "watchlist_playbook": _load_optional_json(watchlist_playbook_path),
            },
        )
        dashboard_summary = dashboard_runner.run_once()
    finally:
        http.close()

    return {
        "generated_at": datetime.now(tz=UTC),
        "revenue_gate_decision": dashboard_summary.get("revenue_gate_decision", "INCONCLUSIVE"),
        "queue_size": int(dashboard_summary.get("queue_size", 0) or 0),
        "observation_tradable_count": int(dashboard_summary.get("observation_tradable_count", 0) or 0),
        "opportunity_tradable_count": int(dashboard_summary.get("opportunity_tradable_count", 0) or 0),
        "open_phase_count": int(dashboard_summary.get("open_phase_count", 0) or 0),
        "watchlist_alert_count": int(dashboard_summary.get("watchlist_alert_count", 0) or 0),
        "dashboard_json_output": str(dashboard_json_output),
        "dashboard_html_output": str(dashboard_html_output),
    }


@app.command("station-dashboard")
def station_dashboard(
    opportunity_report_path: Path | None = typer.Option(None, help="Opportunity report JSON path"),
    observation_latest_path: Path | None = typer.Option(None, help="Observation latest JSON path"),
    observation_summary_path: Path | None = typer.Option(None, help="Observation summary JSON path"),
    queue_path: Path | None = typer.Option(None, help="Manual approval queue JSON path"),
    open_phase_latest_path: Path | None = typer.Option(None, help="Open-phase latest JSON path"),
    open_phase_summary_path: Path | None = typer.Option(None, help="Open-phase summary JSON path"),
    revenue_gate_summary_path: Path | None = typer.Option(None, help="Revenue gate summary JSON path"),
    watchlist_playbook_path: Path | None = typer.Option(None, help="Execution watchlist playbook JSON path"),
    json_output: Path | None = typer.Option(None, help="Dashboard JSON output path"),
    html_output: Path | None = typer.Option(None, help="Dashboard HTML output path"),
    state_path: Path | None = typer.Option(None, help="Dashboard state path"),
) -> None:
    """Render one combined station dashboard from existing signal artifacts."""

    config, _env = load_settings()
    dashboard_config = config.station_dashboard
    runner = _build_station_dashboard_runner(
        opportunity_report_path=_path_or_default(opportunity_report_path, dashboard_config.opportunity_report_path),
        observation_latest_path=_path_or_default(observation_latest_path, dashboard_config.observation_latest_path),
        observation_summary_path=_path_or_default(
            observation_summary_path,
            dashboard_config.observation_summary_path,
        ),
        queue_path=_path_or_default(queue_path, dashboard_config.queue_output_path),
        open_phase_latest_path=_path_or_default(open_phase_latest_path, dashboard_config.open_phase_latest_path),
        open_phase_summary_path=_path_or_default(
            open_phase_summary_path,
            dashboard_config.open_phase_summary_path,
        ),
        revenue_gate_summary_path=_path_or_default(
            revenue_gate_summary_path,
            dashboard_config.revenue_gate_summary_path,
        ),
        watchlist_playbook_path=_path_or_default(
            watchlist_playbook_path,
            dashboard_config.watchlist_playbook_path,
        ),
        interval_seconds=dashboard_config.interval_seconds,
        max_cycles=1,
        json_output_path=_path_or_default(json_output, dashboard_config.json_output_path),
        html_output_path=_path_or_default(html_output, dashboard_config.html_output_path),
        state_path=_path_or_default(state_path, dashboard_config.state_path),
    )
    summary = runner.run_once()

    table = Table(title="Station Dashboard")
    table.add_column("Metric")
    table.add_column("Value")
    for key in [
        "revenue_gate_decision",
        "queue_size",
        "observation_tradable_count",
        "opportunity_tradable_count",
        "open_phase_count",
        "watchlist_alert_count",
    ]:
        table.add_row(key, str(summary.get(key, "")))
    console.print(table)
    console.print(f"Wrote JSON {runner.json_output_path}")
    console.print(f"Wrote HTML {runner.html_output_path}")


@app.command("station-dashboard-daemon")
def station_dashboard_daemon(
    opportunity_report_path: Path | None = typer.Option(None, help="Opportunity report JSON path"),
    observation_latest_path: Path | None = typer.Option(None, help="Observation latest JSON path"),
    observation_summary_path: Path | None = typer.Option(None, help="Observation summary JSON path"),
    queue_path: Path | None = typer.Option(None, help="Manual approval queue JSON path"),
    open_phase_latest_path: Path | None = typer.Option(None, help="Open-phase latest JSON path"),
    open_phase_summary_path: Path | None = typer.Option(None, help="Open-phase summary JSON path"),
    revenue_gate_summary_path: Path | None = typer.Option(None, help="Revenue gate summary JSON path"),
    watchlist_playbook_path: Path | None = typer.Option(None, help="Execution watchlist playbook JSON path"),
    interval: int | None = typer.Option(None, help="Seconds between dashboard refresh cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    json_output: Path | None = typer.Option(None, help="Dashboard JSON output path"),
    html_output: Path | None = typer.Option(None, help="Dashboard HTML output path"),
    state_path: Path | None = typer.Option(None, help="Dashboard state path"),
) -> None:
    """Continuously refresh the combined station dashboard."""

    config, _env = load_settings()
    dashboard_config = config.station_dashboard
    runner = _build_station_dashboard_runner(
        opportunity_report_path=_path_or_default(opportunity_report_path, dashboard_config.opportunity_report_path),
        observation_latest_path=_path_or_default(observation_latest_path, dashboard_config.observation_latest_path),
        observation_summary_path=_path_or_default(
            observation_summary_path,
            dashboard_config.observation_summary_path,
        ),
        queue_path=_path_or_default(queue_path, dashboard_config.queue_output_path),
        open_phase_latest_path=_path_or_default(open_phase_latest_path, dashboard_config.open_phase_latest_path),
        open_phase_summary_path=_path_or_default(
            open_phase_summary_path,
            dashboard_config.open_phase_summary_path,
        ),
        revenue_gate_summary_path=_path_or_default(
            revenue_gate_summary_path,
            dashboard_config.revenue_gate_summary_path,
        ),
        watchlist_playbook_path=_path_or_default(
            watchlist_playbook_path,
            dashboard_config.watchlist_playbook_path,
        ),
        interval_seconds=_resolve_option_value(interval, dashboard_config.interval_seconds),
        max_cycles=_resolve_option_value(max_cycles, dashboard_config.max_cycles) or 0,
        json_output_path=_path_or_default(json_output, dashboard_config.json_output_path),
        html_output_path=_path_or_default(html_output, dashboard_config.html_output_path),
        state_path=_path_or_default(state_path, dashboard_config.state_path),
    )
    console.print(
        "Station dashboard daemon: "
        f"interval={runner.interval_seconds}s, max_cycles={runner.max_cycles}, "
        f"html={runner.html_output_path}"
    )
    runner.run_loop()
    console.print(f"Wrote JSON {runner.json_output_path}")
    console.print(f"Wrote HTML {runner.html_output_path}")


@app.command("station-cycle")
def station_cycle(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    open_window_hours: float = typer.Option(24.0, help="Open-phase window in hours"),
    benchmark_summary_path: Path = typer.Option(
        DEFAULT_BENCHMARK_SUMMARY_PATH,
        help="Recent-core benchmark summary JSON",
    ),
    opportunity_report_path: Path | None = typer.Option(None, help="Opportunity report output path"),
    opportunity_shadow_summary_path: Path | None = typer.Option(None, help="Opportunity shadow summary output path"),
    observation_latest_path: Path | None = typer.Option(None, help="Observation latest output path"),
    observation_summary_path: Path | None = typer.Option(None, help="Observation summary path"),
    observation_alerts_path: Path | None = typer.Option(None, help="Observation alerts output path"),
    queue_path: Path | None = typer.Option(None, help="Manual approval queue path"),
    open_phase_latest_path: Path | None = typer.Option(None, help="Open-phase latest output path"),
    open_phase_summary_path: Path | None = typer.Option(None, help="Open-phase summary path"),
    revenue_gate_summary_path: Path | None = typer.Option(None, help="Revenue gate summary output path"),
    watchlist_playbook_path: Path | None = typer.Option(None, help="Execution watchlist playbook JSON path"),
    dashboard_json_output: Path | None = typer.Option(None, help="Dashboard JSON output path"),
    dashboard_html_output: Path | None = typer.Option(None, help="Dashboard HTML output path"),
    dashboard_state_path: Path | None = typer.Option(None, help="Dashboard state path"),
    state_path: Path | None = typer.Option(None, help="Station cycle state path"),
) -> None:
    """Run one full station cycle from discovery through dashboard rendering."""

    config, _env = load_settings()
    dashboard_config = config.station_dashboard
    observation_config = config.observation_station
    orchestrator_config = config.station_orchestrator
    summary = _run_station_cycle(
        model_path=_resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl")),
        model_name=_resolve_option_value(model_name, TRADING_MODEL_ALIAS),
        cities=_resolve_option_value(cities),
        core_recent_only=bool(_resolve_option_value(core_recent_only, False)),
        market_scope=_resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only),
        markets_path=_resolve_option_value(markets_path),
        horizon=_resolve_option_value(horizon, "policy"),
        min_edge=_resolve_option_value(min_edge),
        open_window_hours=float(_resolve_option_value(open_window_hours, 24.0)),
        benchmark_summary_path=_resolve_option_value(benchmark_summary_path, DEFAULT_BENCHMARK_SUMMARY_PATH),
        opportunity_report_path=_path_or_default(opportunity_report_path, dashboard_config.opportunity_report_path),
        opportunity_shadow_summary_path=_path_or_default(
            opportunity_shadow_summary_path,
            _default_signal_output("opportunity_shadow_summary.json"),
        ),
        observation_latest_path=_path_or_default(observation_latest_path, dashboard_config.observation_latest_path),
        observation_summary_path=_path_or_default(observation_summary_path, dashboard_config.observation_summary_path),
        observation_alerts_path=_path_or_default(observation_alerts_path, observation_config.alerts_output_path),
        queue_path=_path_or_default(queue_path, dashboard_config.queue_output_path),
        open_phase_latest_path=_path_or_default(open_phase_latest_path, dashboard_config.open_phase_latest_path),
        open_phase_summary_path=_path_or_default(open_phase_summary_path, dashboard_config.open_phase_summary_path),
        revenue_gate_summary_path=_path_or_default(
            revenue_gate_summary_path,
            dashboard_config.revenue_gate_summary_path,
        ),
        watchlist_playbook_path=_path_or_default(
            watchlist_playbook_path,
            dashboard_config.watchlist_playbook_path,
        ),
        dashboard_json_output=_path_or_default(dashboard_json_output, dashboard_config.json_output_path),
        dashboard_html_output=_path_or_default(dashboard_html_output, dashboard_config.html_output_path),
        dashboard_state_path=_path_or_default(dashboard_state_path, dashboard_config.state_path),
    )
    effective_state_path = _path_or_default(state_path, orchestrator_config.state_path)
    dump_json(effective_state_path, summary)
    console.print_json(data=json.loads(json.dumps(summary, default=str)))


@app.command("station-daemon")
def station_daemon(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    open_window_hours: float = typer.Option(24.0, help="Open-phase window in hours"),
    benchmark_summary_path: Path = typer.Option(
        DEFAULT_BENCHMARK_SUMMARY_PATH,
        help="Recent-core benchmark summary JSON",
    ),
    opportunity_report_path: Path | None = typer.Option(None, help="Opportunity report output path"),
    opportunity_shadow_summary_path: Path | None = typer.Option(None, help="Opportunity shadow summary output path"),
    observation_latest_path: Path | None = typer.Option(None, help="Observation latest output path"),
    observation_summary_path: Path | None = typer.Option(None, help="Observation summary path"),
    observation_alerts_path: Path | None = typer.Option(None, help="Observation alerts output path"),
    queue_path: Path | None = typer.Option(None, help="Manual approval queue path"),
    open_phase_latest_path: Path | None = typer.Option(None, help="Open-phase latest output path"),
    open_phase_summary_path: Path | None = typer.Option(None, help="Open-phase summary path"),
    revenue_gate_summary_path: Path | None = typer.Option(None, help="Revenue gate summary output path"),
    watchlist_playbook_path: Path | None = typer.Option(None, help="Execution watchlist playbook JSON path"),
    dashboard_json_output: Path | None = typer.Option(None, help="Dashboard JSON output path"),
    dashboard_html_output: Path | None = typer.Option(None, help="Dashboard HTML output path"),
    dashboard_state_path: Path | None = typer.Option(None, help="Dashboard state path"),
    interval: int | None = typer.Option(None, help="Seconds between station cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    state_path: Path | None = typer.Option(None, help="Station cycle state path"),
) -> None:
    """Continuously run the full station cycle."""

    import signal as sig
    import time

    config, _env = load_settings()
    dashboard_config = config.station_dashboard
    observation_config = config.observation_station
    orchestrator_config = config.station_orchestrator
    effective_interval = _resolve_option_value(interval, orchestrator_config.interval_seconds)
    effective_max_cycles = _resolve_option_value(max_cycles, orchestrator_config.max_cycles) or 0
    effective_state_path = _path_or_default(state_path, orchestrator_config.state_path)
    resolved_model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    resolved_model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    resolved_cities = _resolve_option_value(cities)
    resolved_core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    resolved_market_scope = _resolve_market_scope(
        _resolve_option_value(market_scope, "default"),
        core_recent_only=resolved_core_recent_only,
    )
    resolved_markets_path = _resolve_option_value(markets_path)
    resolved_horizon = _resolve_option_value(horizon, "policy")
    resolved_min_edge = _resolve_option_value(min_edge)
    resolved_open_window_hours = float(_resolve_option_value(open_window_hours, 24.0))
    resolved_benchmark_summary_path = _resolve_option_value(benchmark_summary_path, DEFAULT_BENCHMARK_SUMMARY_PATH)
    resolved_opportunity_report_path = _path_or_default(opportunity_report_path, dashboard_config.opportunity_report_path)
    resolved_opportunity_shadow_summary_path = _path_or_default(
        opportunity_shadow_summary_path,
        _default_signal_output("opportunity_shadow_summary.json"),
    )
    resolved_observation_latest_path = _path_or_default(observation_latest_path, dashboard_config.observation_latest_path)
    resolved_observation_summary_path = _path_or_default(observation_summary_path, dashboard_config.observation_summary_path)
    resolved_observation_alerts_path = _path_or_default(observation_alerts_path, observation_config.alerts_output_path)
    resolved_queue_path = _path_or_default(queue_path, dashboard_config.queue_output_path)
    resolved_open_phase_latest_path = _path_or_default(open_phase_latest_path, dashboard_config.open_phase_latest_path)
    resolved_open_phase_summary_path = _path_or_default(open_phase_summary_path, dashboard_config.open_phase_summary_path)
    resolved_revenue_gate_summary_path = _path_or_default(
        revenue_gate_summary_path,
        dashboard_config.revenue_gate_summary_path,
    )
    resolved_watchlist_playbook_path = _path_or_default(
        watchlist_playbook_path,
        dashboard_config.watchlist_playbook_path,
    )
    resolved_dashboard_json_output = _path_or_default(dashboard_json_output, dashboard_config.json_output_path)
    resolved_dashboard_html_output = _path_or_default(dashboard_html_output, dashboard_config.html_output_path)
    resolved_dashboard_state_path = _path_or_default(dashboard_state_path, dashboard_config.state_path)

    running = True

    def _shutdown(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    sig.signal(sig.SIGINT, _shutdown)
    sig.signal(sig.SIGTERM, _shutdown)

    cycle = 0
    console.print(
        "Station daemon: "
        f"interval={effective_interval}s, max_cycles={effective_max_cycles}, "
        f"market_scope={resolved_market_scope}"
    )
    while running:
        summary = _run_station_cycle(
            model_path=resolved_model_path,
            model_name=resolved_model_name,
            cities=resolved_cities,
            core_recent_only=resolved_core_recent_only,
            market_scope=resolved_market_scope,
            markets_path=resolved_markets_path,
            horizon=resolved_horizon,
            min_edge=resolved_min_edge,
            open_window_hours=resolved_open_window_hours,
            benchmark_summary_path=resolved_benchmark_summary_path,
            opportunity_report_path=resolved_opportunity_report_path,
            opportunity_shadow_summary_path=resolved_opportunity_shadow_summary_path,
            observation_latest_path=resolved_observation_latest_path,
            observation_summary_path=resolved_observation_summary_path,
            observation_alerts_path=resolved_observation_alerts_path,
            queue_path=resolved_queue_path,
                open_phase_latest_path=resolved_open_phase_latest_path,
                open_phase_summary_path=resolved_open_phase_summary_path,
                revenue_gate_summary_path=resolved_revenue_gate_summary_path,
                watchlist_playbook_path=resolved_watchlist_playbook_path,
                dashboard_json_output=resolved_dashboard_json_output,
                dashboard_html_output=resolved_dashboard_html_output,
                dashboard_state_path=resolved_dashboard_state_path,
            )
        cycle += 1
        payload = {
            "cycle": cycle,
            **summary,
        }
        dump_json(effective_state_path, payload)
        console.print_json(data=json.loads(json.dumps(payload, default=str)))
        if 0 < effective_max_cycles <= cycle:
            break
        if running:
            time.sleep(effective_interval)


def _build_observation_station_runner(
    *,
    model_path: Path,
    model_name: str,
    cities: list[str] | None,
    market_scope: str,
    markets_path: Path | None,
    interval_seconds: int,
    max_cycles: int,
    output: Path,
    latest_output: Path,
    summary_output: Path,
    alerts_output: Path,
    queue_output: Path,
    state_path: Path,
    horizon: str,
    horizon_policy_path: Path,
    edge_threshold: float | None,
) -> tuple[ObservationShadowRunner, CachedHttpClient]:
    """Build one reusable observation-station runner."""

    config, _env, http, _duckdb, _parquet, openmeteo = _runtime(include_stores=False)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    metar = MetarClient(http, config.metar.base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    horizon_policy = _load_recent_horizon_policy(horizon_policy_path)
    effective_edge_threshold = edge_threshold if edge_threshold is not None else config.backtest.default_edge_threshold
    scoped_cities = _resolve_scoped_cities(cities, market_scope=market_scope)

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_scoped_snapshots(
            markets_path=markets_path,
            cities=scoped_cities,
            market_scope=market_scope,
            active=True,
            closed=False,
        )

    def _evaluator(snapshots: list[MarketSnapshot], observed_at: datetime) -> list[ObservationOpportunity]:
        observations: list[ObservationOpportunity] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                spec,
                now_utc=observed_at,
                horizon=horizon,
                horizon_policy=horizon_policy,
            )
            if horizon_reason == "policy_filtered":
                observations.append(
                    _empty_observation_opportunity(
                        snapshot,
                        observed_at=observed_at,
                        decision_horizon=decision_horizon or "policy",
                        reason="policy_filtered",
                        source_family="aviation",
                        observation_source="aviationweather_metar",
                        station_id=spec.station_id,
                        risk_flags=["policy_filtered"],
                    )
                )
                continue
            if decision_horizon is None:
                continue
            observation = _evaluate_observation_snapshot(
                snapshot,
                builder=builder,
                clob=clob,
                http=http,
                metar=metar,
                model_path=model_path,
                model_name=resolved_model_name,
                config=config,
                observed_at=observed_at,
                decision_horizon=decision_horizon,
                edge_threshold=effective_edge_threshold,
            )
            if observation is not None:
                observations.append(observation)
        return observations

    runner = ObservationShadowRunner(
        config=config,
        interval_seconds=interval_seconds,
        max_cycles=max_cycles,
        state_path=state_path,
        latest_output_path=latest_output,
        history_output_path=output,
        summary_output_path=summary_output,
        alerts_output_path=alerts_output,
        queue_output_path=queue_output,
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )
    return runner, http


@app.command("observation-report")
def observation_report(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path | None = typer.Option(None, help="Latest observation report JSON output"),
    summary_output: Path | None = typer.Option(None, help="Observation summary JSON output"),
    alerts_output: Path | None = typer.Option(None, help="Alert queue JSON output"),
    queue_output: Path | None = typer.Option(None, help="Manual approval queue JSON output"),
) -> None:
    """Generate one observation-driven opportunity snapshot across active markets."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    markets_path = _resolve_option_value(markets_path)
    horizon = _resolve_option_value(horizon, "policy")
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    min_edge = _resolve_option_value(min_edge)
    config, _env = load_settings()
    latest_output = _resolve_option_value(
        output,
        _default_signal_output(config.observation_station.latest_output_path.name),
    )
    summary_output_path = _resolve_option_value(
        summary_output,
        _default_signal_output(config.observation_station.summary_output_path.name),
    )
    alerts_output_path = _resolve_option_value(
        alerts_output,
        _default_signal_output(config.observation_station.alerts_output_path.name),
    )
    queue_output_path = _resolve_option_value(
        queue_output,
        _default_signal_output(config.observation_station.queue_output_path.name),
    )
    runner, http = _build_observation_station_runner(
        model_path=model_path,
        model_name=model_name,
        cities=cities,
        market_scope=market_scope,
        markets_path=markets_path,
        interval_seconds=config.observation_station.interval_seconds,
        max_cycles=1,
        output=_default_signal_output(config.observation_station.history_output_path.name),
        latest_output=latest_output,
        summary_output=summary_output_path,
        alerts_output=alerts_output_path,
        queue_output=queue_output_path,
        state_path=_default_signal_output(config.observation_station.state_path.name),
        horizon=horizon,
        horizon_policy_path=horizon_policy_path,
        edge_threshold=min_edge,
    )
    try:
        summary = runner.run_once()
    finally:
        http.close()

    rows = json.loads(latest_output.read_text()) if latest_output.exists() else []
    summary_payload = json.loads(runner.summary_output_path.read_text()) if runner.summary_output_path.exists() else {}
    table = Table(title="Observation Report")
    table.add_column("City")
    table.add_column("Date")
    table.add_column("Obs")
    table.add_column("Outcome")
    table.add_column("Edge")
    table.add_column("Queue")
    table.add_column("Reason")
    for row in rows[:20]:
        observed_value = row.get("observed_temp_market_unit")
        table.add_row(
            str(row.get("city", "")),
            str(row.get("target_local_date", "")),
            f"{float(observed_value):.1f}" if observed_value is not None else "—",
            str(row.get("outcome_label", "—")),
            f"{float(row['edge']):+.4f}" if row.get("edge") is not None else "—",
            str(row.get("queue_state", "")),
            str(row.get("reason", "")),
        )
    console.print(table)
    if isinstance(summary_payload, dict):
        source_table = Table(title="Observation Source Breakdown")
        source_table.add_column("Source Family")
        source_table.add_column("Observed")
        source_table.add_column("Raw+")
        source_table.add_column("Edge+")
        source_table.add_column("Gate")
        for source_family, source_summary in sorted(dict(summary_payload.get("by_source_family", {})).items()):
            if not isinstance(source_summary, dict):
                continue
            source_table.add_row(
                str(source_family),
                str(int(source_summary.get("markets_evaluated", 0) or 0)),
                str(int(source_summary.get("raw_gap_positive_count", 0) or 0)),
                str(int(source_summary.get("after_cost_edge_positive_count", 0) or 0)),
                str(source_summary.get("gate_decision", "")),
            )
        console.print(source_table)

        observation_source_table = Table(title="Observation Adapter Breakdown")
        observation_source_table.add_column("Observation Source")
        observation_source_table.add_column("Observed")
        observation_source_table.add_column("Manual")
        observation_source_table.add_column("Tradable")
        for observation_source, source_summary in sorted(dict(summary_payload.get("by_observation_source", {})).items()):
            if not isinstance(source_summary, dict):
                continue
            observation_source_table.add_row(
                str(observation_source),
                str(int(source_summary.get("markets_evaluated", 0) or 0)),
                str(int(source_summary.get("manual_review_count", 0) or 0)),
                str(int(source_summary.get("tradable_count", 0) or 0)),
            )
        console.print(observation_source_table)
    console.print(
        "Observation report: "
        f"markets_evaluated={summary['markets_evaluated']}, "
        f"tradable={summary['tradable_count']}, "
        f"manual_review={summary['manual_review_count']}"
    )
    console.print(f"Wrote latest {latest_output}")
    console.print(f"Wrote summary {summary_output_path}")
    console.print(f"Wrote alerts {alerts_output_path}")
    console.print(f"Wrote queue {queue_output_path}")


@app.command("observation-shadow")
def observation_shadow(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between observation cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path | None = typer.Option(None, help="Append-only JSONL output"),
    latest_output: Path | None = typer.Option(None, help="Latest cycle JSON output"),
    summary_output: Path | None = typer.Option(None, help="Summary JSON output"),
    alerts_output: Path | None = typer.Option(None, help="Alert queue JSON output"),
    queue_output: Path | None = typer.Option(None, help="Manual approval queue JSON output"),
    state_path: Path | None = typer.Option(None, help="State JSON path"),
) -> None:
    """Continuously validate observation-driven edges without posting orders."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    horizon = _resolve_option_value(horizon, "policy")
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output)
    latest_output = _resolve_option_value(latest_output)
    summary_output = _resolve_option_value(summary_output)
    alerts_output = _resolve_option_value(alerts_output)
    queue_output = _resolve_option_value(queue_output)
    state_path = _resolve_option_value(state_path)
    config, _env = load_settings()
    runner, http = _build_observation_station_runner(
        model_path=model_path,
        model_name=model_name,
        cities=cities,
        market_scope=market_scope,
        markets_path=markets_path,
        interval_seconds=interval or config.observation_station.interval_seconds,
        max_cycles=(config.observation_station.max_cycles if max_cycles is None else max_cycles) or 0,
        output=output or _default_signal_output(config.observation_station.history_output_path.name),
        latest_output=latest_output or _default_signal_output(config.observation_station.latest_output_path.name),
        summary_output=summary_output or _default_signal_output(config.observation_station.summary_output_path.name),
        alerts_output=alerts_output or _default_signal_output(config.observation_station.alerts_output_path.name),
        queue_output=queue_output or _default_signal_output(config.observation_station.queue_output_path.name),
        state_path=state_path or _default_signal_output(config.observation_station.state_path.name),
        horizon=horizon,
        horizon_policy_path=horizon_policy_path,
        edge_threshold=min_edge,
    )
    try:
        console.print(
            "Observation shadow: "
            f"interval={runner.interval_seconds}s, max_cycles={runner.max_cycles}, horizon={horizon}"
        )
        runner.run_loop()
    finally:
        http.close()

    console.print(f"Wrote latest {runner.latest_output_path}")
    console.print(f"Wrote history {runner.history_output_path}")
    console.print(f"Wrote summary {runner.summary_output_path}")
    console.print(f"Wrote alerts {runner.alerts_output_path}")
    console.print(f"Wrote queue {runner.queue_output_path}")


@app.command("observation-daemon")
def observation_daemon(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    interval: int | None = typer.Option(None, help="Seconds between observation cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
) -> None:
    """Run the observation-station loop using the checked-in default outputs."""

    observation_shadow(
        model_path=model_path,
        model_name=model_name,
        cities=cities,
        core_recent_only=False,
        market_scope=market_scope,
        markets_path=markets_path,
        interval=interval,
        max_cycles=max_cycles,
        horizon=horizon,
        min_edge=min_edge,
        output=None,
        latest_output=None,
        summary_output=None,
        alerts_output=None,
        queue_output=None,
        state_path=None,
    )


@app.command("approve-live-candidate")
def approve_live_candidate(
    manual_approval_token: str = typer.Argument(..., help="Manual approval token from live_pilot_queue.json"),
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    queue_path: Path = typer.Option(
        Path("artifacts/signals/v2/live_pilot_queue.json"),
        help="Manual approval queue JSON path",
    ),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    dry_run: bool = typer.Option(True, help="Preview only; do not post by default"),
    post_order: bool = typer.Option(False, help="Actually post the order after approval"),
) -> None:
    """Revalidate one queued observation candidate and create a live order preview."""

    manual_approval_token = str(_resolve_option_value(manual_approval_token))
    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    queue_path = _resolve_option_value(queue_path, Path("artifacts/signals/v2/live_pilot_queue.json"))
    markets_path = _resolve_option_value(markets_path)
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=False)
    dry_run = bool(_resolve_option_value(dry_run, True))
    post_order = bool(_resolve_option_value(post_order, False))

    if not queue_path.exists():
        raise typer.BadParameter(f"Manual approval queue does not exist: {queue_path}")
    payload = load_json(queue_path)
    if not isinstance(payload, list):
        raise typer.BadParameter(f"Manual approval queue is malformed: {queue_path}")

    queued_row = next(
        (
            ObservationOpportunity.model_validate(item)
            for item in payload
            if isinstance(item, dict) and str(item.get("manual_approval_token", "")) == manual_approval_token
        ),
        None,
    )
    if queued_row is None:
        raise typer.BadParameter(f"Manual approval token not found: {manual_approval_token}")
    if queued_row.approval_expires_at is not None and datetime.now(tz=UTC) > _as_utc_datetime(queued_row.approval_expires_at):
        raise typer.BadParameter(f"Manual approval token expired: {manual_approval_token}")

    config, env, http, _duckdb, _parquet, openmeteo = _runtime(include_stores=False)
    broker = LiveBroker(env)
    preflight = broker.preflight(require_posting=post_order and not dry_run)

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    metar = MetarClient(http, config.metar.base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    snapshots = _load_scoped_snapshots(
        markets_path=markets_path,
        cities=None,
        market_scope=market_scope,
        active=True,
        closed=False,
    )
    snapshot = next(
        (
            item
            for item in snapshots
            if item.spec is not None and item.spec.market_id == queued_row.market_id
        ),
        None,
    )
    if snapshot is None:
        http.close()
        raise typer.BadParameter(f"Active market snapshot not found for {queued_row.market_id}")

    refreshed = _evaluate_observation_snapshot(
        snapshot,
        builder=builder,
        clob=clob,
        http=http,
        metar=metar,
        model_path=model_path,
        model_name=resolved_model_name,
        config=config,
        observed_at=datetime.now(tz=UTC),
        decision_horizon=queued_row.decision_horizon,
        edge_threshold=config.backtest.default_edge_threshold,
    )
    if refreshed is None:
        http.close()
        raise typer.BadParameter(f"Could not re-evaluate live candidate for {queued_row.market_id}")
    if refreshed.queue_state == "blocked":
        http.close()
        raise typer.BadParameter(f"Candidate is no longer live-eligible: {refreshed.reason}")
    if refreshed.outcome_label != queued_row.outcome_label:
        http.close()
        raise typer.BadParameter(
            f"Candidate outcome changed from {queued_row.outcome_label} to {refreshed.outcome_label}"
        )

    signal = TradeSignal(
        market_id=refreshed.market_id,
        token_id=str(refreshed.token_id or ""),
        outcome_label=str(refreshed.outcome_label),
        side="buy",
        fair_probability=float(refreshed.fair_probability or 0.0),
        executable_price=float(refreshed.best_ask or 0.0),
        fee_estimate=float(refreshed.fee_estimate or 0.0),
        slippage_estimate=float(refreshed.slippage_estimate or 0.0),
        edge=float(refreshed.after_cost_edge or 0.0),
        confidence=float(refreshed.fair_probability or 0.0),
        rationale=f"Approved observation candidate {refreshed.outcome_label}",
        mode="live",
        forecast_contract_version=refreshed.forecast_contract_version,
        probability_source=refreshed.probability_source,
        distribution_family=refreshed.distribution_family,
        decision_horizon=refreshed.decision_horizon,
    )
    size_multiplier = (
        config.observation_station.exact_public_size_multiplier
        if refreshed.truth_track == "exact_public"
        else config.observation_station.research_public_size_multiplier
    )
    bankroll = DEFAULT_PILOT_CONSTRAINTS["bankroll"] * size_multiplier
    size_notional = capped_kelly(signal.edge, signal.fair_probability, bankroll, signal.executable_price)
    size_notional = min(size_notional, config.execution.max_city_exposure * size_multiplier)
    size_notional = min(size_notional, config.execution.global_max_exposure * size_multiplier)
    size = size_notional / max(signal.executable_price, 1e-6)
    if size <= 0:
        http.close()
        raise typer.BadParameter("Candidate size collapsed to zero under the pilot guardrails.")

    preview = broker.preview_limit_order(signal, size=size)
    if post_order and not dry_run and preflight.ok:
        preview["post_result"] = broker.post_limit_order(signal, size=size)

    payload = {
        "preflight": preflight.model_dump(mode="json"),
        "queued_candidate": _serialize_observation_opportunity(queued_row),
        "refreshed_candidate": _serialize_observation_opportunity(refreshed),
        "size_notional": round(size_notional, 6),
        "size": round(size, 6),
        "preview": preview,
    }
    http.close()
    dump_json(_default_signal_output("live_pilot_approval_preview.json"), payload)
    console.print_json(data=payload)


@app.command("paper-trader")
def paper_trader(
    model_path: Path = Path("artifacts/models/v2/champion.pkl"),
    model_name: str = DEFAULT_MODEL_NAME,
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    horizon: str = "policy",
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    bankroll: float = 10_000.0,
    min_edge: float | None = None,
    max_spread_bps: int | None = typer.Option(None, help="Maximum spread override"),
    min_liquidity: float | None = typer.Option(None, help="Minimum liquidity override"),
    price_source: str = typer.Option("gamma", help="Price source for edge calculation: 'gamma' (Gamma mid-prices) or 'clob' (live order book)"),
    min_market_price: float = typer.Option(0.0, help="Skip outcomes where Gamma mid-price is below this threshold (e.g. 0.10 to exclude <10% bins)"),
    output: Path = Path("artifacts/signals/v2/paper_signals.json"),
) -> None:
    """Run paper trading over active discovered markets or bundled history."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    markets_path = _resolve_option_value(markets_path)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    horizon = _resolve_option_value(horizon, "policy")
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    bankroll = float(_resolve_option_value(bankroll, 10_000.0))
    min_edge = _resolve_option_value(min_edge)
    max_spread_bps = _resolve_option_value(max_spread_bps)
    min_liquidity = _resolve_option_value(min_liquidity)
    price_source = str(_resolve_option_value(price_source, "gamma"))
    min_market_price = float(_resolve_option_value(min_market_price, 0.0))
    output = _resolve_option_value(output, Path("artifacts/signals/v2/paper_signals.json"))
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    snapshots = _load_scoped_snapshots(
        markets_path=markets_path,
        cities=cities,
        market_scope=market_scope,
        active=True,
        closed=False,
    )
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    horizon_policy = _load_recent_horizon_policy(horizon_policy_path)
    results, _bankroll_remaining = _run_paper_model_evaluation(
        snapshots=snapshots,
        builder=builder,
        clob=clob,
        config=config,
        model_path=model_path,
        model_name=resolved_model_name,
        horizon=horizon,
        horizon_policy=horizon_policy,
        bankroll=bankroll,
        edge_threshold=edge_threshold,
        max_spread_bps=int(max_spread_bps if max_spread_bps is not None else config.execution.max_spread_bps),
        min_liquidity=float(min_liquidity if min_liquidity is not None else config.execution.min_liquidity),
        price_source=price_source,
        min_market_price=min_market_price,
    )
    dump_json(output, results)
    console.print_json(data=results)


@app.command("paper-multimodel-report")
def paper_multimodel_report(
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    bankroll: float = typer.Option(10_000.0, help="Paper bankroll"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    max_spread_bps: int | None = typer.Option(None, help="Maximum spread override"),
    min_liquidity: float | None = typer.Option(None, help="Minimum liquidity override"),
    output_dir: Path | None = typer.Option(None, help="Output directory for per-model JSON and summary"),
    model_labels: Annotated[list[str] | None, typer.Option("--model-label")] = None,
    model_paths: Annotated[list[Path] | None, typer.Option("--model-path")] = None,
    model_names: Annotated[list[str] | None, typer.Option("--model-name")] = None,
) -> None:
    """Compare the active champion against a top-challenger pool on one paper snapshot."""

    markets_path = _resolve_option_value(markets_path)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    horizon = _resolve_option_value(horizon, "policy")
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    bankroll = float(_resolve_option_value(bankroll, 10_000.0))
    min_edge = _resolve_option_value(min_edge)
    max_spread_bps = _resolve_option_value(max_spread_bps)
    min_liquidity = _resolve_option_value(min_liquidity)
    output_dir = _resolve_option_value(output_dir)
    model_labels = _resolve_option_value(model_labels)
    model_paths = _resolve_option_value(model_paths)
    model_names = _resolve_option_value(model_names)

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    scoped_cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    snapshots = _load_scoped_snapshots(
        markets_path=markets_path,
        cities=scoped_cities,
        market_scope=market_scope,
        active=True,
        closed=False,
    )
    specs = _resolve_paper_multimodel_specs(
        model_labels=model_labels,
        model_paths=model_paths,
        model_names=model_names,
    )
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    spread_limit = int(max_spread_bps if max_spread_bps is not None else config.execution.max_spread_bps)
    liquidity_floor = float(min_liquidity if min_liquidity is not None else config.execution.min_liquidity)
    horizon_policy = _load_recent_horizon_policy(horizon_policy_path)
    books_by_market = {
        snapshot.spec.market_id: _load_full_books_for_snapshot(clob, snapshot)
        for snapshot in snapshots
        if snapshot.spec is not None
    }
    run_dir = Path(output_dir) if output_dir is not None else (
        DEFAULT_PAPER_MULTIMODEL_ROOT / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model_summaries: dict[str, dict[str, object]] = {}
    leaderboard_rows: list[dict[str, object]] = []
    try:
        for spec in specs:
            label = str(spec["label"])
            rows, bankroll_remaining = _run_paper_model_evaluation(
                snapshots=snapshots,
                builder=builder,
                clob=clob,
                config=config,
                model_path=Path(str(spec["model_path"])),
                model_name=str(spec["model_name"]),
                horizon=horizon,
                horizon_policy=horizon_policy,
                bankroll=bankroll,
                edge_threshold=edge_threshold,
                max_spread_bps=spread_limit,
                min_liquidity=liquidity_floor,
                book_cache=books_by_market,
            )
            dump_json(run_dir / f"{_safe_slug(label)}.json", rows)
            summary = _summarize_result_rows(rows, bankroll_remaining=bankroll_remaining)
            model_summaries[label] = summary
            leaderboard_rows.append(
                {
                    "model": label,
                    "fills": int(summary.get("fills", 0) or 0),
                    "tradable_rows": int(summary.get("tradable_rows", 0) or 0),
                    "raw_gap_positive_count": int(summary.get("raw_gap_positive_count", 0) or 0),
                    "after_cost_edge_positive_count": int(summary.get("after_cost_edge_positive_count", 0) or 0),
                    "fee_killed_edge": int(dict(summary.get("reason_counts", {})).get("fee_killed_edge", 0) or 0),
                    "raw_gap_non_positive": int(dict(summary.get("reason_counts", {})).get("raw_gap_non_positive", 0) or 0),
                    "policy_filtered": int(dict(summary.get("reason_counts", {})).get("policy_filtered", 0) or 0),
                    "bankroll_remaining": summary.get("bankroll_remaining"),
                }
            )
    finally:
        with suppress(Exception):
            http.close()

    leaderboard_rows.sort(
        key=lambda row: (
            -int(row["fills"]),
            -int(row["tradable_rows"]),
            -int(row["after_cost_edge_positive_count"]),
            -int(row["raw_gap_positive_count"]),
            int(row["raw_gap_non_positive"]),
            str(row["model"]),
        )
    )
    summary_payload = {
        "generated_at": datetime.now(tz=UTC),
        "market_scope": market_scope,
        "horizon": horizon,
        "horizon_policy_path": str(horizon_policy_path),
        "bankroll": bankroll,
        "min_edge": edge_threshold,
        "max_spread_bps": spread_limit,
        "min_liquidity": liquidity_floor,
        "markets_evaluated": len(snapshots),
        "models": model_summaries,
        "leaderboard": leaderboard_rows,
    }
    dump_json(run_dir / "summary.json", summary_payload)
    pd.DataFrame(leaderboard_rows).to_csv(run_dir / "leaderboard.csv", index=False)

    table = Table(title="Paper Multimodel Report")
    table.add_column("Model")
    table.add_column("Fills")
    table.add_column("Tradable")
    table.add_column("Raw+")
    table.add_column("Edge+")
    table.add_column("Fee Killed")
    table.add_column("Raw<=0")
    table.add_column("Policy")
    for row in leaderboard_rows:
        table.add_row(
            str(row["model"]),
            str(row["fills"]),
            str(row["tradable_rows"]),
            str(row["raw_gap_positive_count"]),
            str(row["after_cost_edge_positive_count"]),
            str(row["fee_killed_edge"]),
            str(row["raw_gap_non_positive"]),
            str(row["policy_filtered"]),
        )
    console.print(table)
    console.print(f"Wrote {run_dir / 'summary.json'}")
    console.print(f"Wrote {run_dir / 'leaderboard.csv'}")


@app.command("execution-sensitivity-report")
def execution_sensitivity_report(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    preset_path: Path = typer.Option(
        DEFAULT_PAPER_EXPLORATION_CONFIG_PATH,
        help="Paper exploration YAML preset",
    ),
    market_scopes: Annotated[list[str] | None, typer.Option("--market-scope")] = None,
    min_edges: Annotated[list[float] | None, typer.Option("--min-edge")] = None,
    max_spread_bps_values: Annotated[list[int] | None, typer.Option("--max-spread-bps")] = None,
    min_liquidity_values: Annotated[list[float] | None, typer.Option("--min-liquidity")] = None,
    horizon_policy_paths: Annotated[list[Path] | None, typer.Option("--horizon-policy-path")] = None,
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    bankroll: float = typer.Option(10_000.0, help="Paper bankroll"),
    output_dir: Path | None = typer.Option(None, help="Output directory for sweep results"),
) -> None:
    """Sweep paper-only execution thresholds and horizon policies without touching live guards."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    markets_path = _resolve_option_value(markets_path)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    preset_path = _resolve_option_value(preset_path, DEFAULT_PAPER_EXPLORATION_CONFIG_PATH)
    market_scopes = _resolve_option_value(market_scopes)
    min_edges = _resolve_option_value(min_edges)
    max_spread_bps_values = _resolve_option_value(max_spread_bps_values)
    min_liquidity_values = _resolve_option_value(min_liquidity_values)
    horizon_policy_paths = _resolve_option_value(horizon_policy_paths)
    horizon = _resolve_option_value(horizon, "policy")
    bankroll = float(_resolve_option_value(bankroll, 10_000.0))
    output_dir = _resolve_option_value(output_dir)

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    preset = _load_paper_exploration_preset(preset_path)
    scope_values = list(market_scopes or preset.get("market_scopes") or ["default", "recent_core", "supported_wu_open_phase"])
    edge_values = list(min_edges or preset.get("min_edges") or [config.backtest.default_edge_threshold, 0.1, 0.05])
    spread_values = list(
        max_spread_bps_values
        or preset.get("max_spread_bps")
        or [config.execution.max_spread_bps, max(config.execution.max_spread_bps + 250, config.execution.max_spread_bps), 1000]
    )
    liquidity_values = list(
        min_liquidity_values
        or preset.get("min_liquidity")
        or [config.execution.min_liquidity, max(config.execution.min_liquidity / 2.0, 1.0), 0.0]
    )
    policy_entries: list[dict[str, object]] = []
    if horizon_policy_paths:
        policy_entries = [
            {"label": Path(path).stem, "path": Path(path)}
            for path in horizon_policy_paths
        ]
    else:
        raw_entries = preset.get("horizon_policies")
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                path_value = entry.get("path")
                if path_value is None:
                    continue
                policy_entries.append(
                    {
                        "label": str(entry.get("label") or Path(str(path_value)).stem),
                        "path": Path(str(path_value)),
                    }
                )
    if not policy_entries:
        policy_entries = [
            {"label": "current_policy", "path": DEFAULT_RECENT_HORIZON_POLICY_PATH},
            {"label": "all_supported_policy", "path": DEFAULT_PAPER_ALL_SUPPORTED_HORIZON_POLICY_PATH},
        ]

    run_dir = Path(output_dir) if output_dir is not None else (
        DEFAULT_EXECUTION_SENSITIVITY_ROOT / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    combos_dir = run_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

    snapshots_by_scope: dict[str, list[MarketSnapshot]] = {}
    books_by_scope: dict[str, dict[str, dict[str, BookSnapshot]]] = {}
    combo_rows: list[dict[str, object]] = []
    try:
        for scope in scope_values:
            resolved_scope = _resolve_market_scope(str(scope), core_recent_only=core_recent_only)
            scoped_cities = _resolve_scoped_cities(cities, market_scope=resolved_scope)
            snapshots = _load_scoped_snapshots(
                markets_path=markets_path,
                cities=scoped_cities,
                market_scope=resolved_scope,
                active=True,
                closed=False,
            )
            snapshots_by_scope[resolved_scope] = snapshots
            books_by_scope[resolved_scope] = {
                snapshot.spec.market_id: _load_full_books_for_snapshot(clob, snapshot)
                for snapshot in snapshots
                if snapshot.spec is not None
            }
            for policy_entry in policy_entries:
                policy_label = str(policy_entry["label"])
                policy_path = Path(str(policy_entry["path"]))
                horizon_policy = _load_recent_horizon_policy(policy_path)
                for edge_threshold in edge_values:
                    for spread_limit in spread_values:
                        for liquidity_floor in liquidity_values:
                            combo_slug = _safe_slug(
                                f"{resolved_scope}_{policy_label}_edge_{edge_threshold}_spread_{spread_limit}_liq_{liquidity_floor}"
                            )
                            rows, bankroll_remaining = _run_paper_model_evaluation(
                                snapshots=snapshots,
                                builder=builder,
                                clob=clob,
                                config=config,
                                model_path=model_path,
                                model_name=resolved_model_name,
                                horizon=horizon,
                                horizon_policy=horizon_policy,
                                bankroll=bankroll,
                                edge_threshold=float(edge_threshold),
                                max_spread_bps=int(spread_limit),
                                min_liquidity=float(liquidity_floor),
                                book_cache=books_by_scope[resolved_scope],
                            )
                            rows_path = combos_dir / f"{combo_slug}.json"
                            dump_json(rows_path, rows)
                            summary = _summarize_result_rows(rows, bankroll_remaining=bankroll_remaining)
                            combo_rows.append(
                                {
                                    "combo": combo_slug,
                                    "market_scope": resolved_scope,
                                    "horizon_policy": policy_label,
                                    "horizon_policy_path": str(policy_path),
                                    "min_edge": float(edge_threshold),
                                    "max_spread_bps": int(spread_limit),
                                    "min_liquidity": float(liquidity_floor),
                                    "fills": int(summary.get("fills", 0) or 0),
                                    "tradable_rows": int(summary.get("tradable_rows", 0) or 0),
                                    "raw_gap_positive_count": int(summary.get("raw_gap_positive_count", 0) or 0),
                                    "after_cost_edge_positive_count": int(summary.get("after_cost_edge_positive_count", 0) or 0),
                                    "fee_killed_edge": int(dict(summary.get("reason_counts", {})).get("fee_killed_edge", 0) or 0),
                                    "raw_gap_non_positive": int(dict(summary.get("reason_counts", {})).get("raw_gap_non_positive", 0) or 0),
                                    "policy_filtered": int(dict(summary.get("reason_counts", {})).get("policy_filtered", 0) or 0),
                                    "rows_path": str(rows_path),
                                    }
                                )
    finally:
        with suppress(Exception):
            http.close()

    combo_rows.sort(
        key=lambda row: (
            -int(row["fills"]),
            -int(row["tradable_rows"]),
            -int(row["after_cost_edge_positive_count"]),
            -int(row["raw_gap_positive_count"]),
            int(row["raw_gap_non_positive"]),
            int(row["policy_filtered"]),
            str(row["combo"]),
        )
    )
    summary_payload = {
        "generated_at": datetime.now(tz=UTC),
        "model_path": str(model_path),
        "model_name": resolved_model_name,
        "preset_path": str(preset_path),
        "bankroll": bankroll,
        "horizon": horizon,
        "combinations": combo_rows,
        "top_combinations": combo_rows[:10],
    }
    dump_json(run_dir / "summary.json", summary_payload)
    pd.DataFrame(combo_rows).to_csv(run_dir / "leaderboard.csv", index=False)

    table = Table(title="Execution Sensitivity Report")
    table.add_column("Scope")
    table.add_column("Policy")
    table.add_column("Edge")
    table.add_column("Spread")
    table.add_column("Liquidity")
    table.add_column("Fills")
    table.add_column("Tradable")
    table.add_column("Fee Killed")
    table.add_column("Raw<=0")
    for row in combo_rows[:10]:
        table.add_row(
            str(row["market_scope"]),
            str(row["horizon_policy"]),
            f"{float(row['min_edge']):.3f}",
            str(row["max_spread_bps"]),
            f"{float(row['min_liquidity']):.1f}",
            str(row["fills"]),
            str(row["tradable_rows"]),
            str(row["fee_killed_edge"]),
            str(row["raw_gap_non_positive"]),
        )
    console.print(table)
    console.print(f"Wrote {run_dir / 'summary.json'}")
    console.print(f"Wrote {run_dir / 'leaderboard.csv'}")


def _load_report_rows(path: Path) -> list[dict[str, object]]:
    """Load one row-oriented report from JSON or JSONL."""

    if not path.exists():
        msg = f"Report input does not exist: {path}"
        raise FileNotFoundError(msg)
    if path.suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    payload = load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        nested_rows = payload.get("rows")
        if isinstance(nested_rows, list):
            return [row for row in nested_rows if isinstance(row, dict)]
    msg = f"Unsupported report payload shape in {path}"
    raise ValueError(msg)


def _leaderboard_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows = summary.get("leaderboard", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _summary_model_rows(summary_path: Path, model_label: str) -> list[dict[str, object]]:
    """Load one per-model JSON emitted next to a paper multimodel summary."""

    model_path = summary_path.parent / f"{_safe_slug(model_label)}.json"
    return _load_report_rows(model_path)


def _best_ask_from_row(row: dict[str, object]) -> float | None:
    return _row_float(row, "best_ask") or _row_float(row, "executable_price")


def _break_even_threshold_ask(row: dict[str, object]) -> float | None:
    best_ask = _best_ask_from_row(row)
    after_cost_edge = _row_float(row, "after_cost_edge")
    if best_ask is None or after_cost_edge is None:
        return None
    return round(max(best_ask + after_cost_edge, 0.0), 6)


def _choose_fee_watchlist_model(summary: Mapping[str, object]) -> str | None:
    leaderboard = _leaderboard_rows(summary)
    if not leaderboard:
        return None
    ranked = sorted(
        leaderboard,
        key=lambda row: (
            -int(row.get("fee_killed_edge", 0) or 0),
            -int(row.get("raw_gap_positive_count", 0) or 0),
            int(row.get("raw_gap_non_positive", 0) or 0),
            str(row.get("model", "")),
        ),
    )
    model = ranked[0].get("model")
    return str(model) if model else None


def _top_watchlist_examples(
    rows: list[dict[str, object]],
    *,
    cities: set[str],
    limit: int,
    include_model: str | None = None,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_row in rows:
        row = _normalize_report_row(raw_row)
        city = str(row.get("city") or "")
        if cities and city not in cities:
            continue
        if str(row.get("reason") or "") != "fee_killed_edge":
            continue
        threshold = _break_even_threshold_ask(row)
        best_ask = _best_ask_from_row(row)
        if threshold is None or best_ask is None:
            continue
        key = (
            str(row.get("market_id") or ""),
            str(row.get("outcome_label") or ""),
            str(row.get("decision_horizon") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        payload = {
            "city": city,
            "market_id": row.get("market_id"),
            "target_local_date": row.get("target_local_date"),
            "decision_horizon": row.get("decision_horizon"),
            "outcome_label": row.get("outcome_label"),
            "best_ask": round(best_ask, 6),
            "raw_gap": _row_float(row, "raw_gap"),
            "after_cost_edge": _row_float(row, "after_cost_edge"),
            "fee_estimate": _row_float(row, "fee_estimate"),
            "spread": _row_float(row, "spread"),
            "watch_rule_threshold_ask": threshold,
        }
        if include_model is not None:
            payload["model"] = include_model
        examples.append(payload)
    examples.sort(
        key=lambda item: (
            str(item.get("city", "")),
            str(item.get("target_local_date", "")),
            str(item.get("decision_horizon", "")),
            str(item.get("outcome_label", "")),
        )
    )
    return examples[:limit]


def _bottleneck_cities(
    payload: Mapping[str, object],
    *,
    key: str,
    limit: int,
) -> list[str]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return []
    rows = summary.get(key, [])
    if not isinstance(rows, list):
        return []
    cities: list[str] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        city = row.get("city")
        if city:
            cities.append(str(city))
    return cities


def _execution_sensitivity_headline(summary: Mapping[str, object]) -> dict[str, object]:
    combinations = summary.get("combinations", [])
    if not isinstance(combinations, list):
        combinations = []
    max_fills = max((int(item.get("fills", 0) or 0) for item in combinations if isinstance(item, Mapping)), default=0)
    min_raw_gap_non_positive = min(
        (
            int(item.get("raw_gap_non_positive", 0) or 0)
            for item in combinations
            if isinstance(item, Mapping)
        ),
        default=0,
    )
    return {
        "max_fills": max_fills,
        "min_raw_gap_non_positive": min_raw_gap_non_positive,
    }


def _build_execution_watchlist_playbook(
    *,
    champion_bottleneck: Mapping[str, object],
    challenger_bottleneck: Mapping[str, object],
    fee_watchlist_summary_path: Path,
    policy_watchlist_summary_path: Path | None,
    sensitivity_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    fee_watchlist_summary = load_json(fee_watchlist_summary_path)
    if not isinstance(fee_watchlist_summary, dict):
        msg = f"Unsupported fee watchlist summary payload: {fee_watchlist_summary_path}"
        raise ValueError(msg)
    policy_watchlist_summary = _load_optional_json(policy_watchlist_summary_path) if policy_watchlist_summary_path else None

    tier_a_cities = _bottleneck_cities(challenger_bottleneck, key="fee_sensitive_watchlist", limit=5)
    tier_b_cities = _bottleneck_cities(challenger_bottleneck, key="policy_blocked_watchlist", limit=5)
    tier_c_cities = _bottleneck_cities(challenger_bottleneck, key="raw_edge_desert_watchlist", limit=10)

    fee_model = _choose_fee_watchlist_model(fee_watchlist_summary)
    if fee_model is None:
        msg = f"No leaderboard rows found in {fee_watchlist_summary_path}"
        raise ValueError(msg)
    fee_model_rows = _summary_model_rows(fee_watchlist_summary_path, fee_model)
    tier_a_evidence = _top_watchlist_examples(
        fee_model_rows,
        cities=set(tier_a_cities),
        limit=10,
    )

    tier_b_evidence: list[dict[str, object]] = []
    if isinstance(policy_watchlist_summary, Mapping):
        for row in _leaderboard_rows(policy_watchlist_summary):
            if int(row.get("fee_killed_edge", 0) or 0) <= 0:
                continue
            model_label = row.get("model")
            if not model_label:
                continue
            tier_b_evidence.extend(
                _top_watchlist_examples(
                    _summary_model_rows(policy_watchlist_summary_path, str(model_label)),
                    cities=set(tier_b_cities),
                    limit=3,
                    include_model=str(model_label),
                )
            )
    sensitivity_headline = _execution_sensitivity_headline(sensitivity_summary or {})

    champion_summary = champion_bottleneck.get("summary", {})
    if not isinstance(champion_summary, Mapping):
        champion_summary = {}

    playbook = {
        "generated_at": datetime.now(tz=UTC),
        "objective": "convert zero-fill diagnostics into a concrete monitoring playbook",
        "headline": {
            "champion_dominant_blocker": dict(champion_summary.get("reason_counts", {})),
            "fee_sensitive_subset_leaderboard": _leaderboard_rows(fee_watchlist_summary),
            "policy_subset_leaderboard": _leaderboard_rows(policy_watchlist_summary or {}),
            "sensitivity_max_fills": sensitivity_headline["max_fills"],
            "sensitivity_min_raw_gap_non_positive": sensitivity_headline["min_raw_gap_non_positive"],
            "fee_watchlist_model": fee_model,
        },
        "playbook": [
            {
                "tier": "A",
                "name": "fee_sensitive_watchlist",
                "cities": tier_a_cities,
                "why": "raw gap exists, but the book is pinned near 0.99 and fees kill the edge",
                "rule": "do not trade these unless the best ask falls to roughly 0.90 or lower on the target outcome; current 0.99 anchors are not close",
                "evidence": tier_a_evidence,
            },
            {
                "tier": "B",
                "name": "policy_blocked_watchlist",
                "cities": tier_b_cities,
                "why": "current recent-core policy hides some horizons, but lifting the policy still does not create fills",
                "rule": "keep the live policy as-is; use all-supported policy only for paper observation on these cities",
                "evidence": tier_b_evidence[:10],
            },
            {
                "tier": "C",
                "name": "raw_edge_desert",
                "cities": tier_c_cities,
                "why": "even before fees, the market is not mispriced often enough",
                "rule": "deprioritize these cities for manual monitoring until observation-driven dislocation or a fresh listing appears",
            },
            {
                "tier": "D",
                "name": "threshold_relaxation_result",
                "cities": [],
                "why": "paper-only threshold and policy sweeps still produced zero tradable rows at the tested scope",
                "rule": "do not loosen live thresholds in response to zero fills; the issue is missing raw edge, not guardrails",
            },
        ],
        "next_actions": [
            "Run observation-shadow long enough to replace the current no_observations state with source-family counts.",
            "Keep fee-sensitive cities on the station dashboard, but only alert when best ask drops below the watch_rule_threshold_ask.",
            "Do not widen live spread or lower live edge thresholds until a paper subset produces non-zero tradable rows.",
        ],
    }
    return playbook


def _render_execution_watchlist_playbook_markdown(playbook: Mapping[str, object]) -> str:
    lines = [
        "# Execution Watchlist Playbook",
        "",
        f"Generated: {playbook.get('generated_at', '')}",
        "",
        "## Headline",
        "",
    ]
    headline = playbook.get("headline", {})
    if isinstance(headline, Mapping):
        lines.append(f"- Champion blockers: {dict(headline.get('champion_dominant_blocker', {}))}")
        lines.append(f"- Sensitivity max fills: {headline.get('sensitivity_max_fills', 0)}")
        lines.append(
            f"- Sensitivity min raw<=0: {headline.get('sensitivity_min_raw_gap_non_positive', 0)}"
        )
        lines.append(f"- Fee-watch model: {headline.get('fee_watchlist_model', '')}")
    lines.append("")

    entries = playbook.get("playbook", [])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            lines.append(f"## Tier {entry.get('tier', '?')}: {entry.get('name', '')}")
            lines.append("")
            cities = entry.get("cities", [])
            if isinstance(cities, list) and cities:
                lines.append(f"- Cities: {', '.join(str(city) for city in cities)}")
            lines.append(f"- Rule: {entry.get('rule', '')}")
            evidence = entry.get("evidence", [])
            if isinstance(evidence, list) and evidence:
                lines.append("")
                lines.append("Top examples:")
                for row in evidence[:5]:
                    if not isinstance(row, Mapping):
                        continue
                    alert_threshold = row.get("watch_rule_threshold_ask")
                    current_ask = row.get("best_ask")
                    lines.append(
                        "- "
                        f"{row.get('city', '')} {row.get('target_local_date', '')} {row.get('outcome_label', '')}: "
                        f"ask={current_ask}, after_cost_edge={row.get('after_cost_edge')}, "
                        f"alert only if ask<={alert_threshold}"
                    )
            lines.append("")

    next_actions = playbook.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.append("## Next Actions")
        lines.append("")
        for item in next_actions:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


@app.command("market-bottleneck-report")
def market_bottleneck_report(
    input_path: Path = typer.Option(Path("artifacts/signals/v2/paper_signals.json"), help="Row-oriented report JSON or JSONL"),
    opportunity_summary_path: Path | None = typer.Option(None, help="Opportunity shadow summary JSON"),
    observation_summary_path: Path | None = typer.Option(None, help="Observation shadow summary JSON"),
    output: Path = typer.Option(Path("artifacts/signals/v2/market_bottleneck_report.json"), help="Output JSON"),
) -> None:
    """Summarize the dominant blockers for one paper/opportunity/observation result set."""

    input_path = _resolve_option_value(input_path, Path("artifacts/signals/v2/paper_signals.json"))
    opportunity_summary_path = _resolve_option_value(opportunity_summary_path)
    observation_summary_path = _resolve_option_value(observation_summary_path)
    output = _resolve_option_value(output, Path("artifacts/signals/v2/market_bottleneck_report.json"))

    rows = _load_report_rows(input_path)
    summary = _summarize_result_rows(rows)
    payload = {
        "generated_at": datetime.now(tz=UTC),
        "input_path": str(input_path),
        "summary": summary,
        "shadow_context": {
            "opportunity": _load_optional_json(opportunity_summary_path) if opportunity_summary_path else None,
            "observation": _load_optional_json(observation_summary_path) if observation_summary_path else None,
        },
    }
    dump_json(output, payload)

    table = Table(title="Market Bottleneck Report")
    table.add_column("Bucket")
    table.add_column("Count")
    reason_counts = dict(summary.get("reason_counts", {}))
    for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        table.add_row(str(reason), str(count))
    console.print(table)
    console.print(f"Wrote {output}")


@app.command("execution-watchlist-playbook")
def execution_watchlist_playbook(
    champion_bottleneck_path: Path = typer.Option(
        Path("artifacts/signals/v2/market_bottleneck_report__champion_alias.json"),
        help="Champion market-bottleneck-report JSON",
    ),
    challenger_bottleneck_path: Path = typer.Option(
        Path("artifacts/signals/v2/market_bottleneck_report__mega_neighbor_oof.json"),
        help="Challenger market-bottleneck-report JSON",
    ),
    fee_watchlist_summary_path: Path = typer.Option(
        ...,
        help="paper-multimodel summary.json for the fee-sensitive city subset",
    ),
    policy_watchlist_summary_path: Path | None = typer.Option(
        None,
        help="Optional paper-multimodel summary.json for the policy-blocked city subset",
    ),
    sensitivity_summary_path: Path | None = typer.Option(
        None,
        help="Optional execution-sensitivity summary.json for the guardrail sweep subset",
    ),
    output: Path = typer.Option(
        DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_PATH,
        help="Watchlist playbook JSON output",
    ),
    markdown_output: Path = typer.Option(
        DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_MD_PATH,
        help="Watchlist playbook Markdown output",
    ),
) -> None:
    """Convert zero-fill diagnostics into one execution watchlist playbook."""

    champion_bottleneck_path = _resolve_option_value(
        champion_bottleneck_path,
        Path("artifacts/signals/v2/market_bottleneck_report__champion_alias.json"),
    )
    challenger_bottleneck_path = _resolve_option_value(
        challenger_bottleneck_path,
        Path("artifacts/signals/v2/market_bottleneck_report__mega_neighbor_oof.json"),
    )
    fee_watchlist_summary_path = _resolve_option_value(fee_watchlist_summary_path)
    policy_watchlist_summary_path = _resolve_option_value(policy_watchlist_summary_path)
    sensitivity_summary_path = _resolve_option_value(sensitivity_summary_path)
    output = _resolve_option_value(output, DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_PATH)
    markdown_output = _resolve_option_value(markdown_output, DEFAULT_EXECUTION_WATCHLIST_PLAYBOOK_MD_PATH)

    champion_bottleneck = load_json(champion_bottleneck_path)
    challenger_bottleneck = load_json(challenger_bottleneck_path)
    if not isinstance(champion_bottleneck, dict) or not isinstance(challenger_bottleneck, dict):
        msg = "Bottleneck reports must be JSON objects."
        raise typer.BadParameter(msg)
    sensitivity_summary = _load_optional_json(sensitivity_summary_path) if sensitivity_summary_path else None
    playbook = _build_execution_watchlist_playbook(
        champion_bottleneck=champion_bottleneck,
        challenger_bottleneck=challenger_bottleneck,
        fee_watchlist_summary_path=fee_watchlist_summary_path,
        policy_watchlist_summary_path=policy_watchlist_summary_path,
        sensitivity_summary=sensitivity_summary,
    )
    dump_json(output, playbook)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_render_execution_watchlist_playbook_markdown(playbook))

    tier_a_entry = next(
        (
            entry
            for entry in list(playbook.get("playbook", []))
            if isinstance(entry, dict) and entry.get("name") == "fee_sensitive_watchlist"
        ),
        {},
    )
    table = Table(title="Execution Watchlist Playbook")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("fee_watchlist_model", str(dict(playbook.get("headline", {})).get("fee_watchlist_model", "")))
    table.add_row("tier_a_cities", ", ".join(str(city) for city in list(dict(tier_a_entry).get("cities", []))))
    table.add_row("tier_a_examples", str(len(list(dict(tier_a_entry).get("evidence", [])))))
    table.add_row("json_output", str(output))
    table.add_row("markdown_output", str(markdown_output))
    console.print(table)
    console.print(f"Wrote {output}")
    console.print(f"Wrote {markdown_output}")


@app.command("live-trader")
def live_trader(
    model_path: Path = Path("artifacts/models/v2/champion.pkl"),
    model_name: str = DEFAULT_MODEL_NAME,
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    horizon: str = "policy",
    dry_run: bool = True,
    post_orders: bool = False,
) -> None:
    """Run live preflight and signed-order previews, with optional posting."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    markets_path = _resolve_option_value(markets_path)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    horizon = _resolve_option_value(horizon, "policy")
    dry_run = bool(_resolve_option_value(dry_run, True))
    post_orders = bool(_resolve_option_value(post_orders, False))
    config, env, http, _, _, openmeteo = _runtime(include_stores=False)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    broker = LiveBroker(env)
    preflight = broker.preflight(require_posting=post_orders and not dry_run)

    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    cities = _resolve_recent_core_cities(cities, core_recent_only=core_recent_only)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    horizon_policy = _load_recent_horizon_policy()

    previews: list[dict[str, object]] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        now_utc = datetime.now(tz=UTC)
        if spec is None or spec.target_local_date < now_utc.date():
            continue
        decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
            spec,
            now_utc=now_utc,
            horizon=horizon,
            horizon_policy=horizon_policy,
        )
        if horizon_reason == "policy_filtered":
            previews.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "decision_horizon": decision_horizon,
                    "reason": "policy_filtered",
                }
            )
            continue
        if decision_horizon is None:
            continue
        feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
        forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)
        forecast_rejection_reason = _forecast_contract_rejection_reason(forecast)
        if forecast_rejection_reason is not None:
            previews.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "decision_horizon": decision_horizon,
                    "forecast_contract_version": getattr(forecast, "contract_version", "v2"),
                    "probability_source": getattr(forecast, "probability_source", "raw"),
                    "distribution_family": getattr(forecast, "distribution_family", "gaussian"),
                    "reason": forecast_rejection_reason,
                }
            )
            continue
        books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
        evaluation = _evaluate_market_signal(
            snapshot,
            forecast.outcome_probabilities,
            books,
            mode="live",
            clob=clob,
            default_fee_bps=_default_fee_bps(config),
            edge_threshold=config.backtest.default_edge_threshold,
            max_spread_bps=config.execution.max_spread_bps,
            min_liquidity=config.execution.min_liquidity,
            forecast_contract_version=getattr(forecast, "contract_version", "v2"),
            probability_source=getattr(forecast, "probability_source", "raw"),
            distribution_family=getattr(forecast, "distribution_family", "gaussian"),
            decision_horizon=decision_horizon,
        )
        signal = evaluation["signal"]
        if not isinstance(signal, TradeSignal):
            previews.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "decision_horizon": decision_horizon,
                    "reason": evaluation["reason"],
                    "book_source_counts": evaluation["book_source_counts"],
                }
            )
            continue
        size_notional = capped_kelly(signal.edge, signal.fair_probability, 1000.0, signal.executable_price)
        size = size_notional / max(signal.executable_price, 1e-6)
        if size <= 0:
            continue
        try:
            preview = broker.preview_limit_order(signal, size=size)
        except Exception as exc:  # noqa: BLE001
            preview = {
                "market_id": signal.market_id,
                "token_id": signal.token_id,
                "outcome_label": signal.outcome_label,
                "decision_horizon": decision_horizon,
                "error": str(exc),
            }
        if post_orders and not dry_run and preflight.ok:
            preview["post_result"] = broker.post_limit_order(signal, size=size)
        preview.setdefault("decision_horizon", decision_horizon)
        previews.append(preview)

    payload = {
        "preflight": preflight.model_dump(mode="json"),
        "orders": previews,
    }
    dump_json(_default_signal_output("live_trader_preview.json"), payload)
    console.print_json(data=payload)


@app.command("collect-l2")
def collect_l2(
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    output: Path = Path("artifacts/l2_snapshot.json"),
) -> None:
    """Archive one current book snapshot for discovered active markets."""

    config, _, http, _, _, _ = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    collected: list[dict[str, object]] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        for outcome_label, token_id in zip(spec.outcome_labels(), spec.token_ids, strict=False):
            book = _fetch_book(clob, snapshot, token_id, outcome_label)
            collected.append(book.model_dump(mode="json"))
    dump_json(output, collected)
    console.print(f"Wrote {output}")


@app.command("scan-daemon")
def scan_daemon(
    model_path: Path = typer.Argument(..., help="Path to trained model artifact"),
    model_name: str = typer.Option("gaussian_emos", help="Model name"),
    interval: int = typer.Option(60, help="Seconds between scan cycles"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    bankroll: float = typer.Option(10_000.0, help="Starting bankroll"),
    max_cycles: int = typer.Option(0, help="Max cycles (0 = infinite)"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
) -> None:
    """Run a continuous scanning daemon that monitors markets and manages positions."""

    import logging

    from pmtmax.execution.scanner import ContinuousScanner

    logger = logging.getLogger("pmtmax.cli.scan_daemon")
    resolved_model_name = require_supported_model_name(model_name)

    config, _env, http, _duckdb, _parquet, openmeteo = _runtime(include_stores=False)

    broker = PaperBroker(
        bankroll=bankroll,
        stop_loss_pct=config.execution.stop_loss_pct,
        trailing_stop_rise_pct=config.execution.trailing_stop_rise_pct,
        forecast_exit_buffer=config.execution.forecast_exit_buffer,
    )
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None, snapshot_dir=None, fixture_dir=None,
        models=config.weather.models or None,
    )
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    refresh_interval = config.scanner.snapshot_refresh_interval
    horizon_policy = _load_recent_horizon_policy()

    # -- snapshot management --------------------------------------------------
    snapshot_cache: list[MarketSnapshot] = []

    def _refresh_snapshots() -> list[MarketSnapshot]:
        if markets_path is not None:
            if not snapshot_cache:
                return _filter_snapshots_by_city(load_market_snapshots(markets_path), cities)
            return snapshot_cache
        gamma = GammaClient(http, config.polymarket.gamma_base_url)
        refs = discover_temperature_event_refs_from_gamma(
            gamma,
            supported_cities=cities or config.app.supported_cities,
            active=True,
            closed=False,
            max_pages=config.polymarket.max_pages,
        )
        fetches = fetch_temperature_event_pages(http, refs, use_cache=False)
        return _filter_snapshots_by_city(snapshots_from_temperature_event_fetches(fetches), cities)

    snapshot_cache = _refresh_snapshots()
    logger.info("Initial snapshot load: %d markets", len(snapshot_cache))

    # -- callback: price_fetcher ----------------------------------------------
    def _price_fetcher() -> dict[str, float]:
        prices: dict[str, float] = {}
        for token_id in list(broker.positions):
            try:
                resp = clob.get_price(token_id, side="buy")
                prices[token_id] = float(resp.get("price", 0.0))
            except Exception:  # noqa: BLE001
                logger.warning("price fetch failed for %s", token_id)
        return prices

    # -- callback: forecast_fetcher -------------------------------------------
    def _forecast_fetcher() -> dict[str, ProbForecast]:
        forecasts: dict[str, ProbForecast] = {}
        seen_market_ids = {pos.market_id for pos in broker.positions.values()}
        for snapshot in snapshot_cache:
            spec = snapshot.spec
            if spec is None:
                continue
            now_utc = datetime.now(tz=UTC)
            decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                spec,
                now_utc=now_utc,
                horizon=horizon,
                horizon_policy=horizon_policy,
            )
            if horizon_reason == "policy_filtered":
                continue
            if decision_horizon is None:
                continue
            if spec.market_id not in seen_market_ids:
                continue
            try:
                feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
                forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)
                forecasts[spec.market_id] = forecast
            except Exception:  # noqa: BLE001
                logger.warning("forecast failed for market %s", spec.market_id)
        return forecasts

    # -- callback: entry_evaluator --------------------------------------------
    current_exposure_by_city: dict[str, float] = {}

    def _entry_evaluator(brk: PaperBroker) -> None:
        nonlocal snapshot_cache

        # Refresh snapshots periodically
        if scanner._cycle > 0 and scanner._cycle % refresh_interval == 0:
            try:
                snapshot_cache = _refresh_snapshots()
                logger.info("Refreshed snapshots: %d markets (cycle %d)", len(snapshot_cache), scanner._cycle)
            except Exception:  # noqa: BLE001
                logger.warning("Snapshot refresh failed at cycle %d, keeping cached", scanner._cycle)

        held_market_ids = {pos.market_id for pos in brk.positions.values()}

        for snapshot in snapshot_cache:
            spec = snapshot.spec
            if spec is None:
                continue
            now_utc = datetime.now(tz=UTC)
            if spec.target_local_date < now_utc.date():
                continue
            if spec.market_id in held_market_ids:
                continue
            decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                spec,
                now_utc=now_utc,
                horizon=horizon,
                horizon_policy=horizon_policy,
            )
            if horizon_reason == "policy_filtered":
                continue
            if decision_horizon is None:
                continue
            try:
                feature_frame = builder.build_live_row(spec, horizon=decision_horizon)
                forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)
            except Exception:  # noqa: BLE001
                logger.warning("forecast failed for %s — skipping entry", spec.market_id)
                continue
            if not forecast_fresh(forecast.generated_at.replace(tzinfo=None), config.execution.stale_forecast_minutes):
                continue
            books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
            evaluation = _evaluate_market_signal(
                snapshot,
                forecast.outcome_probabilities,
                books,
                mode="paper",
                clob=clob,
                default_fee_bps=_default_fee_bps(config),
                edge_threshold=edge_threshold,
                max_spread_bps=config.execution.max_spread_bps,
                min_liquidity=config.execution.min_liquidity,
            )
            signal = evaluation["signal"]
            if not isinstance(signal, TradeSignal) or not isinstance(evaluation["book"], BookSnapshot):
                continue
            if evaluation["reason"] != "tradable":
                continue
            size_notional = capped_kelly(signal.edge, signal.fair_probability, brk.bankroll, signal.executable_price)
            size = size_notional / max(signal.executable_price, 1e-6)
            current_city_exposure = current_exposure_by_city.get(spec.city, 0.0)
            if not exposure_ok(current_city_exposure, size_notional, config.execution.max_city_exposure):
                continue
            if not exposure_ok(sum(current_exposure_by_city.values()), size_notional, config.execution.global_max_exposure):
                continue
            fill = brk.simulate_fill(signal, book=evaluation["book"], size=size)
            if fill is not None:
                current_exposure_by_city[spec.city] = current_city_exposure + size_notional
                logger.info("Entry fill: %s %s edge=%.4f", spec.city, signal.outcome_label, signal.edge)

    # -- wire up scanner and run ----------------------------------------------
    state_path = config.scanner.state_path
    effective_interval = interval or config.scanner.interval_seconds
    effective_max_cycles = max_cycles if max_cycles > 0 else config.scanner.max_cycles

    scanner = ContinuousScanner(
        config=config,
        broker=broker,
        interval_seconds=effective_interval,
        max_cycles=effective_max_cycles,
        state_path=state_path,
        price_fetcher=_price_fetcher,
        forecast_fetcher=_forecast_fetcher,
        entry_evaluator=_entry_evaluator,
    )

    console.print(f"Starting scan daemon: interval={effective_interval}s, max_cycles={effective_max_cycles}")
    console.print(f"Monitoring {len(snapshot_cache)} markets, refresh every {refresh_interval} cycles")
    scanner.run_loop()
    console.print("Scan daemon stopped.")


# ===========================================================================
# Track A: L2 monitoring commands
# ===========================================================================


@app.command("monitor-l2")
def monitor_l2(
    interval: int = typer.Option(1800, help="Seconds between collection cycles"),
    window_hours: float = typer.Option(48.0, help="Only collect markets settling within this window"),
    max_cycles: int = typer.Option(0, help="Max cycles (0 = infinite)"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
) -> None:
    """Continuously monitor L2 order books for markets near settlement."""

    import signal as sig
    import time

    from pmtmax.monitoring.l2_monitor import append_records_jsonl, collect_l2_snapshots

    config, _, http, _, _, _ = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    output_dir = config.monitoring.l2_output_dir
    effective_interval = interval or config.monitoring.l2_interval_seconds
    effective_window = window_hours or config.monitoring.l2_settlement_window_hours

    running = True

    def _shutdown(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    sig.signal(sig.SIGINT, _shutdown)
    sig.signal(sig.SIGTERM, _shutdown)

    cycle = 0
    console.print(f"L2 monitor: interval={effective_interval}s, window={effective_window}h, max_cycles={max_cycles}")

    while running:
        snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
        records = collect_l2_snapshots(clob, snapshots, settlement_window_hours=effective_window)

        if records:
            path = append_records_jsonl(records, output_dir)
            console.print(f"Cycle {cycle}: {len(records)} records → {path}")

            table = Table(title=f"L2 Snapshot — Cycle {cycle}")
            table.add_column("City")
            table.add_column("Outcome")
            table.add_column("Bid")
            table.add_column("Ask")
            table.add_column("Spread")
            table.add_column("Hours Left")
            for r in records[:20]:
                table.add_row(
                    r.city,
                    r.outcome_label,
                    f"{r.best_bid:.4f}",
                    f"{r.best_ask:.4f}",
                    f"{r.spread:.4f}",
                    f"{r.hours_to_settlement:.1f}",
                )
            console.print(table)
        else:
            console.print(f"Cycle {cycle}: no markets within {effective_window}h window")

        cycle += 1
        if 0 < max_cycles <= cycle:
            break
        if running:
            time.sleep(effective_interval)

    console.print("L2 monitor stopped.")


@app.command("analyze-l2")
def analyze_l2(
    data_dir: Path = typer.Option(Path("data/l2_timeseries"), help="L2 data directory"),
    output: Path = typer.Option(Path("artifacts/l2_analysis.json"), help="Output analysis JSON"),
) -> None:
    """Analyze collected L2 time-series data by hours-to-settlement buckets."""

    from pmtmax.monitoring.l2_monitor import analyze_l2_timeseries

    config, _, _, _, _, _ = _runtime(include_stores=False)
    effective_dir = data_dir if data_dir != Path("data/l2_timeseries") else config.monitoring.l2_output_dir
    analysis = analyze_l2_timeseries(effective_dir)

    dump_json(output, analysis)

    table = Table(title="L2 Analysis by Hours-to-Settlement")
    table.add_column("Bucket")
    table.add_column("Count")
    table.add_column("Median Spread")
    table.add_column("Mean Bid Depth")
    table.add_column("Mean Ask Depth")
    table.add_column("Tradeable %")
    for bucket in analysis.get("buckets", []):
        table.add_row(
            bucket["bucket"],
            str(bucket["count"]),
            f"{bucket['median_spread']:.4f}" if bucket["median_spread"] is not None else "—",
            f"{bucket['mean_bid_depth']:.2f}" if bucket["mean_bid_depth"] is not None else "—",
            f"{bucket['mean_ask_depth']:.2f}" if bucket["mean_ask_depth"] is not None else "—",
            f"{bucket['tradeable_pct']:.1f}%" if bucket["tradeable_pct"] is not None else "—",
        )
    console.print(table)
    console.print(f"Total records: {analysis.get('total_records', 0)}")
    console.print(f"Output: {output}")


# ===========================================================================
# Track B: forecast information service commands
# ===========================================================================


@app.command("forecast-report")
def forecast_report(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
    telegram: bool = typer.Option(False, help="Send to Telegram"),
    firebase: bool = typer.Option(False, help="Publish to Firebase"),
    output: Path = typer.Option(Path("artifacts/forecast_report.json"), help="Output JSON"),
) -> None:
    """Generate a one-shot forecast report with optional Telegram/Firebase delivery."""

    from pmtmax.services.forecast_summary import build_forecast_summaries

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None,
        snapshot_dir=None, fixture_dir=None, models=config.weather.models or None,
    )
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    summaries = build_forecast_summaries(snapshots, model_path, resolved_model_name, clob, builder, horizon=horizon)

    # Rich table
    table = Table(title="Forecast Report")
    table.add_column("City")
    table.add_column("Date")
    table.add_column("Mean ± Std")
    table.add_column("Top Outcome")
    table.add_column("Top Mispricing")
    for s in summaries:
        top_str = ""
        if s.top_outcomes:
            top = s.top_outcomes[0]
            top_str = f"{top.get('label', '?')} ({top.get('prob', 0):.1%})"
        mis_str = ""
        if s.mispricings:
            mp = s.mispricings[0]
            mis_str = f"{mp.outcome_label} edge={mp.edge:+.1%}"
        table.add_row(s.city, str(s.target_local_date), f"{s.mean_f} ± {s.std_f}", top_str, mis_str)
    console.print(table)

    # Save
    payload = [s.model_dump(mode="json") for s in summaries]
    dump_json(output, payload)
    console.print(f"Wrote {output}")

    # Telegram
    if telegram and config.telegram.enabled:
        from pmtmax.services.telegram_bot import TelegramNotifier

        notifier = TelegramNotifier(config.telegram.bot_token, config.telegram.chat_id)
        notifier.send_forecast_report(summaries)
        console.print("Sent Telegram notifications")
    elif telegram:
        console.print("[yellow]Telegram not configured (set PMTMAX_TELEGRAM_BOT_TOKEN and PMTMAX_TELEGRAM_CHAT_ID)[/]")

    # Firebase
    if firebase and config.firebase.enabled:
        from pmtmax.services.forecast_publisher import ForecastPublisher

        publisher = ForecastPublisher(
            bucket_name=config.firebase.bucket_name,
            prefix=config.firebase.prefix,
            credentials_json=config.firebase.credentials_json or None,
        )
        result = publisher.publish(summaries, dry_run=False)
        console.print(f"Published {result['count']} forecasts to Firebase")
    elif firebase:
        console.print("[yellow]Firebase not configured[/]")


@app.command("opportunity-report")
def opportunity_report(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path = typer.Option(Path("artifacts/signals/v2/opportunity_report.json"), help="Output JSON"),
) -> None:
    """Generate a one-shot active-market opportunity report with explicit book status."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    markets_path = _resolve_option_value(markets_path)
    horizon = _resolve_option_value(horizon, "policy")
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output, Path("artifacts/signals/v2/opportunity_report.json"))
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    cities = _resolve_scoped_cities(cities, market_scope=market_scope)
    snapshots = _load_scoped_snapshots(
        markets_path=markets_path,
        cities=cities,
        market_scope=market_scope,
        active=True,
        closed=False,
    )
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    horizon_policy = _load_recent_horizon_policy(horizon_policy_path)

    try:
        observations: list[OpportunityObservation] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            now_utc = datetime.now(tz=UTC)
            if spec is None or spec.target_local_date < now_utc.date():
                continue
            decision_horizon, horizon_reason = _resolve_signal_horizon_with_reason(
                spec,
                now_utc=now_utc,
                horizon=horizon,
                horizon_policy=horizon_policy,
            )
            if horizon_reason == "policy_filtered":
                observations.append(
                    _empty_opportunity_observation(
                        snapshot,
                        observed_at=now_utc,
                        decision_horizon=decision_horizon or "policy",
                        reason="policy_filtered",
                    )
                )
                continue
            if decision_horizon is None:
                continue
            observation = _evaluate_opportunity_snapshot(
                snapshot,
                builder=builder,
                clob=clob,
                model_path=model_path,
                model_name=resolved_model_name,
                config=config,
                observed_at=now_utc,
                decision_horizon=decision_horizon,
                edge_threshold=edge_threshold,
            )
            if observation is not None:
                observations.append(observation)

        rows = [_serialize_opportunity_observation(observation) for observation in observations]
        rows.sort(key=lambda row: (row.get("reason") != "tradable", -(float(row.get("edge", -999.0)) if row.get("edge") is not None else -999.0)))
        dump_json(output, rows)

        table = Table(title="Opportunity Report")
        table.add_column("City")
        table.add_column("Date")
        table.add_column("Outcome")
        table.add_column("Edge")
        table.add_column("Book")
        table.add_column("Reason")
        for row in rows[:20]:
            edge = row.get("edge")
            table.add_row(
                str(row.get("city", "")),
                str(row.get("target_local_date", "")),
                str(row.get("outcome_label", "—")),
                f"{float(edge):+.4f}" if edge is not None else "—",
                str(row.get("book_source", "—")),
                str(row.get("reason", "")),
            )
        console.print(table)
        console.print(f"Wrote {output}")
    finally:
        http.close()


@app.command("opportunity-shadow")
def opportunity_shadow(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between shadow cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    near_term_days: int | None = typer.Option(None, help="How many days ahead to include beyond local today"),
    horizon_policy_path: Path = typer.Option(
        DEFAULT_RECENT_HORIZON_POLICY_PATH,
        help="City-level horizon policy YAML",
    ),
    output: Path | None = typer.Option(None, help="Append-only JSONL output"),
    latest_output: Path | None = typer.Option(None, help="Latest cycle JSON output"),
    summary_output: Path | None = typer.Option(None, help="Summary JSON output"),
    state_path: Path | None = typer.Option(None, help="State JSON path"),
) -> None:
    """Continuously validate whether the live opportunity path ever becomes tradable."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    near_term_days = _resolve_option_value(near_term_days)
    horizon_policy_path = _resolve_option_value(horizon_policy_path, DEFAULT_RECENT_HORIZON_POLICY_PATH)
    output = _resolve_option_value(output)
    latest_output = _resolve_option_value(latest_output)
    summary_output = _resolve_option_value(summary_output)
    state_path = _resolve_option_value(state_path)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    shadow_config = config.opportunity_shadow
    effective_interval = interval or shadow_config.interval_seconds
    effective_max_cycles = shadow_config.max_cycles if max_cycles is None else max_cycles
    effective_near_term_days = near_term_days if near_term_days is not None else shadow_config.near_term_days
    history_output_path = output or _default_signal_output(shadow_config.history_output_path.name)
    summary_output_path = summary_output or _default_signal_output(shadow_config.summary_output_path.name)
    latest_output_path = latest_output or _default_signal_output(shadow_config.latest_output_path.name)
    effective_state_path = state_path or _default_signal_output(shadow_config.state_path.name)
    horizon_policy = _load_recent_horizon_policy(horizon_policy_path)
    edge_threshold = config.backtest.default_edge_threshold
    cities = _resolve_scoped_cities(cities, market_scope=market_scope)

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_scoped_snapshots(
            markets_path=markets_path,
            cities=cities,
            market_scope=market_scope,
            active=True,
            closed=False,
        )

    def _evaluator(snapshots: list[MarketSnapshot], observed_at: datetime) -> list[OpportunityObservation]:
        observations: list[OpportunityObservation] = []
        for snapshot in snapshots:
            spec = snapshot.spec
            if spec is None:
                continue
            decision_horizon = _resolve_opportunity_shadow_horizon(
                spec,
                now_utc=observed_at,
                near_term_days=effective_near_term_days,
            )
            if decision_horizon is None:
                continue
            if not _policy_allows_horizon(spec, decision_horizon=decision_horizon, horizon_policy=horizon_policy):
                observations.append(
                    _empty_opportunity_observation(
                        snapshot,
                        observed_at=observed_at,
                        decision_horizon=decision_horizon,
                        reason="policy_filtered",
                    )
                )
                continue
            observation = _evaluate_opportunity_snapshot(
                snapshot,
                builder=builder,
                clob=clob,
                model_path=model_path,
                model_name=resolved_model_name,
                config=config,
                observed_at=observed_at,
                decision_horizon=decision_horizon,
                edge_threshold=edge_threshold,
            )
            if observation is not None:
                observations.append(observation)
        return observations

    runner = OpportunityShadowRunner(
        config=config,
        interval_seconds=effective_interval,
        max_cycles=effective_max_cycles or 0,
        state_path=effective_state_path,
        latest_output_path=latest_output_path,
        history_output_path=history_output_path,
        summary_output_path=summary_output_path,
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )

    try:
        console.print(
            "Opportunity shadow: "
            f"interval={effective_interval}s, max_cycles={effective_max_cycles or 0}, "
            f"near_term_days={effective_near_term_days}"
        )
        runner.run_loop()
        console.print(f"Wrote latest {latest_output_path}")
        console.print(f"Wrote history {history_output_path}")
        console.print(f"Wrote summary {summary_output_path}")
    finally:
        http.close()


@app.command("open-phase-shadow")
def open_phase_shadow(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    core_recent_only: bool = typer.Option(False, help="Restrict to Seoul/NYC/London recent-core cities"),
    market_scope: str = typer.Option("default", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between scan cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    open_window_hours: float = typer.Option(24.0, help="Only include markets opened within this many hours"),
    horizon: str = typer.Option("market_open", help="Forecast horizon for open-phase evaluation"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path = typer.Option(Path("artifacts/signals/v2/open_phase_shadow.jsonl"), help="Append-only JSONL output"),
    latest_output: Path = typer.Option(Path("artifacts/signals/v2/open_phase_shadow_latest.json"), help="Latest cycle JSON output"),
    summary_output: Path = typer.Option(Path("artifacts/signals/v2/open_phase_shadow_summary.json"), help="Summary JSON output"),
    state_path: Path = typer.Option(Path("artifacts/signals/v2/open_phase_shadow_state.json"), help="State JSON path"),
) -> None:
    """Continuously watch newly listed markets and score them at listing/open phase."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/champion.pkl"))
    model_name = _resolve_option_value(model_name, DEFAULT_MODEL_NAME)
    cities = _resolve_option_value(cities)
    core_recent_only = bool(_resolve_option_value(core_recent_only, False))
    market_scope = _resolve_market_scope(_resolve_option_value(market_scope, "default"), core_recent_only=core_recent_only)
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    open_window_hours = float(_resolve_option_value(open_window_hours, 24.0))
    horizon = _resolve_option_value(horizon, "market_open")
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output, Path("artifacts/signals/v2/open_phase_shadow.jsonl"))
    latest_output = _resolve_option_value(latest_output, Path("artifacts/signals/v2/open_phase_shadow_latest.json"))
    summary_output = _resolve_option_value(summary_output, Path("artifacts/signals/v2/open_phase_shadow_summary.json"))
    state_path = _resolve_option_value(state_path, Path("artifacts/signals/v2/open_phase_shadow_state.json"))
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http,
        openmeteo=openmeteo,
        duckdb_store=None,
        parquet_store=None,
        snapshot_dir=None,
        fixture_dir=None,
        models=config.weather.models or None,
    )
    effective_interval = interval or config.opportunity_shadow.interval_seconds
    effective_max_cycles = config.opportunity_shadow.max_cycles if max_cycles is None else max_cycles
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    cities = _resolve_scoped_cities(cities, market_scope=market_scope)

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_scoped_snapshots(
            markets_path=markets_path,
            cities=cities,
            market_scope=market_scope,
            active=True,
            closed=False,
        )

    def _evaluator(snapshots: list[MarketSnapshot], observed_at: datetime) -> list[OpenPhaseObservation]:
        observations: list[OpenPhaseObservation] = []
        candidates = select_open_phase_candidates(
            snapshots,
            observed_at=observed_at,
            open_window_hours=open_window_hours,
        )
        for candidate in candidates:
            observation = _evaluate_opportunity_snapshot(
                candidate.snapshot,
                builder=builder,
                clob=clob,
                model_path=model_path,
                model_name=resolved_model_name,
                config=config,
                observed_at=observed_at,
                decision_horizon=horizon,
                edge_threshold=edge_threshold,
            )
            if observation is None:
                continue
            observations.append(
                _open_phase_observation_from_opportunity(
                    observation,
                    market_created_at=candidate.market_created_at,
                    market_deploying_at=candidate.market_deploying_at,
                    market_accepting_orders_at=candidate.market_accepting_orders_at,
                    market_opened_at=candidate.market_opened_at,
                    open_phase_age_hours=candidate.open_phase_age_hours,
                )
            )
        return observations

    runner = OpenPhaseShadowRunner(
        config=config,
        interval_seconds=effective_interval,
        max_cycles=effective_max_cycles or 0,
        state_path=state_path,
        latest_output_path=latest_output,
        history_output_path=output,
        summary_output_path=summary_output,
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )

    try:
        console.print(
            "Open-phase shadow: "
            f"interval={effective_interval}s, max_cycles={effective_max_cycles or 0}, "
            f"open_window_hours={open_window_hours}, horizon={horizon}"
        )
        runner.run_loop()

        rows = json.loads(latest_output.read_text()) if latest_output.exists() else []
        table = Table(title="Open-Phase Shadow")
        table.add_column("City")
        table.add_column("Date")
        table.add_column("Age(h)")
        table.add_column("Outcome")
        table.add_column("Edge")
        table.add_column("Reason")
        for row in rows[:20]:
            edge = row.get("after_cost_edge")
            table.add_row(
                str(row.get("city", "")),
                str(row.get("target_local_date", "")),
                f"{float(row['open_phase_age_hours']):.2f}" if row.get("open_phase_age_hours") is not None else "—",
                str(row.get("outcome_label", "—")),
                f"{float(edge):+.4f}" if edge is not None else "—",
                str(row.get("reason", "")),
            )
        console.print(table)
        console.print(f"Wrote latest {latest_output}")
        console.print(f"Wrote history {output}")
        console.print(f"Wrote summary {summary_output}")
    finally:
        http.close()


@app.command("hope-hunt-report")
def hope_hunt_report(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    market_scope: str = typer.Option("supported_wu_open_phase", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("market_open", help="Forecast horizon for hope-hunt evaluation"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path | None = typer.Option(None, help="Latest cycle JSON output"),
    history_output: Path | None = typer.Option(None, help="Append-only JSONL history output"),
    summary_output: Path | None = typer.Option(None, help="Summary JSON output"),
) -> None:
    """Generate one ranked hope-hunt snapshot across supported WU-family active markets."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    cities = _resolve_option_value(cities)
    market_scope = _resolve_market_scope(
        _resolve_option_value(market_scope, "supported_wu_open_phase"),
        core_recent_only=False,
    )
    markets_path = _resolve_option_value(markets_path)
    horizon = _resolve_option_value(horizon, "market_open")
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output)
    history_output = _resolve_option_value(history_output)
    summary_output = _resolve_option_value(summary_output)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)

    config, _env = load_settings()
    hope_config = config.hope_hunt
    latest_output = output or _default_signal_output(hope_config.latest_output_path.name)
    history_output_path = history_output or _default_signal_output(hope_config.history_output_path.name)
    summary_output_path = summary_output or _default_signal_output(hope_config.summary_output_path.name)
    state_path = _default_signal_output(hope_config.state_path.name)

    runner, http = _build_hope_hunt_runner(
        model_path=model_path,
        model_name=resolved_model_name,
        cities=cities,
        market_scope=market_scope,
        markets_path=markets_path,
        interval_seconds=hope_config.interval_seconds,
        max_cycles=1,
        output=history_output_path,
        latest_output=latest_output,
        summary_output=summary_output_path,
        state_path=state_path,
        horizon=horizon,
        edge_threshold=min_edge,
    )
    try:
        summary = runner.run_once()
    finally:
        http.close()

    rows = json.loads(latest_output.read_text()) if latest_output.exists() else []
    table = Table(title="Hope Hunt")
    table.add_column("City")
    table.add_column("Date")
    table.add_column("Age(h)")
    table.add_column("Day+")
    table.add_column("Volume")
    table.add_column("Score")
    table.add_column("Reason")
    for row in rows[:20]:
        table.add_row(
            str(row.get("city", "")),
            str(row.get("target_local_date", "")),
            f"{float(row['open_phase_age_hours']):.2f}" if row.get("open_phase_age_hours") is not None else "—",
            str(row.get("target_day_distance", "—")),
            f"{float(row['market_volume']):.1f}" if row.get("market_volume") is not None else "—",
            f"{float(row['priority_score']):.1f}" if row.get("priority_score") is not None else "—",
            str(row.get("reason", "")),
        )
    console.print(table)
    console.print(
        "Hope hunt: "
        f"markets_evaluated={summary['markets_evaluated']}, "
        f"candidate_count={summary['candidate_count']}"
    )
    console.print(f"Wrote latest {latest_output}")
    console.print(f"Wrote history {history_output_path}")
    console.print(f"Wrote summary {summary_output_path}")


@app.command("hope-hunt-daemon")
def hope_hunt_daemon(
    model_path: Path = typer.Option(Path("artifacts/models/v2/trading_champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(TRADING_MODEL_ALIAS, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    market_scope: str = typer.Option("supported_wu_open_phase", help="Market scope preset"),
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between hope-hunt cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    horizon: str = typer.Option("market_open", help="Forecast horizon for hope-hunt evaluation"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path | None = typer.Option(None, help="Append-only JSONL output"),
    latest_output: Path | None = typer.Option(None, help="Latest cycle JSON output"),
    summary_output: Path | None = typer.Option(None, help="Summary JSON output"),
    state_path: Path | None = typer.Option(None, help="State JSON path"),
) -> None:
    """Continuously score open-phase hope candidates without placing orders."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/v2/trading_champion.pkl"))
    model_name = _resolve_option_value(model_name, TRADING_MODEL_ALIAS)
    cities = _resolve_option_value(cities)
    market_scope = _resolve_market_scope(
        _resolve_option_value(market_scope, "supported_wu_open_phase"),
        core_recent_only=False,
    )
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    horizon = _resolve_option_value(horizon, "market_open")
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output)
    latest_output = _resolve_option_value(latest_output)
    summary_output = _resolve_option_value(summary_output)
    state_path = _resolve_option_value(state_path)
    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)

    config, _env = load_settings()
    hope_config = config.hope_hunt
    effective_interval = hope_config.interval_seconds if interval is None else int(interval)
    effective_max_cycles = hope_config.max_cycles if max_cycles is None else int(max_cycles)
    history_output = output or _default_signal_output(hope_config.history_output_path.name)
    latest_output_path = latest_output or _default_signal_output(hope_config.latest_output_path.name)
    summary_output_path = summary_output or _default_signal_output(hope_config.summary_output_path.name)
    effective_state_path = state_path or _default_signal_output(hope_config.state_path.name)

    runner, http = _build_hope_hunt_runner(
        model_path=model_path,
        model_name=resolved_model_name,
        cities=cities,
        market_scope=market_scope,
        markets_path=markets_path,
        interval_seconds=effective_interval,
        max_cycles=effective_max_cycles or 0,
        output=history_output,
        latest_output=latest_output_path,
        summary_output=summary_output_path,
        state_path=effective_state_path,
        horizon=horizon,
        edge_threshold=min_edge,
    )
    try:
        console.print(
            "Hope hunt daemon: "
            f"interval={effective_interval}s, max_cycles={effective_max_cycles or 0}, "
            f"scope={market_scope}, horizon={horizon}"
        )
        runner.run_loop()
    finally:
        http.close()

    console.print(f"Wrote latest {latest_output_path}")
    console.print(f"Wrote history {history_output}")
    console.print(f"Wrote summary {summary_output_path}")


@app.command("forecast-daemon")
def forecast_daemon(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    interval: int = typer.Option(21600, help="Seconds between forecast cycles (default 6h)"),
    max_cycles: int = typer.Option(0, help="Max cycles (0 = infinite)"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
    telegram: bool = typer.Option(True, help="Send to Telegram each cycle"),
    firebase: bool = typer.Option(True, help="Publish to Firebase each cycle"),
) -> None:
    """Continuously publish forecast reports on a schedule."""

    import signal as sig
    import time

    from pmtmax.services.forecast_summary import build_forecast_summaries

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None,
        snapshot_dir=None, fixture_dir=None, models=config.weather.models or None,
    )

    running = True

    def _shutdown(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    sig.signal(sig.SIGINT, _shutdown)
    sig.signal(sig.SIGTERM, _shutdown)

    cycle = 0
    console.print(f"Forecast daemon: interval={interval}s, max_cycles={max_cycles}")

    while running:
        try:
            snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
            summaries = build_forecast_summaries(snapshots, model_path, resolved_model_name, clob, builder, horizon=horizon)

            payload = [s.model_dump(mode="json") for s in summaries]
            out = Path("artifacts/forecast_report.json")
            dump_json(out, payload)
            console.print(f"Cycle {cycle}: {len(summaries)} forecasts → {out}")

            if telegram and config.telegram.enabled:
                from pmtmax.services.telegram_bot import TelegramNotifier

                notifier = TelegramNotifier(config.telegram.bot_token, config.telegram.chat_id)
                notifier.send_forecast_report(summaries)

            if firebase and config.firebase.enabled:
                from pmtmax.services.forecast_publisher import ForecastPublisher

                publisher = ForecastPublisher(
                    bucket_name=config.firebase.bucket_name,
                    prefix=config.firebase.prefix,
                    credentials_json=config.firebase.credentials_json or None,
                )
                publisher.publish(summaries, dry_run=False)

        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Cycle {cycle} error: {exc}[/]")

        cycle += 1
        if 0 < max_cycles <= cycle:
            break
        if running:
            time.sleep(interval)

    console.print("Forecast daemon stopped.")


# ===========================================================================
# Track C: market making commands
# ===========================================================================


@app.command("paper-mm")
def paper_mm(
    model_path: Path = typer.Option(Path("artifacts/models/v2/champion.pkl"), help="Model artifact path"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    interval: int = typer.Option(60, help="Seconds between MM cycles"),
    max_cycles: int = typer.Option(0, help="Max cycles (0 = infinite)"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
) -> None:
    """Run paper market-making simulation against live CLOB books."""

    import signal as sig
    import time

    from pmtmax.execution.paper_market_maker import PaperMarketMaker
    from pmtmax.execution.quoter import Quoter

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None,
        snapshot_dir=None, fixture_dir=None, models=config.weather.models or None,
    )

    mm_config = config.market_making
    risk_limits = RiskLimits(
        max_position_per_outcome=mm_config.max_position_per_outcome,
        max_total_exposure=mm_config.max_total_exposure,
        max_loss=mm_config.max_loss,
    )
    quoter = Quoter(
        base_half_spread=mm_config.base_half_spread,
        skew_factor=mm_config.skew_factor,
        base_size=mm_config.base_size,
    )
    mm = PaperMarketMaker(risk_limits=risk_limits)

    running = True

    def _shutdown(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    sig.signal(sig.SIGINT, _shutdown)
    sig.signal(sig.SIGTERM, _shutdown)

    cycle = 0
    console.print(f"Paper MM: interval={interval}s, max_cycles={max_cycles}")

    while running:
        try:
            snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)

            for snapshot in snapshots:
                spec = snapshot.spec
                if spec is None or spec.target_local_date < datetime.now(tz=UTC).date():
                    continue

                feature_frame = builder.build_live_row(spec, horizon=horizon)
                forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)

                books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
                if not books:
                    console.print(f"  {spec.city}: skip missing_token_mapping")
                    continue
                if any(book.source != "clob" for book in books.values()):
                    console.print(f"  {spec.city}: skip missing_book {_book_source_counts(books)}")
                    continue
                token_map = {outcome_label: book.token_id for outcome_label, book in books.items()}

                quotes = quoter.compute_quotes(
                    forecast.outcome_probabilities, token_map, mm.inventory, risk_limits,
                )
                if not quotes:
                    console.print(f"  {spec.city}: skip no_quotes")
                    continue

                fills = mm.simulate_quotes(quotes, books)
                if fills:
                    console.print(f"  {spec.city}: {len(fills)} fills")

        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Cycle {cycle} error: {exc}[/]")

        summary = mm.summary()
        console.print(f"Cycle {cycle}: fills={summary['total_fills']}, pnl={summary['net_pnl']:.4f}, exposure={summary['total_exposure']:.2f}")

        cycle += 1
        if 0 < max_cycles <= cycle:
            break
        if running:
            time.sleep(interval)

    console.print("Paper MM stopped.")
    console.print_json(data=mm.summary())


@app.command("live-mm")
def live_mm(
    model_path: Path = typer.Argument(..., help="Path to trained model artifact"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
    dry_run: bool = typer.Option(True, help="Compute quotes only, do not post"),
    post_orders: bool = typer.Option(False, help="Actually post orders (requires live trading gates)"),
) -> None:
    """Run live market-making with quote computation and optional order posting."""

    from pmtmax.execution.live_market_maker import LiveMarketMaker
    from pmtmax.execution.quoter import Quoter

    model_path, resolved_model_name = _resolve_model_path(model_path, model_name)
    config, env, http, _, _, openmeteo = _runtime(include_stores=False)
    broker = LiveBroker(env)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None,
        snapshot_dir=None, fixture_dir=None, models=config.weather.models or None,
    )

    mm_config = config.market_making
    risk_limits = RiskLimits(
        max_position_per_outcome=mm_config.max_position_per_outcome,
        max_total_exposure=mm_config.max_total_exposure,
        max_loss=mm_config.max_loss,
    )
    quoter = Quoter(
        base_half_spread=mm_config.base_half_spread,
        skew_factor=mm_config.skew_factor,
        base_size=mm_config.base_size,
    )
    live_mm_engine = LiveMarketMaker(broker=broker, risk_limits=risk_limits)

    should_post = post_orders and not dry_run
    if should_post:
        preflight = broker.preflight(require_posting=True)
        if not preflight.ok:
            console.print("[red]Live preflight failed:[/]")
            for msg in preflight.messages:
                console.print(f"  {msg}")
            raise typer.Exit(1)

    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    all_results: list[dict] = []

    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None or spec.target_local_date < datetime.now(tz=UTC).date():
            continue

        try:
            feature_frame = builder.build_live_row(spec, horizon=horizon)
            forecast = predict_market(model_path, resolved_model_name, spec, feature_frame)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Forecast failed for {spec.city}: {exc}[/]")
            continue

        books = _load_books_for_forecast(clob, snapshot, forecast.outcome_probabilities)
        book_source_counts = _book_source_counts(books)
        if not books:
            all_results.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "reason": "missing_token_mapping",
                    "book_source_counts": book_source_counts,
                }
            )
            continue
        if any(book.source != "clob" for book in books.values()):
            all_results.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "reason": "missing_book",
                    "book_source_counts": book_source_counts,
                }
            )
            continue

        token_map = {outcome_label: book.token_id for outcome_label, book in books.items()}

        quotes = quoter.compute_quotes(
            forecast.outcome_probabilities, token_map, live_mm_engine.inventory, risk_limits,
        )
        if not quotes:
            all_results.append(
                {
                    "market_id": spec.market_id,
                    "city": spec.city,
                    "question": spec.question,
                    "reason": "no_quotes",
                    "book_source_counts": book_source_counts,
                }
            )
            continue

        results = live_mm_engine.update_quotes(quotes, market_id=spec.market_id, dry_run=not should_post)
        for result in results:
            result.setdefault("market_id", spec.market_id)
            result.setdefault("city", spec.city)
            result.setdefault("question", spec.question)

        table = Table(title=f"{spec.city} — {spec.target_local_date}")
        table.add_column("Outcome")
        table.add_column("Bid")
        table.add_column("Ask")
        table.add_column("Size")
        table.add_column("Status")
        for q in quotes:
            status = "dry_run" if not should_post else "posted"
            table.add_row(q.outcome_label, f"{q.bid_price:.4f}", f"{q.ask_price:.4f}", f"{q.bid_size:.2f}", status)
        console.print(table)

        all_results.extend(results)

    dump_json(Path("artifacts/live_mm_preview.json"), all_results)
    console.print(f"Wrote artifacts/live_mm_preview.json ({len(all_results)} entries)")


def run() -> None:
    """Entrypoint for the console script."""

    app()


if __name__ == "__main__":
    run()
