"""Typer CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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
from pmtmax.modeling.evaluation import brier_score, crps_from_samples, gaussian_nll, mae, rmse
from pmtmax.modeling.predict import predict_market
from pmtmax.modeling.train import train_model
from pmtmax.monitoring.open_phase import OpenPhaseShadowRunner, select_open_phase_candidates
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
    LegacyRunInventory,
    MarketSnapshot,
    OpenPhaseObservation,
    OpportunityObservation,
    ProbForecast,
    RiskLimits,
    TradeSignal,
)
from pmtmax.storage.warehouse import DataWarehouse, backup_duckdb_file, ordered_legacy_paths
from pmtmax.utils import dump_json, load_json, load_yaml_with_extends, set_global_seed, stable_hash
from pmtmax.weather.openmeteo_client import OpenMeteoClient

app = typer.Typer(help="Polymarket maximum temperature research and trading lab.")
console = Console()
DEFAULT_RECENT_HORIZON_POLICY_PATH = Path("configs/recent-core-horizon-policy.yaml")


def _resolve_option_value(value: Any, fallback: Any = None) -> Any:
    """Return plain default values when Typer OptionInfo leaks into direct calls."""

    if isinstance(value, OptionInfo):
        default = value.default
        return fallback if default is ... else default
    return value


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


def _backfill_pipeline(config: Any, http: CachedHttpClient, openmeteo: OpenMeteoClient) -> BackfillPipeline:
    warehouse = DataWarehouse.from_paths(
        duckdb_path=config.app.duckdb_path,
        parquet_root=config.app.parquet_dir,
        raw_root=config.app.raw_dir,
        manifest_root=config.app.manifest_dir,
        archive_root=config.app.archive_dir,
    )
    return BackfillPipeline(
        http=http,
        openmeteo=openmeteo,
        warehouse=warehouse,
        models=config.weather.models,
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
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
        return float(getattr(execution, "default_fee_bps"))
    if hasattr(execution, "fee_bps"):
        return float(getattr(execution, "fee_bps"))
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
        if book is None or book.source != "clob" or not book.asks:
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
        signal = _trade_signal_from_candidate(snapshot, best_after_cost, mode=mode)
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
        forecast = predict_market(model_path, model_name, spec, feature_frame)
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
    )
    observation = _empty_opportunity_observation(
        snapshot,
        observed_at=observed_at,
        decision_horizon=decision_horizon,
        reason=str(evaluation["reason"]),
        book_source_counts=dict(evaluation["book_source_counts"]),
        forecast_generated_at=forecast.generated_at,
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
        **trade_summary,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics, prediction_frame, trade_frame


def _run_synthetic_backtest(
    frame: pd.DataFrame,
    *,
    model_name: str,
    artifacts_dir: Path,
    bankroll: float,
    default_fee_bps: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run the existing synthetic-book research backtest."""

    broker = PaperBroker(bankroll=bankroll)
    prediction_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for train, test in rolling_origin_splits(frame, min_train_size=1, test_size=1):
        artifact = train_model(model_name, train, artifacts_dir)
        row = test.iloc[0]
        spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
        snapshot = MarketSnapshot(
            captured_at=datetime.now(tz=UTC),
            market={"id": spec.market_id},
            spec=spec,
            outcome_prices=json.loads(str(row.get("market_prices_json", "{}"))),
            clob_token_ids=spec.token_ids,
        )
        forecast = predict_market(Path(artifact.path), model_name, spec, test)
        winning_label = str(row["winning_outcome"])
        prediction_rows.append(
            {
                "target_date": row["target_date"],
                "city": spec.city,
                "y_true": row["realized_daily_max"],
                "y_pred": forecast.mean,
                "std": forecast.std,
                "brier": brier_score(forecast.outcome_probabilities, winning_label),
                "crps": crps_from_samples(pd.Series(forecast.samples).to_numpy(), float(row["realized_daily_max"])),
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
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using official historical market prices."""

    return _run_panel_pricing_backtest(
        frame,
        panel,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
        flat_stake=flat_stake,
        default_fee_bps=default_fee_bps,
        pricing_source="real_history",
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
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    quote_proxy_half_spread: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using last price plus an explicit quote proxy."""

    return _run_panel_pricing_backtest(
        frame,
        panel,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
        flat_stake=flat_stake,
        default_fee_bps=default_fee_bps,
        pricing_source="quote_proxy",
        quote_proxy_half_spread=quote_proxy_half_spread,
    )


def _run_panel_pricing_backtest(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    model_name: str,
    artifacts_dir: Path,
    flat_stake: float,
    default_fee_bps: float,
    pricing_source: Literal["real_history", "quote_proxy"],
    quote_proxy_half_spread: float = 0.0,
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

    for train, test in rolling_origin_splits(frame, min_train_size=1, test_size=1):
        artifact = train_model(model_name, train, artifacts_dir)
        row = test.iloc[0]
        spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
        forecast = predict_market(Path(artifact.path), model_name, spec, test)
        winning_label = str(row["winning_outcome"])
        prediction_rows.append(
            {
                "target_date": row["target_date"],
                "city": spec.city,
                "y_true": row["realized_daily_max"],
                "y_pred": forecast.mean,
                "std": forecast.std,
                "brier": brier_score(forecast.outcome_probabilities, winning_label),
                "crps": crps_from_samples(pd.Series(forecast.samples).to_numpy(), float(row["realized_daily_max"])),
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
    active: bool = True,
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
        active=active,
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
) -> None:
    """Backfill bronze/silver tables, then materialize a gold training dataset."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
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
        )
        pipeline.warehouse.finish_run(run, status="completed", notes=f"Materialized {len(frame)} tabular rows.")
    except Exception as exc:  # noqa: BLE001
        pipeline.warehouse.finish_run(run, status="failed", notes=str(exc))
        raise
    finally:
        pipeline.warehouse.close()
    console.print(
        f"Built dataset with {len(frame)} rows at {config.app.parquet_dir / 'gold' / f'{output_name}.parquet'}"
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
) -> None:
    """Backfill official truth snapshots and normalized daily observations."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities)
    pipeline = _backfill_pipeline(config, http, openmeteo)
    run = pipeline.warehouse.start_run(
        command="backfill-truth",
        config_hash=_config_hash(config, "backfill-truth", markets_path=markets_path, cities=cities),
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
    dataset_path: Path = Path("data/parquet/gold/historical_training_set.parquet"),
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    output_name: str = "historical_backtest_panel",
    max_price_age_minutes: int = 720,
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
        ),
    )
    pipeline.run_id = run.run_id
    try:
        panel = pipeline.materialize_backtest_panel(
            frame,
            output_name=output_name,
            max_price_age_minutes=max_price_age_minutes,
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
            "output_path": str(config.app.parquet_dir / "gold" / f"{output_name}.parquet"),
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
        ),
    )
    pipeline.run_id = run.run_id
    try:
        frame = pipeline.materialize_training_set(
            snapshots,
            output_name=output_name,
            decision_horizons=decision_horizons or config.backtest.decision_horizons or None,
            contract=contract,
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
            "output_path": str(config.app.parquet_dir / "gold" / f"{output_name}.parquet"),
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
    dataset_path: Path = Path("data/parquet/gold/historical_training_set.parquet"),
    model_name: str = "gaussian_emos",
    artifacts_dir: Path = Path("artifacts/models"),
) -> None:
    """Train a baseline probabilistic model."""

    frame = pd.read_parquet(dataset_path)
    artifact = train_model(model_name, frame, artifacts_dir)
    console.print(f"Trained {model_name} -> {artifact.path}")


@app.command("train-advanced")
def train_advanced(
    dataset_path: Path = Path("data/parquet/gold/historical_training_set.parquet"),
    model_name: str = "det2prob_nn",
    artifacts_dir: Path = Path("artifacts/models"),
) -> None:
    """Train an advanced probabilistic model."""

    frame = pd.read_parquet(dataset_path)
    artifact = train_model(model_name, frame, artifacts_dir)
    console.print(f"Trained {model_name} -> {artifact.path}")


@app.command("backtest")
def backtest(
    dataset_path: Path = Path("data/parquet/gold/historical_training_set.parquet"),
    model_name: str = "gaussian_emos",
    artifacts_dir: Path = Path("artifacts/models"),
    bankroll: float = 10_000.0,
    pricing_source: Literal["synthetic", "real_history", "quote_proxy"] = "synthetic",
    panel_path: Path = Path("data/parquet/gold/historical_backtest_panel.parquet"),
    flat_stake: float = 1.0,
    quote_proxy_half_spread: float = 0.02,
) -> None:
    """Run a rolling-origin backtest with synthetic or official historical pricing."""

    config, _ = load_settings()
    default_fee_bps = _default_fee_bps(config)
    frame = pd.read_parquet(dataset_path)
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
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            bankroll=bankroll,
            default_fee_bps=default_fee_bps,
        )
        metrics_output = Path("artifacts/backtest_metrics.json")
        trades_output = Path("artifacts/backtest_trades.json")
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
                model_name=model_name,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=default_fee_bps,
            )
            metrics_output = Path("artifacts/backtest_metrics_real_history.json")
            trades_output = Path("artifacts/backtest_trades_real_history.json")
        else:
            metrics, trade_rows = _run_quote_proxy_backtest(
                frame,
                panel,
                model_name=model_name,
                artifacts_dir=artifacts_dir,
                flat_stake=flat_stake,
                default_fee_bps=default_fee_bps,
                quote_proxy_half_spread=quote_proxy_half_spread,
            )
            metrics_output = Path("artifacts/backtest_metrics_quote_proxy.json")
            trades_output = Path("artifacts/backtest_trades_quote_proxy.json")

    dump_json(metrics_output, metrics)
    dump_json(trades_output, trade_rows)
    console.print_json(data=metrics)


@app.command("paper-trader")
def paper_trader(
    model_path: Path = Path("artifacts/models/gaussian_emos.pkl"),
    model_name: str = "gaussian_emos",
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    horizon: str = "policy",
    bankroll: float = 10_000.0,
    min_edge: float | None = None,
) -> None:
    """Run paper trading over active discovered markets or bundled history."""

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    broker = PaperBroker(bankroll=bankroll)
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
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    horizon_policy = _load_recent_horizon_policy()

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
        if horizon_reason == "policy_filtered":
            results.append(
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
        forecast = predict_market(model_path, model_name, spec, feature_frame)
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
        book = evaluation["book"]
        reason = str(evaluation["reason"])
        fill_payload: dict[str, object] | None = None
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
                "market_id": spec.market_id,
                "city": spec.city,
                "question": spec.question,
                "decision_horizon": decision_horizon,
                "outcome_label": signal.outcome_label if isinstance(signal, TradeSignal) else None,
                "fair_probability": signal.fair_probability if isinstance(signal, TradeSignal) else None,
                "executable_price": signal.executable_price if isinstance(signal, TradeSignal) else None,
                "edge": signal.edge if isinstance(signal, TradeSignal) else None,
                "book_source_counts": evaluation["book_source_counts"],
                "reason": reason,
                "fill": fill_payload,
            }
        )
    dump_json(Path("artifacts/paper_signals.json"), results)
    console.print_json(data=results)


@app.command("live-trader")
def live_trader(
    model_path: Path = Path("artifacts/models/gaussian_emos.pkl"),
    model_name: str = "gaussian_emos",
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    horizon: str = "policy",
    dry_run: bool = True,
    post_orders: bool = False,
) -> None:
    """Run live preflight and signed-order previews, with optional posting."""

    config, env, http, _, _, openmeteo = _runtime(include_stores=False)
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
        forecast = predict_market(model_path, model_name, spec, feature_frame)
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
    dump_json(Path("artifacts/live_trader_preview.json"), payload)
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
                forecast = predict_market(model_path, model_name, spec, feature_frame)
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
                forecast = predict_market(model_path, model_name, spec, feature_frame)
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
    model_path: Path = typer.Option(Path("artifacts/models/ts_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("ts_emos", help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
    telegram: bool = typer.Option(False, help="Send to Telegram"),
    firebase: bool = typer.Option(False, help="Publish to Firebase"),
    output: Path = typer.Option(Path("artifacts/forecast_report.json"), help="Output JSON"),
) -> None:
    """Generate a one-shot forecast report with optional Telegram/Firebase delivery."""

    from pmtmax.services.forecast_summary import build_forecast_summaries

    config, _, http, _, _, openmeteo = _runtime(include_stores=False)
    clob = ClobReadClient(http, config.polymarket.clob_base_url)
    builder = DatasetBuilder(
        http=http, openmeteo=openmeteo, duckdb_store=None, parquet_store=None,
        snapshot_dir=None, fixture_dir=None, models=config.weather.models or None,
    )
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    summaries = build_forecast_summaries(snapshots, model_path, model_name, clob, builder, horizon=horizon)

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
    model_path: Path = typer.Option(Path("artifacts/models/gaussian_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("gaussian_emos", help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("policy", help="Forecast horizon or 'policy'"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path = typer.Option(Path("artifacts/opportunity_report.json"), help="Output JSON"),
) -> None:
    """Generate a one-shot active-market opportunity report with explicit book status."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/gaussian_emos.pkl"))
    model_name = _resolve_option_value(model_name, "gaussian_emos")
    cities = _resolve_option_value(cities)
    markets_path = _resolve_option_value(markets_path)
    horizon = _resolve_option_value(horizon, "policy")
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output, Path("artifacts/opportunity_report.json"))

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
    snapshots = _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)
    edge_threshold = min_edge if min_edge is not None else config.backtest.default_edge_threshold
    horizon_policy = _load_recent_horizon_policy()

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
            model_name=model_name,
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


@app.command("opportunity-shadow")
def opportunity_shadow(
    model_path: Path = typer.Option(Path("artifacts/models/gaussian_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("gaussian_emos", help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between shadow cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    near_term_days: int | None = typer.Option(None, help="How many days ahead to include beyond local today"),
    output: Path | None = typer.Option(None, help="Append-only JSONL output"),
    summary_output: Path | None = typer.Option(None, help="Summary JSON output"),
    state_path: Path | None = typer.Option(None, help="State JSON path"),
) -> None:
    """Continuously validate whether the live opportunity path ever becomes tradable."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/gaussian_emos.pkl"))
    model_name = _resolve_option_value(model_name, "gaussian_emos")
    cities = _resolve_option_value(cities)
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    near_term_days = _resolve_option_value(near_term_days)
    output = _resolve_option_value(output)
    summary_output = _resolve_option_value(summary_output)
    state_path = _resolve_option_value(state_path)

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
    history_output_path = output or shadow_config.history_output_path
    summary_output_path = summary_output or shadow_config.summary_output_path
    latest_output_path = shadow_config.latest_output_path
    effective_state_path = state_path or shadow_config.state_path
    horizon_policy = _load_recent_horizon_policy()
    edge_threshold = config.backtest.default_edge_threshold

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)

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
                model_name=model_name,
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

    console.print(
        "Opportunity shadow: "
        f"interval={effective_interval}s, max_cycles={effective_max_cycles or 0}, "
        f"near_term_days={effective_near_term_days}"
    )
    runner.run_loop()
    console.print(f"Wrote latest {latest_output_path}")
    console.print(f"Wrote history {history_output_path}")
    console.print(f"Wrote summary {summary_output_path}")


@app.command("open-phase-shadow")
def open_phase_shadow(
    model_path: Path = typer.Option(Path("artifacts/models/gaussian_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("gaussian_emos", help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    interval: int | None = typer.Option(None, help="Seconds between scan cycles"),
    max_cycles: int | None = typer.Option(None, help="Maximum cycles (0 = infinite)"),
    open_window_hours: float = typer.Option(24.0, help="Only include markets opened within this many hours"),
    horizon: str = typer.Option("market_open", help="Forecast horizon for open-phase evaluation"),
    min_edge: float | None = typer.Option(None, help="Minimum edge threshold override"),
    output: Path = typer.Option(Path("artifacts/open_phase_shadow.jsonl"), help="Append-only JSONL output"),
    latest_output: Path = typer.Option(Path("artifacts/open_phase_shadow_latest.json"), help="Latest cycle JSON output"),
    summary_output: Path = typer.Option(Path("artifacts/open_phase_shadow_summary.json"), help="Summary JSON output"),
    state_path: Path = typer.Option(Path("artifacts/open_phase_shadow_state.json"), help="State JSON path"),
) -> None:
    """Continuously watch newly listed markets and score them at listing/open phase."""

    model_path = _resolve_option_value(model_path, Path("artifacts/models/gaussian_emos.pkl"))
    model_name = _resolve_option_value(model_name, "gaussian_emos")
    cities = _resolve_option_value(cities)
    markets_path = _resolve_option_value(markets_path)
    interval = _resolve_option_value(interval)
    max_cycles = _resolve_option_value(max_cycles)
    open_window_hours = float(_resolve_option_value(open_window_hours, 24.0))
    horizon = _resolve_option_value(horizon, "market_open")
    min_edge = _resolve_option_value(min_edge)
    output = _resolve_option_value(output, Path("artifacts/open_phase_shadow.jsonl"))
    latest_output = _resolve_option_value(latest_output, Path("artifacts/open_phase_shadow_latest.json"))
    summary_output = _resolve_option_value(summary_output, Path("artifacts/open_phase_shadow_summary.json"))
    state_path = _resolve_option_value(state_path, Path("artifacts/open_phase_shadow_state.json"))

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

    def _snapshot_fetcher() -> list[MarketSnapshot]:
        return _load_snapshots(markets_path=markets_path, cities=cities, active=True, closed=False)

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
                model_name=model_name,
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


@app.command("forecast-daemon")
def forecast_daemon(
    model_path: Path = typer.Option(Path("artifacts/models/ts_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("ts_emos", help="Model name"),
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
            summaries = build_forecast_summaries(snapshots, model_path, model_name, clob, builder, horizon=horizon)

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
    model_path: Path = typer.Option(Path("artifacts/models/ts_emos.pkl"), help="Model artifact path"),
    model_name: str = typer.Option("ts_emos", help="Model name"),
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
                forecast = predict_market(model_path, model_name, spec, feature_frame)

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
    model_name: str = typer.Option("ts_emos", help="Model name"),
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    markets_path: Path | None = typer.Option(None, help="Offline JSON snapshot file"),
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
    dry_run: bool = typer.Option(True, help="Compute quotes only, do not post"),
    post_orders: bool = typer.Option(False, help="Actually post orders (requires live trading gates)"),
) -> None:
    """Run live market-making with quote computation and optional order posting."""

    from pmtmax.execution.live_market_maker import LiveMarketMaker
    from pmtmax.execution.quoter import Quoter

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
            forecast = predict_market(model_path, model_name, spec, feature_frame)
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
