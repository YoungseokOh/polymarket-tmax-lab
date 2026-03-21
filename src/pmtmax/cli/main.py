"""Typer CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

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
from pmtmax.execution.paper_broker import PaperBroker
from pmtmax.execution.sizing import capped_kelly
from pmtmax.execution.slippage import estimate_slippage
from pmtmax.http import CachedHttpClient
from pmtmax.logging_utils import configure_logging
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
    BookLevel,
    BookSnapshot,
    LegacyRunInventory,
    MarketSnapshot,
    TradeSignal,
)
from pmtmax.storage.warehouse import DataWarehouse, backup_duckdb_file, ordered_legacy_paths
from pmtmax.utils import dump_json, load_json, set_global_seed, stable_hash
from pmtmax.weather.openmeteo_client import OpenMeteoClient

app = typer.Typer(help="Polymarket maximum temperature research and trading lab.")
console = Console()


def _runtime(include_stores: bool = True) -> tuple:
    config, env = load_settings()
    configure_logging(env.log_level)
    set_global_seed(config.app.random_seed)
    http = CachedHttpClient(config.app.cache_dir, config.weather.timeout_seconds, config.weather.retries)
    duckdb_store = DuckDBStore(config.app.duckdb_path) if include_stores else None
    parquet_store = ParquetStore(config.app.parquet_dir) if include_stores else None
    openmeteo = OpenMeteoClient(http, config.weather.openmeteo_base_url, config.weather.archive_base_url)
    return config, env, http, duckdb_store, parquet_store, openmeteo


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
                "Seoul/RKSI uses AMO AIR_CALP; other supported cities default to NOAA Global Hourly. "
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


def _synthetic_book(snapshot: MarketSnapshot, outcome_label: str, token_id: str) -> BookSnapshot:
    price = snapshot.outcome_prices.get(outcome_label, 0.5)
    bid = max(price - 0.02, 0.01)
    ask = min(price + 0.02, 0.99)
    return BookSnapshot(
        market_id=snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id")),
        token_id=token_id,
        outcome_label=outcome_label,
        source="fixture",
        timestamp=snapshot.captured_at,
        bids=[BookLevel(price=bid, size=100.0)],
        asks=[BookLevel(price=ask, size=100.0)],
    )


def _book_snapshot_from_payload(
    *,
    snapshot: MarketSnapshot,
    token_id: str,
    outcome_label: str,
    payload: dict[str, Any] | None,
) -> BookSnapshot:
    if payload is None:
        return _synthetic_book(snapshot, outcome_label, token_id)
    bids = [BookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("bids", [])[:5]]
    asks = [BookLevel(price=float(level["price"]), size=float(level["size"])) for level in payload.get("asks", [])[:5]]
    timestamp = payload.get("timestamp")
    parsed_ts = None
    if timestamp:
        try:
            parsed_ts = datetime.fromtimestamp(int(str(timestamp)) / 1000.0, tz=UTC)
        except ValueError:
            parsed_ts = None
    return BookSnapshot(
        market_id=snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id")),
        token_id=token_id,
        outcome_label=outcome_label,
        source="clob",
        timestamp=parsed_ts,
        bids=bids,
        asks=asks,
    )


def _fetch_book(
    clob: ClobReadClient,
    snapshot: MarketSnapshot,
    token_id: str,
    outcome_label: str,
) -> BookSnapshot:
    try:
        payload = clob.get_book(token_id)
    except Exception:  # noqa: BLE001
        payload = None
    return _book_snapshot_from_payload(snapshot=snapshot, token_id=token_id, outcome_label=outcome_label, payload=payload)


def _signal_from_forecast(
    snapshot: MarketSnapshot,
    forecast_probs: dict[str, float],
    book: BookSnapshot,
    *,
    mode: Literal["paper", "live"],
) -> TradeSignal:
    outcome_label, fair_probability = max(forecast_probs.items(), key=lambda item: item[1])
    executable_price = book.best_ask()
    spread = max(book.best_ask() - book.best_bid(), 0.0)
    visible_liquidity = (book.bids[0].size if book.bids else 0.0) + (book.asks[0].size if book.asks else 0.0)
    fee = estimate_fee(executable_price)
    slippage = estimate_slippage(executable_price, spread, visible_liquidity, 1.0)
    edge = compute_edge(fair_probability, executable_price, fee, slippage)
    return TradeSignal(
        market_id=snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id")),
        token_id=book.token_id,
        outcome_label=outcome_label,
        side="buy",
        fair_probability=fair_probability,
        executable_price=executable_price,
        fee_estimate=fee,
        slippage_estimate=slippage,
        edge=edge,
        confidence=fair_probability,
        rationale=f"Top modeled outcome is {outcome_label} with p={fair_probability:.3f}",
        mode=mode,
    )


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

        outcome_label, _ = max(forecast.outcome_probabilities.items(), key=lambda item: item[1])
        token_id = spec.token_ids[spec.outcome_labels().index(outcome_label)] if spec.token_ids else outcome_label
        book = _synthetic_book(snapshot, outcome_label, token_id)
        signal = _signal_from_forecast(snapshot, forecast.outcome_probabilities, book, mode="paper")
        size_notional = capped_kelly(signal.edge, signal.confidence, broker.bankroll)
        size = size_notional / max(signal.executable_price, 1e-6)
        if size <= 0:
            continue
        fill = broker.simulate_fill(
            signal,
            spread=book.best_ask() - book.best_bid(),
            liquidity=(book.bids[0].size if book.bids else 0.0) + (book.asks[0].size if book.asks else 0.0),
            size=size,
        )
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
            fee_paid=estimate_fee(fill.price * fill.size),
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
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Run a decision-time backtest using official historical market prices."""

    if flat_stake <= 0:
        raise typer.BadParameter("flat_stake must be positive.")
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

        outcome_label, fair_probability = max(forecast.outcome_probabilities.items(), key=lambda item: item[1])
        panel_row = working_panel.loc[
            (working_panel["market_id"] == spec.market_id)
            & (working_panel["decision_horizon"] == str(row["decision_horizon"]))
            & (working_panel["outcome_label"] == outcome_label)
        ].copy()
        if panel_row.empty:
            skipped_missing_price += 1
            continue
        selected = panel_row.iloc[-1]
        coverage_status = str(selected["coverage_status"])
        if coverage_status == "stale":
            skipped_stale_price += 1
            continue
        if coverage_status != "ok":
            skipped_missing_price += 1
            continue
        market_price = float(selected["market_price"])
        fee_per_share = estimate_fee(market_price)
        edge = compute_edge(fair_probability, market_price, fee_per_share, 0.0)
        if edge <= 0:
            skipped_non_positive_edge += 1
            continue
        size = flat_stake / max(market_price, 1e-6)
        priced_decision_rows += 1
        age_seconds = selected.get("price_age_seconds")
        if age_seconds is not None and pd.notna(age_seconds):
            price_ages.append(float(age_seconds))
        realized_pnl = settle_position(
            Position(
                outcome_label=outcome_label,
                price=market_price,
                size=size,
                side="buy",
            ),
            winning_label,
            fee_paid=estimate_fee(flat_stake),
        )
        trade_rows.append(
            {
                "market_id": spec.market_id,
                "city": spec.city,
                "decision_horizon": str(row["decision_horizon"]),
                "outcome_label": outcome_label,
                "winning_outcome": winning_label,
                "price": market_price,
                "size": size,
                "edge": edge,
                "price_ts": selected.get("price_ts"),
                "price_age_seconds": age_seconds,
                "realized_pnl": realized_pnl,
                "pricing_source": "real_history",
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
    pricing_source: Literal["synthetic", "real_history"] = "synthetic",
    panel_path: Path = Path("data/parquet/gold/historical_backtest_panel.parquet"),
    flat_stake: float = 1.0,
) -> None:
    """Run a rolling-origin backtest with synthetic or official historical pricing."""

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
        metrics, trade_rows = _run_real_history_backtest(
            frame,
            panel,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            flat_stake=flat_stake,
        )
        metrics_output = Path("artifacts/backtest_metrics_real_history.json")
        trades_output = Path("artifacts/backtest_trades_real_history.json")

    dump_json(metrics_output, metrics)
    dump_json(trades_output, trade_rows)
    console.print_json(data=metrics)


@app.command("paper-trader")
def paper_trader(
    model_path: Path = Path("artifacts/models/gaussian_emos.pkl"),
    model_name: str = "gaussian_emos",
    markets_path: Path | None = None,
    cities: Annotated[list[str] | None, typer.Option("--city")] = None,
    horizon: str = "morning_of",
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

    current_exposure_by_city: dict[str, float] = {}
    results: list[dict[str, object]] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        if spec.target_local_date < datetime.now(tz=UTC).date():
            continue
        feature_frame = builder.build_live_row(spec, horizon=horizon)
        forecast = predict_market(model_path, model_name, spec, feature_frame)
        if not forecast_fresh(forecast.generated_at.replace(tzinfo=None), config.execution.stale_forecast_minutes):
            continue
        outcome_label, _ = max(forecast.outcome_probabilities.items(), key=lambda item: item[1])
        token_id = spec.token_ids[spec.outcome_labels().index(outcome_label)] if spec.token_ids else outcome_label
        book = _fetch_book(clob, snapshot, token_id, outcome_label)
        signal = _signal_from_forecast(snapshot, forecast.outcome_probabilities, book, mode="paper")
        spread = book.best_ask() - book.best_bid()
        liquidity = (book.bids[0].size if book.bids else 0.0) + (book.asks[0].size if book.asks else 0.0)
        reason = "accepted"
        fill_payload: dict[str, object] | None = None
        if signal.edge < edge_threshold:
            reason = "edge_below_threshold"
        elif not spread_ok(book.best_bid(), book.best_ask(), config.execution.max_spread_bps):
            reason = "spread_too_wide"
        elif liquidity < config.execution.min_liquidity:
            reason = "liquidity_too_low"
        else:
            size_notional = capped_kelly(signal.edge, signal.confidence, broker.bankroll)
            size = size_notional / max(signal.executable_price, 1e-6)
            current_city_exposure = current_exposure_by_city.get(spec.city, 0.0)
            if not exposure_ok(current_city_exposure, size_notional, config.execution.max_city_exposure):
                reason = "city_exposure_limit"
            elif not exposure_ok(sum(current_exposure_by_city.values()), size_notional, config.execution.global_max_exposure):
                reason = "global_exposure_limit"
            else:
                fill = broker.simulate_fill(signal, spread=spread, liquidity=liquidity, size=size)
                if fill is None:
                    reason = "broker_rejected"
                else:
                    current_exposure_by_city[spec.city] = current_city_exposure + size_notional
                    fill_payload = fill.model_dump(mode="json")
        results.append(
            {
                "market_id": signal.market_id,
                "city": spec.city,
                "question": spec.question,
                "outcome_label": signal.outcome_label,
                "fair_probability": signal.fair_probability,
                "executable_price": signal.executable_price,
                "edge": signal.edge,
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
    horizon: str = "morning_of",
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

    previews: list[dict[str, object]] = []
    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None or spec.target_local_date < datetime.now(tz=UTC).date():
            continue
        feature_frame = builder.build_live_row(spec, horizon=horizon)
        forecast = predict_market(model_path, model_name, spec, feature_frame)
        outcome_label, _ = max(forecast.outcome_probabilities.items(), key=lambda item: item[1])
        token_id = spec.token_ids[spec.outcome_labels().index(outcome_label)] if spec.token_ids else outcome_label
        book = _fetch_book(clob, snapshot, token_id, outcome_label)
        signal = _signal_from_forecast(snapshot, forecast.outcome_probabilities, book, mode="live")
        size_notional = capped_kelly(signal.edge, signal.confidence, 1000.0)
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
                "error": str(exc),
            }
        if post_orders and not dry_run and preflight.ok:
            preview["post_result"] = broker.post_limit_order(signal, size=size)
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
    horizon: str = typer.Option("morning_of", help="Forecast horizon"),
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
            if spec.market_id not in seen_market_ids:
                continue
            try:
                feature_frame = builder.build_live_row(spec, horizon=horizon)
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
            if spec.target_local_date < datetime.now(tz=UTC).date():
                continue
            if spec.market_id in held_market_ids:
                continue
            try:
                feature_frame = builder.build_live_row(spec, horizon=horizon)
                forecast = predict_market(model_path, model_name, spec, feature_frame)
            except Exception:  # noqa: BLE001
                logger.warning("forecast failed for %s — skipping entry", spec.market_id)
                continue
            if not forecast_fresh(forecast.generated_at.replace(tzinfo=None), config.execution.stale_forecast_minutes):
                continue
            outcome_label, _ = max(forecast.outcome_probabilities.items(), key=lambda item: item[1])
            token_id = spec.token_ids[spec.outcome_labels().index(outcome_label)] if spec.token_ids else outcome_label
            book = _fetch_book(clob, snapshot, token_id, outcome_label)
            signal = _signal_from_forecast(snapshot, forecast.outcome_probabilities, book, mode="paper")
            spread = book.best_ask() - book.best_bid()
            liquidity = (book.bids[0].size if book.bids else 0.0) + (book.asks[0].size if book.asks else 0.0)
            if signal.edge < edge_threshold:
                continue
            if not spread_ok(book.best_bid(), book.best_ask(), config.execution.max_spread_bps):
                continue
            if liquidity < config.execution.min_liquidity:
                continue
            size_notional = capped_kelly(signal.edge, signal.confidence, brk.bankroll)
            size = size_notional / max(signal.executable_price, 1e-6)
            current_city_exposure = current_exposure_by_city.get(spec.city, 0.0)
            if not exposure_ok(current_city_exposure, size_notional, config.execution.max_city_exposure):
                continue
            if not exposure_ok(sum(current_exposure_by_city.values()), size_notional, config.execution.global_max_exposure):
                continue
            fill = brk.simulate_fill(signal, spread=spread, liquidity=liquidity, size=size)
            if fill is not None:
                current_exposure_by_city[spec.city] = current_city_exposure + size_notional
                logger.info("Entry fill: %s %s edge=%.4f", spec.city, outcome_label, signal.edge)

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


def run() -> None:
    """Entrypoint for the console script."""

    app()


if __name__ == "__main__":
    run()
