from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from pmtmax.cli.main import (
    backtest,
    _bootstrap_snapshots,
    _collection_preflight_report,
    _load_snapshots,
    _resolve_opportunity_shadow_horizon,
    live_mm,
    materialize_backtest_panel,
    opportunity_shadow,
    opportunity_report,
    summarize_dataset_readiness,
    summarize_price_history_coverage,
    summarize_truth_coverage,
)
from pmtmax.config.settings import EnvSettings
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.schemas import BookLevel, BookSnapshot, OpportunityObservation


def _future_snapshot(city: str = "Seoul"):
    snapshot = bundled_market_snapshots([city])[0].model_copy(deep=True)
    assert snapshot.spec is not None
    snapshot.spec = snapshot.spec.model_copy(
        update={"target_local_date": datetime.now(tz=UTC).date() + timedelta(days=1)}
    )
    return snapshot


def test_load_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _load_snapshots(markets_path=missing)


def test_bootstrap_snapshots_raises_for_missing_markets_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_snapshots.json"

    with pytest.raises(FileNotFoundError, match="Market snapshot file does not exist"):
        _bootstrap_snapshots(markets_path=missing, cities=None)


def test_collection_preflight_defaults_to_public_archive_for_wu_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PMTMAX_WU_API_KEY", raising=False)

    report = _collection_preflight_report(
        bundled_market_snapshots(["Seoul", "Hong Kong"]),
        EnvSettings(),
    )

    assert report["ready"] is True
    assert report["missing_env"] == []
    assert report["source_counts"] == {"hko": 1, "wunderground": 1}
    assert report["truth_track_counts"] == {"exact_public": 1, "research_public": 1}
    assert report["settlement_eligible_count"] == 1
    assert json.loads(json.dumps(report))["optional_env"] == ["PMTMAX_WU_API_KEY"]


def test_summarize_truth_coverage_command_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWarehouse:
        def close(self) -> None:
            return None

    class _FakePipeline:
        warehouse = _FakeWarehouse()

        def summarize_truth_coverage(self) -> dict[str, pd.DataFrame]:
            return {
                "summary": pd.DataFrame(
                    [{"status": "lag", "truth_track": "research_public", "city": "Seoul", "count": 1}]
                ),
                "details": pd.DataFrame(
                    [{"city": "Seoul", "station_id": "RKSI", "status": "lag", "lag_days": 8}]
                ),
            }

    monkeypatch.setattr("pmtmax.cli.main._runtime", lambda include_stores=False: (None, None, None, None, None, None))
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())

    output = tmp_path / "truth_coverage.json"
    summarize_truth_coverage(output)
    payload = json.loads(output.read_text())
    assert payload["summary"][0]["status"] == "lag"
    assert payload["details"][0]["station_id"] == "RKSI"


def test_summarize_dataset_readiness_command_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWarehouse:
        def close(self) -> None:
            return None

    class _FakePipeline:
        warehouse = _FakeWarehouse()

        def summarize_dataset_readiness(self, snapshots: list[object]) -> dict[str, pd.DataFrame]:
            assert snapshots == ["snapshot"]
            return {
                "summary": pd.DataFrame(
                    [{"city": "Seoul", "snapshot_count": 1, "truth_ok_count": 1, "gold_row_count": 2}]
                ),
                "details": pd.DataFrame(
                    [{"city": "Seoul", "market_id": "101025", "readiness_status": "ready", "gold_row_count": 2}]
                ),
            }

    monkeypatch.setattr("pmtmax.cli.main._runtime", lambda include_stores=False: (None, None, None, None, None, None))
    monkeypatch.setattr("pmtmax.cli.main._bootstrap_snapshots", lambda markets_path=None, cities=None: ["snapshot"])
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())

    output = tmp_path / "dataset_readiness.json"
    summarize_dataset_readiness(markets_path=tmp_path / "snapshots.json", output=output)
    payload = json.loads(output.read_text())
    assert payload["summary"][0]["city"] == "Seoul"
    assert payload["details"][0]["readiness_status"] == "ready"


def test_summarize_price_history_coverage_command_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHttp:
        def close(self) -> None:
            return None

    class _FakeWarehouse:
        def close(self) -> None:
            return None

    class _FakePipeline:
        warehouse = _FakeWarehouse()

        def summarize_price_history_coverage(self, market_ids: set[str] | None = None) -> dict[str, pd.DataFrame]:
            assert market_ids == {"101025"}
            return {
                "request_summary": pd.DataFrame([{"status": "ok", "city": "Seoul", "request_count": 3}]),
                "request_details": pd.DataFrame([{"market_id": "101025", "outcome_label": "11°C"}]),
                "panel_summary": pd.DataFrame([{"city": "Seoul", "decision_horizon": "morning_of", "coverage_status": "ok"}]),
                "market_summary": pd.DataFrame([{"market_id": "101025", "market_ready": True}]),
                "details": pd.DataFrame([{"market_id": "101025", "coverage_status": "ok"}]),
            }

    class _Snapshot:
        def __init__(self) -> None:
            self.spec = type("_Spec", (), {"market_id": "101025"})()

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (None, None, _FakeHttp(), None, None, None),
    )
    monkeypatch.setattr("pmtmax.cli.main._bootstrap_snapshots", lambda markets_path=None, cities=None: [_Snapshot()])
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())

    output = tmp_path / "price_history_coverage.json"
    summarize_price_history_coverage(markets_path=tmp_path / "snapshots.json", output=output)
    payload = json.loads(output.read_text())
    assert payload["request_summary"][0]["status"] == "ok"
    assert payload["panel_summary"][0]["coverage_status"] == "ok"


def test_materialize_backtest_panel_command_filters_market_ids_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHttp:
        def close(self) -> None:
            return None

    class _FakeWarehouse:
        def close(self) -> None:
            return None

        def start_run(self, **_: object) -> object:
            return type("_Run", (), {"run_id": "run-1"})()

        def finish_run(self, run: object, *, status: str, notes: str = "") -> object:
            assert status == "completed"
            return run

    class _FakePipeline:
        warehouse = _FakeWarehouse()
        run_id: str | None = None

        def materialize_backtest_panel(
            self,
            frame: pd.DataFrame,
            *,
            output_name: str = "historical_backtest_panel",
            max_price_age_minutes: int = 720,
        ) -> pd.DataFrame:
            assert list(frame["market_id"].astype(str)) == ["101025"]
            assert output_name == "historical_backtest_panel"
            assert max_price_age_minutes == 720
            return pd.DataFrame([{"market_id": "101025", "coverage_status": "ok"}])

    dataset_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {"market_id": "101025", "market_spec_json": "{}", "decision_time_utc": pd.Timestamp(datetime.now(tz=UTC))},
            {"market_id": "34779", "market_spec_json": "{}", "decision_time_utc": pd.Timestamp(datetime.now(tz=UTC))},
        ]
    ).to_parquet(dataset_path)

    class _Snapshot:
        def __init__(self, market_id: str) -> None:
            self.spec = type("_Spec", (), {"market_id": market_id})()

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (
            type("_Config", (), {"app": type("_App", (), {"parquet_dir": tmp_path})()})(),
            None,
            _FakeHttp(),
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr("pmtmax.cli.main._config_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(
        "pmtmax.cli.main._bootstrap_snapshots",
        lambda markets_path=None, cities=None: [_Snapshot("101025")],
    )
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())

    materialize_backtest_panel(dataset_path=dataset_path, markets_path=tmp_path / "snapshots.json")


def test_backtest_real_history_writes_separate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "training.parquet"
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(
        [
            {
                "market_id": "101025",
                "market_spec_json": "{}",
                "market_prices_json": "{}",
                "winning_outcome": "11°C",
                "realized_daily_max": 11.0,
            },
            {
                "market_id": "101026",
                "market_spec_json": "{}",
                "market_prices_json": "{}",
                "winning_outcome": "12°C",
                "realized_daily_max": 12.0,
            },
        ]
    ).to_parquet(dataset_path)
    pd.DataFrame(
        [
            {
                "market_id": "101025",
                "decision_horizon": "morning_of",
                "outcome_label": "11°C",
                "coverage_status": "ok",
                "market_price": 0.5,
            }
        ]
    ).to_parquet(panel_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pmtmax.cli.main._run_real_history_backtest",
        lambda frame, panel, *, model_name, artifacts_dir, flat_stake: (
            {"mae": 1.0, "rmse": 1.0, "nll": 1.0, "avg_brier": 0.1, "avg_crps": 0.2, "num_trades": 1.0, "pnl": 0.5, "hit_rate": 1.0, "avg_edge": 0.1},
            [{"market_id": "101025", "pricing_source": "real_history"}],
        ),
    )

    backtest(
        dataset_path=dataset_path,
        panel_path=panel_path,
        pricing_source="real_history",
    )

    metrics = json.loads((tmp_path / "artifacts" / "backtest_metrics_real_history.json").read_text())
    trades = json.loads((tmp_path / "artifacts" / "backtest_trades_real_history.json").read_text())
    assert metrics["num_trades"] == 1.0
    assert trades[0]["pricing_source"] == "real_history"


def test_opportunity_report_marks_missing_books_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    outcome_label = snapshot.spec.outcome_labels()[0]
    token_id = snapshot.spec.token_ids[0]

    class _FakeBuilder:
        def __init__(self, **_: object) -> None:
            return None

        def build_live_row(self, spec: object, horizon: str = "morning_of") -> pd.DataFrame:
            return pd.DataFrame([{"market_id": getattr(spec, "market_id", "m1"), "horizon": horizon}])

    class _Forecast:
        generated_at = datetime.now(tz=UTC)
        mean = 11.0
        std = 1.5
        outcome_probabilities = {outcome_label: 0.55}

    config = type(
        "_Config",
        (),
        {
            "polymarket": type("_Poly", (), {"clob_base_url": "https://clob"})(),
            "weather": type("_Weather", (), {"models": []})(),
            "backtest": type("_Backtest", (), {"default_edge_threshold": 0.02})(),
            "execution": type(
                "_Exec",
                (),
                {
                    "max_spread_bps": 500,
                    "min_liquidity": 10.0,
                    "stale_forecast_minutes": 60,
                },
            )(),
        },
    )()

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), object(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main.predict_market", lambda *args, **kwargs: _Forecast())
    monkeypatch.setattr(
        "pmtmax.cli.main._load_snapshots",
        lambda **kwargs: [snapshot],
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._load_books_for_forecast",
        lambda clob, snap, probs, allow_synthetic_fallback=False: {
            outcome_label: BookSnapshot(
                market_id=snapshot.spec.market_id,
                token_id=token_id,
                outcome_label=outcome_label,
                source="missing",
                bids=[],
                asks=[],
            )
        },
    )

    output = tmp_path / "opportunity_report.json"
    opportunity_report(output=output)

    payload = json.loads(output.read_text())
    assert payload[0]["reason"] == "missing_book"
    assert payload[0]["book_source_counts"] == {"missing": 1}


def test_resolve_opportunity_shadow_horizon_uses_market_local_date() -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0].model_copy(deep=True)
    assert snapshot.spec is not None
    spec = snapshot.spec
    now_utc = datetime(2026, 3, 23, 3, 0, tzinfo=UTC)

    same_day = spec.model_copy(update={"target_local_date": datetime(2026, 3, 23, tzinfo=UTC).date()})
    next_day = spec.model_copy(update={"target_local_date": datetime(2026, 3, 24, tzinfo=UTC).date()})
    far_day = spec.model_copy(update={"target_local_date": datetime(2026, 3, 25, tzinfo=UTC).date()})

    assert _resolve_opportunity_shadow_horizon(same_day, now_utc=now_utc, near_term_days=1) == "morning_of"
    assert _resolve_opportunity_shadow_horizon(next_day, now_utc=now_utc, near_term_days=1) == "previous_evening"
    assert _resolve_opportunity_shadow_horizon(far_day, now_utc=now_utc, near_term_days=1) is None


def test_opportunity_shadow_command_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    snapshot.spec = snapshot.spec.model_copy(
        update={"target_local_date": datetime.now(tz=UTC).date()}
    )

    class _FakeBuilder:
        def __init__(self, **_: object) -> None:
            return None

    config = type(
        "_Config",
        (),
        {
            "polymarket": type("_Poly", (), {"clob_base_url": "https://clob"})(),
            "weather": type("_Weather", (), {"models": []})(),
            "backtest": type("_Backtest", (), {"default_edge_threshold": 0.02})(),
            "execution": type("_Exec", (), {"max_spread_bps": 500, "min_liquidity": 10.0, "stale_forecast_minutes": 60})(),
            "opportunity_shadow": type(
                "_Shadow",
                (),
                {
                    "interval_seconds": 1,
                    "max_cycles": 1,
                    "near_term_days": 1,
                    "state_path": tmp_path / "shadow_state.json",
                    "latest_output_path": tmp_path / "shadow_latest.json",
                    "history_output_path": tmp_path / "shadow_history.jsonl",
                    "summary_output_path": tmp_path / "shadow_summary.json",
                },
            )(),
        },
    )()

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), object(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main._load_snapshots", lambda **kwargs: [snapshot])

    def _fake_eval(
        snapshot,
        *,
        builder,
        clob,
        model_path,
        model_name,
        config,
        observed_at,
        decision_horizon,
        edge_threshold,
    ):
        assert decision_horizon == "morning_of"
        return OpportunityObservation(
            observed_at=observed_at,
            market_id=snapshot.spec.market_id,
            city=snapshot.spec.city,
            question=snapshot.spec.question,
            target_local_date=snapshot.spec.target_local_date,
            decision_horizon=decision_horizon,
            reason="tradable",
            market_url=f"https://polymarket.com/event/{snapshot.spec.slug}",
            outcome_label=snapshot.spec.outcome_labels()[0],
            fair_probability=0.62,
            best_bid=0.58,
            best_ask=0.60,
            spread=0.02,
            visible_liquidity=500.0,
            fee_estimate=0.012,
            slippage_estimate=0.011,
            raw_gap=0.02,
            after_cost_edge=-0.003,
            book_source="clob",
        )

    monkeypatch.setattr("pmtmax.cli.main._evaluate_opportunity_snapshot", _fake_eval)

    opportunity_shadow(max_cycles=1)

    latest = json.loads((tmp_path / "shadow_latest.json").read_text())
    summary = json.loads((tmp_path / "shadow_summary.json").read_text())
    state = json.loads((tmp_path / "shadow_state.json").read_text())
    history_lines = (tmp_path / "shadow_history.jsonl").read_text().strip().splitlines()

    assert latest[0]["reason"] == "tradable"
    assert latest[0]["decision_horizon"] == "morning_of"
    assert summary["tradable_count"] == 1
    assert state["tradable_count"] == 1
    assert len(history_lines) == 1


def test_live_mm_uses_inventory_mapping_for_quoter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    outcome_label = snapshot.spec.outcome_labels()[0]
    token_id = snapshot.spec.token_ids[0]

    class _FakeBuilder:
        def __init__(self, **_: object) -> None:
            return None

        def build_live_row(self, spec: object, horizon: str = "morning_of") -> pd.DataFrame:
            return pd.DataFrame([{"market_id": getattr(spec, "market_id", "m1"), "horizon": horizon}])

    class _Forecast:
        generated_at = datetime.now(tz=UTC)
        mean = 14.0
        std = 1.0
        outcome_probabilities = {outcome_label: 0.52}

    config = type(
        "_Config",
        (),
        {
            "polymarket": type("_Poly", (), {"clob_base_url": "https://clob"})(),
            "weather": type("_Weather", (), {"models": []})(),
            "market_making": type(
                "_MM",
                (),
                {
                    "max_position_per_outcome": 100.0,
                    "max_total_exposure": 500.0,
                    "max_loss": 50.0,
                    "base_half_spread": 0.02,
                    "skew_factor": 0.5,
                    "base_size": 10.0,
                },
            )(),
        },
    )()

    called: dict[str, object] = {}

    def _fake_compute_quotes(self, outcome_probs, token_ids, inventory, risk_limits):
        called["inventory_type"] = type(inventory)
        assert isinstance(inventory, dict)
        return [
            type(
                "_Quote",
                (),
                {
                    "token_id": token_id,
                    "outcome_label": outcome_label,
                    "fair_value": 0.52,
                    "bid_price": 0.50,
                    "bid_size": 10.0,
                    "ask_price": 0.54,
                    "ask_size": 10.0,
                },
            )()
        ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), object(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main.predict_market", lambda *args, **kwargs: _Forecast())
    monkeypatch.setattr("pmtmax.cli.main._load_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr(
        "pmtmax.cli.main._load_books_for_forecast",
        lambda clob, snap, probs, allow_synthetic_fallback=False: {
            outcome_label: BookSnapshot(
                market_id=snapshot.spec.market_id,
                token_id=token_id,
                outcome_label=outcome_label,
                source="clob",
                bids=[BookLevel(price=0.49, size=25.0)],
                asks=[BookLevel(price=0.51, size=25.0)],
            )
        },
    )
    monkeypatch.setattr(
        "pmtmax.execution.quoter.Quoter.compute_quotes",
        _fake_compute_quotes,
    )

    live_mm(model_path=Path("artifacts/models/test.pkl"), dry_run=True, post_orders=False)

    assert called["inventory_type"] is dict
    payload = json.loads((tmp_path / "artifacts" / "live_mm_preview.json").read_text())
    assert payload[0]["market_id"] == snapshot.spec.market_id
    assert payload[0]["city"] == snapshot.spec.city
