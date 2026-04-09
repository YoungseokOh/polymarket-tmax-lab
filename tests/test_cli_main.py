from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from pmtmax.cli.main import (
    _bootstrap_snapshots,
    _collection_preflight_report,
    _evaluate_market_signal,
    _load_alias_metadata,
    _load_snapshots,
    _quote_proxy_prices,
    _resolve_opportunity_shadow_horizon,
    _resolve_signal_horizon_with_reason,
    _run_quote_proxy_backtest,
    _run_real_history_backtest,
    backtest,
    bootstrap_lab,
    execution_sensitivity_report,
    execution_watchlist_playbook,
    live_mm,
    market_bottleneck_report,
    materialize_backtest_panel,
    open_phase_shadow,
    opportunity_report,
    opportunity_shadow,
    paper_multimodel_report,
    revenue_gate_report,
    station_cycle,
    station_daemon,
    station_dashboard,
    summarize_dataset_readiness,
    summarize_price_history_coverage,
    summarize_truth_coverage,
)
from pmtmax.config.settings import EnvSettings
from pmtmax.markets.repository import bundled_market_snapshots, load_market_snapshots
from pmtmax.storage.schemas import BookLevel, BookSnapshot, OpportunityObservation


def _future_snapshot(city: str = "Seoul"):
    snapshot = bundled_market_snapshots([city])[0].model_copy(deep=True)
    assert snapshot.spec is not None
    local_today = datetime.now(tz=UTC).astimezone(ZoneInfo(snapshot.spec.timezone)).date()
    snapshot.spec = snapshot.spec.model_copy(
        update={"target_local_date": local_today + timedelta(days=1)}
    )
    return snapshot


def _future_snapshot_days(city: str, days: int):
    if city == "Seoul":
        snapshot = bundled_market_snapshots([city])[0].model_copy(deep=True)
    else:
        snapshot = next(
            snap.model_copy(deep=True)
            for snap in load_market_snapshots(Path("configs/market_inventory/recent_core_temperature_snapshots.json"))
            if snap.spec is not None and snap.spec.city == city
        )
    assert snapshot.spec is not None
    local_today = datetime.now(tz=UTC).astimezone(ZoneInfo(snapshot.spec.timezone)).date()
    snapshot.spec = snapshot.spec.model_copy(
        update={"target_local_date": local_today + timedelta(days=days)}
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


def test_load_alias_metadata_repairs_missing_alias_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "det2prob_nn.pkl").write_bytes(b"model")
    (models_dir / "det2prob_nn.calibrator.pkl").write_bytes(b"calibrator")
    metadata_path = models_dir / "trading_champion.json"
    metadata_path.write_text(
        json.dumps(
            {
                "alias_name": "trading_champion",
                "model_name": "det2prob_nn",
                "alias_path": "/tmp/broken/trading_champion.pkl",
                "alias_calibration_path": "/tmp/broken/trading_champion.calibrator.pkl",
                "source_model_path": "/tmp/broken/det2prob_nn.pkl",
                "source_calibration_path": "/tmp/broken/det2prob_nn.calibrator.pkl",
                "contract_version": "v2",
            }
        )
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._default_model_path",
        lambda model_name: models_dir / f"{model_name}.pkl",
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._default_alias_metadata_path",
        lambda alias_name: models_dir / f"{alias_name}.json",
    )

    payload = _load_alias_metadata("trading_champion")

    assert payload["alias_path"] == str(models_dir / "trading_champion.pkl")
    assert payload["alias_calibration_path"] == str(models_dir / "trading_champion.calibrator.pkl")
    assert payload["source_model_path"] == str(models_dir / "det2prob_nn.pkl")
    assert payload["source_calibration_path"] == str(models_dir / "det2prob_nn.calibrator.pkl")
    assert Path(payload["alias_path"]).exists()
    assert Path(payload["alias_calibration_path"]).exists()


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


def test_bootstrap_lab_passes_forecast_missing_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeHttp:
        def close(self) -> None:
            return None

    class _FakeWarehouse:
        def write_manifest(self) -> Path:
            path = tmp_path / "manifests" / "warehouse_manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            return path

        def start_run(self, **_: object) -> object:
            return type("_Run", (), {"run_id": "run-1"})()

        def compact(self) -> dict[str, int]:
            return {"gold_training_examples_tabular": 1}

        def finish_run(self, run: object, *, status: str, notes: str = "") -> object:
            captured["run_status"] = status
            captured["run_notes"] = notes
            return run

        def close(self) -> None:
            return None

    class _FakePipeline:
        warehouse = _FakeWarehouse()
        run_id: str | None = None

        def backfill_markets(self, snapshots: list[object], source_name: str = "snapshot") -> dict[str, pd.DataFrame]:
            captured["market_snapshots"] = snapshots
            captured["market_source_name"] = source_name
            return {}

        def backfill_forecasts(
            self,
            snapshots: list[object],
            *,
            models: list[str] | None = None,
            allow_fixture_fallback: bool = False,
            strict_archive: bool = True,
            single_run_horizons: list[str] | None = None,
            missing_only: bool = False,
            return_tables: bool = False,
        ) -> dict[str, object]:
            captured["forecast_snapshots"] = snapshots
            captured["forecast_missing_only"] = missing_only
            captured["forecast_single_run_horizons"] = single_run_horizons
            return {
                "bronze_forecast_requests_count": 0,
                "silver_forecast_runs_hourly_count": 0,
                "bronze_forecast_requests": None,
                "silver_forecast_runs_hourly": None,
            }

        def backfill_truth(self, snapshots: list[object]) -> dict[str, pd.DataFrame]:
            captured["truth_snapshots"] = snapshots
            return {}

        def materialize_training_set(
            self,
            snapshots: list[object],
            *,
            output_name: str = "historical_training_set",
            decision_horizons: list[str] | None = None,
            contract: str = "both",
            allow_canonical_overwrite: bool = False,
        ) -> pd.DataFrame:
            captured["materialize_snapshots"] = snapshots
            return pd.DataFrame([{"market_id": "m1"}])

        def summarize_forecast_availability(self) -> dict[str, pd.DataFrame]:
            return {"summary": pd.DataFrame(), "recommended": pd.DataFrame()}

    config = type(
        "_Config",
        (),
        {
            "app": type(
                "_App",
                (),
                {
                    "data_dir": tmp_path / "data",
                    "duckdb_path": tmp_path / "data" / "duckdb" / "warehouse.duckdb",
                    "manifest_dir": tmp_path / "data" / "manifests",
                    "parquet_dir": tmp_path / "data" / "parquet",
                    "raw_dir": tmp_path / "data" / "raw",
                    "archive_dir": tmp_path / "data" / "archive",
                },
            )(),
            "backtest": type("_Backtest", (), {"decision_horizons": ["market_open", "morning_of"]})(),
        },
    )()

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, None, _FakeHttp(), None, None, None),
    )
    monkeypatch.setattr("pmtmax.cli.main._bootstrap_snapshots", lambda markets_path=None, cities=None: ["snapshot"])
    monkeypatch.setattr("pmtmax.cli.main._backfill_pipeline", lambda config, http, openmeteo: _FakePipeline())
    monkeypatch.setattr("pmtmax.cli.main._config_hash", lambda *args, **kwargs: "hash")

    bootstrap_lab(
        markets_path=tmp_path / "snapshots.json",
        forecast_missing_only=True,
        cleanup_legacy=False,
        seed_path=tmp_path / "missing_seed.tar.gz",
    )

    assert captured["forecast_missing_only"] is True
    assert captured["market_snapshots"] == ["snapshot"]
    assert captured["forecast_snapshots"] == ["snapshot"]
    assert captured["truth_snapshots"] == ["snapshot"]
    assert captured["run_status"] == "completed"


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
            allow_canonical_overwrite: bool = False,
        ) -> pd.DataFrame:
            assert list(frame["market_id"].astype(str)) == ["101025"]
            assert output_name == "historical_backtest_panel"
            assert max_price_age_minutes == 720
            assert allow_canonical_overwrite is False
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
        "pmtmax.cli.main.load_settings",
        lambda: (
            type(
                "_Config",
                (),
                {
                    "app": type("_App", (), {"random_seed": 7})(),
                    "execution": type("_Exec", (), {"default_fee_bps": 0.0})(),
                },
            )(),
            EnvSettings(),
        ),
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._run_real_history_backtest",
        lambda frame, panel, *, model_name, variant=None, variant_config=None, artifacts_dir, flat_stake, default_fee_bps, split_policy, seed, min_train_size=None, retrain_stride=1: (
            {"mae": 1.0, "rmse": 1.0, "nll": 1.0, "avg_brier": 0.1, "avg_crps": 0.2, "num_trades": 1.0, "pnl": 0.5, "hit_rate": 1.0, "avg_edge": 0.1},
            [{"market_id": "101025", "pricing_source": "real_history"}],
        ),
    )

    backtest(
        dataset_path=dataset_path,
        panel_path=panel_path,
        pricing_source="real_history",
        model_name="gaussian_emos",
    )

    metrics = json.loads((tmp_path / "artifacts" / "backtests" / "v2" / "backtest_metrics_real_history.json").read_text())
    trades = json.loads((tmp_path / "artifacts" / "backtests" / "v2" / "backtest_trades_real_history.json").read_text())
    assert metrics["num_trades"] == 1.0
    assert trades[0]["pricing_source"] == "real_history"


def test_backtest_quote_proxy_writes_separate_artifacts(
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
        "pmtmax.cli.main.load_settings",
        lambda: (
            type(
                "_Config",
                (),
                {
                    "app": type("_App", (), {"random_seed": 7})(),
                    "execution": type("_Exec", (), {"default_fee_bps": 0.0})(),
                },
            )(),
            EnvSettings(),
        ),
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._run_quote_proxy_backtest",
        lambda frame, panel, *, model_name, variant=None, variant_config=None, artifacts_dir, flat_stake, default_fee_bps, quote_proxy_half_spread, split_policy, seed, min_train_size=None, retrain_stride=1: (
            {
                "mae": 1.0,
                "rmse": 1.0,
                "nll": 1.0,
                "avg_brier": 0.1,
                "avg_crps": 0.2,
                "num_trades": 1.0,
                "pnl": 0.25,
                "hit_rate": 1.0,
                "avg_edge": 0.05,
            },
            [{"market_id": "101025", "pricing_source": "quote_proxy"}],
        ),
    )

    backtest(
        dataset_path=dataset_path,
        panel_path=panel_path,
        pricing_source="quote_proxy",
        quote_proxy_half_spread=0.02,
        model_name="gaussian_emos",
    )

    metrics = json.loads((tmp_path / "artifacts" / "backtests" / "v2" / "backtest_metrics_quote_proxy.json").read_text())
    trades = json.loads((tmp_path / "artifacts" / "backtests" / "v2" / "backtest_trades_quote_proxy.json").read_text())
    assert metrics["num_trades"] == 1.0
    assert trades[0]["pricing_source"] == "quote_proxy"


def test_run_real_history_backtest_counts_missing_coverage_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec
    outcome_label = spec.outcome_labels()[0]

    frame = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-16"),
                "decision_horizon": "morning_of",
            },
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-17"),
                "decision_horizon": "morning_of",
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "decision_horizon": "morning_of",
                "outcome_label": outcome_label,
                "coverage_status": "missing",
                "market_price": 0.5,
            }
        ]
    )

    monkeypatch.setattr(
        "pmtmax.cli.main.train_model",
        lambda model_name, train, artifacts_dir, *, split_policy, seed: type(
            "_Artifact",
            (),
            {"path": str(tmp_path / "model.pkl")},
        )(),
    )
    monkeypatch.setattr(
        "pmtmax.cli.main.predict_market",
        lambda *args, **kwargs: type(
            "_Forecast",
            (),
            {
                "mean": 10.0,
                "std": 1.0,
                "samples": [10.0, 10.0],
                "outcome_probabilities": {outcome_label: 0.55},
            },
        )(),
    )

    metrics, trades = _run_real_history_backtest(
        frame,
        panel,
        model_name="gaussian_emos",
        artifacts_dir=tmp_path,
        flat_stake=1.0,
        default_fee_bps=0.0,
    )

    assert trades == []
    assert metrics["priced_decision_rows"] == 0.0
    assert metrics["skipped_missing_price"] == 1.0
    assert metrics["skipped_non_positive_edge"] == 0.0


def test_run_real_history_backtest_counts_stale_coverage_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec
    outcome_label = spec.outcome_labels()[0]

    frame = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-16"),
                "decision_horizon": "morning_of",
            },
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-17"),
                "decision_horizon": "morning_of",
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "decision_horizon": "morning_of",
                "outcome_label": outcome_label,
                "coverage_status": "stale",
                "market_price": 0.5,
            }
        ]
    )

    monkeypatch.setattr(
        "pmtmax.cli.main.train_model",
        lambda model_name, train, artifacts_dir, *, split_policy, seed: type(
            "_Artifact",
            (),
            {"path": str(tmp_path / "model.pkl")},
        )(),
    )
    monkeypatch.setattr(
        "pmtmax.cli.main.predict_market",
        lambda *args, **kwargs: type(
            "_Forecast",
            (),
            {
                "mean": 10.0,
                "std": 1.0,
                "samples": [10.0, 10.0],
                "outcome_probabilities": {outcome_label: 0.55},
            },
        )(),
    )

    metrics, trades = _run_real_history_backtest(
        frame,
        panel,
        model_name="gaussian_emos",
        artifacts_dir=tmp_path,
        flat_stake=1.0,
        default_fee_bps=0.0,
    )

    assert trades == []
    assert metrics["priced_decision_rows"] == 0.0
    assert metrics["skipped_missing_price"] == 0.0
    assert metrics["skipped_stale_price"] == 1.0


def test_run_quote_proxy_backtest_uses_proxy_execution_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec
    outcome_label = spec.outcome_labels()[0]

    frame = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-16"),
                "decision_horizon": "morning_of",
            },
            {
                "market_id": spec.market_id,
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "winning_outcome": outcome_label,
                "realized_daily_max": 10.0,
                "target_date": pd.Timestamp("2026-03-17"),
                "decision_horizon": "morning_of",
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "decision_horizon": "morning_of",
                "outcome_label": outcome_label,
                "coverage_status": "ok",
                "market_price": 0.5,
                "price_age_seconds": 60.0,
            }
        ]
    )

    monkeypatch.setattr(
        "pmtmax.cli.main.train_model",
        lambda model_name, train, artifacts_dir, *, split_policy, seed: type(
            "_Artifact",
            (),
            {"path": str(tmp_path / "model.pkl")},
        )(),
    )
    monkeypatch.setattr(
        "pmtmax.cli.main.predict_market",
        lambda *args, **kwargs: type(
            "_Forecast",
            (),
            {
                "mean": 10.0,
                "std": 1.0,
                "samples": [10.0, 10.0],
                "outcome_probabilities": {outcome_label: 0.7},
            },
        )(),
    )

    metrics, trades = _run_quote_proxy_backtest(
        frame,
        panel,
        model_name="gaussian_emos",
        artifacts_dir=tmp_path,
        flat_stake=1.0,
        default_fee_bps=0.0,
        quote_proxy_half_spread=0.02,
    )

    assert metrics["num_trades"] == 1.0
    assert metrics["avg_execution_price_premium"] == pytest.approx(0.02)
    assert trades[0]["pricing_source"] == "quote_proxy"
    assert trades[0]["reference_market_price"] == pytest.approx(0.5)
    assert trades[0]["price"] == pytest.approx(0.52)


def test_quote_proxy_prices_clip_to_bounds() -> None:
    bid, ask = _quote_proxy_prices(0.01, half_spread=0.02)
    assert bid == pytest.approx(0.0005)
    assert ask == pytest.approx(0.03)


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
        contract_version = "v2"
        probability_source = "calibrated"
        distribution_family = "gaussian"
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

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main.predict_market", lambda *args, **kwargs: _Forecast())
    monkeypatch.setattr(
        "pmtmax.cli.main._load_scoped_snapshots",
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
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_model_path",
        lambda model_path, model_name: (tmp_path / "champion.pkl", "gaussian_emos"),
    )

    output = tmp_path / "opportunity_report.json"
    opportunity_report(output=output)

    payload = json.loads(output.read_text())
    assert payload[0]["reason"] == "missing_book"
    assert payload[0]["book_source_counts"] == {"missing": 1}


def test_resolve_signal_horizon_with_reason_applies_recent_policy() -> None:
    now_utc = datetime(2026, 3, 23, 3, 0, tzinfo=UTC)
    london_snapshot = _future_snapshot_days("London", 1)
    nyc_snapshot = _future_snapshot_days("NYC", 1)
    seoul_snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert london_snapshot.spec is not None
    assert nyc_snapshot.spec is not None
    assert seoul_snapshot.spec is not None

    policy = {
        "London": ["previous_evening"],
        "NYC": ["market_open", "previous_evening"],
        "Seoul": ["market_open", "previous_evening", "morning_of"],
    }

    london_spec = london_snapshot.spec.model_copy(update={"target_local_date": datetime(2026, 3, 25, tzinfo=UTC).date()})
    nyc_spec = nyc_snapshot.spec.model_copy(update={"target_local_date": datetime(2026, 3, 25, tzinfo=UTC).date()})
    seoul_spec = seoul_snapshot.spec.model_copy(update={"target_local_date": datetime(2026, 3, 23, tzinfo=UTC).date()})

    assert _resolve_signal_horizon_with_reason(
        london_spec,
        now_utc=now_utc,
        horizon="policy",
        horizon_policy=policy,
    ) == ("market_open", "policy_filtered")
    assert _resolve_signal_horizon_with_reason(
        nyc_spec,
        now_utc=now_utc,
        horizon="policy",
        horizon_policy=policy,
    ) == ("market_open", None)
    assert _resolve_signal_horizon_with_reason(
        seoul_spec,
        now_utc=now_utc,
        horizon="policy",
        horizon_policy=policy,
    ) == ("morning_of", None)


def test_opportunity_report_marks_policy_filtered_markets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot_days("London", 2)
    assert snapshot.spec is not None

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

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main._load_scoped_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr(
        "pmtmax.cli.main._load_recent_horizon_policy",
        lambda path=Path("configs/recent-core-horizon-policy.yaml"): {"London": ["previous_evening"]},
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_model_path",
        lambda model_path, model_name: (tmp_path / "champion.pkl", "gaussian_emos"),
    )

    output = tmp_path / "opportunity_report_policy.json"
    opportunity_report(output=output)

    payload = json.loads(output.read_text())
    assert payload[0]["reason"] == "policy_filtered"
    assert payload[0]["decision_horizon"] == "market_open"


def test_evaluate_market_signal_identifies_fee_killed_edge() -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    outcome_label = snapshot.spec.outcome_labels()[0]
    token_id = snapshot.spec.token_ids[0]
    books = {
        outcome_label: BookSnapshot(
            market_id=snapshot.spec.market_id,
            token_id=token_id,
            outcome_label=outcome_label,
            bids=[BookLevel(price=0.59, size=10.0)],
            asks=[BookLevel(price=0.60, size=10.0)],
        )
    }

    class _Clob:
        def get_fee_rate(self, token_id: str) -> float:
            assert token_id
            return 100.0

    evaluation = _evaluate_market_signal(
        snapshot,
        {outcome_label: 0.605},
        books,
        mode="paper",
        clob=_Clob(),
        default_fee_bps=30.0,
        edge_threshold=0.0,
        max_spread_bps=10_000,
        min_liquidity=0.0,
    )

    assert evaluation["reason"] == "fee_killed_edge"


def test_evaluate_market_signal_identifies_slippage_killed_edge() -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    outcome_label = snapshot.spec.outcome_labels()[0]
    token_id = snapshot.spec.token_ids[0]
    books = {
        outcome_label: BookSnapshot(
            market_id=snapshot.spec.market_id,
            token_id=token_id,
            outcome_label=outcome_label,
            bids=[BookLevel(price=0.59, size=10.0)],
            asks=[
                BookLevel(price=0.60, size=0.10),
                BookLevel(price=0.80, size=0.90),
            ],
        )
    }

    class _Clob:
        def get_fee_rate(self, token_id: str) -> float:
            assert token_id
            return 0.0

    evaluation = _evaluate_market_signal(
        snapshot,
        {outcome_label: 0.62},
        books,
        mode="paper",
        clob=_Clob(),
        default_fee_bps=30.0,
        edge_threshold=0.0,
        max_spread_bps=10_000,
        min_liquidity=0.0,
    )

    assert evaluation["reason"] == "slippage_killed_edge"


def test_evaluate_market_signal_distinguishes_positive_edge_from_spread_guardrail() -> None:
    snapshot = _future_snapshot("Seoul")
    assert snapshot.spec is not None
    outcome_label = snapshot.spec.outcome_labels()[0]
    token_id = snapshot.spec.token_ids[0]
    books = {
        outcome_label: BookSnapshot(
            market_id=snapshot.spec.market_id,
            token_id=token_id,
            outcome_label=outcome_label,
            bids=[BookLevel(price=0.01, size=100.0)],
            asks=[BookLevel(price=0.51, size=100.0)],
        )
    }

    class _Clob:
        def get_fee_rate(self, token_id: str) -> float:
            assert token_id
            return 0.0

    evaluation = _evaluate_market_signal(
        snapshot,
        {outcome_label: 0.80},
        books,
        mode="paper",
        clob=_Clob(),
        default_fee_bps=30.0,
        edge_threshold=0.0,
        max_spread_bps=500,
        min_liquidity=0.0,
    )

    assert evaluation["reason"] == "after_cost_positive_but_spread_too_wide"


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
    local_today = datetime.now(tz=UTC).astimezone(ZoneInfo(snapshot.spec.timezone)).date()
    snapshot.spec = snapshot.spec.model_copy(
        update={"target_local_date": local_today}
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

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main._load_scoped_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_model_path",
        lambda model_path, model_name: (tmp_path / "champion.pkl", "gaussian_emos"),
    )

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

    latest = json.loads((tmp_path / "artifacts" / "signals" / "v2" / "shadow_latest.json").read_text())
    summary = json.loads((tmp_path / "artifacts" / "signals" / "v2" / "shadow_summary.json").read_text())
    state = json.loads((tmp_path / "artifacts" / "signals" / "v2" / "shadow_state.json").read_text())
    history_lines = (tmp_path / "artifacts" / "signals" / "v2" / "shadow_history.jsonl").read_text().strip().splitlines()

    assert latest[0]["reason"] == "tradable"
    assert latest[0]["decision_horizon"] == "morning_of"
    assert summary["tradable_count"] == 1
    assert state["tradable_count"] == 1
    assert len(history_lines) == 1


def test_open_phase_shadow_command_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot_days("NYC", 2)
    assert snapshot.spec is not None
    opened_at = datetime.now(tz=UTC) - timedelta(hours=2)
    snapshot.market["componentMarkets"] = [
        {
            "createdAt": (opened_at - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            "created_at": (opened_at - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            "deployingTimestamp": (opened_at - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "acceptingOrdersTimestamp": opened_at.isoformat().replace("+00:00", "Z"),
        }
    ]

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
            "opportunity_shadow": type("_Shadow", (), {"interval_seconds": 1, "max_cycles": 1})(),
        },
    )()

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main._load_scoped_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_model_path",
        lambda model_path, model_name: (tmp_path / "champion.pkl", "gaussian_emos"),
    )

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
        assert decision_horizon == "market_open"
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

    open_phase_shadow(
        max_cycles=1,
        open_window_hours=24.0,
        output=tmp_path / "open_phase_history.jsonl",
        latest_output=tmp_path / "open_phase_latest.json",
        summary_output=tmp_path / "open_phase_summary.json",
        state_path=tmp_path / "open_phase_state.json",
    )

    latest = json.loads((tmp_path / "open_phase_latest.json").read_text())
    summary = json.loads((tmp_path / "open_phase_summary.json").read_text())
    state = json.loads((tmp_path / "open_phase_state.json").read_text())
    history_lines = (tmp_path / "open_phase_history.jsonl").read_text().strip().splitlines()

    assert latest[0]["reason"] == "tradable"
    assert latest[0]["decision_horizon"] == "market_open"
    assert latest[0]["open_phase_age_hours"] > 0
    assert summary["tradable_count"] == 1
    assert state["tradable_count"] == 1
    assert len(history_lines) == 1


def test_paper_multimodel_report_writes_summary_and_leaderboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot("Seoul")

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
            "execution": type("_Exec", (), {"max_spread_bps": 500, "min_liquidity": 10.0})(),
        },
    )()

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr("pmtmax.cli.main._load_scoped_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr("pmtmax.cli.main._load_full_books_for_snapshot", lambda clob, snapshot: {})
    monkeypatch.setattr("pmtmax.cli.main._load_recent_horizon_policy", lambda path=Path("configs/recent-core-horizon-policy.yaml"): {})
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_paper_multimodel_specs",
        lambda **kwargs: [
            {"label": "champion_alias", "model_path": tmp_path / "champion.pkl", "model_name": "lgbm_emos"},
            {"label": "challenger", "model_path": tmp_path / "challenger.pkl", "model_name": "lgbm_emos"},
        ],
    )

    def _fake_eval(*, model_path, **kwargs):
        if Path(model_path).stem == "champion":
            return (
                [
                    {
                        "market_id": "m1",
                        "city": "Seoul",
                        "decision_horizon": "previous_evening",
                        "reason": "raw_gap_non_positive",
                    }
                ],
                10_000.0,
            )
        return (
            [
                {
                    "market_id": "m1",
                    "city": "Seoul",
                    "decision_horizon": "previous_evening",
                    "reason": "fee_killed_edge",
                    "raw_gap": 0.02,
                    "after_cost_edge": -0.001,
                }
            ],
            9_999.0,
        )

    monkeypatch.setattr("pmtmax.cli.main._run_paper_model_evaluation", _fake_eval)

    output_dir = tmp_path / "paper_multi"
    paper_multimodel_report(output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text())
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")

    assert set(summary["models"]) == {"champion_alias", "challenger"}
    assert leaderboard.shape[0] == 2
    assert (output_dir / "champion_alias.json").exists()
    assert (output_dir / "challenger.json").exists()


def test_execution_sensitivity_report_writes_combo_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _future_snapshot("Seoul")

    class _FakeBuilder:
        def __init__(self, **_: object) -> None:
            return None

    config = type(
        "_Config",
        (),
        {
            "polymarket": type("_Poly", (), {"clob_base_url": "https://clob"})(),
            "weather": type("_Weather", (), {"models": []})(),
            "backtest": type("_Backtest", (), {"default_edge_threshold": 0.15})(),
            "execution": type("_Exec", (), {"max_spread_bps": 500, "min_liquidity": 10.0})(),
        },
    )()

    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
    )
    monkeypatch.setattr("pmtmax.cli.main._resolve_model_path", lambda model_path, model_name: (tmp_path / "champion.pkl", "lgbm_emos"))
    monkeypatch.setattr("pmtmax.cli.main.ClobReadClient", lambda http, base_url: object())
    monkeypatch.setattr("pmtmax.cli.main.DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr(
        "pmtmax.cli.main._load_paper_exploration_preset",
        lambda path=Path("configs/paper-exploration.yaml"): {
            "market_scopes": ["default"],
            "min_edges": [0.15, 0.05],
            "max_spread_bps": [500],
            "min_liquidity": [10.0],
            "horizon_policies": [{"label": "current_policy", "path": "configs/recent-core-horizon-policy.yaml"}],
        },
    )
    monkeypatch.setattr("pmtmax.cli.main._load_recent_horizon_policy", lambda path=Path("configs/recent-core-horizon-policy.yaml"): {})
    monkeypatch.setattr("pmtmax.cli.main._load_scoped_snapshots", lambda **kwargs: [snapshot])
    monkeypatch.setattr("pmtmax.cli.main._load_full_books_for_snapshot", lambda clob, snapshot: {})

    def _fake_eval(*, edge_threshold, **kwargs):
        tradable = edge_threshold <= 0.05
        return (
            [
                {
                    "market_id": "m1",
                    "city": "Seoul",
                    "decision_horizon": "previous_evening",
                    "reason": "tradable" if tradable else "fee_killed_edge",
                    "raw_gap": 0.02,
                    "after_cost_edge": 0.01 if tradable else -0.001,
                    "fill": {"price": 0.50, "size": 1.0} if tradable else None,
                }
            ],
            9_999.0,
        )

    monkeypatch.setattr("pmtmax.cli.main._run_paper_model_evaluation", _fake_eval)

    output_dir = tmp_path / "sensitivity"
    execution_sensitivity_report(output_dir=output_dir)

    summary = json.loads((output_dir / "summary.json").read_text())
    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")

    assert len(summary["combinations"]) == 2
    assert leaderboard["fills"].max() == 1
    assert (output_dir / "combos").exists()


def test_market_bottleneck_report_includes_shadow_context(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "paper_rows.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "market_id": "m1",
                    "city": "Seoul",
                    "decision_horizon": "previous_evening",
                    "reason": "fee_killed_edge",
                    "raw_gap": 0.03,
                    "after_cost_edge": -0.001,
                },
                {
                    "market_id": "m2",
                    "city": "London",
                    "decision_horizon": "market_open",
                    "reason": "policy_filtered",
                },
            ]
        )
    )
    opportunity_summary_path = tmp_path / "opportunity_summary.json"
    opportunity_summary_path.write_text(json.dumps({"gate_decision": "INCONCLUSIVE"}))
    observation_summary_path = tmp_path / "observation_summary.json"
    observation_summary_path.write_text(json.dumps({"gate_reason": "no_observations"}))

    output = tmp_path / "bottlenecks.json"
    market_bottleneck_report(
        input_path=input_path,
        opportunity_summary_path=opportunity_summary_path,
        observation_summary_path=observation_summary_path,
        output=output,
    )

    payload = json.loads(output.read_text())

    assert payload["summary"]["reason_counts"]["fee_killed_edge"] == 1
    assert payload["summary"]["policy_blocked_watchlist"][0]["city"] == "London"
    assert payload["shadow_context"]["opportunity"]["gate_decision"] == "INCONCLUSIVE"


def test_revenue_gate_report_writes_combined_summary(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "decision": "GO",
                "decision_reason": "positive_policy_pnl_in_real_and_proxy",
                "aggregate_policy_real_history_metrics": {"num_trades": 20.0, "pnl": 24.0},
                "aggregate_policy_quote_proxy_metrics": {"num_trades": 20.0, "pnl": 12.0},
                "aggregate_panel_coverage": {"rows": 55, "coverage": {"ok": 40, "missing": 15}},
            }
        )
    )
    opportunity_path = tmp_path / "opportunity.json"
    opportunity_path.write_text(
        json.dumps(
            {
                "cycles": 3,
                "markets_evaluated": 6,
                "raw_gap_positive_count": 2,
                "after_cost_edge_positive_count": 2,
            }
        )
    )
    open_phase_path = tmp_path / "open_phase.json"
    open_phase_path.write_text(
        json.dumps(
            {
                "cycles": 2,
                "markets_evaluated": 4,
                "raw_gap_positive_count": 1,
                "after_cost_edge_positive_count": 0,
            }
        )
    )

    output = tmp_path / "revenue_gate.json"
    revenue_gate_report(
        benchmark_summary_path=benchmark_path,
        opportunity_summary_path=opportunity_path,
        open_phase_summary_path=open_phase_path,
        output=output,
    )

    payload = json.loads(output.read_text())
    assert payload["decision"] == "GO"
    assert payload["required_model_alias"] == "trading_champion"
    assert payload["opportunity_shadow_gate"]["decision"] == "GO"


def test_station_dashboard_command_writes_json_and_html(tmp_path: Path) -> None:
    opportunity_path = tmp_path / "opportunity.json"
    observation_path = tmp_path / "observation.json"
    observation_summary_path = tmp_path / "observation_summary.json"
    queue_path = tmp_path / "queue.json"
    open_phase_path = tmp_path / "open_phase.json"
    open_phase_summary_path = tmp_path / "open_phase_summary.json"
    revenue_gate_path = tmp_path / "revenue_gate.json"
    watchlist_playbook_path = tmp_path / "execution_watchlist_playbook.json"

    opportunity_path.write_text(
        json.dumps(
            [
                {
                    "market_id": "m-seoul-1",
                    "city": "Seoul",
                    "target_local_date": "2026-04-05",
                    "decision_horizon": "morning_of",
                    "outcome_label": "11°C",
                    "edge": 0.02,
                    "reason": "tradable",
                    "best_ask": 0.89,
                }
            ]
        )
    )
    observation_path.write_text(
        json.dumps(
            [
                {
                    "city": "Taipei",
                    "target_local_date": "2026-04-05",
                    "decision_horizon": "morning_of",
                    "queue_state": "tradable",
                    "outcome_label": "21°C",
                    "edge": 0.01,
                    "source_family": "official_intraday",
                    "observation_source": "cwa_codis_report_month",
                }
            ]
        )
    )
    observation_summary_path.write_text(
        json.dumps(
            {
                "gate_decision": "GO",
                "gate_reason": "x",
                "by_source_family": {
                    "official_intraday": {
                        "markets_evaluated": 1,
                        "tradable_count": 1,
                        "manual_review_count": 0,
                        "after_cost_edge_positive_count": 1,
                        "gate_decision": "GO",
                    }
                },
                "by_observation_source": {
                    "cwa_codis_report_month": {
                        "markets_evaluated": 1,
                        "tradable_count": 1,
                        "manual_review_count": 0,
                        "after_cost_edge_positive_count": 1,
                        "gate_decision": "GO",
                    }
                },
            }
        )
    )
    queue_path.write_text(observation_path.read_text())
    open_phase_path.write_text(
        json.dumps(
            [
                {
                    "city": "London",
                    "target_local_date": "2026-04-05",
                    "decision_horizon": "market_open",
                    "outcome_label": "14°C",
                    "edge": 0.015,
                    "reason": "tradable",
                    "open_phase_age_hours": 2.0,
                }
            ]
        )
    )
    open_phase_summary_path.write_text(json.dumps({"gate_decision": "INCONCLUSIVE", "gate_reason": "x"}))
    revenue_gate_path.write_text(
        json.dumps(
            {
                "decision": "GO",
                "decision_reason": "x",
                "eligible_for_live_pilot": True,
                "required_model_alias": "trading_champion",
            }
        )
    )
    watchlist_playbook_path.write_text(
        json.dumps(
            {
                "playbook": [
                    {
                        "tier": "A",
                        "name": "fee_sensitive_watchlist",
                        "cities": ["Seoul"],
                        "evidence": [
                            {
                                "city": "Seoul",
                                "market_id": "m-seoul-1",
                                "target_local_date": "2026-04-05",
                                "decision_horizon": "morning_of",
                                "outcome_label": "11°C",
                                "watch_rule_threshold_ask": 0.901,
                            }
                        ],
                    }
                ]
            }
        )
    )

    json_output = tmp_path / "station_dashboard.json"
    html_output = tmp_path / "station_dashboard.html"
    station_dashboard(
        opportunity_report_path=opportunity_path,
        observation_latest_path=observation_path,
        observation_summary_path=observation_summary_path,
        queue_path=queue_path,
        open_phase_latest_path=open_phase_path,
        open_phase_summary_path=open_phase_summary_path,
        revenue_gate_summary_path=revenue_gate_path,
        watchlist_playbook_path=watchlist_playbook_path,
        json_output=json_output,
        html_output=html_output,
        state_path=tmp_path / "station_dashboard_state.json",
    )

    payload = json.loads(json_output.read_text())
    assert payload["overview"]["revenue_gate_decision"] == "GO"
    assert payload["overview"]["watchlist_alert_count"] == 1
    assert payload["observation_panel"]["source_family_breakdown"][0]["name"] == "official_intraday"
    assert "PMTMAX Station Dashboard" in html_output.read_text()


def test_execution_watchlist_playbook_writes_outputs(tmp_path: Path) -> None:
    champion_bottleneck_path = tmp_path / "market_bottleneck_report__champion_alias.json"
    challenger_bottleneck_path = tmp_path / "market_bottleneck_report__mega_neighbor_oof.json"
    fee_summary_path = tmp_path / "fee_watchlist" / "summary.json"
    policy_summary_path = tmp_path / "policy_watchlist" / "summary.json"
    sensitivity_summary_path = tmp_path / "execution_sensitivity" / "summary.json"
    fee_summary_path.parent.mkdir(parents=True, exist_ok=True)
    policy_summary_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_summary_path.parent.mkdir(parents=True, exist_ok=True)

    champion_bottleneck_path.write_text(
        json.dumps(
            {
                "summary": {
                    "reason_counts": {
                        "raw_gap_non_positive": 113,
                        "policy_filtered": 4,
                        "fee_killed_edge": 3,
                    }
                }
            }
        )
    )
    challenger_bottleneck_path.write_text(
        json.dumps(
            {
                "summary": {
                    "fee_sensitive_watchlist": [
                        {"city": "Taipei", "fee_killed_edge": 3},
                        {"city": "Chongqing", "fee_killed_edge": 2},
                    ],
                    "policy_blocked_watchlist": [
                        {"city": "London", "policy_filtered": 3},
                        {"city": "NYC", "policy_filtered": 1},
                    ],
                    "raw_edge_desert_watchlist": [
                        {"city": "Hong Kong", "raw_gap_non_positive": 4},
                        {"city": "Paris", "raw_gap_non_positive": 4},
                    ],
                }
            }
        )
    )
    fee_summary_path.write_text(
        json.dumps(
            {
                "leaderboard": [
                    {
                        "model": "mega_neighbor_oof",
                        "fee_killed_edge": 5,
                        "raw_gap_positive_count": 5,
                        "raw_gap_non_positive": 1,
                    }
                ]
            }
        )
    )
    (fee_summary_path.parent / "mega_neighbor_oof.json").write_text(
        json.dumps(
            [
                {
                    "market_id": "m1",
                    "city": "Taipei",
                    "target_local_date": "2026-04-05",
                    "decision_horizon": "morning_of",
                    "outcome_label": "27°C",
                    "reason": "fee_killed_edge",
                    "best_ask": 0.999,
                    "raw_gap": 0.001,
                    "after_cost_edge": -0.0989,
                    "fee_estimate": 0.0999,
                    "spread": 0.999,
                },
                {
                    "market_id": "m2",
                    "city": "Chongqing",
                    "target_local_date": "2026-04-06",
                    "decision_horizon": "market_open",
                    "outcome_label": "23°C",
                    "reason": "fee_killed_edge",
                    "best_ask": 0.99,
                    "raw_gap": 0.01,
                    "after_cost_edge": -0.089,
                    "fee_estimate": 0.099,
                    "spread": 0.98,
                },
            ]
        )
    )
    policy_summary_path.write_text(
        json.dumps(
            {
                "leaderboard": [
                    {
                        "model": "neighbor_oof_half_life20",
                        "fee_killed_edge": 1,
                        "raw_gap_positive_count": 1,
                        "raw_gap_non_positive": 7,
                    }
                ]
            }
        )
    )
    (policy_summary_path.parent / "neighbor_oof_half_life20.json").write_text(
        json.dumps(
            [
                {
                    "market_id": "m-nyc",
                    "city": "NYC",
                    "target_local_date": "2026-04-05",
                    "decision_horizon": "morning_of",
                    "outcome_label": "68-69°F",
                    "reason": "fee_killed_edge",
                    "best_ask": 0.999,
                    "after_cost_edge": -0.0989,
                }
            ]
        )
    )
    sensitivity_summary_path.write_text(
        json.dumps(
            {
                "combinations": [
                    {"fills": 0, "raw_gap_non_positive": 4},
                    {"fills": 0, "raw_gap_non_positive": 6},
                ]
            }
        )
    )

    output = tmp_path / "execution_watchlist_playbook.json"
    markdown_output = tmp_path / "execution_watchlist_playbook.md"
    execution_watchlist_playbook(
        champion_bottleneck_path=champion_bottleneck_path,
        challenger_bottleneck_path=challenger_bottleneck_path,
        fee_watchlist_summary_path=fee_summary_path,
        policy_watchlist_summary_path=policy_summary_path,
        sensitivity_summary_path=sensitivity_summary_path,
        output=output,
        markdown_output=markdown_output,
    )

    payload = json.loads(output.read_text())
    assert payload["headline"]["fee_watchlist_model"] == "mega_neighbor_oof"
    assert payload["playbook"][0]["cities"] == ["Taipei", "Chongqing"]
    thresholds = sorted(row["watch_rule_threshold_ask"] for row in payload["playbook"][0]["evidence"])
    assert thresholds == pytest.approx([0.9001, 0.901])
    assert "Tier A" in markdown_output.read_text()


def test_station_cycle_command_writes_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_station_cycle(**kwargs):
        calls.append(kwargs)
        return {
            "generated_at": "2026-04-05T00:00:00+00:00",
            "revenue_gate_decision": "GO",
            "queue_size": 2,
            "observation_tradable_count": 1,
            "opportunity_tradable_count": 1,
            "open_phase_count": 3,
            "dashboard_json_output": str(kwargs["dashboard_json_output"]),
            "dashboard_html_output": str(kwargs["dashboard_html_output"]),
        }

    monkeypatch.setattr(
        "pmtmax.cli.main._run_station_cycle",
        _fake_run_station_cycle,
    )

    state_path = tmp_path / "station_cycle_state.json"
    benchmark_summary_path = tmp_path / "benchmark_summary.json"
    benchmark_summary_path.write_text(json.dumps({"decision": "GO"}))
    dashboard_json_output = tmp_path / "station_dashboard_custom.json"
    dashboard_html_output = tmp_path / "station_dashboard_custom.html"
    dashboard_state_path = tmp_path / "station_dashboard_custom_state.json"
    station_cycle(
        benchmark_summary_path=benchmark_summary_path,
        dashboard_json_output=dashboard_json_output,
        dashboard_html_output=dashboard_html_output,
        dashboard_state_path=dashboard_state_path,
        state_path=state_path,
    )

    assert len(calls) == 1
    assert calls[0]["benchmark_summary_path"] == benchmark_summary_path
    assert calls[0]["dashboard_json_output"] == dashboard_json_output
    assert calls[0]["dashboard_html_output"] == dashboard_html_output
    assert calls[0]["dashboard_state_path"] == dashboard_state_path
    payload = json.loads(state_path.read_text())
    assert payload["revenue_gate_decision"] == "GO"
    assert payload["queue_size"] == 2


def test_station_daemon_runs_single_cycle_and_writes_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_station_cycle(**kwargs):
        calls.append(kwargs)
        return {
            "generated_at": "2026-04-05T00:00:00+00:00",
            "revenue_gate_decision": "INCONCLUSIVE",
            "queue_size": 0,
            "observation_tradable_count": 0,
            "opportunity_tradable_count": 0,
            "open_phase_count": 0,
            "dashboard_json_output": str(kwargs["dashboard_json_output"]),
            "dashboard_html_output": str(kwargs["dashboard_html_output"]),
        }

    monkeypatch.setattr("pmtmax.cli.main._run_station_cycle", _fake_run_station_cycle)

    state_path = tmp_path / "station_cycle_state.json"
    benchmark_summary_path = tmp_path / "benchmark_summary.json"
    benchmark_summary_path.write_text(json.dumps({"decision": "GO"}))
    dashboard_json_output = tmp_path / "station_dashboard_custom.json"
    dashboard_html_output = tmp_path / "station_dashboard_custom.html"
    dashboard_state_path = tmp_path / "station_dashboard_custom_state.json"
    station_daemon(
        benchmark_summary_path=benchmark_summary_path,
        max_cycles=1,
        interval=1,
        dashboard_json_output=dashboard_json_output,
        dashboard_html_output=dashboard_html_output,
        dashboard_state_path=dashboard_state_path,
        state_path=state_path,
    )

    assert len(calls) == 1
    assert calls[0]["benchmark_summary_path"] == benchmark_summary_path
    assert calls[0]["dashboard_json_output"] == dashboard_json_output
    assert calls[0]["dashboard_html_output"] == dashboard_html_output
    assert calls[0]["dashboard_state_path"] == dashboard_state_path
    payload = json.loads(state_path.read_text())
    assert payload["cycle"] == 1
    assert payload["revenue_gate_decision"] == "INCONCLUSIVE"


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
    class _FakeHttp:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "pmtmax.cli.main._runtime",
        lambda include_stores=False: (config, EnvSettings(), _FakeHttp(), None, None, object()),
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

    live_mm(model_path=Path("artifacts/models/test.pkl"), model_name="gaussian_emos", dry_run=True, post_orders=False)

    assert called["inventory_type"] is dict
    payload = json.loads((tmp_path / "artifacts" / "live_mm_preview.json").read_text())
    assert payload[0]["market_id"] == snapshot.spec.market_id
    assert payload[0]["city"] == snapshot.spec.city
