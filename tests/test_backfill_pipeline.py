from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from pmtmax.backfill import BackfillPipeline
from pmtmax.http import CachedHttpClient
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.warehouse import DataWarehouse


class _BrokenOpenMeteoClient:
    def historical_forecast(self, **_: object) -> dict:
        raise RuntimeError("force fixture fallback")

    def forecast(self, **_: object) -> dict:
        raise RuntimeError("force fixture fallback")


class _FailIfForecastFetchedClient:
    def historical_forecast(self, **_: object) -> dict:
        raise AssertionError("unexpected historical_forecast call")

    def forecast(self, **_: object) -> dict:
        raise AssertionError("unexpected forecast call")

    def single_run(self, **_: object) -> dict:
        raise AssertionError("unexpected single_run call")


def _load_fixture(city: str) -> dict:
    fixture_name = city.lower().replace(" ", "_") + "_daily.json"
    return json.loads((Path("tests/fixtures/openmeteo") / fixture_name).read_text())


class _SingleRunOpenMeteoClient:
    def historical_forecast(self, *, latitude: float, longitude: float, model: str, hourly: list[str], start_date: str, end_date: str, timezone: str) -> dict:  # noqa: ARG002
        return _load_fixture("Seoul")

    def forecast(self, **_: object) -> dict:
        return _load_fixture("Seoul")

    def single_run(self, *, latitude: float, longitude: float, model: str, hourly: list[str], run: str, forecast_days: int, timezone: str) -> dict:  # noqa: ARG002
        payload = _load_fixture("Seoul")
        payload["hourly"]["temperature_2m"] = [value + 1.0 for value in payload["hourly"]["temperature_2m"]]
        return payload


class _BadTruthHttp:
    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:  # noqa: ARG002
        return "<html>no daily max here</html>"


class _LagTruthHttp:
    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:  # noqa: ARG002
        if params and params.get("yyyymm") == "202508":
            return (
                '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'
                '"20250823","113","559","250","1411","315"\n'
                '"20250824","113","601","248","1342","302"\n'
            )
        return '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'


class _FakeClobClient:
    def __init__(
        self,
        payloads: dict[str, dict[str, object]],
        *,
        last_trade_payloads: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.payloads = payloads
        self.last_trade_payloads = last_trade_payloads or {}
        self.last_trade_calls: list[str] = []

    def get_prices_history(
        self,
        market: str,
        *,
        interval: str = "max",
        fidelity: int | None = 60,
        use_cache: bool = True,
    ) -> dict[str, object]:
        assert interval == "max"
        assert fidelity == 60
        assert use_cache is True
        return self.payloads.get(market, {"history": []})

    def get_last_trade_price(self, token_id: str) -> dict[str, object]:
        self.last_trade_calls.append(token_id)
        return self.last_trade_payloads.get(token_id, {})


def test_backfill_pipeline_builds_bronze_silver_gold_tables(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots()
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "test.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025", "ecmwf_aifs025_single"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    market_tables = pipeline.backfill_markets(snapshots, source_name="test")
    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=True,
        strict_archive=False,
        return_tables=True,
    )
    truth_tables = pipeline.backfill_truth(snapshots)
    gold = pipeline.materialize_training_set(
        snapshots,
        output_name="test_training_set",
        decision_horizons=["market_open", "morning_of"],
    )

    assert len(market_tables["bronze_market_snapshots"]) == 4
    assert len(market_tables["silver_market_specs"]) == 4
    assert int(forecast_tables["bronze_forecast_requests_count"]) == 16
    assert int(forecast_tables["silver_forecast_runs_hourly_count"]) > 0
    assert forecast_tables["bronze_forecast_requests"] is not None
    assert forecast_tables["silver_forecast_runs_hourly"] is not None
    assert len(forecast_tables["bronze_forecast_requests"]) == 16
    assert len(forecast_tables["silver_forecast_runs_hourly"]) > 0
    assert len(truth_tables["bronze_truth_snapshots"]) == 4
    assert len(truth_tables["silver_observations_daily"]) == 4
    assert len(gold) == 8
    assert (tmp_path / "parquet" / "silver" / "silver_forecast_runs_hourly.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "v2" / "test_training_set.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "v2" / "test_training_set_sequence.parquet").exists()
    assert (tmp_path / "manifests" / "warehouse_manifest.json").exists()
    assert any((tmp_path / "raw").rglob("*.json"))


def test_backfill_pipeline_materializes_gold_contracts(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "test_v2.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025", "ecmwf_aifs025_single"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets(snapshots, source_name="test_v2")
    pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=True,
        strict_archive=False,
    )
    pipeline.backfill_truth(snapshots)
    gold = pipeline.materialize_training_set(
        snapshots,
        output_name="test_training_set_v2",
        decision_horizons=["market_open", "morning_of"],
    )

    assert gold["contract_version"].eq("v2").all()
    assert gold["split_group"].eq(gold["group_id"]).all()
    assert (tmp_path / "parquet" / "gold" / "v2" / "test_training_set_v2.parquet").exists()
    assert (tmp_path / "parquet" / "gold" / "v2" / "test_training_set_v2_sequence.parquet").exists()


def test_backfill_pipeline_strict_archive_skips_fixture_fallback(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots()
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "strict.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=False,
        strict_archive=True,
        return_tables=True,
    )

    assert int(forecast_tables["bronze_forecast_requests_count"]) == 8
    assert int(forecast_tables["silver_forecast_runs_hourly_count"]) == 0
    assert forecast_tables["bronze_forecast_requests"] is not None
    assert forecast_tables["silver_forecast_runs_hourly"] is not None
    assert len(forecast_tables["bronze_forecast_requests"]) == 8
    assert forecast_tables["silver_forecast_runs_hourly"].empty
    assert set(forecast_tables["bronze_forecast_requests"]["request_kind"]) == {"probe", "full"}


def test_backfill_forecasts_can_skip_table_readback(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "lightweight.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    result = pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=True,
        strict_archive=False,
    )

    assert result["bronze_forecast_requests"] is None
    assert result["silver_forecast_runs_hourly"] is None
    assert int(result["bronze_forecast_requests_count"]) == 2
    assert int(result["silver_forecast_runs_hourly_count"]) > 0
    assert (tmp_path / "parquet" / "silver" / "silver_forecast_runs_hourly.parquet").exists()
    assert (tmp_path / "manifests" / "warehouse_manifest.json").exists()


def test_backfill_forecasts_missing_only_skips_existing_request_keys(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "missing_only.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    first = pipeline.backfill_forecasts(
        snapshots,
        strict_archive=True,
        single_run_horizons=["morning_of"],
    )
    pipeline.openmeteo = _FailIfForecastFetchedClient()  # type: ignore[assignment]
    second = pipeline.backfill_forecasts(
        snapshots,
        strict_archive=True,
        single_run_horizons=["morning_of"],
        missing_only=True,
    )

    assert int(first["bronze_forecast_requests_count"]) == int(second["bronze_forecast_requests_count"])
    assert int(first["silver_forecast_runs_hourly_count"]) == int(second["silver_forecast_runs_hourly_count"])


def test_backfill_pipeline_single_run_horizon_overrides_generic_rows(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "single_run.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets(snapshots, source_name="single_run_test")
    forecast_tables = pipeline.backfill_forecasts(
        snapshots,
        strict_archive=True,
        single_run_horizons=["morning_of"],
        return_tables=True,
    )
    pipeline.backfill_truth(snapshots)
    gold = pipeline.materialize_training_set(
        snapshots,
        output_name="single_run_training_set",
        decision_horizons=["morning_of"],
    )

    assert forecast_tables["silver_forecast_runs_hourly"] is not None
    single_run_rows = forecast_tables["silver_forecast_runs_hourly"].loc[
        forecast_tables["silver_forecast_runs_hourly"]["endpoint_kind"] == "single_run"
    ]
    assert not single_run_rows.empty
    assert set(single_run_rows["decision_horizon"].dropna().astype(str)) == {"morning_of"}
    assert len(gold) == 1
    row = gold.iloc[0]
    assert row["forecast_source_kind"] == "single_run"
    assert json.loads(str(row["selected_models_json"])) == ["ecmwf_ifs025"]
    assert pd.notna(row["issue_time_utc"])

    availability = pipeline.summarize_forecast_availability(top_k=2)
    assert not availability["summary"].empty
    assert not availability["recommended"].empty


def test_normalize_forecast_rows_handles_dst_nonexistent_local_times(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["NYC"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "dst.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )
    spec = snapshots[0].spec
    assert spec is not None

    rows = pipeline._normalize_forecast_rows(
        spec=spec.model_copy(update={"target_local_date": date(2026, 3, 8)}),
        model_name="ecmwf_ifs025",
        endpoint_kind="single_run",
        payload={
            "hourly": {
                "time": ["2026-03-08T01:00", "2026-03-08T02:00", "2026-03-08T03:00"],
                "temperature_2m": [40.0, 41.0, 42.0],
            }
        },
        raw_path="raw.json",
        raw_hash="hash",
        availability_status="available",
        source_variables=["temperature_2m"],
        retrieved_at=datetime.now(tz=UTC),
    )

    assert len(rows) == 3
    assert rows[1]["forecast_time_local"] == "2026-03-08T03:00:00-04:00"
    assert rows[1]["forecast_time_utc"] == "2026-03-08T07:00:00+00:00"
    features = pipeline._features_from_hourly_rows(pd.DataFrame(rows), date(2026, 3, 8), spec.timezone)
    assert features["num_hours"] == 3.0


def test_summarize_dataset_readiness_reports_city_level_progress(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "readiness.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_BrokenOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025", "ecmwf_aifs025_single"],
        truth_snapshot_dir=Path("tests/fixtures/truth"),
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets(snapshots, source_name="readiness_test")
    pipeline.backfill_forecasts(
        snapshots,
        allow_fixture_fallback=True,
        strict_archive=False,
    )
    pipeline.backfill_truth(snapshots)
    pipeline.materialize_training_set(
        snapshots,
        output_name="readiness_training_set",
        decision_horizons=["market_open", "morning_of"],
    )

    readiness = pipeline.summarize_dataset_readiness(snapshots)

    assert readiness["summary"].iloc[0]["city"] == "Seoul"
    assert int(readiness["summary"].iloc[0]["snapshot_count"]) == 1
    assert int(readiness["summary"].iloc[0]["forecast_ready_count"]) == 1
    assert int(readiness["summary"].iloc[0]["truth_ok_count"]) == 1
    assert int(readiness["summary"].iloc[0]["gold_market_count"]) == 1
    assert int(readiness["summary"].iloc[0]["gold_row_count"]) == 2
    assert readiness["details"].iloc[0]["readiness_status"] == "ready"


def test_backfill_pipeline_fails_closed_when_truth_rows_are_missing(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "missing_truth.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=_BadTruthHttp(),  # type: ignore[arg-type]
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets(snapshots, source_name="missing_truth_test")
    pipeline.backfill_forecasts(
        snapshots,
        strict_archive=True,
        single_run_horizons=["morning_of"],
    )
    truth_tables = pipeline.backfill_truth(snapshots)

    assert len(truth_tables["bronze_truth_snapshots"]) == 1
    assert truth_tables["silver_observations_daily"].empty
    with pytest.raises(ValueError, match="public truth response"):
        pipeline.materialize_training_set(
            snapshots,
            output_name="missing_truth_training_set",
            decision_horizons=["morning_of"],
        )


def test_backfill_pipeline_classifies_truth_archive_lag_and_summarizes_it(tmp_path: Path) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    lagged_snapshot = snapshot.model_copy(
        update={
            "spec": snapshot.spec.model_copy(update={"target_local_date": date(2025, 9, 1)}),  # type: ignore[union-attr]
        }
    )
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "truth_lag.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=_LagTruthHttp(),  # type: ignore[arg-type]
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_markets([lagged_snapshot], source_name="truth_lag_test")
    pipeline.backfill_forecasts(
        [lagged_snapshot],
        strict_archive=True,
        single_run_horizons=["morning_of"],
    )
    truth_tables = pipeline.backfill_truth([lagged_snapshot])
    summary = pipeline.summarize_truth_coverage()

    assert truth_tables["silver_observations_daily"].empty
    assert truth_tables["bronze_truth_snapshots"].iloc[0]["status"] == "lag"
    assert str(truth_tables["bronze_truth_snapshots"].iloc[0]["latest_available_date"]).startswith("2025-08-24")
    assert not summary["summary"].empty
    assert not summary["details"].empty
    assert set(summary["details"]["status"].astype(str)) == {"lag"}
    with pytest.raises(ValueError, match="Public archive lag detected: Seoul/RKSI: latest 2025-08-24"):
        pipeline.materialize_training_set(
            [lagged_snapshot],
            output_name="truth_lag_training_set",
            decision_horizons=["morning_of"],
        )


def test_empty_materialization_message_filters_truth_lag_to_requested_snapshots(tmp_path: Path) -> None:
    snapshots = bundled_market_snapshots(["Seoul", "NYC"])
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "lag_filter.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=_LagTruthHttp(),  # type: ignore[arg-type]
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )
    warehouse.upsert_table(
        "bronze_truth_snapshots",
        pd.DataFrame(
            [
                {
                    "market_id": snapshots[0].spec.market_id,  # type: ignore[union-attr]
                    "status": "lag",
                    "city": "Seoul",
                    "station_id": "RKSI",
                    "target_local_date": pd.Timestamp("2025-09-01"),
                    "latest_available_date": pd.Timestamp("2025-08-24"),
                },
                {
                    "market_id": snapshots[1].spec.market_id,  # type: ignore[union-attr]
                    "status": "lag",
                    "city": "NYC",
                    "station_id": "KLGA",
                    "target_local_date": pd.Timestamp("2025-09-01"),
                    "latest_available_date": pd.Timestamp("2025-08-27"),
                },
            ]
        ),
    )

    message = pipeline._empty_materialization_message([snapshots[0]], pd.DataFrame(), pd.DataFrame())

    assert "Seoul/RKSI: latest 2025-08-24" in message
    assert "NYC/KLGA" not in message


def test_backfill_price_history_and_panel_materialization_capture_coverage_states(tmp_path: Path) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    spec = snapshot.spec
    assert spec is not None
    token_ids = spec.token_ids
    payloads = {
        token_ids[0]: {
            "history": [
                {"t": int(datetime(2025, 1, 1, 8, 0, tzinfo=UTC).timestamp()), "p": 0.42},
                {"t": int(datetime(2025, 1, 1, 9, 0, tzinfo=UTC).timestamp()), "p": 0.55},
            ]
        },
        token_ids[1]: {
            "history": [
                {"t": int(datetime(2025, 1, 1, 0, 0, tzinfo=UTC).timestamp()), "p": 0.10},
            ]
        },
    }
    empty_token_ids = [token_id for token_id in token_ids if token_id not in payloads]
    clob = _FakeClobClient(
        payloads,
        last_trade_payloads={token_id: {"price": "0.11", "side": "BUY"} for token_id in empty_token_ids},
    )
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "prices.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    history_tables = pipeline.backfill_price_history([snapshot], clob=clob)
    bronze = history_tables["bronze_price_history_requests"]
    empty_request = bronze.loc[bronze["status"].astype(str) == "empty"].iloc[0]

    assert len(bronze) == len(token_ids)
    assert set(bronze["status"].astype(str)) == {"ok", "empty"}
    assert empty_request["empty_reason"] == "history_unavailable_last_trade_present"
    assert float(empty_request["last_trade_price"]) == 0.11
    assert empty_request["last_trade_side"] == "BUY"
    assert bool(empty_request["last_trade_present"]) is True
    assert set(clob.last_trade_calls) == set(empty_token_ids)
    assert len(history_tables["silver_price_timeseries"]) == 3

    dataset = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "city": spec.city,
                "target_date": pd.Timestamp(spec.target_local_date),
                "decision_horizon": "morning_of",
                "decision_time_utc": pd.Timestamp(datetime(2025, 1, 1, 9, 30, tzinfo=UTC)),
                "market_spec_json": spec.model_dump_json(),
                "winning_outcome": spec.outcome_labels()[0],
                "realized_daily_max": 10.0,
            }
        ]
    )
    panel = pipeline.materialize_backtest_panel(
        dataset,
        output_name="test_backtest_panel",
        max_price_age_minutes=60,
    )
    summary = pipeline.summarize_price_history_coverage({spec.market_id})

    assert len(panel) == len(token_ids)
    assert set(panel["coverage_status"].astype(str)) == {"missing", "ok", "stale"}
    assert set(summary["panel_summary"]["coverage_status"].astype(str)) == {"missing", "ok", "stale"}
    empty_summary = summary["request_summary"].loc[summary["request_summary"]["status"].astype(str) == "empty"].iloc[0]
    assert int(empty_summary["last_trade_probe_count"]) == len(empty_token_ids)
    assert int(empty_summary["last_trade_present_count"]) == len(empty_token_ids)
    assert int(empty_summary["history_empty_with_last_trade_count"]) == len(empty_token_ids)
    assert bool(summary["market_summary"].iloc[0]["market_ready"]) is True


def test_materialize_backtest_panel_fails_closed_when_price_history_table_is_empty(tmp_path: Path) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    spec = snapshot.spec
    assert spec is not None
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "empty_prices.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )
    dataset = pd.DataFrame(
        [
            {
                "market_id": spec.market_id,
                "city": spec.city,
                "target_date": pd.Timestamp(spec.target_local_date),
                "decision_horizon": "morning_of",
                "decision_time_utc": pd.Timestamp(datetime(2025, 1, 1, 9, 30, tzinfo=UTC)),
                "market_spec_json": spec.model_dump_json(),
                "winning_outcome": spec.outcome_labels()[0],
                "realized_daily_max": 10.0,
            }
        ]
    )

    panel = pipeline.materialize_backtest_panel(
        dataset,
        output_name="empty_price_backtest_panel",
        max_price_age_minutes=60,
    )

    assert len(panel) == len(spec.token_ids)
    assert set(panel["coverage_status"].astype(str)) == {"missing"}


def test_summarize_price_history_coverage_uses_latest_request_per_token(tmp_path: Path) -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    spec = snapshot.spec
    assert spec is not None
    token_ids = spec.token_ids
    payloads = {
        token_ids[0]: {
            "history": [
                {"t": int(datetime(2025, 1, 1, 8, 0, tzinfo=UTC).timestamp()), "p": 0.42},
            ]
        }
    }
    empty_token_ids = [token_id for token_id in token_ids if token_id not in payloads]
    warehouse = DataWarehouse.from_paths(
        duckdb_path=tmp_path / "duckdb" / "summary_latest.duckdb",
        parquet_root=tmp_path / "parquet",
        raw_root=tmp_path / "raw",
        manifest_root=tmp_path / "manifests",
        archive_root=tmp_path / "archive",
    )
    pipeline = BackfillPipeline(
        http=CachedHttpClient(tmp_path / "cache"),
        openmeteo=_SingleRunOpenMeteoClient(),  # type: ignore[arg-type]
        warehouse=warehouse,
        models=["ecmwf_ifs025"],
        truth_snapshot_dir=None,
        forecast_fixture_dir=Path("tests/fixtures/openmeteo"),
    )

    pipeline.backfill_price_history([snapshot], clob=_FakeClobClient(payloads))
    pipeline.backfill_price_history(
        [snapshot],
        clob=_FakeClobClient(
            payloads,
            last_trade_payloads={token_id: {"price": "0.11", "side": "BUY"} for token_id in empty_token_ids},
        ),
    )
    summary = pipeline.summarize_price_history_coverage({spec.market_id})

    assert len(summary["request_details"]) == len(token_ids)
    empty_summary = summary["request_summary"].loc[summary["request_summary"]["status"].astype(str) == "empty"].iloc[0]
    assert int(empty_summary["history_empty_with_last_trade_count"]) == len(empty_token_ids)
    assert int(empty_summary["history_empty_without_last_trade_count"]) == 0
