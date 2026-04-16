from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pmtmax.cli.main import (
    autoresearch_init,
    autoresearch_promote,
    autoresearch_step,
    train_advanced,
)
from pmtmax.examples import example_market_specs
from pmtmax.modeling.autoresearch import load_lgbm_autoresearch_spec
from pmtmax.storage.schemas import ModelArtifact


def _dataset_frame(num_rows: int = 48) -> pd.DataFrame:
    spec_template = example_market_specs(["Seoul"])[0]
    rows: list[dict[str, object]] = []
    for idx in range(num_rows):
        target_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx)
        spec = spec_template.model_copy(
            update={
                "market_id": f"m{idx:03d}",
                "target_local_date": target_date.date(),
            }
        )
        realized = 8.0 + float(idx % 5)
        rows.append(
            {
                "market_id": spec.market_id,
                "station_id": spec.station_id,
                "target_date": target_date,
                "decision_horizon": "morning_of",
                "decision_time_utc": target_date.tz_localize("UTC"),
                "market_spec_json": spec.model_dump_json(),
                "market_prices_json": "{}",
                "realized_daily_max": realized,
                "winning_outcome": spec.outcome_labels()[0],
                "lead_hours": float(8 + idx % 6),
                "model_daily_max": realized - 0.2,
                "ecmwf_ifs025_model_daily_max": realized - 0.1,
            }
        )
    return pd.DataFrame(rows)


def _panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_id": "m000",
                "decision_horizon": "morning_of",
                "outcome_label": "8°C",
                "coverage_status": "ok",
                "market_price": 0.5,
            }
        ]
    )


def _write_candidate_spec(path: Path, *, run_tag: str, candidate_name: str = "candidate_alpha") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"run_tag: {run_tag}",
                f"candidate_name: {candidate_name}",
                "model_name: lgbm_emos",
                "base_variant: recency_neighbor_oof",
                "description: tune learning rate and leaves",
                "params:",
                "  num_leaves: 79",
                "  learning_rate: 0.025",
                "",
            ]
        )
    )
    return path


def test_train_advanced_supports_variant_spec(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "historical_training_set.parquet"
    _dataset_frame().to_parquet(dataset_path)
    spec_path = _write_candidate_spec(tmp_path / "candidate.yaml", run_tag="20260404-lgbm")

    calls: dict[str, object] = {}

    def _fake_train_model(model_name, frame, artifacts_dir, *, split_policy, seed, variant=None, variant_config=None):
        calls["model_name"] = model_name
        calls["variant"] = variant
        calls["variant_config"] = variant_config
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / f"{variant}.pkl"
        model_path.write_bytes(b"model")
        return ModelArtifact(
            model_name=model_name,
            version="0.1.0",
            trained_at=datetime.now(tz=UTC),
            features=["model_daily_max"],
            metrics={"fit_rows": float(len(frame))},
            path=str(model_path),
            contract_version="v2",
            seed=seed,
            dataset_signature="sig",
            split_policy=split_policy,
            variant=variant,
            status="experimental",
        )

    config = type("_Config", (), {"app": type("_App", (), {"random_seed": 7})()})()
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (config, None))
    monkeypatch.setattr("pmtmax.cli.main.train_model", _fake_train_model)

    train_advanced(
        dataset_path=dataset_path,
        model_name="lgbm_emos",
        variant_spec=spec_path,
        artifacts_dir=tmp_path / "models",
    )

    assert calls["model_name"] == "lgbm_emos"
    assert calls["variant"] == "candidate_alpha"
    assert calls["variant_config"].name == "candidate_alpha"


def test_autoresearch_init_writes_scaffold(tmp_path: Path) -> None:
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    _dataset_frame().to_parquet(dataset_path)
    _panel_frame().to_parquet(panel_path)

    autoresearch_init(
        run_tag="20260404-lgbm",
        dataset_path=dataset_path,
        panel_path=panel_path,
        root_dir=tmp_path / "runs",
    )

    run_dir = tmp_path / "runs" / "20260404-lgbm"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "program.md").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "candidates" / "candidate_template.yaml").exists()


def test_autoresearch_step_appends_result(tmp_path: Path, monkeypatch) -> None:
    run_tag = "20260404-lgbm"
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    _dataset_frame().to_parquet(dataset_path)
    _panel_frame().to_parquet(panel_path)
    autoresearch_init(run_tag=run_tag, dataset_path=dataset_path, panel_path=panel_path, root_dir=tmp_path / "runs")

    spec_path = _write_candidate_spec(tmp_path / "candidate.yaml", run_tag=run_tag)

    config = type("_Config", (), {"app": type("_App", (), {"random_seed": 7})()})()
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (config, None))

    def _fake_train_model(model_name, frame, artifacts_dir, *, split_policy, seed, variant=None, variant_config=None):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / f"{variant or model_name}.pkl"
        model_path.write_bytes(b"model")
        calibrator_path = artifacts_dir / f"{(variant or model_name)}.calibrator.pkl"
        calibrator_path.write_bytes(b"cal")
        return ModelArtifact(
            model_name=model_name,
            version="0.1.0",
            trained_at=datetime.now(tz=UTC),
            features=["model_daily_max"],
            metrics={"fit_rows": float(len(frame))},
            path=str(model_path),
            contract_version="v2",
            seed=seed,
            dataset_signature="sig",
            split_policy=split_policy,
            variant=variant,
            calibration_path=str(calibrator_path),
            status="experimental",
        )

    def _fake_evaluate_saved_model(model_path: Path, holdout: pd.DataFrame):
        assert not holdout.empty
        if "recency_neighbor_oof" in model_path.stem:
            return {"n": 10.0, "mae": 1.0, "crps": 0.8, "brier": 0.2}
        return {"n": 10.0, "mae": 0.9, "crps": 0.7, "brier": 0.18}

    monkeypatch.setattr("pmtmax.cli.main.train_model", _fake_train_model)
    monkeypatch.setattr("pmtmax.cli.main.evaluate_saved_model", _fake_evaluate_saved_model)

    autoresearch_step(
        spec_path=spec_path,
        dataset_path=dataset_path,
        root_dir=tmp_path / "runs",
    )

    result_path = tmp_path / "runs" / run_tag / "analysis" / "quick_eval__candidate_alpha.json"
    payload = json.loads(result_path.read_text())
    assert payload["status"] == "keep"
    assert payload["candidate_name"] == "candidate_alpha"
    results_jsonl = (tmp_path / "runs" / run_tag / "results.jsonl").read_text().strip().splitlines()
    assert len(results_jsonl) == 1


def test_autoresearch_promote_writes_promoted_spec_without_publishing_alias(tmp_path: Path, monkeypatch) -> None:
    run_tag = "20260404-lgbm"
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    _dataset_frame().to_parquet(dataset_path)
    _panel_frame().to_parquet(panel_path)
    autoresearch_init(run_tag=run_tag, dataset_path=dataset_path, panel_path=panel_path, root_dir=tmp_path / "runs")

    spec_path = _write_candidate_spec(tmp_path / "candidate.yaml", run_tag=run_tag)
    spec = load_lgbm_autoresearch_spec(spec_path)
    run_dir = tmp_path / "runs" / run_tag
    gate_summary_path = run_dir / "analysis" / f"gate_summary__{spec.candidate_name}.json"
    gate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    gate_summary_path.write_text(json.dumps({"benchmark_gate_passed": True}))
    paper_dir = run_dir / "analysis" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / f"paper_analysis_summary__{spec.candidate_name}.json").write_text(
        json.dumps({"overall_gate_decision": "GO"})
    )

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lgbm_emos__{spec.candidate_name}.pkl"
    model_path.write_bytes(b"model")
    model_path.with_name(f"{model_path.stem}.calibrator.pkl").write_bytes(b"cal")

    monkeypatch.setattr("pmtmax.cli.main.promoted_lgbm_emos_spec_path", lambda candidate_name: tmp_path / "promoted" / f"{candidate_name}.yaml")

    autoresearch_promote(
        spec_path=spec_path,
        root_dir=tmp_path / "runs",
    )

    assert (tmp_path / "promoted" / "candidate_alpha.yaml").exists()
    summary = json.loads(
        (tmp_path / "runs" / run_tag / "analysis" / "promotion_summary__candidate_alpha.json").read_text()
    )
    assert summary["published_aliases"] == {}
