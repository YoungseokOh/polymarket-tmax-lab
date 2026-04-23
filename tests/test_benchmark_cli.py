from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import typer

from pmtmax.cli.main import benchmark_ablations, benchmark_models, publish_champion


def test_benchmark_models_writes_leaderboard_without_publishing_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    pd.DataFrame(
        [
            {
                "market_id": "m1",
                "target_date": pd.Timestamp("2026-03-24"),
                "decision_horizon": "morning_of",
                "market_spec_json": "{}",
                "market_prices_json": "{}",
                "winning_outcome": "9°C",
                "realized_daily_max": 9.0,
                "model_daily_max": 8.9,
            }
        ]
    ).to_parquet(dataset_path)
    pd.DataFrame(
        [
            {
                "market_id": "m1",
                "decision_horizon": "morning_of",
                "outcome_label": "9°C",
                "coverage_status": "ok",
                "market_price": 0.5,
            }
        ]
    ).to_parquet(panel_path)

    config = type(
        "_Config",
        (),
        {
            "app": type("_App", (), {"random_seed": 7, "workspace_name": "historical_real", "dataset_profile": "real_market"})(),
            "models": type("_Models", (), {"benchmark_ladder": ["gaussian_emos", "det2prob_nn"]})(),
            "execution": type("_Exec", (), {"default_fee_bps": 30.0})(),
        },
    )()
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (config, None))

    def _fake_real_history_backtest(
        frame,
        panel,
        *,
        model_name,
        variant=None,
        variant_config=None,
        artifacts_dir,
        flat_stake,
        default_fee_bps,
        split_policy,
        seed,
        retrain_stride=1,
    ):
        assert variant is None
        assert variant_config is None
        assert retrain_stride >= 1
        base = 1.0 if model_name == "det2prob_nn" else 2.0
        return (
            {
                "mae": base,
                "rmse": base + 0.1,
                "nll": base + 0.2,
                "avg_brier": 0.05 * base,
                "avg_crps": 0.4 * base,
                "calibration_gap": 0.01 * base,
                "num_trades": 10.0,
                "pnl": 50.0 if model_name == "det2prob_nn" else 10.0,
                "hit_rate": 0.6 if model_name == "det2prob_nn" else 0.4,
                "avg_edge": 0.1,
                "priced_decision_rows": 12.0,
            },
            [],
        )

    def _fake_quote_proxy_backtest(
        frame,
        panel,
        *,
        model_name,
        variant=None,
        variant_config=None,
        artifacts_dir,
        flat_stake,
        default_fee_bps,
        quote_proxy_half_spread,
        split_policy,
        seed,
        retrain_stride=1,
    ):
        assert variant is None
        assert variant_config is None
        assert retrain_stride >= 1
        return (
            {
                "num_trades": 10.0,
                "pnl": 40.0 if model_name == "det2prob_nn" else 5.0,
                "hit_rate": 0.55 if model_name == "det2prob_nn" else 0.35,
                "avg_edge": 0.08,
                "priced_decision_rows": 12.0,
            },
            [],
        )

    monkeypatch.setattr("pmtmax.cli.main._run_real_history_backtest", _fake_real_history_backtest)
    monkeypatch.setattr("pmtmax.cli.main._run_quote_proxy_backtest", _fake_quote_proxy_backtest)

    leaderboard_output = tmp_path / "benchmarks" / "leaderboard.json"
    leaderboard_csv_output = tmp_path / "benchmarks" / "leaderboard.csv"
    summary_output = tmp_path / "benchmarks" / "summary.json"

    benchmark_models(
        dataset_path=dataset_path,
        panel_path=panel_path,
        models=["gaussian_emos", "det2prob_nn"],
        seeds=[7, 11],
        artifacts_dir=tmp_path / "models",
        leaderboard_output=leaderboard_output,
        leaderboard_csv_output=leaderboard_csv_output,
        summary_output=summary_output,
    )

    leaderboard = json.loads(leaderboard_output.read_text())
    summary = json.loads(summary_output.read_text())

    assert leaderboard[0]["model_name"] == "det2prob_nn"
    assert "execution_candidate_score" in leaderboard[0]
    assert summary["champion_model_name"] == "det2prob_nn"
    assert summary["execution_candidate_model_name"] == "det2prob_nn"
    assert summary["champion_published"] is False
    assert leaderboard_csv_output.exists()


def test_publish_champion_copies_public_alias_after_recent_core_go(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "workspace_models" / "lgbm_emos__candidate_alpha.pkl"
    calibrator_path = model_path.with_name("lgbm_emos__candidate_alpha.calibrator.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    calibrator_path.write_bytes(b"cal")

    recent_core_summary_path = tmp_path / "recent_core_benchmark_summary.json"
    recent_core_summary_path.write_text(
        json.dumps(
            {
                "decision": "GO",
                "decision_reason": "positive_policy_pnl_real_history_with_city_gates",
                "sample_adequacy": {"passes": True},
                "city_gate_details": {
                    "Seoul": {"passes": True},
                    "NYC": {"passes": True},
                    "London": {"passes": True},
                },
                "aggregate_policy_real_history_metrics": {"pnl": 12.0},
                "aggregate_policy_quote_proxy_metrics": {"pnl": 9.0},
                "aggregate_panel_coverage": {"ok_ratio": 0.31},
            }
        )
    )

    config = type(
        "_Config",
        (),
        {"app": type("_App", (), {"workspace_name": "recent_core_eval", "dataset_profile": "real_market"})()},
    )()
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (config, None))
    monkeypatch.setattr(
        "pmtmax.cli.main._default_model_path",
        lambda model_name: tmp_path / "public_models" / f"{model_name}.pkl",
    )
    monkeypatch.setattr(
        "pmtmax.cli.main._default_alias_metadata_path",
        lambda alias_name: tmp_path / "public_models" / f"{alias_name}.json",
    )

    publish_champion(
        model_path=model_path,
        recent_core_summary_path=recent_core_summary_path,
    )

    metadata = json.loads((tmp_path / "public_models" / "champion.json").read_text())
    assert metadata["alias_name"] == "champion"
    assert metadata["model_name"] == "lgbm_emos"
    assert metadata["workspace_name"] == "recent_core_eval"
    assert metadata["dataset_profile"] == "real_market"
    assert metadata["publish_gate"]["decision"] == "GO"
    assert metadata["variant"] == "candidate_alpha"
    assert Path(metadata["alias_path"]).exists()
    assert Path(metadata["alias_calibration_path"]).exists()


def test_publish_champion_rejects_diagnostic_reduced_core_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "workspace_models" / "lgbm_emos__candidate_alpha.pkl"
    calibrator_path = model_path.with_name("lgbm_emos__candidate_alpha.calibrator.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    calibrator_path.write_bytes(b"cal")

    recent_core_summary_path = tmp_path / "recent_core_benchmark_summary.json"
    recent_core_summary_path.write_text(
        json.dumps(
            {
                "decision": "INCONCLUSIVE",
                "decision_reason": "city_panel_coverage_inadequate",
                "sample_adequacy": {"passes": True},
                "city_gate_details": {
                    "Seoul": {"passes": True},
                    "NYC": {"passes": True},
                    "London": {"passes": False},
                },
                "reduced_core_candidate": {
                    "decision": "GO",
                    "publish_eligible": False,
                    "coverage_eligible_cities": ["Seoul", "NYC"],
                },
            }
        )
    )

    with pytest.raises(typer.BadParameter, match="diagnostic only and cannot be published"):
        publish_champion(
            model_path=model_path,
            model_name="lgbm_emos",
            recent_core_summary_path=recent_core_summary_path,
        )


def test_benchmark_ablations_writes_variant_leaderboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    pd.DataFrame(
        [
            {
                "market_id": "m1",
                "target_date": pd.Timestamp("2026-03-24"),
                "decision_horizon": "morning_of",
                "market_spec_json": "{}",
                "market_prices_json": "{}",
                "winning_outcome": "9°C",
                "realized_daily_max": 9.0,
                "model_daily_max": 8.9,
            }
        ]
    ).to_parquet(dataset_path)
    pd.DataFrame(
        [
            {
                "market_id": "m1",
                "decision_horizon": "morning_of",
                "outcome_label": "9°C",
                "coverage_status": "ok",
                "market_price": 0.5,
            }
        ]
    ).to_parquet(panel_path)

    config = type(
        "_Config",
        (),
        {
            "app": type("_App", (), {"random_seed": 7})(),
            "execution": type("_Exec", (), {"default_fee_bps": 30.0})(),
        },
    )()
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (config, None))
    monkeypatch.setattr(
        "pmtmax.cli.main.supported_ablation_variants",
        lambda model_name: ("legacy_fixed2", "current_gate3_scale"),
    )
    monkeypatch.setattr("pmtmax.cli.main.require_supported_variant", lambda model_name, variant: variant)

    def _fake_run_grouped_holdout_ablation(
        frame,
        panel,
        *,
        model_name,
        variant,
        artifacts_dir,
        flat_stake,
        default_fee_bps,
        quote_proxy_half_spread,
        split_policy,
        seed,
    ):
        base = 0.8 if variant == "legacy_fixed2" else 1.2
        return (
            {
                "mae": base,
                "rmse": base + 0.1,
                "nll": base + 0.2,
                "avg_brier": base / 10.0,
                "avg_crps": base / 5.0,
                "calibration_gap": base / 20.0,
                "num_trades": 5.0,
                "pnl": 20.0 if variant == "legacy_fixed2" else 5.0,
                "hit_rate": 0.5,
                "avg_edge": 0.1,
                "priced_decision_rows": 7.0,
            },
            {
                "num_trades": 5.0,
                "pnl": 10.0 if variant == "legacy_fixed2" else 2.0,
                "hit_rate": 0.4,
                "avg_edge": 0.08,
                "priced_decision_rows": 7.0,
            },
            {
                "artifact_path": str(artifacts_dir / f"{model_name}__{variant}.pkl"),
                "artifact_variant": variant,
                "artifact_status": "experimental",
                "artifact_diagnostics": {"pred_std_p90": 2.0 if variant == "legacy_fixed2" else 4.0},
                "calibration_path": str(artifacts_dir / f"{model_name}__{variant}.calibrator.pkl"),
            },
        )

    monkeypatch.setattr("pmtmax.cli.main._run_grouped_holdout_ablation", _fake_run_grouped_holdout_ablation)

    leaderboard_output = tmp_path / "benchmarks" / "ablation_leaderboard.json"
    leaderboard_csv_output = tmp_path / "benchmarks" / "ablation_leaderboard.csv"
    summary_output = tmp_path / "benchmarks" / "ablation_summary.json"

    benchmark_ablations(
        dataset_path=dataset_path,
        panel_path=panel_path,
        model_name="tuned_ensemble",
        variants=["legacy_fixed2", "current_gate3_scale"],
        split_policies=["market_day"],
        seeds=[7, 11],
        artifacts_dir=tmp_path / "models",
        leaderboard_output=leaderboard_output,
        leaderboard_csv_output=leaderboard_csv_output,
        summary_output=summary_output,
    )

    leaderboard = json.loads(leaderboard_output.read_text())
    summary = json.loads(summary_output.read_text())

    assert leaderboard[0]["variant"] == "legacy_fixed2"
    assert leaderboard[0]["model_family"] == "tuned_ensemble"
    assert leaderboard[0]["split_policy"] == "market_day"
    assert leaderboard[0]["diag_pred_std_p90_mean"] == 2.0
    assert summary["model_family"] == "tuned_ensemble"
    assert summary["variants"] == ["legacy_fixed2", "current_gate3_scale"]
    assert leaderboard_csv_output.exists()
