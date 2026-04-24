from __future__ import annotations

import json
from pathlib import Path

from pmtmax.modeling.autoresearch import path_signature
from pmtmax.modeling.research_agent import (
    ModelResearchLogEntry,
    ResearchOps,
    append_log_entry,
    next_auto_candidate_spec_path,
    parse_log,
    render_status_markdown,
    run_model_research_agent,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_append_model_research_log_entry_round_trips(tmp_path: Path) -> None:
    log_path = tmp_path / "model_research_log.md"

    append_log_entry(
        log_path,
        ModelResearchLogEntry(
            run_date="2026-04-24",
            run_tag="20260424-lgbm-high_neighbor_oof-agent",
            actions="train_baseline, init_run, create_candidate",
            outcome="keep",
            candidate="mr_20260424_01_lr_down_leaves_up",
            notes="baseline trained and first candidate queued",
        ),
    )

    entries = parse_log(log_path)

    assert len(entries) == 1
    assert entries[0].run_tag == "20260424-lgbm-high_neighbor_oof-agent"
    assert entries[0].candidate == "mr_20260424_01_lr_down_leaves_up"


def test_next_auto_candidate_spec_path_creates_unique_specs(tmp_path: Path) -> None:
    root_dir = tmp_path / "autoresearch"
    run_tag = "20260424-lgbm-high_neighbor_oof-agent"
    first = next_auto_candidate_spec_path(
        run_tag=run_tag,
        root_dir=root_dir,
        baseline_variant="high_neighbor_oof",
        auto_candidate_limit=5,
    )
    second = next_auto_candidate_spec_path(
        run_tag=run_tag,
        root_dir=root_dir,
        baseline_variant="high_neighbor_oof",
        auto_candidate_limit=5,
    )

    assert first is not None and first.exists()
    assert second is not None and second.exists()
    assert first.name != second.name
    assert "high_neighbor_oof" in first.read_text(encoding="utf-8")


def test_run_model_research_agent_bootstraps_and_processes_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "historical_training_set.parquet"
    panel_path = tmp_path / "historical_backtest_panel.parquet"
    dataset_path.write_bytes(b"dataset")
    panel_path.write_bytes(b"panel")
    markets_path = tmp_path / "inventory.json"
    markets_path.write_text("[]", encoding="utf-8")

    status_path = tmp_path / "checker" / "model_research_status.md"
    log_path = tmp_path / "checker" / "model_research_log.md"
    model_artifacts_dir = tmp_path / "artifacts" / "models" / "v2"
    autoresearch_root = tmp_path / "artifacts" / "autoresearch"
    champion_path = tmp_path / "artifacts" / "public_models" / "champion.json"
    promoted_dir = tmp_path / "configs" / "autoresearch" / "lgbm_emos" / "promoted"

    monkeypatch.setattr(
        "pmtmax.modeling.research_agent._verify_model_research_workspace",
        lambda markets_path: None,
    )
    monkeypatch.setattr(
        "pmtmax.modeling.research_agent.DEFAULT_CHAMPION_METADATA_PATH",
        champion_path,
    )
    monkeypatch.setattr(
        "pmtmax.modeling.research_agent.promoted_lgbm_emos_spec_path",
        lambda candidate_name: promoted_dir / f"{candidate_name}.yaml",
    )

    def fake_train_baseline(*, dataset_path: Path, model_name: str, artifacts_dir: Path, variant: str, pretrained_weather_model: Path | None = None, **_: object) -> None:
        artifact_path = artifacts_dir / f"{model_name}__{variant}.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"baseline")
        artifact_path.with_suffix(".calibrator.pkl").write_bytes(b"cal")
        _write_json(
            artifact_path.with_suffix(".json"),
            {
                "path": str(artifact_path),
                "variant": variant,
                "dataset_signature": path_signature(dataset_path),
                "pretrained_weather_model": str(pretrained_weather_model) if pretrained_weather_model is not None else None,
            },
        )

    def fake_init_run(*, run_tag: str, dataset_path: Path, panel_path: Path, baseline_variant: str, root_dir: Path, **_: object) -> None:
        run_dir = root_dir / run_tag
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis" / "paper").mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "manifest.json",
            {
                "run_tag": run_tag,
                "baseline_variant": baseline_variant,
                "dataset_signature": path_signature(dataset_path),
                "panel_signature": path_signature(panel_path),
            },
        )
        (run_dir / "results.jsonl").write_text("", encoding="utf-8")
        (run_dir / "candidates" / "candidate_template.yaml").write_text("{}", encoding="utf-8")

    def fake_step(*, spec_path: Path, dataset_path: Path, root_dir: Path, **_: object) -> None:
        text = spec_path.read_text(encoding="utf-8")
        candidate_name = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("candidate_name:"))
        run_tag = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("run_tag:"))
        model_path = root_dir / run_tag / "models" / f"lgbm_emos__{candidate_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"candidate")
        model_path.with_suffix(".calibrator.pkl").write_bytes(b"cal")
        _write_json(
            root_dir / run_tag / "analysis" / f"quick_eval__{candidate_name}.json",
            {
                "candidate_name": candidate_name,
                "status": "keep",
                "candidate_artifact_path": str(model_path),
                "dataset_signature": path_signature(dataset_path),
            },
        )
        with (root_dir / run_tag / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"candidate_name": candidate_name, "status": "keep"}) + "\n")

    def fake_gate(*, spec_path: Path, root_dir: Path, dataset_path: Path, panel_path: Path, **_: object) -> None:
        text = spec_path.read_text(encoding="utf-8")
        candidate_name = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("candidate_name:"))
        run_tag = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("run_tag:"))
        leaderboard_path = root_dir / run_tag / "analysis" / f"gate_leaderboard__{candidate_name}.json"
        leaderboard_csv_path = leaderboard_path.with_suffix(".csv")
        _write_json(leaderboard_path, {"rows": []})
        leaderboard_csv_path.write_text("variant\n", encoding="utf-8")
        _write_json(
            root_dir / run_tag / "analysis" / f"gate_summary__{candidate_name}.json",
            {
                "candidate_name": candidate_name,
                "benchmark_gate_passed": True,
                "dataset_signature": path_signature(dataset_path),
                "panel_signature": path_signature(panel_path),
                "leaderboard_path": str(leaderboard_path),
                "leaderboard_csv_path": str(leaderboard_csv_path),
            },
        )

    def fake_analyze_paper(*, spec_path: Path, root_dir: Path, **_: object) -> None:
        text = spec_path.read_text(encoding="utf-8")
        candidate_name = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("candidate_name:"))
        run_tag = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("run_tag:"))
        _write_json(
            root_dir / run_tag / "analysis" / "paper" / f"paper_analysis_summary__{candidate_name}.json",
            {
                "candidate_name": candidate_name,
                "analysis_completed": True,
                "overall_gate_decision": "GO",
            },
        )

    def fake_promote(*, spec_path: Path, root_dir: Path, **_: object) -> None:
        text = spec_path.read_text(encoding="utf-8")
        candidate_name = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("candidate_name:"))
        run_tag = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("run_tag:"))
        target_path = promoted_dir / f"{candidate_name}.yaml"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")
        _write_json(
            root_dir / run_tag / "analysis" / f"promotion_summary__{candidate_name}.json",
            {
                "candidate_name": candidate_name,
                "promoted_spec_path": str(target_path),
            },
        )

    def fake_publish(*, model_path: Path, variant: str, recent_core_summary_path: Path, **_: object) -> None:
        _write_json(
            champion_path,
            {
                "model_name": "lgbm_emos",
                "variant": variant,
                "alias_path": str(model_path),
                "publish_gate": {"decision": "GO", "recent_core_summary_path": str(recent_core_summary_path)},
            },
        )

    recent_core_summary_path = tmp_path / "recent_core_benchmark_summary.json"
    _write_json(recent_core_summary_path, {"decision": "GO"})

    summary = run_model_research_agent(
        status_path=status_path,
        log_path=log_path,
        markets_path=markets_path,
        dataset_path=dataset_path,
        panel_path=panel_path,
        model_name="lgbm_emos",
        baseline_variant="high_neighbor_oof",
        model_artifacts_dir=model_artifacts_dir,
        autoresearch_root=autoresearch_root,
        pretrained_weather_model=None,
        auto_candidate_limit=3,
        max_candidates=1,
        enable_training=True,
        enable_gate=True,
        enable_paper=True,
        enable_promote=True,
        enable_publish=True,
        recent_core_summary_path=recent_core_summary_path,
        force_baseline_train=False,
        run_tag_override=None,
        ops=ResearchOps(
            train_baseline=fake_train_baseline,
            init_run=fake_init_run,
            step=fake_step,
            gate=fake_gate,
            analyze_paper=fake_analyze_paper,
            promote=fake_promote,
            publish=fake_publish,
        ),
    )

    assert summary.baseline_trained is True
    assert summary.run_initialized is True
    assert summary.auto_candidate_created is True
    assert summary.run_tag.startswith("2026")
    assert summary.promoted_candidate is not None
    assert summary.published_candidate == summary.promoted_candidate
    assert status_path.exists()
    assert "current public champion" in status_path.read_text(encoding="utf-8")
    assert log_path.exists()
    assert "publish:" in log_path.read_text(encoding="utf-8")


def test_render_status_markdown_reports_publish_blocker(tmp_path: Path) -> None:
    states = []
    markdown = render_status_markdown(
        run_tag="20260424-lgbm-high_neighbor_oof-agent",
        dataset_path=tmp_path / "dataset.parquet",
        panel_path=tmp_path / "panel.parquet",
        dataset_signature="dataset_sig",
        panel_signature="panel_sig",
        baseline_variant="high_neighbor_oof",
        baseline_artifact_path=tmp_path / "models" / "lgbm_emos__high_neighbor_oof.pkl",
        pretrained_weather_model=None,
        states=states,
        champion_metadata={"variant": "high_neighbor_oof"},
        log_entries=[],
        recent_core_summary_path=None,
        publish_enabled=True,
    )

    assert "Publish remains blocked until a candidate-specific recent-core GO summary is supplied." in markdown
