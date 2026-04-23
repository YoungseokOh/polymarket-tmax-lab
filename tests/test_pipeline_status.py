from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_pipeline_status_module():
    path = Path("scripts/pipeline_status.py")
    spec = importlib.util.spec_from_file_location("pipeline_status_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfill_progress_reads_checkpoint_and_single_run_logs(tmp_path: Path) -> None:
    module = _load_pipeline_status_module()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    inventory = tmp_path / "historical_temperature_snapshots.json"
    inventory.write_text(
        '[{"captured_at": "2026-01-01T00:00:00Z"}, {"captured_at": "2026-01-02T00:00:00Z"}]'
    )
    log_path = log_dir / "single_run_20260422T000000.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message": "single_run market done",
                        "cumulative_ok": 3,
                        "cumulative_err": 1,
                    }
                ),
                json.dumps(
                    {
                        "message": "backfill_forecasts checkpoint",
                        "processed": 1,
                        "total": 2,
                        "pct": 50.0,
                        "single_run_ok": 4,
                        "single_run_err": 1,
                    }
                ),
            ]
        )
    )

    progress = module.backfill_progress(log_dir, inventory_path=inventory)

    assert progress["processed"] == 1
    assert progress["total"] == 2
    assert progress["pct"] == 50.0
    assert progress["single_run_ok"] == 4
    assert progress["single_run_err"] == 1
