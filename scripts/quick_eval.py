"""Quick fixed-split evaluation of saved models — no retraining.

Loads each saved pkl from artifacts/models/v2/, runs predict() on the held-out
last 20% of the dataset, and prints a comparison table.

Usage:
    uv run python scripts/quick_eval.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pmtmax.markets.market_spec import MarketSpec
from pmtmax.modeling.bin_mapper import map_normal_to_outcomes, map_samples_to_outcomes
from pmtmax.modeling.evaluation import brier_score, calibration_gap, crps_from_samples, mae
from pmtmax.modeling.sampling import sample_gaussian_mixture, sample_normal
from pmtmax.modeling.train import load_model, sanitize_model_frame

MODELS_DIR = Path("artifacts/models/v2")
DATASET = Path("data/parquet/gold/v2/historical_training_set.parquet")
HOLDOUT_FRAC = 0.20
NUM_SAMPLES = 500


def _predict_row(model, row: pd.Series, model_name: str) -> dict[str, float] | None:
    spec = MarketSpec.model_validate_json(str(row["market_spec_json"]))
    frame = row.to_frame().T.reset_index(drop=True)
    try:
        prediction = model.predict(frame)
    except Exception:
        return None

    if len(prediction) == 2:
        mean_arr, std_arr = prediction
        mean = float(np.asarray(mean_arr).reshape(-1)[0])
        std = float(max(float(np.asarray(std_arr).reshape(-1)[0]), 0.1))
        samples = sample_normal(mean, std, num_samples=NUM_SAMPLES)
        probs = map_normal_to_outcomes(spec, mean, std)
    else:
        w, m, s = prediction
        samples = sample_gaussian_mixture(
            np.asarray(w)[0], np.asarray(m)[0], np.asarray(s)[0], num_samples=NUM_SAMPLES
        )
        probs = map_samples_to_outcomes(spec, samples)

    return {"samples": samples, "probs": probs}


def eval_model(model_path: Path, holdout: pd.DataFrame) -> dict | None:
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"  SKIP {model_path.name}: {e}")
        return None

    maes, crps_scores, briers, winning_probs, winning_labels = [], [], [], [], []
    y_true_all = []

    for _, row in holdout.iterrows():
        result = _predict_row(model, row, model_path.stem)
        if result is None:
            continue
        y_true = float(row["realized_daily_max"])
        winner = str(row["winning_outcome"])
        probs = result["probs"]
        samples = result["samples"]

        maes.append(abs(y_true - float(np.mean(samples))))
        crps_scores.append(crps_from_samples(samples, y_true))
        briers.append(brier_score(probs, winner))
        winning_probs.append(probs.get(winner, 0.0))
        winning_labels.append(winner)
        y_true_all.append(y_true)

    if not maes:
        return None

    return {
        "n": len(maes),
        "mae": float(np.mean(maes)),
        "crps": float(np.mean(crps_scores)),
        "brier": float(np.mean(briers)),
    }


def main() -> None:
    df = pd.read_parquet(DATASET)
    df = sanitize_model_frame(df)

    # Fixed temporal split — last 20% by target_date
    if "target_date" in df.columns:
        dates = df["target_date"].sort_values().unique()
        cutoff = dates[int(len(dates) * 0.80)]
        holdout = df[df["target_date"] >= cutoff].reset_index(drop=True)
        train_end = df[df["target_date"] < cutoff]
    else:
        cutoff_idx = int(len(df) * 0.80)
        holdout = df.iloc[cutoff_idx:].reset_index(drop=True)
        train_end = df.iloc[:cutoff_idx]

    print(f"\nDataset: {len(df)} rows total | holdout: {len(holdout)} rows (last 20%)")
    print(f"Cutoff date: {cutoff}\n")

    models_to_eval = [
        # Baseline: current champion (in-sample scale, clip=2.0 floor applied)
        ("lgbm_emos__ultra_high_neighbor_fast", MODELS_DIR / "lgbm_emos__ultra_high_neighbor_fast.pkl"),
        # OOF scale variants: honest sigma via 4-fold expanding-window OOF residuals
        ("lgbm_emos__recency_neighbor_oof", MODELS_DIR / "lgbm_emos__recency_neighbor_oof.pkl"),       # existing: num_leaves=63
        ("lgbm_emos__high_neighbor_oof", MODELS_DIR / "lgbm_emos__high_neighbor_oof.pkl"),             # new: num_leaves=95
        ("lgbm_emos__ultra_high_neighbor_oof", MODELS_DIR / "lgbm_emos__ultra_high_neighbor_oof.pkl"), # new: num_leaves=127
        ("lgbm_emos__mega_neighbor_oof", MODELS_DIR / "lgbm_emos__mega_neighbor_oof.pkl"),             # new: num_leaves=150
    ]

    results = []
    for name, path in models_to_eval:
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        print(f"  Evaluating {name}...", end=" ", flush=True)
        r = eval_model(path, holdout)
        if r:
            r["model"] = name
            results.append(r)
            print(f"MAE={r['mae']:.4f}  CRPS={r['crps']:.4f}  Brier={r['brier']:.4f}  (n={r['n']})")
        else:
            print("FAILED")

    if not results:
        print("No results.")
        return

    res_df = pd.DataFrame(results)[["model", "mae", "crps", "brier", "n"]]
    res_df = res_df.sort_values("crps").reset_index(drop=True)

    print("\n=== QUICK EVAL (fixed 80/20 split, no retraining) ===")
    print(res_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  --> BEST CRPS: {res_df.iloc[0]['model']}")
    print(f"  --> BEST MAE:  {res_df.sort_values('mae').iloc[0]['model']}")


if __name__ == "__main__":
    main()
