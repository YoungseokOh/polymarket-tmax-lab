"""Quick fixed-split evaluation of saved models — no retraining.

Loads each saved pkl from artifacts/models/v2/, runs predict() on the held-out
last 20% of the dataset, and prints a comparison table.

Usage:
    uv run python scripts/quick_eval.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmtmax.modeling.quick_eval import evaluate_saved_model, quick_eval_holdout
from pmtmax.modeling.train import sanitize_model_frame

MODELS_DIR = Path("artifacts/models/v2")
DATASET = Path("data/parquet/gold/v2/historical_training_set.parquet")


def main() -> None:
    df = pd.read_parquet(DATASET)
    df = sanitize_model_frame(df)

    # Fixed temporal split — last 20% by target_date
    _, holdout = quick_eval_holdout(df)
    cutoff = holdout["target_date"].min() if "target_date" in holdout.columns and not holdout.empty else "n/a"

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
        r = evaluate_saved_model(path, holdout)
        if r:
            r["model"] = name
            results.append(r)
            print(
                f"MAE={r['mae']:.4f}  CRPS={r['crps']:.4f}  "
                f"Brier={r['brier']:.4f}  DirAcc={r.get('dir_acc', float('nan')):.3f}  "
                f"ECE={r.get('ece', float('nan')):.4f}  (n={int(r['n'])})"
            )
        else:
            print("FAILED")

    if not results:
        print("No results.")
        return

    cols = ["model", "crps", "brier", "dir_acc", "ece", "mae", "n"]
    cols = [c for c in cols if c in results[0]]
    res_df = pd.DataFrame(results)[cols]
    res_df = res_df.sort_values("crps").reset_index(drop=True)

    print("\n=== QUICK EVAL (fixed 80/20 split, no retraining) ===")
    print(res_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  --> BEST CRPS:    {res_df.iloc[0]['model']}")
    print(f"  --> BEST DirAcc:  {res_df.sort_values('dir_acc', ascending=False).iloc[0]['model']}")
    print(f"  --> LOWEST ECE:   {res_df.sort_values('ece').iloc[0]['model']}")
    print(f"  --> BEST MAE:     {res_df.sort_values('mae').iloc[0]['model']}")


if __name__ == "__main__":
    main()
