"""Training orchestration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmtmax.modeling.advanced.aifs_nwp_blend import AifsNwpBlendModel
from pmtmax.modeling.advanced.tuned_ensemble import TunedEnsembleModel
from pmtmax.modeling.advanced.det2prob_nn import Det2ProbNNModel
from pmtmax.modeling.advanced.flexible_flow_nn import FlexibleProbNNModel
from pmtmax.modeling.advanced.pinn_postproc import PermutationInvariantNNModel
from pmtmax.modeling.advanced.spatial_gnn import SpatialGNNModel
from pmtmax.modeling.advanced.transformer_postproc import TransformerPostprocModel
from pmtmax.modeling.baselines.climatology import ClimatologyModel
from pmtmax.modeling.baselines.gaussian_emos import GaussianEMOSModel
from pmtmax.modeling.baselines.heteroscedastic_linear import HeteroscedasticLinearModel
from pmtmax.modeling.baselines.leadtime_continuous import LeadTimeContinuousModel
from pmtmax.modeling.baselines.raw_nwp import RawBestModelBaseline, RawMultiModelAverageBaseline
from pmtmax.modeling.baselines.ts_emos import TSEmosModel
from pmtmax.storage.schemas import ModelArtifact


def sanitize_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric feature values so legacy-merged datasets remain trainable."""

    clean = frame.copy()
    numeric_columns = clean.select_dtypes(include=["number"]).columns
    clean.loc[:, numeric_columns] = clean.loc[:, numeric_columns].fillna(0.0)
    return clean


def default_feature_names(frame: pd.DataFrame) -> list[str]:
    """Infer the default modeling features from a dataset."""

    excluded = {
        "market_id",
        "station_id",
        "target_date",
        "realized_daily_max",
        "winning_outcome",
        "source_priority",
        "settlement_eligible",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].nunique(dropna=False) > 1
        and not column.startswith("kma_ldps_")
        and not column.endswith("_num_hours")
    ]


def train_model(model_name: str, frame: pd.DataFrame, artifacts_dir: Path) -> ModelArtifact:
    """Train a named model and persist it to disk."""

    clean_frame = sanitize_model_frame(frame)
    features = default_feature_names(clean_frame)
    model: Any
    if model_name == "climatology":
        model = ClimatologyModel()
        model.fit(clean_frame)
    elif model_name == "raw_best_model":
        model = RawBestModelBaseline()
        model.fit(clean_frame)
    elif model_name == "raw_multimodel_average":
        model = RawMultiModelAverageBaseline([name for name in features if name.endswith("_daily_max")][:3] or ["model_daily_max"])
        model.fit(clean_frame)
    elif model_name == "gaussian_emos":
        model = GaussianEMOSModel(features)
        model.fit(clean_frame)
    elif model_name == "ts_emos":
        model = TSEmosModel(features)
        model.fit(clean_frame)
    elif model_name == "leadtime_continuous":
        model = LeadTimeContinuousModel([name for name in features if name != "lead_hours"])
        model.fit(clean_frame)
    elif model_name == "det2prob_nn":
        model = Det2ProbNNModel(features)
        model.fit(clean_frame)
    elif model_name == "aifs_nwp_blend":
        nwp = [name for name in features if "ifs" in name or "kma" in name]
        ai = [name for name in features if "aifs" in name]
        model = AifsNwpBlendModel(nwp_features=nwp, ai_features=ai)
        model.fit(clean_frame)
    elif model_name == "heteroscedastic_linear":
        model = HeteroscedasticLinearModel(features)
        model.fit(clean_frame)
    elif model_name == "flexible_flow_nn":
        model = FlexibleProbNNModel(features)
        model.fit(clean_frame)
    elif model_name == "spatial_gnn":
        gnn_features = [f for f in features if f not in ("neighbor_mean_temp", "neighbor_spread")]
        model = SpatialGNNModel(gnn_features)
        model.fit(clean_frame)
    elif model_name == "transformer_postproc":
        model = _train_transformer(clean_frame)
    elif model_name == "pinn_postproc":
        model = _train_pinn(clean_frame)
    elif model_name == "tuned_ensemble":
        model = TunedEnsembleModel(features)
        model.fit(clean_frame)
    else:
        msg = f"Unsupported trainable model: {model_name}"
        raise ValueError(msg)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{model_name}.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)

    return ModelArtifact(
        model_name=model_name,
        version="0.1.0",
        trained_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        features=features,
        metrics={},
        path=str(path),
    )


def _extract_sequences(frame: pd.DataFrame, column: str, seq_len: int) -> np.ndarray:
    """Extract fixed-length sequences from a JSON column, zero-padding if needed."""

    rows = []
    for raw in frame[column]:
        arr = json.loads(raw) if isinstance(raw, str) else list(raw)
        arr = [float(v) if v is not None else 0.0 for v in arr]
        if len(arr) < seq_len:
            arr = arr + [0.0] * (seq_len - len(arr))
        rows.append(arr[:seq_len])
    return np.array(rows, dtype=np.float32)


def _train_transformer(frame: pd.DataFrame) -> TransformerPostprocModel:
    """Train transformer postprocessor from tabular data with daily_max features as pseudo-sequence."""

    daily_max_cols = sorted(col for col in frame.columns if col.endswith("_model_daily_max"))
    if not daily_max_cols:
        daily_max_cols = ["model_daily_max"]
    seq_len = len(daily_max_cols)
    sequences = frame[daily_max_cols].to_numpy(dtype=np.float32)
    if sequences.ndim == 1:
        sequences = sequences.reshape(-1, 1)
    targets = frame["realized_daily_max"].to_numpy(dtype=np.float32)
    model = TransformerPostprocModel(sequence_length=seq_len)
    model.fit(sequences, targets)
    model.feature_names = daily_max_cols
    return model


def _train_pinn(frame: pd.DataFrame) -> PermutationInvariantNNModel:
    """Train permutation-invariant NN from NWP ensemble member daily_max columns."""

    daily_max_cols = sorted(col for col in frame.columns if col.endswith("_model_daily_max"))
    if not daily_max_cols:
        daily_max_cols = ["model_daily_max"]
    member_dim = len(daily_max_cols)
    ensemble = frame[daily_max_cols].to_numpy(dtype=np.float32)
    if ensemble.ndim == 1:
        ensemble = ensemble.reshape(-1, 1)
    # PINN expects (batch, members, member_dim) — use (batch, members, 1)
    ensemble_3d = ensemble.reshape(ensemble.shape[0], member_dim, 1)
    targets = frame["realized_daily_max"].to_numpy(dtype=np.float32)
    model = PermutationInvariantNNModel(member_dim=1)
    model.fit(ensemble_3d, targets)
    model.feature_names = daily_max_cols
    return model


def load_model(path: Path) -> Any:
    """Load a serialized model artifact."""

    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301
