"""Training orchestration."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from pmtmax.modeling.advanced.aifs_nwp_blend import AifsNwpBlendModel
from pmtmax.modeling.advanced.det2prob_nn import Det2ProbNNModel
from pmtmax.modeling.baselines.climatology import ClimatologyModel
from pmtmax.modeling.baselines.gaussian_emos import GaussianEMOSModel
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

    excluded = {"market_id", "station_id", "target_date", "realized_daily_max", "winning_outcome"}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
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


def load_model(path: Path) -> Any:
    """Load a serialized model artifact."""

    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301
