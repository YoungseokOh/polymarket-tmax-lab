"""Probability calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class OutcomeCalibrator:
    """Isotonic calibrator for one-vs-all outcome probabilities."""

    def __init__(self) -> None:
        self.models: dict[str, IsotonicRegression] = {}

    def fit(self, probabilities: dict[str, np.ndarray], winners: np.ndarray) -> None:
        """Fit per-outcome isotonic calibrators."""

        for label, probs in probabilities.items():
            target = (winners == label).astype(float)
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(probs, target)
            self.models[label] = model

    def calibrate(self, probabilities: dict[str, float]) -> dict[str, float]:
        """Calibrate a single probability vector."""

        calibrated = {}
        for label, value in probabilities.items():
            model = self.models.get(label)
            calibrated[label] = float(model.predict([value])[0]) if model else value
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {key: value / total for key, value in calibrated.items()}
        return calibrated

