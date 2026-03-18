import numpy as np

from pmtmax.modeling.daily_max import daily_max_from_hourly, daily_max_from_samples


def test_daily_max_from_hourly() -> None:
    assert daily_max_from_hourly(np.array([1.0, 4.0, 2.5])) == 4.0


def test_daily_max_from_samples() -> None:
    hourly = np.array([[1.0, 4.0, 3.0], [2.0, 5.0, 1.0]])
    maxima = daily_max_from_samples(hourly)
    assert maxima.tolist() == [4.0, 5.0]

