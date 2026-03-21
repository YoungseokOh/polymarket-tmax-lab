from datetime import UTC, datetime

from pmtmax.execution.forecast_exit import should_forecast_exit
from pmtmax.storage.schemas import ProbForecast


def _make_forecast(outcome_probs: dict[str, float]) -> ProbForecast:
    return ProbForecast(
        target_market="m1",
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        mean=10.0,
        std=2.0,
        outcome_probabilities=outcome_probs,
    )


def test_forecast_exit_triggers_when_prob_below_buffer() -> None:
    forecast = _make_forecast({"8°C": 0.03, "9°C": 0.97})
    assert should_forecast_exit("8°C", forecast, buffer=0.05) is True


def test_forecast_exit_does_not_trigger_when_prob_above_buffer() -> None:
    forecast = _make_forecast({"8°C": 0.10, "9°C": 0.90})
    assert should_forecast_exit("8°C", forecast, buffer=0.05) is False


def test_forecast_exit_triggers_for_missing_label() -> None:
    forecast = _make_forecast({"9°C": 1.0})
    assert should_forecast_exit("8°C", forecast, buffer=0.05) is True


def test_forecast_exit_at_boundary() -> None:
    forecast = _make_forecast({"8°C": 0.05})
    assert should_forecast_exit("8°C", forecast, buffer=0.05) is False
