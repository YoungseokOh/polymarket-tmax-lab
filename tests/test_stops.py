from datetime import UTC, datetime

from pmtmax.execution.stops import (
    evaluate_stops,
    should_stop_loss,
    should_trailing_stop,
    update_high_water_mark,
)
from pmtmax.storage.schemas import PaperPosition


def _make_position(**overrides) -> PaperPosition:
    defaults = {
        "market_id": "m1",
        "token_id": "t1",
        "outcome_label": "8°C",
        "side": "buy",
        "avg_price": 0.50,
        "size": 100.0,
        "high_water_mark": 0.50,
        "trailing_stop_active": False,
        "opened_at": datetime(2025, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return PaperPosition(**defaults)


def test_stop_loss_triggers_at_20pct_drop() -> None:
    # 0.50 → 0.39: 22% drop, clearly above threshold
    assert should_stop_loss(entry_price=0.50, current_price=0.39, threshold=0.20) is True


def test_stop_loss_does_not_trigger_at_19pct_drop() -> None:
    assert should_stop_loss(entry_price=0.50, current_price=0.405, threshold=0.20) is False


def test_trailing_stop_triggers_after_rise_and_fall() -> None:
    pos = _make_position(high_water_mark=0.625, trailing_stop_active=True)
    # Price falls 20% from HWM of 0.625 → 0.50
    assert should_trailing_stop(pos, current_price=0.50) is True


def test_trailing_stop_inactive_when_no_rise() -> None:
    pos = _make_position(high_water_mark=0.50, trailing_stop_active=False)
    assert should_trailing_stop(pos, current_price=0.40) is False


def test_update_high_water_mark_activates_trailing() -> None:
    pos = _make_position(high_water_mark=0.50, trailing_stop_active=False)
    updated = update_high_water_mark(pos, 0.60)
    assert updated.high_water_mark == 0.60
    assert updated.trailing_stop_active is True


def test_evaluate_stops_stop_loss() -> None:
    pos = _make_position(avg_price=0.50)
    updated, reason = evaluate_stops(pos, current_price=0.39, threshold=0.20)
    assert reason == "stop_loss"


def test_evaluate_stops_trailing_stop() -> None:
    pos = _make_position(avg_price=0.50, high_water_mark=0.625, trailing_stop_active=True)
    # Price at 0.50 is exactly 20% below HWM of 0.625
    updated, reason = evaluate_stops(pos, current_price=0.50, threshold=0.20)
    assert reason == "trailing_stop"


def test_evaluate_stops_no_trigger() -> None:
    pos = _make_position(avg_price=0.50)
    updated, reason = evaluate_stops(pos, current_price=0.45, threshold=0.20)
    assert reason is None
