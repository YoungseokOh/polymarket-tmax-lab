"""Stop-loss and trailing-stop evaluation — pure functions."""

from __future__ import annotations

from pmtmax.storage.schemas import PaperPosition


def should_stop_loss(entry_price: float, current_price: float, threshold: float = 0.20) -> bool:
    """Return ``True`` when the position has lost more than *threshold* of entry price."""

    if entry_price <= 0:
        return False
    loss_pct = (entry_price - current_price) / entry_price
    return loss_pct >= threshold


def update_high_water_mark(position: PaperPosition, current_price: float) -> PaperPosition:
    """Return a copy of *position* with its high-water mark updated."""

    new_hwm = max(position.high_water_mark, current_price)
    trailing_active = new_hwm > position.avg_price
    return position.model_copy(
        update={"high_water_mark": new_hwm, "trailing_stop_active": trailing_active},
    )


def should_trailing_stop(position: PaperPosition, current_price: float) -> bool:
    """Return ``True`` when the price has fallen *trailing_stop_rise_pct* from the high-water mark.

    The trailing stop only activates once the price has risen above the entry price at some
    point (i.e. ``position.trailing_stop_active`` is ``True``).
    """

    if not position.trailing_stop_active or position.high_water_mark <= 0:
        return False
    drawdown = (position.high_water_mark - current_price) / position.high_water_mark
    return drawdown >= 0.20


def evaluate_stops(
    position: PaperPosition,
    current_price: float,
    threshold: float = 0.20,
) -> tuple[PaperPosition, str | None]:
    """Run stop-loss and trailing-stop checks.

    Returns the (possibly updated) position and an exit reason string if a stop
    was triggered, or ``None`` otherwise.
    """

    if should_stop_loss(position.avg_price, current_price, threshold):
        return position, "stop_loss"

    position = update_high_water_mark(position, current_price)

    if should_trailing_stop(position, current_price):
        return position, "trailing_stop"

    return position, None
