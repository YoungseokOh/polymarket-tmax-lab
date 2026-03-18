from pmtmax.execution.paper_broker import PaperBroker
from pmtmax.storage.schemas import TradeSignal


def test_paper_broker_simulate_fill() -> None:
    broker = PaperBroker(bankroll=100.0)
    signal = TradeSignal(
        market_id="m1",
        token_id="t1",
        outcome_label="8°C",
        side="buy",
        fair_probability=0.7,
        executable_price=0.4,
        fee_estimate=0.0,
        slippage_estimate=0.0,
        edge=0.3,
        confidence=0.8,
        rationale="test",
        mode="paper",
    )
    fill = broker.simulate_fill(signal, spread=0.02, liquidity=1000.0, size=10.0)
    assert fill is not None
    assert len(broker.inventory) == 1
