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


def test_paper_broker_uses_per_share_fee_for_edge_gate() -> None:
    broker = PaperBroker(bankroll=1_000.0)
    signal = TradeSignal(
        market_id="m2",
        token_id="t2",
        outcome_label="3°C or below",
        side="buy",
        fair_probability=0.95,
        executable_price=0.02,
        fee_estimate=0.0001,
        slippage_estimate=0.0208,
        edge=0.9091,
        confidence=0.95,
        rationale="high-confidence tail",
        mode="paper",
    )

    fill = broker.simulate_fill(signal, spread=0.04, liquidity=200.0, size=25_000.0)

    assert fill is not None
