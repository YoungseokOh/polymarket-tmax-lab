from pmtmax.execution.edge import compute_edge


def test_compute_edge_after_costs() -> None:
    edge = compute_edge(0.62, 0.55, fee_estimate=0.01, slippage_estimate=0.02)
    assert round(edge, 2) == 0.04

