from __future__ import annotations

import pytest

from pmtmax.markets.clob_read_client import ClobReadClient


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None, bool]] = []

    def get_json(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> dict[str, object]:
        self.calls.append((url, params, use_cache))
        return {"history": []}


def test_get_prices_history_uses_interval_and_cache() -> None:
    http = _FakeHttp()
    client = ClobReadClient(http, "https://clob.polymarket.com")

    payload = client.get_prices_history("token-1", interval="max", fidelity=60, use_cache=True)

    assert payload == {"history": []}
    assert http.calls == [
        (
            "https://clob.polymarket.com/prices-history",
            {"market": "token-1", "interval": "max", "fidelity": 60},
            True,
        )
    ]


def test_get_prices_history_rejects_invalid_window_combinations() -> None:
    http = _FakeHttp()
    client = ClobReadClient(http, "https://clob.polymarket.com")

    with pytest.raises(ValueError, match="interval cannot be combined"):
        client.get_prices_history("token-1", interval="max", start_ts=1, fidelity=60)

    with pytest.raises(ValueError, match="Provide either interval or a start/end timestamp window"):
        client.get_prices_history("token-1", interval=None, fidelity=60)
