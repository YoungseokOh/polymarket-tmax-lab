from __future__ import annotations

from pmtmax.config.settings import EnvSettings
from pmtmax.execution.live_broker import LiveBroker


class _DummyClient:
    def get_ok(self) -> dict[str, str]:
        return {"status": "ok"}

    def get_address(self) -> str:
        return "0xabc"


def test_live_broker_preflight_fails_closed_without_private_key(monkeypatch) -> None:
    broker = LiveBroker(EnvSettings())
    monkeypatch.setattr(broker, "_build_client", lambda require_level_2=False: _DummyClient())

    report = broker.preflight(require_posting=True)

    assert not report.ok
    assert any("Missing PMTMAX_POLY_PRIVATE_KEY." in message for message in report.messages)
