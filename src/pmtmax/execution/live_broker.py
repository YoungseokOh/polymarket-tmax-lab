"""Live trading broker behind explicit feature flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, OrderArgs

from pmtmax.config.settings import EnvSettings
from pmtmax.storage.schemas import PreflightReport, TradeSignal


@dataclass
class LiveBroker:
    """Thin py-clob-client wrapper guarded behind explicit confirmations."""

    env: EnvSettings

    def preflight(self, *, require_posting: bool = False) -> PreflightReport:
        """Run a fail-closed live-trading preflight."""

        messages: list[str] = []
        ok = True
        geoblock_ok: bool | None = None
        balance_ok: bool | None = None
        allowance_ok: bool | None = None

        try:
            client = self._build_client()
            client.get_ok()
            address = client.get_address()
            messages.append("CLOB health check succeeded.")
        except Exception as exc:  # noqa: BLE001
            client = None
            address = None
            ok = False
            messages.append(f"CLOB health check failed: {exc}")

        api_key_present = bool(self.env.poly_api_key and self.env.poly_api_secret and self.env.poly_passphrase)
        if not self.env.poly_private_key:
            ok = False
            messages.append("Missing PMTMAX_POLY_PRIVATE_KEY.")

        if require_posting and not self._posting_enabled():
            ok = False
            messages.append(
                "Live posting is disabled. Set PMTMAX_LIVE_TRADING=true and "
                "PMTMAX_CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND."
            )

        if require_posting and not api_key_present:
            ok = False
            messages.append("Missing L2 API credentials for posting.")

        if client is not None and api_key_present:
            try:
                balance = client.get_balance_allowance(BalanceAllowanceParams())
                balance_ok = True
                allowance_ok = True
                messages.append(f"Balance/allowance check returned: {balance}")
            except Exception as exc:  # noqa: BLE001
                balance_ok = False
                allowance_ok = False
                text = str(exc)
                if "geo" in text.lower() or "restricted" in text.lower():
                    geoblock_ok = False
                messages.append(f"Balance/allowance check failed: {exc}")

        return PreflightReport(
            ok=ok,
            mode="live" if require_posting else "dry_run",
            address=address,
            api_key_present=api_key_present,
            geoblock_ok=geoblock_ok,
            balance_ok=balance_ok,
            allowance_ok=allowance_ok,
            messages=messages,
        )

    def preview_limit_order(self, signal: TradeSignal, *, size: float) -> dict[str, Any]:
        """Create a signed limit order preview without posting it."""

        if not self.env.poly_private_key:
            msg = "Missing private key for signed order preview."
            raise RuntimeError(msg)
        client = self._build_client()
        order = client.create_order(self._order_args(signal, size))
        return {
            "market_id": signal.market_id,
            "token_id": signal.token_id,
            "outcome_label": signal.outcome_label,
            "side": signal.side,
            "size": size,
            "price": signal.executable_price,
            "signed_order": getattr(order, "__dict__", str(order)),
        }

    def post_limit_order(self, signal: TradeSignal, *, size: float) -> dict[str, Any]:
        """Create and post a limit order when live trading is explicitly enabled."""

        self._assert_enabled()
        client = self._build_client(require_level_2=True)
        order = client.create_order(self._order_args(signal, size))
        return cast(dict[str, Any], client.post_order(order))

    def cancel_orders(self, order_ids: list[str]) -> Any:
        """Cancel live orders when L2 auth is configured."""

        self._assert_enabled()
        client = self._build_client(require_level_2=True)
        return client.cancel_orders(order_ids)

    def _assert_enabled(self) -> None:
        if not self._posting_enabled():
            msg = (
                "Live trading is disabled. Set PMTMAX_LIVE_TRADING=true and "
                "PMTMAX_CONFIRM_LIVE_TRADING=YES_I_UNDERSTAND to enable it."
            )
            raise RuntimeError(msg)
        required = [
            self.env.poly_private_key,
            self.env.poly_api_key,
            self.env.poly_api_secret,
            self.env.poly_passphrase,
        ]
        if not all(required):
            msg = "Missing required Polymarket live-trading credentials."
            raise RuntimeError(msg)

    def _build_client(self, *, require_level_2: bool = False) -> ClobClient:
        creds = None
        if self.env.poly_api_key and self.env.poly_api_secret and self.env.poly_passphrase:
            creds = ApiCreds(
                api_key=self.env.poly_api_key,
                api_secret=self.env.poly_api_secret,
                api_passphrase=self.env.poly_passphrase,
            )
        if require_level_2 and creds is None:
            msg = "Missing API credentials for L2 authenticated requests."
            raise RuntimeError(msg)

        client = ClobClient(
            self.env.clob_host,
            chain_id=self.env.poly_chain_id,
            key=self.env.poly_private_key or None,
            creds=creds,
            signature_type=self.env.poly_signature_type,
            funder=self.env.poly_funder_address or self.env.poly_proxy_address or None,
        )
        return client

    def _order_args(self, signal: TradeSignal, size: float) -> OrderArgs:
        return OrderArgs(
            token_id=signal.token_id,
            price=signal.executable_price,
            size=size,
            side=signal.side.upper(),
        )

    def _posting_enabled(self) -> bool:
        return self.env.live_trading and self.env.confirm_live_trading == "YES_I_UNDERSTAND"
