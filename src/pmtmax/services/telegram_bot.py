"""Telegram notification service using httpx."""

from __future__ import annotations

import httpx

from pmtmax.logging_utils import get_logger
from pmtmax.storage.schemas import ForecastSummary

LOGGER = get_logger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    """Send forecast reports via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def _format_summary(self, summary: ForecastSummary) -> str:
        lines = [
            f"{summary.city} {summary.target_local_date}",
            f"Model: {summary.mean_f} ± {summary.std_f}°",
        ]

        if summary.top_outcomes:
            top_parts = []
            for item in summary.top_outcomes[:3]:
                label = item.get("label", "?")
                prob = item.get("prob", 0.0)
                top_parts.append(f"{label} ({prob:.1%})")
            lines.append(f"Top: {', '.join(top_parts)}")

        for mp in summary.mispricings[:3]:
            lines.append(
                f"Mispricing: {mp.outcome_label} "
                f"model={mp.model_prob:.0%} "
                f"market={mp.market_price:.2f}¢ "
                f"edge={mp.edge:+.0%}"
            )

        return "\n".join(lines)

    def send_message(self, text: str) -> dict:
        """Send a text message via Telegram Bot API."""

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        response = httpx.post(
            url,
            json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def send_forecast_report(self, summaries: list[ForecastSummary]) -> list[dict]:
        """Send one Telegram message per market forecast summary."""

        results = []
        for summary in summaries:
            text = self._format_summary(summary)
            try:
                result = self.send_message(text)
                results.append(result)
                LOGGER.info("Sent Telegram message for %s", summary.city)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Failed to send Telegram message for %s", summary.city)
        return results
