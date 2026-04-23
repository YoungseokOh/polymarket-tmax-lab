"""Build forecast summaries with mispricing analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pmtmax.logging_utils import get_logger
from pmtmax.markets.book_utils import fetch_book
from pmtmax.markets.clob_read_client import ClobReadClient
from pmtmax.modeling.predict import predict_market
from pmtmax.storage.schemas import ForecastSummary, MarketSnapshot, OutcomeMispricing

LOGGER = get_logger(__name__)


def build_forecast_summaries(
    snapshots: list[MarketSnapshot],
    model_path: Path,
    model_name: str,
    clob: ClobReadClient,
    builder: object,
    *,
    horizon: str = "morning_of",
) -> list[ForecastSummary]:
    """Generate forecast summaries with mispricing detection for each market."""

    summaries: list[ForecastSummary] = []

    for snapshot in snapshots:
        spec = snapshot.spec
        if spec is None:
            continue
        if spec.target_local_date < datetime.now(tz=UTC).date():
            continue

        try:
            feature_frame = builder.build_live_row(spec, horizon=horizon)  # type: ignore[attr-defined]
            forecast = predict_market(model_path, model_name, spec, feature_frame)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Forecast failed for market %s (%s)", spec.market_id, spec.city)
            continue

        # Top-3 outcomes by model probability
        sorted_probs = sorted(
            forecast.outcome_probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_outcomes = [
            {"label": label, "prob": round(prob, 4)}
            for label, prob in sorted_probs[:3]
        ]

        # Mispricing analysis via CLOB
        mispricings: list[OutcomeMispricing] = []
        for outcome_label, model_prob in forecast.outcome_probabilities.items():
            token_idx = spec.outcome_labels().index(outcome_label) if outcome_label in spec.outcome_labels() else -1
            if token_idx < 0 or token_idx >= len(spec.token_ids):
                continue
            token_id = spec.token_ids[token_idx]
            book = fetch_book(
                clob,
                snapshot,
                token_id,
                outcome_label,
            )
            if book.source != "clob" or not book.asks:
                continue
            market_price = book.best_ask()
            edge = model_prob - market_price
            if abs(edge) > 0.02:
                mispricings.append(
                    OutcomeMispricing(
                        outcome_label=outcome_label,
                        model_prob=round(model_prob, 4),
                        market_price=round(market_price, 4),
                        edge=round(edge, 4),
                    )
                )

        mispricings.sort(key=lambda x: abs(x.edge), reverse=True)

        summaries.append(
            ForecastSummary(
                market_id=spec.market_id,
                city=spec.city,
                target_local_date=spec.target_local_date,
                question=spec.question,
                generated_at=datetime.now(tz=UTC),
                model_name=model_name,
                mean_f=round(forecast.mean, 2),
                std_f=round(forecast.std, 2),
                top_outcomes=top_outcomes,
                mispricings=mispricings,
            )
        )

    LOGGER.info("Built %d forecast summaries", len(summaries))
    return summaries
