from __future__ import annotations

from datetime import UTC, date, datetime

from pmtmax.cli.main import (
    _fetch_live_observation,
    _is_target_day_observation,
    _observation_adjusted_probabilities,
    _observation_queue_state,
)
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.storage.schemas import MetarObservation


class _FakeIntradayHttp:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:
        return self.payload


class _FakeMetarClient:
    def __init__(self, observation: MetarObservation | None) -> None:
        self.observation = observation

    def fetch_latest(self, icao_code: str) -> MetarObservation | None:
        return self.observation


def test_observation_adjusted_probabilities_zeroes_impossible_lower_bins() -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec
    labels = spec.outcome_labels()
    forecast_probs = {label: 1.0 / len(labels) for label in labels}

    adjusted, impossible, removed_mass = _observation_adjusted_probabilities(
        spec,
        forecast_probs,
        observed_market_unit=11.1,
    )

    assert adjusted[labels[0]] == 0.0
    assert adjusted[labels[1]] == 0.0
    assert impossible
    assert round(sum(adjusted.values()), 8) == 1.0
    assert removed_mass > 0


def test_observation_queue_state_routes_reviewable_edges_to_manual_review() -> None:
    assert _observation_queue_state(
        reason="tradable",
        risk_flags=[],
    ) == "tradable"
    assert _observation_queue_state(
        reason="tradable",
        risk_flags=["research_public"],
    ) == "manual_review"
    assert _observation_queue_state(
        reason="missing_observation",
        risk_flags=["missing_observation"],
    ) == "blocked"


def test_fetch_live_observation_prefers_stronger_intraday_lower_bound_over_metar() -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec.model_copy(update={"target_local_date": date(2026, 4, 5)})

    selected = _fetch_live_observation(
        spec,
        http=_FakeIntradayHttp(
            '"TM","STN_ID","TMP_MNM_TM","TMP_MNM","TMP_MAX_TM","TMP_MAX"\n'
            '"20260405","113","0605","31","1412","121"\n'
        ),  # type: ignore[arg-type]
        metar=_FakeMetarClient(
            MetarObservation(
                station_id="RKSI",
                observed_at=datetime(2026, 4, 5, 5, 40, tzinfo=UTC),
                temp_c=9.8,
                raw_metar="METAR RKSI",
            )
        ),  # type: ignore[arg-type]
        observed_at=datetime(2026, 4, 5, 6, 0, tzinfo=UTC),
        allow_metar=True,
    )

    assert selected is not None
    assert selected.observation_source == "amo_air_calp_intraday"
    assert selected.source_family == "research_intraday"
    assert selected.lower_bound_temp_c == 12.1


def test_is_target_day_observation_uses_market_timezone() -> None:
    snapshot = bundled_market_snapshots(["Seoul"])[0]
    assert snapshot.spec is not None
    spec = snapshot.spec.model_copy(update={"target_local_date": date(2026, 4, 5)})

    assert _is_target_day_observation(spec, observed_at=datetime(2026, 4, 5, 0, 30, tzinfo=UTC)) is True
    assert _is_target_day_observation(spec, observed_at=datetime(2026, 4, 4, 14, 30, tzinfo=UTC)) is False
