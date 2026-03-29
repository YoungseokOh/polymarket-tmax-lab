from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from pmtmax.cli.main import (
    _hope_hunt_observation_from_opportunity,
    _load_scoped_snapshots,
    _supported_wu_open_phase_cities,
    hope_hunt_daemon,
    hope_hunt_report,
)
from pmtmax.config.settings import EnvSettings, RepoConfig
from pmtmax.markets.repository import bundled_market_snapshots
from pmtmax.markets.station_registry import lookup_station
from pmtmax.monitoring.hope_hunt import HopeHuntRunner, summarize_hope_hunt_history
from pmtmax.storage.schemas import HopeHuntObservation, OpportunityObservation


def _scoped_snapshot(
    *,
    city: str,
    days: int,
    official_source_name: str = "Wunderground",
    truth_track: str = "research_public",
    research_priority: str = "expansion",
    market_volume: float = 300.0,
):
    snapshot = bundled_market_snapshots(["Seoul"])[0].model_copy(deep=True)
    assert snapshot.spec is not None
    reference_spec = snapshot.spec
    station = lookup_station(city) or lookup_station("Seoul")
    assert station is not None
    local_today = datetime.now(tz=UTC).astimezone(ZoneInfo(station.timezone)).date()
    snapshot.spec = reference_spec.model_copy(
        update={
            "market_id": f"{city.lower().replace(' ', '-')}-{days}",
            "slug": f"highest-temperature-in-{city.lower().replace(' ', '-')}-{days}",
            "question": f"Highest temperature in {city} on sample date?",
            "city": city,
            "target_local_date": local_today + timedelta(days=days),
            "timezone": station.timezone,
            "official_source_name": official_source_name,
            "station_id": station.station_id,
            "station_name": station.station_name,
            "truth_track": truth_track,
            "research_priority": research_priority,
            "settlement_eligible": truth_track != "research_public",
            "public_truth_source_name": station.public_truth_source_name,
            "public_truth_station_id": station.public_truth_station_id,
        }
    )
    snapshot.market["componentMarkets"] = [{"volumeNum": market_volume}]
    return snapshot


def test_supported_wu_open_phase_cities_include_expected_examples() -> None:
    cities = _supported_wu_open_phase_cities()

    for city in ["Ankara", "Dallas", "Miami", "Toronto", "NYC", "Seoul", "London"]:
        assert city in cities
    for city in ["Istanbul", "Mexico City", "Hong Kong"]:
        assert city not in cities


def test_load_scoped_snapshots_filters_supported_wu_open_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    supported = _scoped_snapshot(city="Miami", days=3)
    unsupported = _scoped_snapshot(city="Istanbul", days=3)
    unsupported.spec = unsupported.spec.model_copy(update={"city": "Istanbul"})  # type: ignore[union-attr]
    non_wu = _scoped_snapshot(
        city="Hong Kong",
        days=2,
        official_source_name="Hong Kong Observatory Daily Extract",
        truth_track="exact_public",
    )

    monkeypatch.setattr(
        "pmtmax.cli.main._load_snapshots",
        lambda **kwargs: [supported, unsupported, non_wu],
    )

    snapshots = _load_scoped_snapshots(
        markets_path=None,
        cities=None,
        market_scope="supported_wu_open_phase",
        active=True,
        closed=False,
    )

    assert [snapshot.spec.city for snapshot in snapshots if snapshot.spec is not None] == ["Miami"]


def test_hope_hunt_observation_prioritizes_fresh_listings() -> None:
    observed_at = datetime(2026, 3, 29, 12, 0, tzinfo=UTC)
    fresh_snapshot = _scoped_snapshot(city="Miami", days=3, market_volume=320.0)
    mature_snapshot = _scoped_snapshot(city="Seoul", days=0, research_priority="core", market_volume=18_000.0)

    fresh_observation = OpportunityObservation(
        observed_at=observed_at,
        market_id="fresh",
        city="Miami",
        question="q1",
        target_local_date=fresh_snapshot.spec.target_local_date,  # type: ignore[union-attr]
        decision_horizon="market_open",
        reason="after_cost_positive_but_spread_too_wide",
        raw_gap=0.03,
        after_cost_edge=-0.004,
        spread=0.02,
    )
    mature_observation = OpportunityObservation(
        observed_at=observed_at,
        market_id="mature",
        city="Seoul",
        question="q2",
        target_local_date=mature_snapshot.spec.target_local_date,  # type: ignore[union-attr]
        decision_horizon="market_open",
        reason="raw_gap_non_positive",
        raw_gap=-0.02,
        after_cost_edge=-0.05,
        spread=0.05,
    )

    fresh = _hope_hunt_observation_from_opportunity(
        fresh_snapshot,
        fresh_observation,
        market_created_at=observed_at - timedelta(hours=4, minutes=5),
        market_deploying_at=observed_at - timedelta(hours=4, minutes=2),
        market_accepting_orders_at=observed_at - timedelta(hours=4),
        market_opened_at=observed_at - timedelta(hours=4),
        open_phase_age_hours=4.0,
        observed_at=observed_at,
    )
    mature = _hope_hunt_observation_from_opportunity(
        mature_snapshot,
        mature_observation,
        market_created_at=observed_at - timedelta(hours=80),
        market_deploying_at=observed_at - timedelta(hours=79),
        market_accepting_orders_at=observed_at - timedelta(hours=72),
        market_opened_at=observed_at - timedelta(hours=72),
        open_phase_age_hours=72.0,
        observed_at=observed_at,
    )

    assert fresh.priority_bucket == "fresh_listing_spread_blocked"
    assert fresh.candidate_alert is True
    assert mature.priority_bucket == "mature_core"
    assert mature.candidate_alert is False
    assert fresh.priority_score is not None
    assert mature.priority_score is not None
    assert fresh.priority_score > mature.priority_score


def test_summarize_hope_hunt_history_groups_age_buckets(tmp_path: Path) -> None:
    history_path = tmp_path / "hope_hunt_history.jsonl"
    rows = [
        HopeHuntObservation(
            observed_at=datetime(2026, 3, 29, 1, 0, tzinfo=UTC),
            market_id="m1",
            city="Miami",
            question="q1",
            target_local_date=datetime(2026, 4, 1, tzinfo=UTC).date(),
            decision_horizon="market_open",
            reason="tradable",
            open_phase_age_hours=5.0,
            open_phase_age_bucket="0-6h",
            target_day_distance=3,
            market_volume=400.0,
            priority_bucket="fresh_listing",
            priority_score=75.0,
            candidate_alert=True,
            raw_gap=0.04,
            after_cost_edge=0.01,
        ),
        HopeHuntObservation(
            observed_at=datetime(2026, 3, 29, 2, 0, tzinfo=UTC),
            market_id="m2",
            city="Dallas",
            question="q2",
            target_local_date=datetime(2026, 4, 1, tzinfo=UTC).date(),
            decision_horizon="market_open",
            reason="after_cost_positive_but_spread_too_wide",
            open_phase_age_hours=8.0,
            open_phase_age_bucket="6-24h",
            target_day_distance=3,
            market_volume=600.0,
            priority_bucket="fresh_listing_spread_blocked",
            priority_score=72.0,
            candidate_alert=True,
            raw_gap=0.03,
            after_cost_edge=0.005,
        ),
    ]
    with history_path.open("a") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    summary = summarize_hope_hunt_history(history_path)

    assert summary["candidate_count"] == 2
    assert summary["fresh_listing_count"] == 2
    assert summary["by_open_phase_age_bucket"]["0-6h"]["candidate_count"] == 1
    assert summary["by_open_phase_age_bucket"]["6-24h"]["candidate_count"] == 1
    assert summary["gate_decision"] == "GO"


def test_hope_hunt_runner_writes_latest_history_summary_and_state(tmp_path: Path) -> None:
    config = RepoConfig()

    def _snapshot_fetcher():
        return ["m1"]

    def _evaluator(snapshots, observed_at):
        assert snapshots == ["m1"]
        return [
            HopeHuntObservation(
                observed_at=observed_at,
                market_id="m1",
                city="Miami",
                question="Highest temperature in Miami on April 1?",
                target_local_date=datetime(2026, 4, 1, tzinfo=UTC).date(),
                decision_horizon="market_open",
                reason="after_cost_positive_but_spread_too_wide",
                open_phase_age_hours=6.0,
                open_phase_age_bucket="0-6h",
                target_day_distance=3,
                market_volume=320.0,
                priority_bucket="fresh_listing_spread_blocked",
                priority_score=71.0,
                candidate_alert=True,
                raw_gap=0.03,
                after_cost_edge=-0.002,
            )
        ]

    runner = HopeHuntRunner(
        config=config,
        interval_seconds=1,
        max_cycles=1,
        state_path=tmp_path / "state.json",
        latest_output_path=tmp_path / "latest.json",
        history_output_path=tmp_path / "history.jsonl",
        summary_output_path=tmp_path / "summary.json",
        snapshot_fetcher=_snapshot_fetcher,
        evaluator=_evaluator,
    )

    summary = runner.run_once()

    assert summary["markets_total"] == 1
    assert summary["candidate_count"] == 1
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["by_priority_bucket"]["fresh_listing_spread_blocked"]["candidate_count"] == 1


def test_hope_hunt_report_and_daemon_write_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pmtmax.cli.main.load_settings", lambda: (RepoConfig(), EnvSettings()))
    monkeypatch.setattr(
        "pmtmax.cli.main._resolve_model_path",
        lambda model_path, model_name: (Path("artifacts/models/v2/trading_champion.pkl"), "gaussian_emos"),
    )

    captured: dict[str, Path] = {}

    class _FakeHttp:
        def close(self) -> None:
            return None

    class _FakeRunner:
        def __init__(self, *, latest_output: Path, history_output: Path, summary_output: Path, state_path: Path) -> None:
            self.latest_output = latest_output
            self.history_output = history_output
            self.summary_output = summary_output
            self.state_path = state_path

        def run_once(self) -> dict[str, int]:
            self.latest_output.parent.mkdir(parents=True, exist_ok=True)
            self.history_output.parent.mkdir(parents=True, exist_ok=True)
            self.summary_output.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_output.write_text(json.dumps([{"city": "Miami", "target_local_date": "2026-04-01"}]))
            self.history_output.write_text("{}\n")
            self.summary_output.write_text(json.dumps({"candidate_count": 1}))
            self.state_path.write_text(json.dumps({"cycle": 1}))
            return {"markets_evaluated": 1, "candidate_count": 1}

        def run_loop(self) -> None:
            self.latest_output.parent.mkdir(parents=True, exist_ok=True)
            self.history_output.parent.mkdir(parents=True, exist_ok=True)
            self.summary_output.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_output.write_text(json.dumps([{"city": "Miami"}]))
            self.history_output.write_text("{}\n")
            self.summary_output.write_text(json.dumps({"candidate_count": 1}))
            self.state_path.write_text(json.dumps({"cycle": 1}))

    def _fake_build_runner(**kwargs):
        captured["history_output"] = kwargs["output"]
        captured["latest_output"] = kwargs["latest_output"]
        captured["summary_output"] = kwargs["summary_output"]
        captured["state_path"] = kwargs["state_path"]
        return (
            _FakeRunner(
                latest_output=kwargs["latest_output"],
                history_output=kwargs["output"],
                summary_output=kwargs["summary_output"],
                state_path=kwargs["state_path"],
            ),
            _FakeHttp(),
        )

    monkeypatch.setattr("pmtmax.cli.main._build_hope_hunt_runner", _fake_build_runner)

    hope_hunt_report(
        output=tmp_path / "hope_latest.json",
        summary_output=tmp_path / "hope_summary.json",
    )
    assert (tmp_path / "hope_latest.json").exists()
    assert (tmp_path / "hope_summary.json").exists()
    assert captured["history_output"].resolve() == (
        tmp_path / "artifacts" / "signals" / "v2" / "hope_hunt_history.jsonl"
    ).resolve()

    hope_hunt_daemon(
        max_cycles=1,
        output=tmp_path / "daemon_history.jsonl",
        latest_output=tmp_path / "daemon_latest.json",
        summary_output=tmp_path / "daemon_summary.json",
        state_path=tmp_path / "daemon_state.json",
    )
    assert (tmp_path / "daemon_history.jsonl").exists()
    assert (tmp_path / "daemon_latest.json").exists()
    assert (tmp_path / "daemon_summary.json").exists()
    assert (tmp_path / "daemon_state.json").exists()
