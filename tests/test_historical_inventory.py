from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from pmtmax.examples import EXAMPLE_MARKETS
from pmtmax.markets.inventory import (
    HistoricalCollectionStatusEntry,
    HistoricalCollectionStatusReport,
    HistoricalEventCandidateEntry,
    HistoricalEventCandidateReport,
    HistoricalEventFetch,
    HistoricalEventPage,
    HistoricalEventPageFetchEntry,
    HistoricalEventPageFetchReport,
    TemperatureEventRef,
    TruthProbeResult,
    build_active_weather_watchlist,
    build_fetches_from_report,
    build_historical_collection_status,
    build_historical_inventory_from_pages,
    discover_temperature_event_refs,
    discover_temperature_event_refs_from_gamma,
    fetch_historical_event_page_report,
    merge_historical_collection_status_reports,
    merge_historical_event_candidates,
    preserve_existing_capture_times,
    snapshot_from_temperature_event_page,
    validate_historical_inventory,
)
from pmtmax.markets.repository import bundled_market_snapshots


def _event_page_html(
    *,
    title: str,
    slug: str,
    event_id: str,
    active: bool,
    closed: bool,
    description: str | None = None,
) -> str:
    description = description or EXAMPLE_MARKETS["Seoul"]["description"]
    event = {
        "id": event_id,
        "title": title,
        "slug": slug,
        "negRiskMarketID": f"neg-{event_id}",
        "markets": [
            {
                "question": "Will the highest temperature in Seoul be 4°C or below on December 11?",
                "conditionId": f"{event_id}-4",
                "questionID": f"{event_id}-4",
                "groupItemTitle": "4°C or below",
                "description": description,
                "resolutionSource": "",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.03", "0.97"],
                "clobTokenIds": [f"{event_id}-4-yes", f"{event_id}-4-no"],
                "active": active,
                "closed": closed,
            },
            {
                "question": "Will the highest temperature in Seoul be 5°C on December 11?",
                "conditionId": f"{event_id}-5",
                "questionID": f"{event_id}-5",
                "groupItemTitle": "5°C",
                "description": description,
                "resolutionSource": "",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.08", "0.92"],
                "clobTokenIds": [f"{event_id}-5-yes", f"{event_id}-5-no"],
                "active": active,
                "closed": closed,
            },
            {
                "question": "Will the highest temperature in Seoul be 10°C or higher on December 11?",
                "conditionId": f"{event_id}-10",
                "questionID": f"{event_id}-10",
                "groupItemTitle": "10°C or higher",
                "description": description,
                "resolutionSource": "",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.15", "0.85"],
                "clobTokenIds": [f"{event_id}-10-yes", f"{event_id}-10-no"],
                "active": active,
                "closed": closed,
            },
        ],
    }
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "state": {
                                "data": event,
                            }
                        }
                    ]
                }
            }
        }
    }
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">'
        f"{json.dumps(payload)}"
        "</script></body></html>"
    )


def test_snapshot_from_temperature_event_page_builds_multi_outcome_snapshot() -> None:
    snapshot = snapshot_from_temperature_event_page(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        captured_at=datetime(2026, 3, 19, tzinfo=UTC),
    )

    assert snapshot.parse_error is None
    assert snapshot.spec is not None
    assert snapshot.spec.market_id == "evt-seoul"
    assert snapshot.spec.city == "Seoul"
    assert snapshot.outcome_prices == {
        "4°C or below": 0.03,
        "5°C": 0.08,
        "10°C or higher": 0.15,
    }
    assert snapshot.clob_token_ids == ["evt-seoul-4-yes", "evt-seoul-5-yes", "evt-seoul-10-yes"]
    assert snapshot.market["componentMarkets"][0]["groupItemTitle"] == "4°C or below"


def test_build_historical_inventory_from_pages_skips_open_events() -> None:
    closed_page = HistoricalEventPage(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    open_page = HistoricalEventPage(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-march-22-2026",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11-open",
            event_id="evt-seoul-open",
            active=True,
            closed=False,
        ),
        fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
    )

    snapshots, report = build_historical_inventory_from_pages(
        [closed_page, open_page],
        supported_cities=["Seoul", "NYC", "London", "Hong Kong", "Taipei"],
        source_manifest="test_urls.json",
        as_of_date=date(2026, 3, 19),
    )

    assert len(snapshots) == 1
    assert snapshots[0].spec is not None
    assert snapshots[0].spec.market_id == "evt-seoul"
    assert {issue.reason for issue in report.issues} == {"not_closed"}


def test_build_historical_inventory_from_pages_filters_truth_unready_snapshots() -> None:
    seoul_page = HistoricalEventPage(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    nyc_page = HistoricalEventPage(
        url="https://polymarket.com/event/highest-temperature-in-nyc-on-december-20",
        html=_event_page_html(
            title="Highest temperature in NYC on December 20?",
            slug="highest-temperature-in-nyc-on-december-20",
            event_id="evt-nyc",
            active=False,
            closed=True,
            description=EXAMPLE_MARKETS["NYC"]["description"],
        ),
        fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
    )

    snapshots, report = build_historical_inventory_from_pages(
        [seoul_page, nyc_page],
        supported_cities=["Seoul", "NYC"],
        source_manifest="test_urls.json",
        as_of_date=date(2026, 3, 19),
        truth_probe=lambda snapshot: (
            TruthProbeResult(status="truth_source_lag", detail="no public archive rows")
            if snapshot.spec is not None and snapshot.spec.city == "Seoul"
            else TruthProbeResult(status="ready", detail="")
        ),
    )

    assert len(snapshots) == 1
    assert snapshots[0].spec is not None
    assert snapshots[0].spec.city == "NYC"
    assert report.issue_counts["truth_source_lag"] == 1
    assert any(issue.reason == "truth_source_lag" and issue.city == "Seoul" for issue in report.issues)


def test_preserve_existing_capture_times_reuses_previous_snapshot_timestamp() -> None:
    fresh_snapshot = snapshot_from_temperature_event_page(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        captured_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    existing_snapshot = fresh_snapshot.model_copy(update={"captured_at": datetime(2026, 3, 1, tzinfo=UTC)})

    preserved = preserve_existing_capture_times([fresh_snapshot], existing_snapshots=[existing_snapshot])

    assert preserved[0].captured_at == datetime(2026, 3, 1, tzinfo=UTC)


def test_validate_historical_inventory_flags_examples_and_duplicates() -> None:
    valid_snapshot = snapshot_from_temperature_event_page(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        captured_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    duplicate_snapshot = valid_snapshot.model_copy(deep=True)
    example_snapshot = bundled_market_snapshots(["Seoul"])[0]

    report = validate_historical_inventory(
        [valid_snapshot, duplicate_snapshot, example_snapshot],
        supported_cities=["Seoul", "NYC", "London", "Hong Kong", "Taipei"],
        source_manifest="test_inventory.json",
    )

    assert report.duplicate_market_ids == ["evt-seoul"]
    assert {issue.reason for issue in report.issues} == {"duplicate_market_id", "example_market_id"}


def test_validate_historical_inventory_flags_truth_unready_snapshot() -> None:
    snapshot = snapshot_from_temperature_event_page(
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        html=_event_page_html(
            title="Highest temperature in Seoul on December 11?",
            slug="highest-temperature-in-seoul-on-december-11",
            event_id="evt-seoul",
            active=False,
            closed=True,
        ),
        captured_at=datetime(2026, 3, 19, tzinfo=UTC),
    )

    report = validate_historical_inventory(
        [snapshot],
        supported_cities=["Seoul"],
        source_manifest="test_inventory.json",
        truth_probe=lambda _: TruthProbeResult(status="truth_source_lag", detail="lagging source"),
    )

    assert report.issue_counts["truth_source_lag"] == 1
    assert report.issues[0].reason == "truth_source_lag"


def test_discover_temperature_event_refs_filters_supported_titles() -> None:
    refs = discover_temperature_event_refs(
        [
            {
                "id": "evt-london",
                "slug": "highest-temperature-in-london-on-march-17-2026",
                "title": "Highest temperature in London on March 17, 2026?",
                "active": False,
                "closed": True,
            },
            {
                "id": "evt-toronto",
                "slug": "highest-temperature-in-toronto-on-march-17-2026",
                "title": "Highest temperature in Toronto on March 17, 2026?",
                "active": True,
                "closed": False,
            },
            {
                "id": "evt-other",
                "slug": "where-will-2026-rank-among-the-hottest-years-on-record",
                "title": "Where will 2026 rank among the hottest years on record?",
                "active": True,
                "closed": False,
            },
        ],
        supported_cities=["London", "NYC"],
    )

    assert [ref.city for ref in refs] == ["London"]


class _FakeGammaClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_events(
        self,
        *,
        active: bool | None = None,  # noqa: ARG002
        closed: bool | None = None,  # noqa: ARG002
        tag_slug: str | None = None,
        limit: int = 100,  # noqa: ARG002
        offset: int = 0,
    ) -> list[dict[str, object]]:
        self.calls.append(f"{tag_slug}:{offset}")
        if offset > 0:
            return []
        if tag_slug == "weather":
            return [
                {"id": "evt-london", "slug": "highest-temperature-in-london-on-march-21-2026", "title": "Highest temperature in London on March 21?", "active": True, "closed": False},
                {"id": "evt-nyc", "slug": "highest-temperature-in-nyc-on-march-21-2026", "title": "Highest temperature in NYC on March 21?", "active": True, "closed": False},
            ]
        if tag_slug == "temperature":
            return [
                {"id": "evt-london", "slug": "highest-temperature-in-london-on-march-21-2026", "title": "Highest temperature in London on March 21?", "active": True, "closed": False},
                {"id": "evt-toronto", "slug": "highest-temperature-in-toronto-on-march-21-2026", "title": "Highest temperature in Toronto on March 21?", "active": True, "closed": False},
            ]
        return []


def test_discover_temperature_event_refs_from_gamma_dedupes_fallback_tags() -> None:
    refs = discover_temperature_event_refs_from_gamma(
        _FakeGammaClient(),  # type: ignore[arg-type]
        supported_cities=["London", "NYC", "Toronto"],
        active=True,
        closed=False,
        max_pages=2,
    )

    assert [ref.city for ref in refs] == ["London", "NYC", "Toronto"]
    assert refs[0].url == "https://polymarket.com/event/highest-temperature-in-london-on-march-21-2026"


def test_merge_historical_event_candidates_preserves_first_seen_and_adds_new_refs() -> None:
    existing_report = HistoricalEventCandidateReport(
        generated_at=datetime(2026, 3, 18, tzinfo=UTC),
        supported_cities=["Seoul", "London"],
        total_discovered=1,
        candidate_count=1,
        city_counts={"Seoul": 1},
        entries=[
            HistoricalEventCandidateEntry(
                event_id="evt-seoul",
                slug="highest-temperature-in-seoul-on-december-11",
                title="Highest temperature in Seoul on December 11?",
                url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
                city="Seoul",
                active=False,
                closed=True,
                discovered_at=datetime(2026, 3, 18, 1, tzinfo=UTC),
                first_seen_at=datetime(2026, 3, 18, 1, tzinfo=UTC),
                last_seen_at=datetime(2026, 3, 18, 1, tzinfo=UTC),
            )
        ],
    )

    merged = merge_historical_event_candidates(
        [
            TemperatureEventRef(
                event_id="evt-seoul-updated",
                slug="highest-temperature-in-seoul-on-december-11",
                title="Highest temperature in Seoul on December 11?",
                url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
                city="Seoul",
                closed=True,
            ),
            TemperatureEventRef(
                event_id="evt-london",
                slug="highest-temperature-in-london-on-december-11",
                title="Highest temperature in London on December 11?",
                url="https://polymarket.com/event/highest-temperature-in-london-on-december-11",
                city="London",
                closed=True,
            ),
        ],
        supported_cities=["Seoul", "London"],
        existing_report=existing_report,
        generated_at=datetime(2026, 3, 19, tzinfo=UTC),
    )

    seoul = next(entry for entry in merged.entries if entry.city == "Seoul")
    london = next(entry for entry in merged.entries if entry.city == "London")
    assert seoul.first_seen_at == datetime(2026, 3, 18, 1, tzinfo=UTC)
    assert seoul.last_seen_at == datetime(2026, 3, 19, tzinfo=UTC)
    assert seoul.event_id == "evt-seoul-updated"
    assert london.first_seen_at == datetime(2026, 3, 19, tzinfo=UTC)
    assert merged.candidate_count == 2


class _FakeFetchHttp:
    def __init__(self, responses: dict[str, str], *, failures: set[str] | None = None) -> None:
        self.responses = responses
        self.failures = failures or set()
        self.cache: dict[str, str] = {}
        self.calls: list[str] = []

    def get_text(self, url: str, params: dict[str, object] | None = None, use_cache: bool = True) -> str:
        self.calls.append(url)
        if url in self.failures:
            msg = f"timeout:{url}"
            raise RuntimeError(msg)
        text = self.responses[url]
        self.cache[url] = text
        return text

    def load_cached_text(self, url: str, *, params: dict[str, object] | None = None) -> str | None:
        return self.cache.get(url)


def test_fetch_historical_event_page_report_resumes_cached_entries_and_retries_failures() -> None:
    seoul = HistoricalEventCandidateEntry(
        event_id="evt-seoul",
        slug="highest-temperature-in-seoul-on-december-11",
        title="Highest temperature in Seoul on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        city="Seoul",
        active=False,
        closed=True,
        discovered_at=datetime(2026, 3, 18, tzinfo=UTC),
        first_seen_at=datetime(2026, 3, 18, tzinfo=UTC),
        last_seen_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    london = HistoricalEventCandidateEntry(
        event_id="evt-london",
        slug="highest-temperature-in-london-on-december-11",
        title="Highest temperature in London on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-london-on-december-11",
        city="London",
        active=False,
        closed=True,
        discovered_at=datetime(2026, 3, 18, tzinfo=UTC),
        first_seen_at=datetime(2026, 3, 18, tzinfo=UTC),
        last_seen_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    nyc = HistoricalEventCandidateEntry(
        event_id="evt-nyc",
        slug="highest-temperature-in-nyc-on-december-11",
        title="Highest temperature in NYC on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-nyc-on-december-11",
        city="NYC",
        active=False,
        closed=True,
        discovered_at=datetime(2026, 3, 19, tzinfo=UTC),
        first_seen_at=datetime(2026, 3, 19, tzinfo=UTC),
        last_seen_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    existing_report = HistoricalEventPageFetchReport(
        generated_at=datetime(2026, 3, 18, tzinfo=UTC),
        supported_cities=["Seoul", "London", "NYC"],
        total_candidates=2,
        processed_this_run=2,
        status_counts={"fetch_failed": 1, "fetched": 1},
        entries=[
            HistoricalEventPageFetchEntry(
                event_id=seoul.event_id,
                slug=seoul.slug,
                title=seoul.title,
                url=seoul.url,
                city=seoul.city,
                fetch_status="fetched",
                discovered_at=seoul.discovered_at,
                first_seen_at=seoul.first_seen_at,
                fetched_at=datetime(2026, 3, 18, tzinfo=UTC),
                last_attempted_at=datetime(2026, 3, 18, tzinfo=UTC),
                attempt_count=1,
                content_hash="cached",
            ),
            HistoricalEventPageFetchEntry(
                event_id=london.event_id,
                slug=london.slug,
                title=london.title,
                url=london.url,
                city=london.city,
                fetch_status="fetch_failed",
                detail="timeout",
                discovered_at=london.discovered_at,
                first_seen_at=london.first_seen_at,
                fetched_at=None,
                last_attempted_at=datetime(2026, 3, 18, tzinfo=UTC),
                attempt_count=1,
            ),
        ],
    )
    http = _FakeFetchHttp(
        {
            seoul.url: _event_page_html(
                title=seoul.title,
                slug=seoul.slug,
                event_id=seoul.event_id,
                active=False,
                closed=True,
            ),
            london.url: _event_page_html(
                title=london.title,
                slug=london.slug,
                event_id=london.event_id,
                active=False,
                closed=True,
                description=Path("tests/fixtures/markets/london_rules.txt").read_text(),
            ),
            nyc.url: _event_page_html(
                title=nyc.title,
                slug=nyc.slug,
                event_id=nyc.event_id,
                active=False,
                closed=True,
                description=EXAMPLE_MARKETS["NYC"]["description"],
            ),
        }
    )
    http.cache[seoul.url] = http.responses[seoul.url]

    report = fetch_historical_event_page_report(
        http,  # type: ignore[arg-type]
        [seoul, london, nyc],
        existing_report=existing_report,
        resume=True,
        max_workers=1,
    )

    assert http.calls == [london.url, nyc.url]
    assert report.processed_this_run == 2
    by_url = {entry.url: entry for entry in report.entries}
    assert by_url[seoul.url].attempt_count == 1
    assert by_url[london.url].attempt_count == 2
    assert by_url[nyc.url].fetch_status == "fetched"


def test_build_fetches_from_report_uses_cached_html_without_network() -> None:
    url = "https://polymarket.com/event/highest-temperature-in-seoul-on-december-11"
    http = _FakeFetchHttp(
        {
            url: _event_page_html(
                title="Highest temperature in Seoul on December 11?",
                slug="highest-temperature-in-seoul-on-december-11",
                event_id="evt-seoul",
                active=False,
                closed=True,
            )
        }
    )
    http.cache[url] = http.responses[url]
    report = HistoricalEventPageFetchReport(
        generated_at=datetime(2026, 3, 19, tzinfo=UTC),
        supported_cities=["Seoul"],
        total_candidates=1,
        processed_this_run=1,
        status_counts={"fetched": 1},
        entries=[
            HistoricalEventPageFetchEntry(
                event_id="evt-seoul",
                slug="highest-temperature-in-seoul-on-december-11",
                title="Highest temperature in Seoul on December 11?",
                url=url,
                city="Seoul",
                fetch_status="fetched",
                discovered_at=datetime(2026, 3, 19, tzinfo=UTC),
                first_seen_at=datetime(2026, 3, 19, tzinfo=UTC),
                fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
                last_attempted_at=datetime(2026, 3, 19, tzinfo=UTC),
                attempt_count=1,
            )
        ],
    )

    fetches = build_fetches_from_report(http, report, supported_cities=["Seoul"])  # type: ignore[arg-type]

    assert len(fetches) == 1
    assert fetches[0].fetch_error is None
    assert fetches[0].html == http.responses[url]


def test_build_historical_collection_status_classifies_truth_blocked_and_duplicates() -> None:
    collected_ref = TemperatureEventRef(
        event_id="evt-seoul",
        slug="highest-temperature-in-seoul-on-december-11",
        title="Highest temperature in Seoul on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        city="Seoul",
        closed=True,
    )
    blocked_ref = TemperatureEventRef(
        event_id="evt-seoul-blocked",
        slug="highest-temperature-in-seoul-on-december-12",
        title="Highest temperature in Seoul on December 12?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-12",
        city="Seoul",
        closed=True,
    )
    duplicate_ref = TemperatureEventRef(
        event_id="evt-seoul-duplicate",
        slug="highest-temperature-in-seoul-on-december-11-duplicate",
        title="Highest temperature in Seoul on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        city="Seoul",
        closed=True,
    )
    fetches = [
        HistoricalEventFetch(
            ref=collected_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=collected_ref.title,
                slug=collected_ref.slug,
                event_id="evt-seoul",
                active=False,
                closed=True,
            ),
        ),
        HistoricalEventFetch(
            ref=blocked_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=blocked_ref.title,
                slug=blocked_ref.slug,
                event_id="evt-seoul-blocked",
                active=False,
                closed=True,
            ),
        ),
        HistoricalEventFetch(
            ref=duplicate_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=duplicate_ref.title,
                slug=duplicate_ref.slug,
                event_id="evt-seoul-duplicate",
                active=False,
                closed=True,
            ),
        ),
    ]

    snapshots, report = build_historical_collection_status(
        fetches,
        supported_cities=["Seoul", "NYC", "London", "Hong Kong", "Taipei"],
        truth_probe=lambda snapshot: "truth_missing" if snapshot.spec and snapshot.spec.market_id == "evt-seoul-blocked" else None,
        source_manifest="historical_temperature_event_urls.json",
        as_of_date=date(2026, 3, 19),
    )

    assert len(snapshots) == 1
    assert snapshots[0].spec is not None
    assert snapshots[0].spec.market_id == "evt-seoul"
    assert report.status_counts == {"collected": 1, "duplicate": 1, "truth_blocked": 1}


def test_build_historical_collection_status_classifies_retryable_truth_states() -> None:
    lag_ref = TemperatureEventRef(
        event_id="evt-lag",
        slug="highest-temperature-in-seoul-on-december-13",
        title="Highest temperature in Seoul on December 13?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-13",
        city="Seoul",
        closed=True,
    )
    request_ref = TemperatureEventRef(
        event_id="evt-request",
        slug="highest-temperature-in-seoul-on-december-14",
        title="Highest temperature in Seoul on December 14?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-14",
        city="Seoul",
        closed=True,
    )
    collected_ref = TemperatureEventRef(
        event_id="evt-collected",
        slug="highest-temperature-in-seoul-on-december-15",
        title="Highest temperature in Seoul on December 15?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-15",
        city="Seoul",
        closed=True,
    )
    fetches = [
        HistoricalEventFetch(
            ref=lag_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=lag_ref.title,
                slug=lag_ref.slug,
                event_id=lag_ref.event_id,
                active=False,
                closed=True,
            ),
        ),
        HistoricalEventFetch(
            ref=request_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=request_ref.title,
                slug=request_ref.slug,
                event_id=request_ref.event_id,
                active=False,
                closed=True,
            ),
        ),
        HistoricalEventFetch(
            ref=collected_ref,
            fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
            html=_event_page_html(
                title=collected_ref.title,
                slug=collected_ref.slug,
                event_id=collected_ref.event_id,
                active=False,
                closed=True,
            ),
        ),
    ]

    def _truth_probe(snapshot):  # type: ignore[no-untyped-def]
        if snapshot.spec is None:
            return TruthProbeResult(status="truth_blocked", detail="missing_spec")
        if snapshot.spec.market_id == lag_ref.event_id:
            return TruthProbeResult(status="truth_source_lag", detail="No HKO record for 2026-03-17")
        if snapshot.spec.market_id == request_ref.event_id:
            return TruthProbeResult(status="truth_request_failed", detail="504 Gateway Timeout")
        return TruthProbeResult(status="ready")

    snapshots, report = build_historical_collection_status(
        fetches,
        supported_cities=["Seoul", "NYC", "London", "Hong Kong", "Taipei"],
        truth_probe=_truth_probe,
        source_manifest="historical_temperature_event_urls.json",
        as_of_date=date(2026, 3, 19),
        truth_workers=2,
    )

    assert len(snapshots) == 1
    assert report.status_counts == {"collected": 1, "truth_request_failed": 1, "truth_source_lag": 1}
    assert {entry.status for entry in report.entries if entry.status != "collected"} == {"truth_request_failed", "truth_source_lag"}


def test_merge_historical_collection_status_reports_overwrites_selected_urls_only() -> None:
    existing = HistoricalCollectionStatusReport(
        generated_at=datetime(2026, 3, 18, tzinfo=UTC),
        source_manifest="historical_temperature_event_urls.json",
        supported_cities=["Seoul"],
        total_discovered=2,
        processed_this_run=1,
        collected_urls=["https://polymarket.com/event/highest-temperature-in-seoul-on-december-11"],
        status_counts={"collected": 1, "truth_source_lag": 1},
        entries=[
            HistoricalCollectionStatusEntry(
                event_id="evt-collected",
                slug="highest-temperature-in-seoul-on-december-11",
                title="Highest temperature in Seoul on December 11?",
                url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
                city="Seoul",
                market_id="evt-collected",
                status="collected",
                status_reason="collected",
                terminal=True,
            ),
            HistoricalCollectionStatusEntry(
                event_id="evt-lag",
                slug="highest-temperature-in-seoul-on-december-12",
                title="Highest temperature in Seoul on December 12?",
                url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-12",
                city="Seoul",
                market_id="evt-lag",
                status="truth_source_lag",
                status_reason="No HKO record",
                terminal=False,
            ),
        ],
    )
    updated = HistoricalCollectionStatusReport(
        generated_at=datetime(2026, 3, 19, tzinfo=UTC),
        source_manifest="historical_temperature_event_urls.json",
        supported_cities=["Seoul"],
        total_discovered=1,
        processed_this_run=1,
        collected_urls=["https://polymarket.com/event/highest-temperature-in-seoul-on-december-12"],
        status_counts={"collected": 1},
        entries=[
            HistoricalCollectionStatusEntry(
                event_id="evt-lag",
                slug="highest-temperature-in-seoul-on-december-12",
                title="Highest temperature in Seoul on December 12?",
                url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-12",
                city="Seoul",
                market_id="evt-lag",
                status="collected",
                status_reason="collected",
                terminal=True,
            )
        ],
    )

    merged = merge_historical_collection_status_reports(
        existing_report=existing,
        updated_report=updated,
        total_discovered=2,
        supported_cities=["Seoul"],
        source_manifest="historical_temperature_event_urls.json",
    )

    assert merged.total_discovered == 2
    assert merged.status_counts == {"collected": 2}
    assert sorted(merged.collected_urls) == [
        "https://polymarket.com/event/highest-temperature-in-seoul-on-december-11",
        "https://polymarket.com/event/highest-temperature-in-seoul-on-december-12",
    ]


def test_build_active_weather_watchlist_reports_ready_and_parse_failures() -> None:
    london_description = Path("tests/fixtures/markets/london_rules.txt").read_text()
    ready_ref = TemperatureEventRef(
        event_id="evt-london",
        slug="highest-temperature-in-london-on-march-18-2026",
        title="Highest temperature in London on March 18, 2026?",
        url="https://polymarket.com/event/highest-temperature-in-london-on-march-18-2026",
        city="London",
        active=True,
    )
    broken_ref = TemperatureEventRef(
        event_id="evt-broken",
        slug="highest-temperature-in-seoul-on-december-11-broken",
        title="Highest temperature in Seoul on December 11?",
        url="https://polymarket.com/event/highest-temperature-in-seoul-on-december-11-broken",
        city="Seoul",
        active=True,
    )
    report = build_active_weather_watchlist(
        [
            HistoricalEventFetch(
                ref=ready_ref,
                fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
                html=_event_page_html(
                    title=ready_ref.title,
                    slug=ready_ref.slug,
                    event_id="evt-london",
                    active=True,
                    closed=False,
                    description=london_description,
                ),
            ),
            HistoricalEventFetch(
                ref=broken_ref,
                fetched_at=datetime(2026, 3, 19, tzinfo=UTC),
                html="<html><body>broken</body></html>",
            ),
        ],
        supported_cities=["Seoul", "NYC", "London", "Hong Kong", "Taipei"],
    )

    assert report.status_counts == {"parse_failed": 1, "ready": 1}
    ready_entry = next(entry for entry in report.entries if entry.status == "ready")
    assert ready_entry.city == "London"
    assert ready_entry.parse_ready is True
    assert ready_entry.target_local_date == date(2026, 3, 18)
