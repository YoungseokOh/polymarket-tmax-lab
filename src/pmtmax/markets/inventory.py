"""Curated historical market inventory helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from threading import Semaphore
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from pmtmax.http import CachedHttpClient
from pmtmax.markets.gamma_client import GammaClient
from pmtmax.markets.market_filter import TEMP_MARKET_RE
from pmtmax.markets.normalization import extract_outcome_labels
from pmtmax.markets.repository import snapshot_from_market
from pmtmax.markets.station_registry import canonical_city
from pmtmax.storage.schemas import MarketSnapshot
from pmtmax.utils import stable_hash_bytes
from pmtmax.weather.truth_sources import make_truth_source

NEXT_DATA_TAG = '<script id="__NEXT_DATA__" type="application/json" crossorigin="anonymous">'
YES_OUTCOME_INDEX = 0
QUESTION_LABEL_RE = re.compile(r"be (?P<label>.+?) on ", re.IGNORECASE)
EVENT_TITLE_RE = re.compile(r"highest temperature in (?P<city>.+?) on (?P<date>.+?)\??$", re.IGNORECASE)
EVENT_URL_PREFIX = "https://polymarket.com/event/"
HistoricalCollectionStatus = Literal[
    "collected",
    "parse_failed",
    "truth_blocked",
    "truth_source_lag",
    "truth_request_failed",
    "unsupported_rule",
    "duplicate",
]
WatchlistStatus = Literal["ready", "parse_failed", "unsupported_rule", "duplicate"]
HistoricalFetchStatus = Literal["pending", "fetched", "fetch_failed"]
TruthProbeStatus = Literal["ready", "truth_blocked", "truth_source_lag", "truth_request_failed"]
TERMINAL_COLLECTION_STATUSES = frozenset({"collected", "parse_failed", "truth_blocked", "unsupported_rule", "duplicate"})
NON_TERMINAL_COLLECTION_STATUSES = frozenset({"truth_source_lag", "truth_request_failed"})


@dataclass(frozen=True)
class HistoricalEventPage:
    """Fetched Polymarket event page used to build a curated inventory."""

    url: str
    html: str
    fetched_at: datetime


class TemperatureEventRef(BaseModel):
    """Minimal grouped Polymarket event reference."""

    event_id: str
    slug: str
    title: str
    url: str
    city: str
    active: bool = False
    closed: bool = False


class TruthProbeResult(BaseModel):
    """Outcome of one exact-source truth readiness probe."""

    status: TruthProbeStatus
    detail: str = ""


class HistoricalEventCandidateEntry(BaseModel):
    """One supported grouped event discovered from Gamma."""

    event_id: str
    slug: str
    title: str
    url: str
    city: str
    active: bool = False
    closed: bool = False
    discovered_at: datetime
    first_seen_at: datetime
    last_seen_at: datetime


class HistoricalEventCandidateReport(BaseModel):
    """Persisted supported-city closed-event backlog."""

    generated_at: datetime
    supported_cities: list[str] = Field(default_factory=list)
    total_discovered: int = 0
    candidate_count: int = 0
    city_counts: dict[str, int] = Field(default_factory=dict)
    entries: list[HistoricalEventCandidateEntry] = Field(default_factory=list)


class HistoricalEventPageFetchEntry(BaseModel):
    """One event page fetch attempt for the historical refresh pipeline."""

    event_id: str
    slug: str
    title: str
    url: str
    city: str
    fetch_status: HistoricalFetchStatus
    detail: str = ""
    http_status: int | None = None
    content_hash: str | None = None
    discovered_at: datetime
    first_seen_at: datetime
    fetched_at: datetime | None = None
    last_attempted_at: datetime
    attempt_count: int = 0
    next_retry_after: datetime | None = None


class HistoricalEventPageFetchReport(BaseModel):
    """Persisted fetch state for Polymarket grouped event pages."""

    generated_at: datetime
    supported_cities: list[str] = Field(default_factory=list)
    total_candidates: int = 0
    processed_this_run: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    entries: list[HistoricalEventPageFetchEntry] = Field(default_factory=list)


@dataclass(frozen=True)
class HistoricalEventFetch:
    """Fetched event page or its fetch error."""

    ref: TemperatureEventRef
    fetched_at: datetime
    html: str | None = None
    fetch_error: str | None = None


class InventoryIssue(BaseModel):
    """Validation or build issue for a curated market entry."""

    url: str = ""
    reason: str
    detail: str = ""
    market_id: str | None = None
    city: str | None = None


class HistoricalInventoryReport(BaseModel):
    """Report emitted after building or validating a curated inventory."""

    generated_at: datetime
    source_manifest: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    total_inputs: int = 0
    snapshot_count: int = 0
    supported_cities: list[str] = Field(default_factory=list)
    city_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_market_ids: list[str] = Field(default_factory=list)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[InventoryIssue] = Field(default_factory=list)


class HistoricalCollectionStatusEntry(BaseModel):
    """One closed grouped event and its collection status."""

    event_id: str
    slug: str
    title: str
    url: str
    city: str | None = None
    target_local_date: date | None = None
    official_source_family: str | None = None
    official_source_name: str | None = None
    truth_track: str | None = None
    settlement_eligible: bool | None = None
    research_priority: str | None = None
    market_id: str | None = None
    status: HistoricalCollectionStatus
    status_reason: str = ""
    discovered_at: datetime | None = None
    last_attempted_at: datetime | None = None
    attempt_count: int = 0
    terminal: bool = True
    detail: str = ""


class HistoricalCollectionStatusReport(BaseModel):
    """Collection audit for discoverable closed temperature events."""

    generated_at: datetime
    source_manifest: str | None = None
    supported_cities: list[str] = Field(default_factory=list)
    total_discovered: int = 0
    processed_this_run: int = 0
    collected_urls: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    entries: list[HistoricalCollectionStatusEntry] = Field(default_factory=list)


class ActiveWeatherWatchEntry(BaseModel):
    """One active grouped event in the supported-city watchlist."""

    event_id: str
    slug: str
    title: str
    url: str
    city: str | None = None
    target_local_date: date | None = None
    official_source_family: str | None = None
    official_source_name: str | None = None
    truth_track: str | None = None
    settlement_eligible: bool | None = None
    research_priority: str | None = None
    market_id: str | None = None
    parse_ready: bool = False
    status: WatchlistStatus
    detail: str = ""


class ActiveWeatherWatchlistReport(BaseModel):
    """Artifact for active supported-city grouped temperature events."""

    generated_at: datetime
    supported_cities: list[str] = Field(default_factory=list)
    total_discovered: int = 0
    city_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    entries: list[ActiveWeatherWatchEntry] = Field(default_factory=list)


def _iter_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_nodes(value)


def extract_next_data_payload(html: str) -> dict[str, Any]:
    """Extract the Next.js dehydrated payload from an event page."""

    try:
        start = html.index(NEXT_DATA_TAG) + len(NEXT_DATA_TAG)
        end = html.index("</script>", start)
    except ValueError as exc:
        msg = "Could not locate __NEXT_DATA__ payload in event page"
        raise ValueError(msg) from exc
    return json.loads(html[start:end])


def find_temperature_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Find the event node that contains grouped temperature markets."""

    for candidate in _iter_nodes(payload):
        title = str(candidate.get("title", "")).strip()
        slug = str(candidate.get("slug", "")).strip()
        markets = candidate.get("markets")
        if not isinstance(markets, list) or not markets:
            continue
        if TEMP_MARKET_RE.search(title) or slug.startswith("highest-temperature-in-"):
            return candidate
    msg = "Could not locate grouped temperature event payload"
    raise ValueError(msg)


def _event_label(component_market: dict[str, Any]) -> str:
    label = str(component_market.get("groupItemTitle", "")).strip()
    if label:
        return label
    question = str(component_market.get("question", "")).strip()
    match = QUESTION_LABEL_RE.search(question)
    if match:
        return match.group("label").strip()
    msg = "Could not determine grouped outcome label"
    raise ValueError(msg)


def _coerce_prices(raw_prices: Any) -> list[str]:
    if isinstance(raw_prices, list):
        return [str(price) for price in raw_prices]
    if isinstance(raw_prices, str):
        try:
            decoded = json.loads(raw_prices)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [str(price) for price in decoded]
    return []


def aggregate_event_market_payload(event: dict[str, Any], *, source_url: str) -> dict[str, Any]:
    """Collapse a grouped event page into a Gamma-like multi-outcome market payload."""

    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        msg = "Temperature event payload is missing grouped component markets"
        raise ValueError(msg)

    tokens: list[dict[str, str]] = []
    outcome_prices: list[str] = []
    clob_token_ids: list[str] = []
    labels: list[str] = []
    for component_market in markets:
        if not isinstance(component_market, dict):
            msg = "Component market payload is malformed"
            raise ValueError(msg)
        label = _event_label(component_market)
        raw_prices = _coerce_prices(component_market.get("outcomePrices"))
        if len(raw_prices) <= YES_OUTCOME_INDEX:
            msg = f"Component market is missing Yes outcome prices for {label}"
            raise ValueError(msg)
        raw_token_ids = component_market.get("clobTokenIds") or []
        if not isinstance(raw_token_ids, list) or len(raw_token_ids) <= YES_OUTCOME_INDEX:
            msg = f"Component market is missing Yes token ids for {label}"
            raise ValueError(msg)
        yes_token_id = str(raw_token_ids[YES_OUTCOME_INDEX])
        labels.append(label)
        outcome_prices.append(raw_prices[YES_OUTCOME_INDEX])
        clob_token_ids.append(yes_token_id)
        tokens.append({"outcome": label, "token_id": yes_token_id})

    first_market = markets[0]
    aggregated_id = str(event.get("id") or event.get("slug") or first_market.get("questionID") or source_url)
    condition_id = str(
        event.get("negRiskMarketID")
        or event.get("questionID")
        or first_market.get("questionID")
        or first_market.get("conditionId")
        or aggregated_id
    )
    description = str(first_market.get("description") or event.get("description") or "").strip()
    if not description:
        msg = "Temperature event payload is missing a rule description"
        raise ValueError(msg)

    return {
        "id": aggregated_id,
        "slug": str(event.get("slug", "")).strip(),
        "question": str(event.get("title", "")).strip(),
        "conditionId": condition_id,
        "description": description,
        "resolutionSource": str(first_market.get("resolutionSource") or event.get("resolutionSource") or ""),
        "outcomes": labels,
        "outcomePrices": outcome_prices,
        "clobTokenIds": clob_token_ids,
        "tokens": tokens,
        "componentMarkets": markets,
        "sourceUrl": source_url,
    }


def snapshot_from_temperature_event_page(
    *,
    url: str,
    html: str,
    captured_at: datetime | None = None,
) -> MarketSnapshot:
    """Parse an event page and aggregate it into a multi-outcome market snapshot."""

    payload = extract_next_data_payload(html)
    event = find_temperature_event(payload)
    market = aggregate_event_market_payload(event, source_url=url)
    return snapshot_from_market(market, captured_at=captured_at)


def _snapshot_identity(snapshot: MarketSnapshot) -> dict[str, object]:
    spec = snapshot.spec
    labels = extract_outcome_labels(snapshot.market)
    return {
        "market_id": spec.market_id if spec is not None else None,
        "city": spec.city if spec is not None else None,
        "station_id": spec.station_id if spec is not None else None,
        "official_source_name": spec.official_source_name if spec is not None else None,
        "official_source_url": spec.official_source_url if spec is not None else None,
        "target_local_date": spec.target_local_date.isoformat() if spec is not None else None,
        "timezone": spec.timezone if spec is not None else None,
        "unit": spec.unit if spec is not None else None,
        "labels": labels,
    }


def preserve_existing_capture_times(
    snapshots: list[MarketSnapshot],
    *,
    existing_snapshots: list[MarketSnapshot],
) -> list[MarketSnapshot]:
    """Reuse stable capture timestamps for unchanged curated snapshots."""

    by_url: dict[str, datetime] = {}
    by_market_id: dict[str, datetime] = {}
    for snapshot in existing_snapshots:
        url = str(snapshot.market.get("sourceUrl") or "").strip()
        if url:
            by_url[url] = snapshot.captured_at
        market_id = snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id") or "").strip()
        if market_id:
            by_market_id[market_id] = snapshot.captured_at

    preserved: list[MarketSnapshot] = []
    for snapshot in snapshots:
        url = str(snapshot.market.get("sourceUrl") or "").strip()
        market_id = snapshot.spec.market_id if snapshot.spec is not None else str(snapshot.market.get("id") or "").strip()
        captured_at = by_url.get(url) or by_market_id.get(market_id) or snapshot.captured_at
        if captured_at == snapshot.captured_at:
            preserved.append(snapshot)
            continue
        preserved.append(snapshot.model_copy(update={"captured_at": captured_at}))
    return preserved


def event_url_from_slug(slug: str) -> str:
    """Return a canonical Polymarket event page URL for one slug."""

    return f"{EVENT_URL_PREFIX}{slug.strip()}"


def temperature_event_ref_from_event(
    event: dict[str, Any],
    *,
    supported_cities: list[str] | None = None,
) -> TemperatureEventRef | None:
    """Return a normalized grouped event ref when the title matches the target family."""

    title = str(event.get("title") or "").strip()
    slug = str(event.get("slug") or "").strip()
    match = EVENT_TITLE_RE.search(title)
    if not match or not slug:
        return None
    city = canonical_city(match.group("city"))
    if supported_cities and city.lower() not in {item.lower() for item in supported_cities}:
        return None
    return TemperatureEventRef(
        event_id=str(event.get("id") or slug),
        slug=slug,
        title=title,
        url=event_url_from_slug(slug),
        city=city,
        active=bool(event.get("active")),
        closed=bool(event.get("closed")),
    )


def discover_temperature_event_refs(
    events: list[dict[str, Any]],
    *,
    supported_cities: list[str] | None = None,
) -> list[TemperatureEventRef]:
    """Filter raw Gamma grouped events down to supported temperature event refs."""

    refs: list[TemperatureEventRef] = []
    for event in events:
        ref = temperature_event_ref_from_event(event, supported_cities=supported_cities)
        if ref is not None:
            refs.append(ref)
    return refs


def discover_temperature_event_refs_from_gamma(
    gamma: GammaClient,
    *,
    supported_cities: list[str] | None = None,
    active: bool | None = None,
    closed: bool | None = None,
    tag_slug: str = "weather",
    fallback_tag_slugs: list[str] | None = None,
    max_pages: int = 20,
    page_size: int = 100,
) -> list[TemperatureEventRef]:
    """Query Gamma grouped events and return supported temperature refs."""

    refs_by_url: dict[str, TemperatureEventRef] = {}
    tag_slugs = [tag_slug, *(fallback_tag_slugs or ["temperature"])]
    for current_tag in dict.fromkeys(tag_slugs):
        for page in range(max_pages):
            offset = page * page_size
            batch = gamma.fetch_events(
                active=active,
                closed=closed,
                tag_slug=current_tag,
                limit=page_size,
                offset=offset,
            )
            if not batch:
                break
            for ref in discover_temperature_event_refs(batch, supported_cities=supported_cities):
                refs_by_url.setdefault(ref.url, ref)
            if len(batch) < page_size:
                break
    return sorted(refs_by_url.values(), key=lambda ref: (ref.city, ref.url))


def fetch_temperature_event_pages(
    http: CachedHttpClient,
    refs: list[TemperatureEventRef],
    *,
    use_cache: bool = True,
) -> list[HistoricalEventFetch]:
    """Fetch grouped temperature event pages into reusable in-memory bundles."""

    fetched_at = datetime.now(tz=UTC)
    fetches: list[HistoricalEventFetch] = []
    for ref in refs:
        try:
            html = http.get_text(ref.url, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            fetches.append(HistoricalEventFetch(ref=ref, fetched_at=fetched_at, html=None, fetch_error=str(exc)))
            continue
        fetches.append(HistoricalEventFetch(ref=ref, fetched_at=fetched_at, html=html, fetch_error=None))
    return fetches


def snapshots_from_temperature_event_fetches(fetches: list[HistoricalEventFetch]) -> list[MarketSnapshot]:
    """Parse fetched grouped event pages into MarketSnapshot objects."""

    snapshots: list[MarketSnapshot] = []
    for fetch in fetches:
        if fetch.fetch_error:
            snapshots.append(
                MarketSnapshot(
                    captured_at=fetch.fetched_at,
                    market={
                        "id": fetch.ref.event_id,
                        "slug": fetch.ref.slug,
                        "question": fetch.ref.title,
                        "sourceUrl": fetch.ref.url,
                    },
                    spec=None,
                    parse_error=fetch.fetch_error,
                )
            )
            continue
        if fetch.html is None:
            snapshots.append(
                MarketSnapshot(
                    captured_at=fetch.fetched_at,
                    market={
                        "id": fetch.ref.event_id,
                        "slug": fetch.ref.slug,
                        "question": fetch.ref.title,
                        "sourceUrl": fetch.ref.url,
                    },
                    spec=None,
                    parse_error="missing_event_html",
                )
            )
            continue
        try:
            snapshot = snapshot_from_temperature_event_page(
                url=fetch.ref.url,
                html=fetch.html,
                captured_at=fetch.fetched_at,
            )
        except Exception as exc:  # noqa: BLE001
            snapshot = MarketSnapshot(
                captured_at=fetch.fetched_at,
                market={
                    "id": fetch.ref.event_id,
                    "slug": fetch.ref.slug,
                    "question": fetch.ref.title,
                    "sourceUrl": fetch.ref.url,
                },
                spec=None,
                parse_error=str(exc),
            )
        snapshots.append(snapshot)
    return snapshots


def probe_truth_readiness(snapshot: MarketSnapshot, http: CachedHttpClient) -> TruthProbeResult:
    """Return a structured truth-readiness result for one snapshot."""

    spec = snapshot.spec
    if spec is None:
        return TruthProbeResult(status="truth_blocked", detail="missing_spec")
    try:
        truth_source = make_truth_source(spec, http, snapshot_dir=None)
        truth_source.fetch_observation_bundle(spec, spec.target_local_date)
    except httpx.HTTPError as exc:
        return TruthProbeResult(status="truth_request_failed", detail=str(exc))
    except RuntimeError as exc:
        return TruthProbeResult(status="truth_request_failed", detail=str(exc))
    except ValueError as exc:
        status: TruthProbeStatus = "truth_source_lag" if _is_truth_source_lag_error(str(exc)) else "truth_blocked"
        return TruthProbeResult(status=status, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TruthProbeResult(status="truth_blocked", detail=str(exc))
    return TruthProbeResult(status="ready", detail="")


def _coerce_truth_result(value: TruthProbeResult | str | None) -> TruthProbeResult:
    if isinstance(value, TruthProbeResult):
        return value
    if value is None:
        return TruthProbeResult(status="ready", detail="")
    return TruthProbeResult(status="truth_blocked", detail=str(value))


def _filter_truth_ready_snapshots(
    snapshots: list[MarketSnapshot],
    *,
    truth_probe: Callable[[MarketSnapshot], TruthProbeResult | str | None] | None,
    truth_workers: int = 1,
    truth_per_source_limit: int | None = None,
) -> tuple[list[MarketSnapshot], list[InventoryIssue]]:
    """Return only truth-ready snapshots and issues for lagged or blocked entries."""

    if truth_probe is None:
        return snapshots, []

    issues: list[InventoryIssue] = []
    ready_snapshots: list[MarketSnapshot] = []
    candidates = [snapshot for snapshot in snapshots if snapshot.spec is not None]
    if not candidates:
        return ready_snapshots, issues

    if truth_workers <= 1:
        truth_results = [_coerce_truth_result(truth_probe(snapshot)) for snapshot in candidates]
    else:
        truth_results = [TruthProbeResult(status="truth_blocked", detail="unreachable")] * len(candidates)
        semaphores: dict[str, Semaphore] = {}
        if truth_per_source_limit and truth_per_source_limit > 0:
            families = {
                snapshot.spec.adapter_key() if snapshot.spec is not None else "unknown"
                for snapshot in candidates
            }
            semaphores = {family: Semaphore(truth_per_source_limit) for family in families}

        def _probe_with_limits(snapshot: MarketSnapshot) -> TruthProbeResult | str | None:
            if not semaphores or snapshot.spec is None:
                return truth_probe(snapshot)
            family = snapshot.spec.adapter_key()
            with semaphores[family]:
                return truth_probe(snapshot)

        with ThreadPoolExecutor(max_workers=truth_workers) as executor:
            future_to_index = {
                executor.submit(_probe_with_limits, snapshot): index
                for index, snapshot in enumerate(candidates)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    truth_results[index] = _coerce_truth_result(future.result())
                except Exception as exc:  # noqa: BLE001
                    truth_results[index] = TruthProbeResult(status="truth_blocked", detail=str(exc))

    for snapshot, truth_result in zip(candidates, truth_results, strict=True):
        spec = snapshot.spec
        if spec is None:
            continue
        if truth_result.status == "ready":
            ready_snapshots.append(snapshot)
            continue
        issues.append(
            InventoryIssue(
                url=str(snapshot.market.get("sourceUrl") or ""),
                reason=truth_result.status,
                detail=truth_result.detail,
                market_id=spec.market_id,
                city=spec.city,
            )
        )
    return ready_snapshots, issues


def _collection_entry(
    ref: TemperatureEventRef,
    *,
    status: HistoricalCollectionStatus,
    detail: str = "",
    snapshot: MarketSnapshot | None = None,
    discovered_at: datetime | None = None,
    last_attempted_at: datetime | None = None,
    attempt_count: int = 0,
) -> HistoricalCollectionStatusEntry:
    spec = snapshot.spec if snapshot is not None else None
    return HistoricalCollectionStatusEntry(
        event_id=ref.event_id,
        slug=ref.slug,
        title=ref.title,
        url=ref.url,
        city=spec.city if spec is not None else ref.city,
        target_local_date=spec.target_local_date if spec is not None else None,
        official_source_family=spec.adapter_key() if spec is not None else None,
        official_source_name=spec.official_source_name if spec is not None else None,
        truth_track=spec.truth_track if spec is not None else None,
        settlement_eligible=spec.settlement_eligible if spec is not None else None,
        research_priority=spec.research_priority if spec is not None else None,
        market_id=spec.market_id if spec is not None else None,
        status=status,
        status_reason=detail or status,
        discovered_at=discovered_at,
        last_attempted_at=last_attempted_at,
        attempt_count=attempt_count,
        terminal=status in TERMINAL_COLLECTION_STATUSES,
        detail=detail,
    )


def _watch_entry(
    ref: TemperatureEventRef,
    *,
    status: WatchlistStatus,
    detail: str = "",
    snapshot: MarketSnapshot | None = None,
) -> ActiveWeatherWatchEntry:
    spec = snapshot.spec if snapshot is not None else None
    return ActiveWeatherWatchEntry(
        event_id=ref.event_id,
        slug=ref.slug,
        title=ref.title,
        url=ref.url,
        city=spec.city if spec is not None else ref.city,
        target_local_date=spec.target_local_date if spec is not None else None,
        official_source_family=spec.adapter_key() if spec is not None else None,
        official_source_name=spec.official_source_name if spec is not None else None,
        truth_track=spec.truth_track if spec is not None else None,
        settlement_eligible=spec.settlement_eligible if spec is not None else None,
        research_priority=spec.research_priority if spec is not None else None,
        market_id=spec.market_id if spec is not None else None,
        parse_ready=status == "ready",
        status=status,
        detail=detail,
    )


def _status_counts(entries: list[HistoricalCollectionStatusEntry | ActiveWeatherWatchEntry]) -> dict[str, int]:
    return dict(sorted(Counter(entry.status for entry in entries).items()))


def _city_counts(entries: list[ActiveWeatherWatchEntry]) -> dict[str, int]:
    return dict(sorted(Counter(entry.city for entry in entries if entry.city).items()))


def _candidate_city_counts(entries: list[HistoricalEventCandidateEntry]) -> dict[str, int]:
    return dict(sorted(Counter(entry.city for entry in entries if entry.city).items()))


def _entry_sort_key(entry: TemperatureEventRef | HistoricalEventCandidateEntry | HistoricalEventPageFetchEntry) -> tuple[str, str]:
    city = entry.city or ""
    return (city, entry.url)


def _status_entry_sort_key(entry: HistoricalCollectionStatusEntry | ActiveWeatherWatchEntry) -> tuple[str, str]:
    city = entry.city or ""
    return (city, entry.url)


def _ref_from_candidate_entry(entry: HistoricalEventCandidateEntry) -> TemperatureEventRef:
    return TemperatureEventRef(
        event_id=entry.event_id,
        slug=entry.slug,
        title=entry.title,
        url=entry.url,
        city=entry.city,
        active=entry.active,
        closed=entry.closed,
    )


def _fetch_entry_from_candidate(
    entry: HistoricalEventCandidateEntry,
    *,
    fetch_status: HistoricalFetchStatus,
    detail: str = "",
    http_status: int | None = None,
    content_hash: str | None = None,
    fetched_at: datetime | None = None,
    last_attempted_at: datetime,
    attempt_count: int,
) -> HistoricalEventPageFetchEntry:
    return HistoricalEventPageFetchEntry(
        event_id=entry.event_id,
        slug=entry.slug,
        title=entry.title,
        url=entry.url,
        city=entry.city,
        fetch_status=fetch_status,
        detail=detail,
        http_status=http_status,
        content_hash=content_hash,
        discovered_at=entry.discovered_at,
        first_seen_at=entry.first_seen_at,
        fetched_at=fetched_at,
        last_attempted_at=last_attempted_at,
        attempt_count=attempt_count,
        next_retry_after=None,
    )


def _http_status_from_exception(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return int(exc.response.status_code)
    return None


def _is_truth_source_lag_error(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "no hko record",
        "no cwa codis daily row",
        "no wunderground historical observations found",
        "no wunderground historical temperature values found",
        "could not parse wunderground daily max",
        "no noaa global hourly rows",
        "could not parse cwa daily max",
        "missing airtemperature.maximum",
        "no record for ",
    )
    return any(marker in lowered for marker in markers)


def _is_unsupported_rule_error(message: str) -> bool:
    lowered = message.lower()
    return "unsupported official source" in lowered or "unsupported question format" in lowered


def merge_historical_event_candidates(
    discovered_refs: list[TemperatureEventRef],
    *,
    supported_cities: list[str],
    existing_report: HistoricalEventCandidateReport | None = None,
    generated_at: datetime | None = None,
) -> HistoricalEventCandidateReport:
    """Merge newly discovered grouped events into a persisted backlog manifest."""

    now = generated_at or datetime.now(tz=UTC)
    existing_by_url = {entry.url: entry for entry in (existing_report.entries if existing_report else [])}
    merged: dict[str, HistoricalEventCandidateEntry] = dict(existing_by_url)
    for ref in discovered_refs:
        existing = merged.get(ref.url)
        if existing is None:
            merged[ref.url] = HistoricalEventCandidateEntry(
                event_id=ref.event_id,
                slug=ref.slug,
                title=ref.title,
                url=ref.url,
                city=ref.city,
                active=ref.active,
                closed=ref.closed,
                discovered_at=now,
                first_seen_at=now,
                last_seen_at=now,
            )
            continue
        merged[ref.url] = existing.model_copy(
            update={
                "event_id": ref.event_id,
                "slug": ref.slug,
                "title": ref.title,
                "city": ref.city,
                "active": ref.active,
                "closed": ref.closed,
                "last_seen_at": now,
            }
        )

    entries = sorted(merged.values(), key=_entry_sort_key)
    return HistoricalEventCandidateReport(
        generated_at=now,
        supported_cities=supported_cities,
        total_discovered=len(discovered_refs),
        candidate_count=len(entries),
        city_counts=_candidate_city_counts(entries),
        entries=entries,
    )


def filter_historical_event_candidates(
    report: HistoricalEventCandidateReport,
    *,
    supported_cities: list[str] | None = None,
) -> list[HistoricalEventCandidateEntry]:
    """Return candidate entries optionally filtered to one city subset."""

    if not supported_cities:
        return list(report.entries)
    supported = {city.lower() for city in supported_cities}
    return [entry for entry in report.entries if entry.city.lower() in supported]


def fetch_historical_event_page_report(
    http: CachedHttpClient,
    candidates: list[HistoricalEventCandidateEntry],
    *,
    existing_report: HistoricalEventPageFetchReport | None = None,
    use_cache: bool = True,
    resume: bool = True,
    max_workers: int = 1,
    max_events: int | None = None,
    generated_at: datetime | None = None,
) -> HistoricalEventPageFetchReport:
    """Fetch grouped event pages into the shared cache and persist fetch state."""

    now = generated_at or datetime.now(tz=UTC)
    existing_by_url = {entry.url: entry for entry in (existing_report.entries if existing_report else [])}
    selected_candidates = sorted(candidates, key=_entry_sort_key)
    pending: list[HistoricalEventCandidateEntry] = []
    for candidate in selected_candidates:
        existing = existing_by_url.get(candidate.url)
        if resume and existing is not None and existing.fetch_status == "fetched" and http.load_cached_text(candidate.url) is not None:
            continue
        pending.append(candidate)
    if max_events is not None:
        pending = pending[:max_events]

    def _fetch_one(candidate: HistoricalEventCandidateEntry) -> HistoricalEventPageFetchEntry:
        existing = existing_by_url.get(candidate.url)
        attempt_count = (existing.attempt_count if existing is not None else 0) + 1
        attempted_at = datetime.now(tz=UTC)
        try:
            html = http.get_text(candidate.url, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            return _fetch_entry_from_candidate(
                candidate,
                fetch_status="fetch_failed",
                detail=str(exc),
                http_status=_http_status_from_exception(exc),
                fetched_at=existing.fetched_at if existing is not None else None,
                last_attempted_at=attempted_at,
                attempt_count=attempt_count,
            )
        return _fetch_entry_from_candidate(
            candidate,
            fetch_status="fetched",
            content_hash=stable_hash_bytes(html.encode("utf-8")),
            fetched_at=attempted_at,
            last_attempted_at=attempted_at,
            attempt_count=attempt_count,
        )

    updated_by_url: dict[str, HistoricalEventPageFetchEntry] = {}
    if max_workers <= 1:
        for candidate in pending:
            updated_by_url[candidate.url] = _fetch_one(candidate)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(_fetch_one, candidate): candidate.url for candidate in pending}
            for future in as_completed(future_to_url):
                updated_by_url[future_to_url[future]] = future.result()

    merged: dict[str, HistoricalEventPageFetchEntry] = dict(existing_by_url)
    for candidate in selected_candidates:
        existing = merged.get(candidate.url)
        if candidate.url in updated_by_url:
            merged[candidate.url] = updated_by_url[candidate.url]
            continue
        if existing is not None:
            merged[candidate.url] = existing.model_copy(
                update={
                    "event_id": candidate.event_id,
                    "slug": candidate.slug,
                    "title": candidate.title,
                    "city": candidate.city,
                    "discovered_at": candidate.discovered_at,
                    "first_seen_at": candidate.first_seen_at,
                }
            )
            continue
        merged[candidate.url] = _fetch_entry_from_candidate(
            candidate,
            fetch_status="pending",
            detail="pending_fetch",
            fetched_at=None,
            last_attempted_at=now,
            attempt_count=0,
        )

    entries = sorted(merged.values(), key=_entry_sort_key)
    return HistoricalEventPageFetchReport(
        generated_at=now,
        supported_cities=existing_report.supported_cities if existing_report is not None else sorted({entry.city for entry in candidates}),
        total_candidates=len(selected_candidates),
        processed_this_run=len(pending),
        status_counts=dict(sorted(Counter(entry.fetch_status for entry in entries).items())),
        entries=entries,
    )


def build_fetches_from_report(
    http: CachedHttpClient,
    fetch_report: HistoricalEventPageFetchReport,
    *,
    supported_cities: list[str] | None = None,
) -> list[HistoricalEventFetch]:
    """Rehydrate successful event fetches from the shared HTTP cache without network requests."""

    supported = {city.lower() for city in supported_cities} if supported_cities else None
    fetches: list[HistoricalEventFetch] = []
    for entry in fetch_report.entries:
        if entry.fetch_status != "fetched":
            continue
        if supported and entry.city.lower() not in supported:
            continue
        html = http.load_cached_text(entry.url)
        ref = TemperatureEventRef(
            event_id=entry.event_id,
            slug=entry.slug,
            title=entry.title,
            url=entry.url,
            city=entry.city,
            active=False,
            closed=True,
        )
        if html is None:
            fetches.append(
                HistoricalEventFetch(
                    ref=ref,
                    fetched_at=entry.last_attempted_at,
                    html=None,
                    fetch_error="missing_cached_event_html",
                )
            )
            continue
        fetches.append(
            HistoricalEventFetch(
                ref=ref,
                fetched_at=entry.fetched_at or entry.last_attempted_at,
                html=html,
                fetch_error=None,
            )
        )
    return fetches


def merge_historical_collection_status_reports(
    *,
    existing_report: HistoricalCollectionStatusReport | None,
    updated_report: HistoricalCollectionStatusReport,
    total_discovered: int,
    supported_cities: list[str],
    source_manifest: str | None,
) -> HistoricalCollectionStatusReport:
    """Merge one partial classification pass into the persisted collection report."""

    merged: dict[str, HistoricalCollectionStatusEntry] = {}
    if existing_report is not None:
        merged.update({entry.url: entry for entry in existing_report.entries})
    merged.update({entry.url: entry for entry in updated_report.entries})
    entries = sorted(merged.values(), key=_status_entry_sort_key)
    return HistoricalCollectionStatusReport(
        generated_at=updated_report.generated_at,
        source_manifest=source_manifest,
        supported_cities=supported_cities,
        total_discovered=total_discovered,
        processed_this_run=updated_report.processed_this_run,
        collected_urls=[entry.url for entry in entries if entry.status == "collected"],
        status_counts=_status_counts(entries),
        entries=entries,
    )


def build_historical_collection_status(
    fetches: list[HistoricalEventFetch],
    *,
    supported_cities: list[str],
    truth_probe: Callable[[MarketSnapshot], TruthProbeResult | str | None],
    source_manifest: str | None = None,
    as_of_date: date | None = None,
    existing_market_ids: set[str] | None = None,
    attempt_counts: dict[str, int] | None = None,
    discovered_at_by_url: dict[str, datetime] | None = None,
    attempted_at_by_url: dict[str, datetime] | None = None,
    truth_workers: int = 1,
    truth_per_source_limit: int | None = None,
) -> tuple[list[MarketSnapshot], HistoricalCollectionStatusReport]:
    """Classify discoverable closed events and return the truth-ready subset."""

    entries: list[HistoricalCollectionStatusEntry] = []
    provisional_snapshots: list[MarketSnapshot] = []
    seen_urls: set[str] = set()
    seen_market_ids: set[str] = set(existing_market_ids or set())
    historical_cutoff = as_of_date or date.today()
    supported = {city.lower() for city in supported_cities}
    pending_truth: list[tuple[TemperatureEventRef, MarketSnapshot, datetime]] = []

    def _coerce_truth_result(value: TruthProbeResult | str | None) -> TruthProbeResult:
        if isinstance(value, TruthProbeResult):
            return value
        if value is None:
            return TruthProbeResult(status="ready", detail="")
        return TruthProbeResult(status="truth_blocked", detail=str(value))

    for fetch in fetches:
        ref = fetch.ref
        discovered_at = discovered_at_by_url.get(ref.url) if discovered_at_by_url is not None else None
        attempt_count = attempt_counts.get(ref.url, 0) if attempt_counts is not None else 0
        attempted_at = attempted_at_by_url.get(ref.url) if attempted_at_by_url is not None else fetch.fetched_at
        if ref.url in seen_urls:
            entries.append(
                _collection_entry(
                    ref,
                    status="duplicate",
                    detail="duplicate_url",
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        seen_urls.add(ref.url)
        if fetch.fetch_error:
            entries.append(
                _collection_entry(
                    ref,
                    status="parse_failed",
                    detail=fetch.fetch_error,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        if fetch.html is None:
            entries.append(
                _collection_entry(
                    ref,
                    status="parse_failed",
                    detail="missing_event_html",
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        try:
            snapshot = snapshot_from_temperature_event_page(
                url=ref.url,
                html=fetch.html,
                captured_at=fetch.fetched_at,
            )
        except Exception as exc:  # noqa: BLE001
            status: HistoricalCollectionStatus = "unsupported_rule" if _is_unsupported_rule_error(str(exc)) else "parse_failed"
            entries.append(
                _collection_entry(
                    ref,
                    status=status,
                    detail=str(exc),
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        spec = snapshot.spec
        if snapshot.parse_error or spec is None:
            detail = snapshot.parse_error or "missing_spec"
            status = "unsupported_rule" if _is_unsupported_rule_error(detail) else "parse_failed"
            entries.append(
                _collection_entry(
                    ref,
                    status=status,
                    detail=detail,
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        if spec.city.lower() not in supported:
            entries.append(
                _collection_entry(
                    ref,
                    status="unsupported_rule",
                    detail="unsupported_city",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        if spec.target_local_date >= historical_cutoff:
            entries.append(
                _collection_entry(
                    ref,
                    status="parse_failed",
                    detail="not_historical",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        component_markets = snapshot.market.get("componentMarkets") or []
        if not isinstance(component_markets, list) or not component_markets:
            entries.append(
                _collection_entry(
                    ref,
                    status="parse_failed",
                    detail="missing_component_markets",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        if any(not bool(component.get("closed")) for component in component_markets):
            entries.append(
                _collection_entry(
                    ref,
                    status="parse_failed",
                    detail="not_closed",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        if spec.market_id in seen_market_ids:
            entries.append(
                _collection_entry(
                    ref,
                    status="duplicate",
                    detail="duplicate_market_id",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=attempted_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        pending_truth.append((ref, snapshot, attempted_at))

    if truth_workers <= 1:
        truth_results = [_coerce_truth_result(truth_probe(snapshot)) for _, snapshot, _ in pending_truth]
    else:
        truth_results = [TruthProbeResult(status="truth_blocked", detail="unreachable")] * len(pending_truth)
        semaphores: dict[str, Semaphore] = {}
        if truth_per_source_limit and truth_per_source_limit > 0:
            families = {
                snapshot.spec.adapter_key() if snapshot.spec is not None else "unknown"
                for _, snapshot, _ in pending_truth
            }
            semaphores = {family: Semaphore(truth_per_source_limit) for family in families}

        def _probe_with_limits(snapshot: MarketSnapshot) -> TruthProbeResult | str | None:
            if not semaphores or snapshot.spec is None:
                return truth_probe(snapshot)
            family = snapshot.spec.adapter_key()
            with semaphores[family]:
                return truth_probe(snapshot)

        with ThreadPoolExecutor(max_workers=truth_workers) as executor:
            future_to_index = {
                executor.submit(_probe_with_limits, snapshot): index
                for index, (_, snapshot, _) in enumerate(pending_truth)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    truth_results[index] = _coerce_truth_result(future.result())
                except Exception as exc:  # noqa: BLE001
                    truth_results[index] = TruthProbeResult(status="truth_blocked", detail=str(exc))

    for (ref, snapshot, fetched_at), truth_result in zip(pending_truth, truth_results, strict=True):
        discovered_at = discovered_at_by_url.get(ref.url) if discovered_at_by_url is not None else None
        attempt_count = attempt_counts.get(ref.url, 0) if attempt_counts is not None else 0
        if truth_result.status != "ready":
            entries.append(
                _collection_entry(
                    ref,
                    status=truth_result.status,
                    detail=truth_result.detail,
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=fetched_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        spec = snapshot.spec
        if spec is None:
            entries.append(
                _collection_entry(
                    ref,
                    status="truth_blocked",
                    detail="missing_spec",
                    snapshot=snapshot,
                    discovered_at=discovered_at,
                    last_attempted_at=fetched_at,
                    attempt_count=attempt_count,
                )
            )
            continue
        seen_market_ids.add(spec.market_id)
        provisional_snapshots.append(snapshot)
        entries.append(
            _collection_entry(
                ref,
                status="collected",
                snapshot=snapshot,
                discovered_at=discovered_at,
                last_attempted_at=fetched_at,
                attempt_count=attempt_count,
            )
        )

    validation = validate_historical_inventory(
        provisional_snapshots,
        supported_cities=supported_cities,
        source_manifest=source_manifest,
    )
    invalid_market_ids: set[str] = set()
    for issue in validation.issues:
        if not issue.market_id:
            continue
        invalid_market_ids.add(issue.market_id)
        for index, entry in enumerate(entries):
            if entry.market_id != issue.market_id or entry.status != "collected":
                continue
            downgraded_status: HistoricalCollectionStatus
            if issue.reason == "duplicate_market_id":
                downgraded_status = "duplicate"
            elif issue.reason == "unsupported_city":
                downgraded_status = "unsupported_rule"
            else:
                downgraded_status = "parse_failed"
            entries[index] = entry.model_copy(
                update={
                    "status": downgraded_status,
                    "detail": issue.detail or issue.reason,
                }
            )
    snapshots = [
        snapshot
        for snapshot in provisional_snapshots
        if snapshot.spec is not None and snapshot.spec.market_id not in invalid_market_ids
    ]
    report = HistoricalCollectionStatusReport(
        generated_at=datetime.now(tz=UTC),
        source_manifest=source_manifest,
        supported_cities=supported_cities,
        total_discovered=len(fetches),
        processed_this_run=len(fetches),
        collected_urls=[entry.url for entry in entries if entry.status == "collected"],
        status_counts=_status_counts(entries),
        entries=sorted(entries, key=_status_entry_sort_key),
    )
    return snapshots, report


def build_active_weather_watchlist(
    fetches: list[HistoricalEventFetch],
    *,
    supported_cities: list[str],
) -> ActiveWeatherWatchlistReport:
    """Build the supported-city active event watchlist artifact."""

    entries: list[ActiveWeatherWatchEntry] = []
    seen_urls: set[str] = set()
    supported = {city.lower() for city in supported_cities}
    for fetch in fetches:
        ref = fetch.ref
        if ref.url in seen_urls:
            entries.append(_watch_entry(ref, status="duplicate", detail="duplicate_url"))
            continue
        seen_urls.add(ref.url)
        if fetch.fetch_error:
            entries.append(_watch_entry(ref, status="parse_failed", detail=fetch.fetch_error))
            continue
        if fetch.html is None:
            entries.append(_watch_entry(ref, status="parse_failed", detail="missing_event_html"))
            continue
        try:
            snapshot = snapshot_from_temperature_event_page(
                url=ref.url,
                html=fetch.html,
                captured_at=fetch.fetched_at,
            )
        except Exception as exc:  # noqa: BLE001
            status: WatchlistStatus = "unsupported_rule" if _is_unsupported_rule_error(str(exc)) else "parse_failed"
            entries.append(_watch_entry(ref, status=status, detail=str(exc)))
            continue
        spec = snapshot.spec
        if snapshot.parse_error or spec is None:
            detail = snapshot.parse_error or "missing_spec"
            status = "unsupported_rule" if _is_unsupported_rule_error(detail) else "parse_failed"
            entries.append(_watch_entry(ref, status=status, detail=detail, snapshot=snapshot))
            continue
        if spec.city.lower() not in supported:
            entries.append(_watch_entry(ref, status="unsupported_rule", detail="unsupported_city", snapshot=snapshot))
            continue
        entries.append(_watch_entry(ref, status="ready", snapshot=snapshot))
    return ActiveWeatherWatchlistReport(
        generated_at=datetime.now(tz=UTC),
        supported_cities=supported_cities,
        total_discovered=len(fetches),
        city_counts=_city_counts(entries),
        status_counts=_status_counts(entries),
        entries=entries,
    )


def validate_historical_inventory(
    snapshots: list[MarketSnapshot],
    *,
    supported_cities: list[str],
    source_manifest: str | None = None,
    source_urls: list[str] | None = None,
    truth_probe: Callable[[MarketSnapshot], TruthProbeResult | str | None] | None = None,
    truth_workers: int = 1,
    truth_per_source_limit: int | None = None,
) -> HistoricalInventoryReport:
    """Validate a curated historical snapshot inventory."""

    issues: list[InventoryIssue] = []
    duplicate_market_ids: list[str] = []
    city_counts: dict[str, int] = {}
    seen_market_ids: set[str] = set()
    supported = {city.lower() for city in supported_cities}
    structurally_valid_snapshots: list[MarketSnapshot] = []

    for snapshot in snapshots:
        market_id = str(snapshot.market.get("id") or "")
        source_url = str(snapshot.market.get("sourceUrl") or "")
        if market_id in seen_market_ids and market_id:
            duplicate_market_ids.append(market_id)
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="duplicate_market_id",
                    market_id=market_id,
                )
            )
            continue
        seen_market_ids.add(market_id)
        if market_id.startswith("example-"):
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="example_market_id",
                    market_id=market_id,
                )
            )
        if snapshot.parse_error:
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="parse_error",
                    detail=snapshot.parse_error,
                    market_id=market_id,
                )
            )
            continue
        spec = snapshot.spec
        if spec is None:
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="missing_spec",
                    market_id=market_id,
                )
            )
            continue
        if spec.city.lower() not in supported:
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="unsupported_city",
                    city=spec.city,
                    market_id=market_id,
                )
            )
        city_counts[spec.city] = city_counts.get(spec.city, 0) + 1
        reparsed = snapshot_from_market(snapshot.market, captured_at=snapshot.captured_at)
        if reparsed.parse_error or reparsed.spec is None:
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="reparse_failed",
                    detail=reparsed.parse_error or "missing_spec",
                    market_id=market_id,
                    city=spec.city,
                )
            )
            continue
        if _snapshot_identity(reparsed) != _snapshot_identity(snapshot):
            issues.append(
                InventoryIssue(
                    url=source_url,
                    reason="spec_mismatch",
                    detail="Embedded spec does not match reparsed raw market payload",
                    market_id=market_id,
                    city=spec.city,
                )
            )
            continue
        structurally_valid_snapshots.append(snapshot)

    _, truth_issues = _filter_truth_ready_snapshots(
        structurally_valid_snapshots,
        truth_probe=truth_probe,
        truth_workers=truth_workers,
        truth_per_source_limit=truth_per_source_limit,
    )
    issues.extend(truth_issues)

    issue_counts = dict(sorted(Counter(issue.reason for issue in issues).items()))

    return HistoricalInventoryReport(
        generated_at=datetime.now(tz=UTC),
        source_manifest=source_manifest,
        source_urls=source_urls or [],
        total_inputs=len(snapshots),
        snapshot_count=len(snapshots),
        supported_cities=supported_cities,
        city_counts=city_counts,
        duplicate_market_ids=sorted(set(duplicate_market_ids)),
        issue_counts=issue_counts,
        issues=issues,
    )


def build_historical_inventory_from_pages(
    pages: list[HistoricalEventPage],
    *,
    supported_cities: list[str],
    source_manifest: str | None = None,
    as_of_date: date | None = None,
    truth_probe: Callable[[MarketSnapshot], TruthProbeResult | str | None] | None = None,
    truth_workers: int = 1,
    truth_per_source_limit: int | None = None,
) -> tuple[list[MarketSnapshot], HistoricalInventoryReport]:
    """Build a curated historical inventory from fetched event pages."""

    snapshots: list[MarketSnapshot] = []
    issues: list[InventoryIssue] = []
    historical_cutoff = as_of_date or date.today()
    for page in pages:
        try:
            snapshot = snapshot_from_temperature_event_page(
                url=page.url,
                html=page.html,
                captured_at=page.fetched_at,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                InventoryIssue(
                    url=page.url,
                    reason="page_parse_failed",
                    detail=str(exc),
                )
            )
            continue
        spec = snapshot.spec
        if spec is None:
            issues.append(
                InventoryIssue(
                    url=page.url,
                    reason="missing_spec",
                    market_id=str(snapshot.market.get("id") or ""),
                )
            )
            continue
        if spec.target_local_date >= historical_cutoff:
            issues.append(
                InventoryIssue(
                    url=page.url,
                    reason="not_historical",
                    market_id=spec.market_id,
                    city=spec.city,
                )
            )
            continue
        component_markets = snapshot.market.get("componentMarkets") or []
        if not isinstance(component_markets, list) or not component_markets:
            issues.append(
                InventoryIssue(
                    url=page.url,
                    reason="missing_component_markets",
                    market_id=spec.market_id,
                    city=spec.city,
                )
            )
            continue
        if any(not bool(component.get("closed")) for component in component_markets):
            issues.append(
                InventoryIssue(
                    url=page.url,
                    reason="not_closed",
                    market_id=spec.market_id,
                    city=spec.city,
                )
            )
            continue
        snapshots.append(snapshot)

    snapshots, truth_issues = _filter_truth_ready_snapshots(
        snapshots,
        truth_probe=truth_probe,
        truth_workers=truth_workers,
        truth_per_source_limit=truth_per_source_limit,
    )
    issues.extend(truth_issues)

    report = validate_historical_inventory(
        snapshots,
        supported_cities=supported_cities,
        source_manifest=source_manifest,
        source_urls=[page.url for page in pages],
    )
    report.total_inputs = len(pages)
    report.issues.extend(issues)
    report.issue_counts = dict(sorted(Counter(issue.reason for issue in report.issues).items()))
    return snapshots, report
