"""Cross-cutting domain schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from pmtmax.markets.market_spec import MarketSpec


class StationRef(BaseModel):
    station_id: str
    station_name: str
    lat: float | None = None
    lon: float | None = None


class RawArtifactRef(BaseModel):
    relative_path: str
    content_hash: str
    media_type: str


class IngestRun(BaseModel):
    run_id: str
    command: str
    status: Literal["running", "completed", "failed"]
    config_hash: str
    code_version: str
    started_at: datetime
    completed_at: datetime | None = None
    notes: str = ""


class ForecastRun(BaseModel):
    provider: str
    model_name: str
    issue_time_utc: datetime
    forecast_time_utc: datetime
    target_station: StationRef
    variables: dict[str, list[float] | float | str] = Field(default_factory=dict)


class ObservationRecord(BaseModel):
    source: str
    station_id: str
    local_date: date
    hourly_temp: list[float] = Field(default_factory=list)
    daily_max: float
    unit: Literal["C", "F"]
    finalized_at: datetime | None = None
    revision_status: Literal["preliminary", "final"] = "final"


class CalibrationMetadata(BaseModel):
    model_name: str
    calibration_method: str
    fitted_at: datetime
    notes: str = ""


class ProbForecast(BaseModel):
    target_market: str
    generated_at: datetime
    samples: list[float] = Field(default_factory=list)
    mean: float
    std: float
    daily_max_distribution: dict[str, Any] = Field(default_factory=dict)
    outcome_probabilities: dict[str, float] = Field(default_factory=dict)
    calibration_metadata: CalibrationMetadata | None = None


class TradeSignal(BaseModel):
    market_id: str
    token_id: str
    outcome_label: str
    side: Literal["buy", "sell"]
    fair_probability: float
    executable_price: float
    fee_estimate: float
    slippage_estimate: float
    edge: float
    confidence: float
    rationale: str
    mode: Literal["paper", "live"] = "paper"


class ExecutionFill(BaseModel):
    market_id: str
    token_id: str
    outcome_label: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    mode: Literal["paper", "live"]
    timestamp: datetime


class BacktestResult(BaseModel):
    metrics: dict[str, float]
    pnl: float
    num_trades: int
    by_city: dict[str, dict[str, float]] = Field(default_factory=dict)


class ModelArtifact(BaseModel):
    model_name: str
    version: str
    trained_at: datetime
    features: list[str]
    metrics: dict[str, float]
    path: str


class SequenceExample(BaseModel):
    market_id: str
    station_id: str
    city: str
    target_date: date
    decision_horizon: str
    model_name: str
    sequence_length: int
    time_index: list[str] = Field(default_factory=list)
    temperature_2m: list[float] = Field(default_factory=list)
    dew_point_2m: list[float] = Field(default_factory=list)
    relative_humidity_2m: list[float] = Field(default_factory=list)
    wind_speed_10m: list[float] = Field(default_factory=list)
    cloud_cover: list[float] = Field(default_factory=list)
    valid_mask: list[int] = Field(default_factory=list)
    realized_daily_max: float
    winning_outcome: str
    issue_time_utc: datetime
    forecast_source_kind: str


class MarketSnapshot(BaseModel):
    captured_at: datetime
    market: dict[str, Any]
    spec: MarketSpec | None = None
    parse_error: str | None = None
    outcome_prices: dict[str, float] = Field(default_factory=dict)
    clob_token_ids: list[str] = Field(default_factory=list)


class DecisionPoint(BaseModel):
    horizon: str
    decision_time_utc: datetime
    issue_time_utc: datetime
    lead_hours: float


class BookLevel(BaseModel):
    price: float
    size: float


class BookSnapshot(BaseModel):
    market_id: str
    token_id: str
    outcome_label: str
    source: Literal["clob", "fixture"] = "clob"
    timestamp: datetime | None = None
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)

    def best_bid(self) -> float:
        """Return the best bid or zero when the book is empty."""

        return self.bids[0].price if self.bids else 0.0

    def best_ask(self) -> float:
        """Return the best ask or one when the book is empty."""

        return self.asks[0].price if self.asks else 1.0


class PaperPosition(BaseModel):
    market_id: str
    token_id: str
    outcome_label: str
    side: Literal["buy", "sell"]
    avg_price: float
    size: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class PreflightReport(BaseModel):
    ok: bool
    mode: Literal["dry_run", "live"]
    address: str | None = None
    api_key_present: bool = False
    geoblock_ok: bool | None = None
    balance_ok: bool | None = None
    allowance_ok: bool | None = None
    messages: list[str] = Field(default_factory=list)


class WarehouseManifest(BaseModel):
    generated_at: datetime
    duckdb_path: str
    raw_root: str
    parquet_root: str
    manifest_root: str
    tables: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SeedManifest(BaseModel):
    generated_at: datetime
    archive_path: str
    data_root: str
    included_files: list[str] = Field(default_factory=list)
    file_count: int = 0


class LegacyRunEntry(BaseModel):
    category: Literal["raw", "parquet"]
    path: str
    kind: Literal["file", "dir"]
    reason: str
    file_count: int = 0
    size_bytes: int = 0


class LegacyRunInventory(BaseModel):
    generated_at: datetime
    manifest_path: str | None = None
    active_roots: list[str] = Field(default_factory=list)
    entries: list[LegacyRunEntry] = Field(default_factory=list)


class BootstrapManifest(BaseModel):
    generated_at: datetime
    seed_path: str | None = None
    seed_restored: bool = False
    archived_legacy_paths: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)
    warehouse_counts: dict[str, int] = Field(default_factory=dict)


class RemoteSyncManifest(BaseModel):
    generated_at: datetime
    backend: Literal["firebase"]
    bucket_name: str
    prefix: str
    uploaded_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    dry_run: bool = True
