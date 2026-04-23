"""Application settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pmtmax.markets.station_registry import supported_cities as catalog_supported_cities
from pmtmax.utils import load_yaml_with_extends


class AppConfig(BaseModel):
    env: Literal["research", "paper", "live"] = "research"
    random_seed: int = 42
    supported_cities: list[str] = Field(default_factory=catalog_supported_cities)
    workspace_name: str = "default"
    dataset_profile: Literal["real_market", "weather_real"] = "real_market"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    raw_dir: Path = Path("data/raw")
    parquet_dir: Path = Path("data/parquet")
    run_dir: Path = Path("data/runs")
    archive_dir: Path = Path("data/archive")
    manifest_dir: Path = Path("data/manifests")
    duckdb_path: Path = Path("data/duckdb/warehouse.duckdb")
    artifacts_dir: Path = Path("artifacts")
    public_model_dir: Path = Path("artifacts/public_models")
    llm_rule_parser: bool = False


class PolymarketConfig(BaseModel):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    ws_market_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    refresh_interval_seconds: int = 300
    max_pages: int = 20


class WeatherConfig(BaseModel):
    openmeteo_base_url: str = "https://api.open-meteo.com"
    archive_base_url: str = "https://historical-forecast-api.open-meteo.com"
    timeout_seconds: int = 30
    retries: int = 3
    lagged_run_offsets_hours: list[int] = Field(default_factory=lambda: [0, 6, 12, 24])
    models: list[str] = Field(default_factory=list)


class MetarConfig(BaseModel):
    base_url: str = "https://aviationweather.gov/api/data"
    enabled: bool = True
    stale_minutes: int = 30


class BacktestConfig(BaseModel):
    decision_horizons: list[str] = Field(default_factory=list)
    default_edge_threshold: float = 0.03
    min_liquidity: float = 250.0
    slippage_bps: int = 50


class ExecutionConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    live_trading: bool = False
    confirm_live_trading: str = ""
    default_fee_bps: float = Field(30.0, validation_alias=AliasChoices("default_fee_bps", "fee_bps"))
    max_city_exposure: float = 500.0
    global_max_exposure: float = 2_000.0
    min_liquidity: float = 250.0
    max_spread_bps: int = 700
    stale_forecast_minutes: int = 180
    stop_loss_pct: float = 0.20
    trailing_stop_rise_pct: float = 0.20
    forecast_exit_buffer: float = 0.05


class ModelsConfig(BaseModel):
    benchmark_ladder: list[str] = Field(default_factory=list)


class ScannerConfig(BaseModel):
    interval_seconds: int = 60
    max_cycles: int = 0
    state_path: Path = Path("artifacts/scanner_state.json")
    snapshot_refresh_interval: int = 10


class OpportunityShadowConfig(BaseModel):
    interval_seconds: int = 60
    max_cycles: int = 0
    near_term_days: int = 1
    state_path: Path = Path("artifacts/opportunity_shadow_state.json")
    latest_output_path: Path = Path("artifacts/opportunity_shadow_latest.json")
    history_output_path: Path = Path("artifacts/opportunity_shadow.jsonl")
    summary_output_path: Path = Path("artifacts/opportunity_shadow_summary.json")


class HopeHuntConfig(BaseModel):
    interval_seconds: int = 300
    max_cycles: int = 0
    state_path: Path = Path("artifacts/signals/v2/hope_hunt_state.json")
    latest_output_path: Path = Path("artifacts/signals/v2/hope_hunt_latest.json")
    history_output_path: Path = Path("artifacts/signals/v2/hope_hunt_history.jsonl")
    summary_output_path: Path = Path("artifacts/signals/v2/hope_hunt_summary.json")


class ObservationStationConfig(BaseModel):
    interval_seconds: int = 300
    max_cycles: int = 0
    observation_hours: int = 24
    observation_stale_minutes: int = 120
    candidate_max_age_minutes: int = 20
    approval_ttl_minutes: int = 20
    exact_public_size_multiplier: float = 1.0
    research_public_size_multiplier: float = 0.5
    state_path: Path = Path("artifacts/signals/v2/observation_shadow_state.json")
    latest_output_path: Path = Path("artifacts/signals/v2/observation_shadow_latest.json")
    history_output_path: Path = Path("artifacts/signals/v2/observation_shadow.jsonl")
    summary_output_path: Path = Path("artifacts/signals/v2/observation_shadow_summary.json")
    alerts_output_path: Path = Path("artifacts/signals/v2/observation_alerts_latest.json")
    queue_output_path: Path = Path("artifacts/signals/v2/live_pilot_queue.json")


class StationDashboardConfig(BaseModel):
    interval_seconds: int = 60
    max_cycles: int = 0
    opportunity_report_path: Path = Path("artifacts/signals/v2/opportunity_report.json")
    observation_latest_path: Path = Path("artifacts/signals/v2/observation_shadow_latest.json")
    observation_summary_path: Path = Path("artifacts/signals/v2/observation_shadow_summary.json")
    queue_output_path: Path = Path("artifacts/signals/v2/live_pilot_queue.json")
    open_phase_latest_path: Path = Path("artifacts/signals/v2/open_phase_shadow_latest.json")
    open_phase_summary_path: Path = Path("artifacts/signals/v2/open_phase_shadow_summary.json")
    revenue_gate_summary_path: Path = Path("artifacts/signals/v2/revenue_gate_summary.json")
    watchlist_playbook_path: Path = Path("artifacts/signals/v2/execution_watchlist_playbook.json")
    json_output_path: Path = Path("artifacts/signals/v2/station_dashboard.json")
    html_output_path: Path = Path("artifacts/signals/v2/station_dashboard.html")
    state_path: Path = Path("artifacts/signals/v2/station_dashboard_state.json")


class StationOrchestratorConfig(BaseModel):
    interval_seconds: int = 300
    max_cycles: int = 0
    state_path: Path = Path("artifacts/signals/v2/station_cycle_state.json")


class FirebaseConfig(BaseModel):
    enabled: bool = False
    bucket_name: str = ""
    prefix: str = "pmtmax"
    credentials_json: str = ""


class MonitoringConfig(BaseModel):
    l2_interval_seconds: int = 900
    l2_settlement_window_hours: float = 48.0
    l2_output_dir: Path = Path("data/l2_timeseries")


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class MarketMakingConfig(BaseModel):
    enabled: bool = False
    base_half_spread: float = 0.02
    skew_factor: float = 0.5
    base_size: float = 10.0
    requote_threshold: float = 0.01
    max_position_per_outcome: float = 100.0
    max_total_exposure: float = 1000.0
    max_loss: float = 500.0


class RepoConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    metar: MetarConfig = Field(default_factory=MetarConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    opportunity_shadow: OpportunityShadowConfig = Field(default_factory=OpportunityShadowConfig)
    hope_hunt: HopeHuntConfig = Field(default_factory=HopeHuntConfig)
    observation_station: ObservationStationConfig = Field(default_factory=ObservationStationConfig)
    station_dashboard: StationDashboardConfig = Field(default_factory=StationDashboardConfig)
    station_orchestrator: StationOrchestratorConfig = Field(default_factory=StationOrchestratorConfig)
    firebase: FirebaseConfig = Field(default_factory=FirebaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    market_making: MarketMakingConfig = Field(default_factory=MarketMakingConfig)


class EnvSettings(BaseSettings):
    """Environment overrides for file-based repo configuration."""

    model_config = SettingsConfigDict(env_prefix="PMTMAX_", env_file=".env", extra="ignore")

    env: str = "research"
    config: Path = Path("configs/research.yaml")
    workspace_name: str = "default"
    dataset_profile: Literal["real_market", "weather_real"] = "real_market"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    raw_dir: Path = Path("data/raw")
    run_dir: Path = Path("data/runs")
    archive_dir: Path = Path("data/archive")
    manifest_dir: Path = Path("data/manifests")
    duckdb_path: Path = Path("data/duckdb/warehouse.duckdb")
    parquet_dir: Path = Path("data/parquet")
    artifacts_dir: Path = Path("artifacts")
    public_model_dir: Path = Path("artifacts/public_models")
    log_level: str = "INFO"
    random_seed: int = 42
    live_trading: bool = False
    confirm_live_trading: str = ""
    poly_host: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    ws_host: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    poly_chain_id: int = 137
    poly_signature_type: int | None = None
    poly_funder_address: str = ""
    http_timeout: int = 30
    http_retries: int = 3
    llm_rule_parser: bool = False
    wu_api_key: str = ""
    poly_private_key: str = ""
    poly_proxy_address: str = ""
    poly_api_key: str = ""
    poly_api_secret: str = ""
    poly_passphrase: str = ""
    firebase_enabled: bool = False
    firebase_bucket_name: str = ""
    firebase_prefix: str = "pmtmax"
    firebase_credentials_json: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


def _rebase_rooted_path(path: Path, *, source_root: str, target_root: Path) -> Path:
    """Move one repo-relative rooted path under a workspace-specific root."""

    if path.is_absolute():
        return path
    parts = path.parts
    if not parts or parts[0] != source_root:
        return path
    relative = Path(*parts[1:]) if len(parts) > 1 else Path()
    return target_root / relative


def load_settings(config_path: Path | None = None) -> tuple[RepoConfig, EnvSettings]:
    """Load layered repo config plus environment settings."""

    env = EnvSettings()
    final_config_path = config_path or env.config
    payload = load_yaml_with_extends(final_config_path)
    config = RepoConfig.model_validate(payload)

    config.app.workspace_name = env.workspace_name
    config.app.dataset_profile = env.dataset_profile
    config.app.data_dir = env.data_dir
    config.app.cache_dir = env.cache_dir
    config.app.raw_dir = env.raw_dir
    config.app.run_dir = env.run_dir
    config.app.archive_dir = env.archive_dir
    config.app.manifest_dir = env.manifest_dir
    config.app.duckdb_path = env.duckdb_path
    config.app.parquet_dir = env.parquet_dir
    config.app.artifacts_dir = env.artifacts_dir
    config.app.public_model_dir = env.public_model_dir
    config.app.llm_rule_parser = env.llm_rule_parser or config.app.llm_rule_parser
    config.app.random_seed = env.random_seed
    config.polymarket.gamma_base_url = env.poly_host
    config.polymarket.clob_base_url = env.clob_host
    config.polymarket.ws_market_url = env.ws_host
    config.weather.timeout_seconds = env.http_timeout
    config.weather.retries = env.http_retries
    config.execution.live_trading = env.live_trading
    config.execution.confirm_live_trading = env.confirm_live_trading
    config.execution.mode = "live" if env.live_trading else config.execution.mode
    config.firebase.enabled = env.firebase_enabled or config.firebase.enabled
    config.firebase.bucket_name = env.firebase_bucket_name or config.firebase.bucket_name
    config.firebase.prefix = env.firebase_prefix or config.firebase.prefix
    config.firebase.credentials_json = env.firebase_credentials_json or config.firebase.credentials_json
    config.telegram.bot_token = env.telegram_bot_token or config.telegram.bot_token
    config.telegram.chat_id = env.telegram_chat_id or config.telegram.chat_id
    if config.telegram.bot_token and config.telegram.chat_id:
        config.telegram.enabled = True

    config.scanner.state_path = _rebase_rooted_path(config.scanner.state_path, source_root="artifacts", target_root=env.artifacts_dir)
    config.opportunity_shadow.state_path = _rebase_rooted_path(
        config.opportunity_shadow.state_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.opportunity_shadow.latest_output_path = _rebase_rooted_path(
        config.opportunity_shadow.latest_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.opportunity_shadow.history_output_path = _rebase_rooted_path(
        config.opportunity_shadow.history_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.opportunity_shadow.summary_output_path = _rebase_rooted_path(
        config.opportunity_shadow.summary_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.hope_hunt.state_path = _rebase_rooted_path(config.hope_hunt.state_path, source_root="artifacts", target_root=env.artifacts_dir)
    config.hope_hunt.latest_output_path = _rebase_rooted_path(
        config.hope_hunt.latest_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.hope_hunt.history_output_path = _rebase_rooted_path(
        config.hope_hunt.history_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.hope_hunt.summary_output_path = _rebase_rooted_path(
        config.hope_hunt.summary_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.state_path = _rebase_rooted_path(
        config.observation_station.state_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.latest_output_path = _rebase_rooted_path(
        config.observation_station.latest_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.history_output_path = _rebase_rooted_path(
        config.observation_station.history_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.summary_output_path = _rebase_rooted_path(
        config.observation_station.summary_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.alerts_output_path = _rebase_rooted_path(
        config.observation_station.alerts_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.observation_station.queue_output_path = _rebase_rooted_path(
        config.observation_station.queue_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.opportunity_report_path = _rebase_rooted_path(
        config.station_dashboard.opportunity_report_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.observation_latest_path = _rebase_rooted_path(
        config.station_dashboard.observation_latest_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.observation_summary_path = _rebase_rooted_path(
        config.station_dashboard.observation_summary_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.queue_output_path = _rebase_rooted_path(
        config.station_dashboard.queue_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.open_phase_latest_path = _rebase_rooted_path(
        config.station_dashboard.open_phase_latest_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.open_phase_summary_path = _rebase_rooted_path(
        config.station_dashboard.open_phase_summary_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.revenue_gate_summary_path = _rebase_rooted_path(
        config.station_dashboard.revenue_gate_summary_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.watchlist_playbook_path = _rebase_rooted_path(
        config.station_dashboard.watchlist_playbook_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.json_output_path = _rebase_rooted_path(
        config.station_dashboard.json_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.html_output_path = _rebase_rooted_path(
        config.station_dashboard.html_output_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_dashboard.state_path = _rebase_rooted_path(
        config.station_dashboard.state_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.station_orchestrator.state_path = _rebase_rooted_path(
        config.station_orchestrator.state_path,
        source_root="artifacts",
        target_root=env.artifacts_dir,
    )
    config.monitoring.l2_output_dir = _rebase_rooted_path(
        config.monitoring.l2_output_dir,
        source_root="data",
        target_root=env.data_dir,
    )
    return config, env
