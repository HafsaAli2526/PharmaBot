"""
pharmaai/core/config.py
Centralised configuration loader – reads settings.yaml and merges with
environment variables.  A single Settings singleton is exposed via
`get_settings()`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class GPUConfig(BaseModel):
    device: str = "cuda:0"
    mixed_precision: bool = True
    max_memory_gb: int = 60


class ModelConfig(BaseModel):
    path: str
    hf_id: str
    dim: int = 768
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9


class ModelsConfig(BaseModel):
    biobert: ModelConfig
    clinicalbert: ModelConfig
    chemberta: ModelConfig
    slm: ModelConfig
    slm_finetuned: ModelConfig | None = None


class IndexConfig(BaseModel):
    type: str = "faiss"
    dim: int = 2304
    nlist: int = 256
    nprobe: int = 32
    path: str = "./data/index.faiss"
    id_map_path: str = "./data/index_id_map.json"


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "pharmaai"
    user: str = "pharmaai"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 10

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ttl_seconds: int = 3600

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class RabbitMQConfig(BaseModel):
    host: str = "localhost"
    port: int = 5672
    user: str = "guest"
    password: str = "guest"
    vhost: str = "/"
    queues: dict[str, str] = Field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}{self.vhost}"


class ExternalAPIConfig(BaseModel):
    base_url: str
    api_key: str = ""
    rate_limit_per_second: float = 2.0


class ApisConfig(BaseModel):
    pubmed: ExternalAPIConfig
    openfda: ExternalAPIConfig
    clinical_trials: ExternalAPIConfig
    newsapi: ExternalAPIConfig
    google_cse: ExternalAPIConfig
    event_registry: ExternalAPIConfig | None = None


class TwilioConfig(BaseModel):
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""


class SlackConfig(BaseModel):
    webhook_url: str = ""
    channel: str = "#pharmaai-alerts"


class SendGridConfig(BaseModel):
    api_key: str = ""
    from_email: str = "alerts@pharmaai.internal"


class FCMConfig(BaseModel):
    server_key: str = ""


class NotificationsConfig(BaseModel):
    twilio: TwilioConfig = Field(default_factory=TwilioConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    sendgrid: SendGridConfig = Field(default_factory=SendGridConfig)
    fcm: FCMConfig = Field(default_factory=FCMConfig)


class APIServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit: dict[str, int] = Field(default_factory=dict)


class IngestionConfig(BaseModel):
    batch_size: int = 64
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0
    progress_file: str = "./data/ingestion_progress.json"


# ─────────────────────────────────────────────────────────────────────────────
# Root settings
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseModel):
    """Single source of truth for all configuration."""

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    models: ModelsConfig
    index: IndexConfig = Field(default_factory=IndexConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    apis: ApisConfig
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    api: APIServerConfig = Field(default_factory=APIServerConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


def _resolve_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR:-default} placeholders in config values."""
    if isinstance(value, str):
        import re
        pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
        def replace(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return pattern.sub(replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve_env(raw)


@lru_cache(maxsize=1)
def get_settings(config_path: str = "configs/settings.yaml") -> Settings:
    """Load and cache settings from YAML + env vars."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent.parent / config_path
    data = _load_yaml(path)
    return Settings(**data)