"""Centralized configuration via Pydantic Settings (environment-driven)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/stratum_sourcing",
        alias="DATABASE_URL",
    )

    @property
    def sync_database_url(self) -> str:
        """Synchronous DB URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "").replace("asyncpg://", "postgresql://")

    # --- Slack ---
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_signing_secret: str = Field(default="", alias="SLACK_SIGNING_SECRET")
    slack_app_token: str = Field(default="", alias="SLACK_APP_TOKEN")
    slack_channel_sourcing: str = Field(default="#sourcing", alias="SLACK_CHANNEL_SOURCING")

    # --- Notion ---
    notion_api_key: str = Field(default="", alias="NOTION_API_KEY")
    notion_ocean_database_id: str = Field(default="", alias="NOTION_OCEAN_DATABASE_ID")

    # --- LLM ---
    # Option 1 (production): OpenClaw gateway with Codex OAuth
    openclaw_gateway_url: str = Field(default="", alias="OPENCLAW_GATEWAY_URL")
    openclaw_gateway_token: str = Field(default="", alias="OPENCLAW_GATEWAY_TOKEN")
    openclaw_internal_port: int = Field(default=9080, alias="OPENCLAW_INTERNAL_PORT")
    # Option 2 (fallback): Direct Anthropic API
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Option 3 (fallback): Direct OpenAI API key
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    # OAuth minter (used by OpenClaw for Codex auth on Railway)
    oauth_minter_url: str = Field(default="", alias="OAUTH_MINTER_URL")
    oauth_minter_key: str = Field(default="", alias="OAUTH_MINTER_KEY")
    llm_model: str = Field(default="gpt-5.4", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # --- Cron / Security ---
    cron_secret: str = Field(default="", alias="CRON_SECRET")

    # --- Persistent volume ---
    data_dir: str = Field(default="/data", alias="DATA_DIR")

    # --- Scraping ---
    fetch_timeout_seconds: int = Field(default=30, alias="FETCH_TIMEOUT_SECONDS")
    fetch_concurrency: int = Field(default=5, alias="FETCH_CONCURRENCY")
    browser_rate_limit_seconds: float = Field(default=10.0, alias="BROWSER_RATE_LIMIT_SECONDS")

    # --- Scoring weights ---
    score_weight_vertical: float = Field(default=0.40)
    score_weight_geographic: float = Field(default=0.15)
    score_weight_stage: float = Field(default=0.15)
    score_weight_recency: float = Field(default=0.15)
    score_weight_authority: float = Field(default=0.15)


settings = Settings()
