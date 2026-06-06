from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llm_platform"
    postgres_user: str = "llmplatform"
    postgres_password: str = "llmplatform"

    redis_host: str = "localhost"
    redis_port: int = 6379

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    duckdb_path: str = "/data/warehouse/llm_logs.duckdb"
    log_level: str = "INFO"
    environment: str = "development"

    model_config = {"env_file": ".env", "populate_by_name": True}

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


config = APIConfig()
