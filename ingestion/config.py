from pydantic_settings import BaseSettings


class IngestionConfig(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llm_platform"
    postgres_user: str = "llmplatform"
    postgres_password: str = "llmplatform"

    redis_host: str = "localhost"
    redis_port: int = 6379

    groq_api_key: str = ""

    duckdb_path: str = "/data/warehouse/llm_logs.duckdb"
    daily_cost_budget_usd: float = 10.0

    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50
    minhash_num_perm: int = 128
    min_chunk_quality: float = 0.3

    model_config = {"env_file": ".env", "populate_by_name": True}

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


config = IngestionConfig()
