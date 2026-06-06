from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    price_per_1k_tokens: float = 0.00002     # $0.02 per 1M = $0.00002 per 1K
    max_batch_size: int = 2048
    cache_ttl_seconds: int = 604800          # 7 days

    redis_host: str = "localhost"
    redis_port: int = 6379

    model_config = {"env_file": ".env", "populate_by_name": True}


config = EmbeddingConfig()
