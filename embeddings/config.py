from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    # Local sentence-transformers model — runs on CPU, no API key, free.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384       # all-MiniLM-L6-v2 output size
    max_batch_size: int = 64              # encode() batch size
    cache_ttl_seconds: int = 604800       # 7 days

    redis_host: str = "localhost"
    redis_port: int = 6379

    model_config = {"env_file": ".env", "populate_by_name": True}


config = EmbeddingConfig()
