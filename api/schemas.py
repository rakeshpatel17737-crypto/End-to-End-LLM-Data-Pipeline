from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)


class SourceChunk(BaseModel):
    chunk_id: str
    content: str
    source_uri: str
    page_number: int | None
    cosine_sim: float | None


class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[SourceChunk]
    mmr_diversity: float
    mean_cosine_sim: float
    latency_ms: float
    cost_usd: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    db: str
    redis: str
    duckdb: str
