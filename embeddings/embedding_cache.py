"""Redis-backed embedding cache keyed on SHA-256(text)."""
from __future__ import annotations

import hashlib
import json

import redis as redis_lib


def cache_key(text: str) -> str:
    return f"embed:{hashlib.sha256(text.encode()).hexdigest()}"


class EmbeddingCache:
    def __init__(self, redis_client: redis_lib.Redis, ttl: int = 604800) -> None:
        self._r = redis_client
        self._ttl = ttl

    def get(self, text: str) -> list[float] | None:
        val = self._r.get(cache_key(text))
        if val is None:
            return None
        return json.loads(val)

    def set(self, text: str, embedding: list[float]) -> None:
        self._r.setex(cache_key(text), self._ttl, json.dumps(embedding))

    def mget(self, texts: list[str]) -> list[list[float] | None]:
        keys = [cache_key(t) for t in texts]
        values = self._r.mget(keys)
        return [json.loads(v) if v else None for v in values]

    def mset(self, texts: list[str], embeddings: list[list[float]]) -> None:
        pipe = self._r.pipeline()
        for text, emb in zip(texts, embeddings):
            pipe.setex(cache_key(text), self._ttl, json.dumps(emb))
        pipe.execute()
