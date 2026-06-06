"""LLM-based Root Cause Analysis for embedding drift, powered by Groq.

- Async Groq client (OpenAI-compatible) + tenacity retry
- Function calling (tool use) for structured output
- Rule-based fallback if the API is unavailable or tool use fails
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from groq import AsyncGroq, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .embedding_drift_detector import EmbeddingDriftResult

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are an expert MLOps engineer specializing in embedding-based RAG systems.
You diagnose root causes of embedding drift in production vector stores.

Your knowledge covers:
- Local sentence-transformers embedding model behavior and failure modes
- pgvector approximate nearest neighbor index degradation
- Document ingestion pipeline failures (PDF, web scraping, API loaders)
- Chunk quality issues: PII contamination, language mix, near-duplicates
- Semantic drift from topic distribution shifts in ingested documents
- Redis embedding cache poisoning or staleness

Common root cause patterns:
- Centroid shift > 0.15 → major topic distribution change in ingested documents
- KS significant + Z-score anomalous → embedding model change or quantization artifact
- Norm IQR spike → chunk quality degradation (short chunks, garbled text)
- Gradual cosine sim decline → seasonal content drift without baseline update

When diagnosing, always consider:
1. Whether the drift is data-driven (new document topics) or pipeline-driven (bug, model change)
2. Urgency: can we still serve relevant results?
3. Remediation priority: fix pipeline first, then re-embed if needed, then update baseline

You MUST respond by calling the submit_embedding_rca_diagnosis function."""

RCA_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_embedding_rca_diagnosis",
        "description": "Submit structured root cause analysis for embedding drift",
        "parameters": {
            "type": "object",
            "properties": {
                "probable_cause": {"type": "string", "description": "1-2 sentence root cause description"},
                "cause_category": {
                    "type": "string",
                    "enum": [
                        "topic_distribution_shift",
                        "embedding_model_change",
                        "data_quality_degradation",
                        "ingestion_pipeline_bug",
                        "seasonal_content_shift",
                        "baseline_staleness",
                        "unknown",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "remediation_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of remediation actions",
                },
                "urgency": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "estimated_impact": {"type": "string", "description": "Impact on RAG retrieval quality"},
            },
            "required": ["probable_cause", "cause_category", "confidence", "remediation_steps", "urgency", "estimated_impact"],
        },
    },
}


@dataclass
class EmbeddingRCADiagnosis:
    probable_cause: str
    cause_category: str
    confidence: float
    remediation_steps: list[str]
    urgency: str
    estimated_impact: str
    model_used: str
    tokens_used: int = 0
    fallback_used: bool = False
    computed_at: float = field(default_factory=time.time)


class EmbeddingRCAEngine:
    def __init__(self) -> None:
        self._client: Optional[AsyncGroq] = None

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._client = AsyncGroq(api_key=api_key)
        return self._client

    async def analyze(self, result: EmbeddingDriftResult) -> EmbeddingRCADiagnosis:
        try:
            return await self._analyze_with_llm(result)
        except Exception as exc:
            logger.warning("LLM RCA failed (%s), using rule-based fallback", exc)
            return self._rule_based_fallback(result)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIConnectionError),
        reraise=False,
    )
    async def _analyze_with_llm(self, result: EmbeddingDriftResult) -> EmbeddingRCADiagnosis:
        client = self._get_client()
        context = _build_context(result)

        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this embedding drift report:\n\n{context}"},
            ],
            tools=[RCA_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_embedding_rca_diagnosis"}},
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("No tool_call in response")

        args = json.loads(message.tool_calls[0].function.arguments)

        return EmbeddingRCADiagnosis(
            probable_cause=args["probable_cause"],
            cause_category=args["cause_category"],
            confidence=float(args["confidence"]),
            remediation_steps=args["remediation_steps"],
            urgency=args["urgency"],
            estimated_impact=args["estimated_impact"],
            model_used=GROQ_MODEL,
            tokens_used=response.usage.total_tokens,
            fallback_used=False,
        )

    def _rule_based_fallback(self, result: EmbeddingDriftResult) -> EmbeddingRCADiagnosis:
        shift = result.centroid_shift
        ks_sig = result.ks_p_value < 0.05
        z = abs(result.z_score_norm_mean)
        cosine = result.cosine_sim_mean

        if shift > 0.15:
            cause = "topic_distribution_shift"
            probable = (
                f"Major embedding centroid shift ({shift:.3f}). "
                "The ingested documents have significantly different topic distribution than the baseline."
            )
            steps = [
                "1. Review recently ingested document sources for topic changes",
                "2. Update the drift baseline after confirming the new distribution is intentional",
                "3. Re-evaluate RAG retrieval quality with test queries",
                "4. Consider re-embedding baseline documents if topic scope changed permanently",
            ]
            urgency = "high"
            confidence = 0.75
        elif ks_sig and z > 3.0:
            cause = "data_quality_degradation"
            probable = (
                f"Significant norm distribution change (KS significant, Z={z:.2f}). "
                "Likely cause: ingestion of low-quality chunks (very short, garbled, or non-text content)."
            )
            steps = [
                "1. Check chunk quality_score distribution in PostgreSQL chunks table",
                "2. Review dead_letter_queue for recent ingestion failures",
                "3. Increase min_chunk_quality threshold in ingestion config",
                "4. Re-run ingestion with stricter quality filtering",
            ]
            urgency = "medium"
            confidence = 0.65
        elif ks_sig:
            cause = "seasonal_content_shift"
            probable = (
                f"Statistically significant norm distribution change (KS p={result.ks_p_value:.4f}) "
                "without major centroid shift. Likely gradual seasonal content variation."
            )
            steps = [
                "1. Monitor over next 7 days for continued drift",
                "2. If drift persists, update baseline embeddings",
                "3. Compare ingested document topics this week vs last month",
            ]
            urgency = "low"
            confidence = 0.50
        else:
            cause = "unknown"
            probable = "Minor drift detected. Insufficient signal for confident diagnosis."
            steps = ["1. Continue monitoring", "2. Collect more data before taking action"]
            urgency = "low"
            confidence = 0.30

        return EmbeddingRCADiagnosis(
            probable_cause=probable,
            cause_category=cause,
            confidence=confidence,
            remediation_steps=steps,
            urgency=urgency,
            estimated_impact=f"RAG retrieval quality may be degraded. Mean cosine similarity: {cosine:.3f}",
            model_used="rule_based_fallback",
            fallback_used=True,
        )


def _build_context(result: EmbeddingDriftResult) -> str:
    ctx = {
        "drift_metrics": {
            "centroid_shift": result.centroid_shift,
            "cosine_sim_mean": result.cosine_sim_mean,
            "cosine_sim_std": result.cosine_sim_std,
            "norm_iqr": result.norm_iqr,
            "norm_mean": result.norm_mean,
            "norm_std": result.norm_std,
        },
        "statistical_tests": {
            "ks_statistic": result.ks_statistic,
            "ks_p_value": result.ks_p_value,
            "ks_significant": result.ks_p_value < 0.05,
            "z_score_norm_mean": result.z_score_norm_mean,
            "z_score_anomalous": abs(result.z_score_norm_mean) > 3.0,
        },
        "severity": result.drift_severity,
        "health_score": result.health_score,
        "sample_sizes": {
            "current": result.chunk_sample_size,
            "baseline": result.baseline_size,
        },
    }
    return json.dumps(ctx, indent=2)


rca_engine = EmbeddingRCAEngine()
