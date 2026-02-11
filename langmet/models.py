"""Core domain models used by the analytics engine."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence


@dataclass(frozen=True)
class CompletionEvent:
    """A single completion invocation."""

    provider: str
    model: Optional[str]
    latency_ms: Optional[float]
    tokens_total: Optional[int]
    error_message: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class RagEvent:
    """A single RAG pipeline execution event."""

    top_k: Optional[int]
    top_n: Optional[int]
    retrieval_scores: Sequence[float]
    rerank_scores: Sequence[float]
    retrieval_latency_ms: Optional[float]
    rerank_latency_ms: Optional[float]
    created_at: datetime


@dataclass(frozen=True)
class CitationMessageEvent:
    """Assistant message with a pre-aggregated evidence count."""

    message_id: str
    evidence_count: int
    created_at: datetime
