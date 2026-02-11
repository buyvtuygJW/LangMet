"""LangMet - reusable analytics for LLM and RAG systems."""

from .analytics import (
    compute_citation_coverage,
    compute_operational_metrics,
    compute_rag_metrics,
    detect_categorical_drift,
    detect_numeric_drift,
    detect_numeric_drift_windowed,
)
from .service import AnalyticsService

__all__ = [
    "AnalyticsService",
    "compute_operational_metrics",
    "compute_rag_metrics",
    "compute_citation_coverage",
    "detect_numeric_drift",
    "detect_numeric_drift_windowed",
    "detect_categorical_drift",
]

__version__ = "0.1.0"
