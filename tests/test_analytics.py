from datetime import datetime, timedelta, timezone

from langmet.analytics import (
    compute_citation_coverage,
    compute_operational_metrics,
    compute_rag_metrics,
    detect_categorical_drift,
    detect_numeric_drift,
    detect_numeric_drift_windowed,
)
from langmet.models import CitationMessageEvent, CompletionEvent, RagEvent


def test_compute_operational_metrics_basic():
    now = datetime.now(timezone.utc)
    events = [
        CompletionEvent(
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=100,
            tokens_total=50,
            error_message=None,
            created_at=now,
        ),
        CompletionEvent(
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=300,
            tokens_total=150,
            error_message="timeout",
            created_at=now,
        ),
        CompletionEvent(
            provider="anthropic",
            model="claude",
            latency_ms=200,
            tokens_total=100,
            error_message=None,
            created_at=now,
        ),
    ]

    result = compute_operational_metrics(events)

    assert result["overview"]["total_completions"] == 3
    assert result["overview"]["error_count"] == 1
    assert result["overview"]["success_count"] == 2
    assert result["overview"]["total_tokens"] == 300
    assert result["overview"]["avg_latency_ms"] == 200
    assert result["overview"]["latency_percentiles_ms"]["p50"] == 200
    assert result["overview"]["latency_percentiles_ms"]["p90"] == 280
    assert result["overview"]["latency_percentiles_ms"]["p95"] == 290
    assert result["overview"]["latency_percentiles_ms"]["p99"] == 298
    assert result["by_provider"]["openai"]["count"] == 2
    assert result["by_provider"]["openai"]["errors"] == 1


def test_compute_rag_metrics_basic():
    now = datetime.now(timezone.utc)
    events = [
        RagEvent(
            top_k=10,
            top_n=3,
            retrieval_scores=[0.8, 0.7],
            rerank_scores=[0.9],
            retrieval_latency_ms=40,
            rerank_latency_ms=20,
            created_at=now,
        ),
        RagEvent(
            top_k=6,
            top_n=2,
            retrieval_scores=[0.6],
            rerank_scores=[0.7, 0.8],
            retrieval_latency_ms=60,
            rerank_latency_ms=30,
            created_at=now,
        ),
    ]

    result = compute_rag_metrics(events)

    assert result["overview"]["total_queries"] == 2
    assert result["overview"]["avg_top_k"] == 8
    assert result["overview"]["avg_top_n"] == 2.5
    assert result["overview"]["avg_retrieval_latency_ms"] == 50
    assert result["overview"]["avg_rerank_latency_ms"] == 25
    assert result["overview"]["retrieval_latency_percentiles_ms"]["p50"] == 50
    assert result["overview"]["retrieval_latency_percentiles_ms"]["p95"] == 59
    assert result["overview"]["rerank_latency_percentiles_ms"]["p50"] == 25
    assert result["overview"]["rerank_latency_percentiles_ms"]["p95"] == 29.5
    assert result["scores"]["retrieval_score_count"] == 3
    assert result["scores"]["rerank_score_count"] == 3


def test_compute_citation_coverage_basic():
    now = datetime.now(timezone.utc)
    events = [
        CitationMessageEvent(message_id="1", evidence_count=2, created_at=now),
        CitationMessageEvent(message_id="2", evidence_count=0, created_at=now),
        CitationMessageEvent(message_id="3", evidence_count=1, created_at=now),
    ]

    result = compute_citation_coverage(events)

    assert result["total_messages"] == 3
    assert result["messages_with_evidence"] == 2
    assert result["messages_without_evidence"] == 1
    assert result["avg_evidence_per_message"] == 1.0


def test_detect_numeric_drift_detects_shift():
    baseline = [100, 102, 98, 101, 99, 100, 103, 97, 100, 99]
    current = [155, 160, 150, 158, 162, 149, 151, 157, 159, 154]

    result = detect_numeric_drift(baseline, current)

    assert result["status"] == "ok"
    assert result["drift_detected"] is True
    assert result["alerts"]["psi_alert"] is True
    assert result["psi"] > 0.2


def test_detect_categorical_drift_detects_provider_mix_change():
    baseline = ["openai"] * 70 + ["anthropic"] * 30
    current = ["openai"] * 35 + ["anthropic"] * 65

    result = detect_categorical_drift(baseline, current)

    assert result["status"] == "ok"
    assert result["drift_detected"] is True
    assert result["tvd"] > 0.15


def test_detect_numeric_drift_windowed_uses_last_hour_vs_trailing_week():
    ref = datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc)

    baseline_points = [
        (ref - timedelta(hours=2 + i), 100.0 + (i % 3)) for i in range(30)
    ]
    current_points = [
        (ref - timedelta(minutes=59 - i), 170.0 + (i % 2)) for i in range(25)
    ]
    observations = baseline_points + current_points

    result = detect_numeric_drift_windowed(
        observations=observations,
        reference_time=ref,
        min_samples_per_window=20,
    )

    assert result["status"] == "ok"
    assert result["drift_detected"] is True
    assert result["baseline_count"] >= 20
    assert result["current_count"] >= 20
    assert result["windows"]["current_window_seconds"] == 3600
    assert result["windows"]["baseline_window_seconds"] == 604800


def test_detect_numeric_drift_windowed_flags_insufficient_samples():
    ref = datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc)
    observations = [
        (ref - timedelta(hours=2), 100.0),
        (ref - timedelta(minutes=30), 130.0),
    ]

    result = detect_numeric_drift_windowed(
        observations=observations,
        reference_time=ref,
        min_samples_per_window=5,
    )

    assert result["status"] == "insufficient_data"
    assert result["drift_detected"] is False
