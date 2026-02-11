from datetime import datetime, timezone

from langmet.models import CitationMessageEvent, CompletionEvent, RagEvent
from langmet.service import AnalyticsService


class FakeRepo:
    def __init__(self):
        self.last_completion_range = None
        self.last_rag_range = None
        self.last_citation_range = None

    def fetch_completion_events(self, start_date, end_date):
        self.last_completion_range = (start_date, end_date)
        return [
            CompletionEvent(
                provider="openai",
                model="gpt-4o-mini",
                latency_ms=100,
                tokens_total=20,
                error_message=None,
                created_at=end_date,
            )
        ]

    def fetch_rag_events(self, start_date, end_date):
        self.last_rag_range = (start_date, end_date)
        return [
            RagEvent(
                top_k=5,
                top_n=2,
                retrieval_scores=[0.9],
                rerank_scores=[0.8],
                retrieval_latency_ms=10,
                rerank_latency_ms=5,
                created_at=end_date,
            )
        ]

    def fetch_citation_message_events(self, start_date, end_date):
        self.last_citation_range = (start_date, end_date)
        return [CitationMessageEvent(message_id="m1", evidence_count=1, created_at=end_date)]


def test_service_uses_explicit_period():
    repo = FakeRepo()
    service = AnalyticsService(repo)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)

    op = service.get_operational_metrics(start, end)
    rag = service.get_rag_metrics(start, end)
    cit = service.get_citation_coverage(start, end)

    assert repo.last_completion_range == (start, end)
    assert repo.last_rag_range == (start, end)
    assert repo.last_citation_range == (start, end)

    assert op["overview"]["total_completions"] == 1
    assert rag["overview"]["total_queries"] == 1
    assert cit["messages_with_evidence"] == 1
