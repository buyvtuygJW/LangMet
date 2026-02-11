"""Port definitions for fetching analytics events from any source."""

from datetime import datetime
from typing import Protocol, Sequence

from .models import CitationMessageEvent, CompletionEvent, RagEvent


class MetricsRepository(Protocol):
    """Repository interface that adapters can implement for any backend."""

    def fetch_completion_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[CompletionEvent]:
        """Return completion events for a period."""

    def fetch_rag_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[RagEvent]:
        """Return RAG events for a period."""

    def fetch_citation_message_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[CitationMessageEvent]:
        """Return citation/evidence events for a period."""
