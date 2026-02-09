"""
Metrics tracking module.

Tracks performance metrics for all published posts and aggregates them
for analytics, feedback loops, and reporting. Acts as the data backbone
for continuous improvement of the content strategy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.types import (
    ContentGoal,
    ContentTone,
    ContentType,
    PerformanceMetrics,
    PostDraft,
)

logger = logging.getLogger(__name__)


@dataclass
class PostRecord:
    """Record of a published post with its draft and metrics."""

    draft: PostDraft
    published_at: datetime
    metrics: PerformanceMetrics | None = None
    last_measured: datetime | None = None


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple posts."""

    period: str  # "day", "week", "month"
    total_posts: int = 0
    total_impressions: int = 0
    total_engagements: int = 0
    avg_engagement_rate: float = 0.0
    avg_likes: float = 0.0
    avg_replies: float = 0.0
    avg_reposts: float = 0.0
    total_new_followers: int = 0
    top_post_id: str = ""
    top_post_engagement: int = 0
    worst_post_id: str = ""
    worst_post_engagement: int = 0


class MetricsTracker:
    """
    Central metrics tracking and aggregation system.

    Tracks every published post and its performance over time.
    Provides aggregated views for different time periods, content types,
    tones, and goals.

    This module is analogous to the CacheRequestInfoSideEffect in
    home-mixer - it runs as a side effect after content is published
    and continuously updates metrics.
    """

    def __init__(self):
        self._posts: dict[str, PostRecord] = {}  # post_id -> record
        self._daily_metrics: dict[str, AggregateMetrics] = {}  # date -> metrics

    @property
    def name(self) -> str:
        return "MetricsTracker"

    def enabled(self, _) -> bool:
        return True

    def record_publication(self, draft: PostDraft, published_at: datetime | None = None) -> None:
        """Record that a draft was published."""
        pub_time = published_at or datetime.now()
        self._posts[draft.id] = PostRecord(draft=draft, published_at=pub_time)
        logger.info("Recorded publication of post %s at %s", draft.id, pub_time.isoformat())

    def update_metrics(self, post_id: str, metrics: PerformanceMetrics) -> None:
        """Update performance metrics for a published post."""
        record = self._posts.get(post_id)
        if not record:
            logger.warning("No record found for post %s", post_id)
            return

        record.metrics = metrics
        record.last_measured = datetime.now()

        # Update daily aggregation
        date_key = record.published_at.strftime("%Y-%m-%d")
        self._update_daily_aggregate(date_key)

        logger.debug(
            "Updated metrics for post %s: %d impressions, %.2f%% engagement",
            post_id,
            metrics.impressions,
            metrics.engagement_rate * 100,
        )

    def get_aggregate(
        self,
        period: str = "week",
        content_type: ContentType | None = None,
        tone: ContentTone | None = None,
        goal: ContentGoal | None = None,
    ) -> AggregateMetrics:
        """
        Get aggregated metrics for a time period, optionally filtered.

        Args:
            period: "day", "week", "month"
            content_type: Filter by content type
            tone: Filter by tone
            goal: Filter by goal
        """
        now = datetime.now()
        if period == "day":
            cutoff = now - timedelta(days=1)
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(weeks=1)

        # Filter posts
        relevant_posts = [
            r
            for r in self._posts.values()
            if r.published_at >= cutoff and r.metrics is not None
        ]

        if content_type:
            relevant_posts = [
                r for r in relevant_posts
                if r.draft.pieces and r.draft.pieces[0].content_type == content_type
            ]
        if tone:
            relevant_posts = [r for r in relevant_posts if r.draft.tone == tone]
        if goal:
            relevant_posts = [r for r in relevant_posts if r.draft.goal == goal]

        return self._compute_aggregate(relevant_posts, period)

    def get_top_posts(self, count: int = 10, days: int = 30) -> list[PostRecord]:
        """Get top performing posts by engagement."""
        cutoff = datetime.now() - timedelta(days=days)
        with_metrics = [
            r for r in self._posts.values()
            if r.published_at >= cutoff and r.metrics is not None
        ]
        return sorted(
            with_metrics,
            key=lambda r: r.metrics.total_engagements if r.metrics else 0,
            reverse=True,
        )[:count]

    def get_worst_posts(self, count: int = 10, days: int = 30) -> list[PostRecord]:
        """Get worst performing posts by engagement rate."""
        cutoff = datetime.now() - timedelta(days=days)
        with_metrics = [
            r for r in self._posts.values()
            if r.published_at >= cutoff and r.metrics is not None
        ]
        return sorted(
            with_metrics,
            key=lambda r: r.metrics.engagement_rate if r.metrics else 0,
        )[:count]

    def get_performance_by_type(self) -> dict[str, float]:
        """Get average engagement rate by content type."""
        type_metrics: dict[str, list[float]] = defaultdict(list)
        for record in self._posts.values():
            if record.metrics and record.draft.pieces:
                ct = record.draft.pieces[0].content_type.value
                type_metrics[ct].append(record.metrics.engagement_rate)

        return {
            ct: sum(rates) / len(rates) for ct, rates in type_metrics.items() if rates
        }

    def get_performance_by_tone(self) -> dict[str, float]:
        """Get average engagement rate by content tone."""
        tone_metrics: dict[str, list[float]] = defaultdict(list)
        for record in self._posts.values():
            if record.metrics:
                tone_metrics[record.draft.tone.value].append(record.metrics.engagement_rate)

        return {
            tone: sum(rates) / len(rates) for tone, rates in tone_metrics.items() if rates
        }

    def _compute_aggregate(
        self, posts: list[PostRecord], period: str
    ) -> AggregateMetrics:
        """Compute aggregate metrics from a list of post records."""
        agg = AggregateMetrics(period=period, total_posts=len(posts))

        if not posts:
            return agg

        for record in posts:
            m = record.metrics
            if not m:
                continue
            agg.total_impressions += m.impressions
            agg.total_engagements += m.total_engagements
            agg.total_new_followers += m.new_followers

            if m.total_engagements > agg.top_post_engagement:
                agg.top_post_engagement = m.total_engagements
                agg.top_post_id = record.draft.id

            if not agg.worst_post_id or m.total_engagements < agg.worst_post_engagement:
                agg.worst_post_engagement = m.total_engagements
                agg.worst_post_id = record.draft.id

        n = len(posts)
        metrics_posts = [r for r in posts if r.metrics]
        if metrics_posts:
            agg.avg_engagement_rate = sum(
                r.metrics.engagement_rate for r in metrics_posts if r.metrics
            ) / len(metrics_posts)
            agg.avg_likes = sum(r.metrics.likes for r in metrics_posts if r.metrics) / len(metrics_posts)
            agg.avg_replies = sum(r.metrics.replies for r in metrics_posts if r.metrics) / len(metrics_posts)
            agg.avg_reposts = sum(r.metrics.reposts for r in metrics_posts if r.metrics) / len(metrics_posts)

        return agg

    def _update_daily_aggregate(self, date_key: str) -> None:
        """Recompute daily aggregate for a given date."""
        day_posts = [
            r for r in self._posts.values()
            if r.published_at.strftime("%Y-%m-%d") == date_key and r.metrics
        ]
        self._daily_metrics[date_key] = self._compute_aggregate(day_posts, "day")
