"""
Competitor analysis module.

Monitors competitor accounts on X to identify successful content patterns,
gaps in the market, and opportunities for differentiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class CompetitorPost:
    """A post from a competitor account."""

    post_id: str
    author_handle: str
    text: str
    created_at: datetime
    likes: int = 0
    replies: int = 0
    reposts: int = 0
    quotes: int = 0
    views: int = 0
    has_media: bool = False
    has_link: bool = False
    hashtags: list[str] = field(default_factory=list)

    @property
    def engagement_rate(self) -> float:
        total = self.likes + self.replies + self.reposts + self.quotes
        return total / max(self.views, 1)


@dataclass
class CompetitorProfile:
    """Analysis profile for a competitor account."""

    handle: str
    follower_count: int = 0
    avg_engagement_rate: float = 0.0
    posting_frequency: float = 0.0  # Posts per day
    top_topics: list[str] = field(default_factory=list)
    top_hashtags: list[str] = field(default_factory=list)
    peak_posting_hours: list[int] = field(default_factory=list)
    content_type_distribution: dict[str, float] = field(default_factory=dict)
    avg_post_length: float = 0.0
    media_usage_rate: float = 0.0  # % of posts with media
    recent_posts: list[CompetitorPost] = field(default_factory=list)


@dataclass
class CompetitorInsight:
    """Actionable insight derived from competitor analysis."""

    insight_type: str  # "content_gap", "trending_format", "timing", "topic"
    description: str
    confidence: float = 0.0  # 0.0 - 1.0
    source_competitors: list[str] = field(default_factory=list)
    suggested_action: str = ""


class CompetitorAnalyzer:
    """
    Analyzes competitor accounts to derive content strategy insights.

    Analysis dimensions:
    1. Content Performance - What types of posts get the most engagement?
    2. Topic Coverage      - What topics are competitors covering?
    3. Posting Patterns    - When and how often do they post?
    4. Format Preferences  - Threads vs short posts vs media-heavy?
    5. Gap Analysis        - What topics/formats are underserved?
    """

    def __init__(self, config: AgentConfig, competitor_handles: list[str] | None = None):
        self.config = config
        self._competitors: list[str] = competitor_handles or []
        self._profiles: dict[str, CompetitorProfile] = {}

    async def analyze_competitors(self) -> list[CompetitorInsight]:
        """Run full competitor analysis and return actionable insights."""
        insights: list[CompetitorInsight] = []

        for handle in self._competitors:
            profile = await self._build_profile(handle)
            self._profiles[handle] = profile

        # Derive insights from all profiles
        insights.extend(self._find_content_gaps())
        insights.extend(self._find_trending_formats())
        insights.extend(self._find_timing_opportunities())
        insights.extend(self._find_underserved_topics())

        # Sort by confidence
        insights.sort(key=lambda i: i.confidence, reverse=True)

        logger.info(
            "CompetitorAnalyzer generated %d insights from %d competitors",
            len(insights),
            len(self._competitors),
        )
        return insights

    async def get_top_performing_posts(
        self, handle: str, count: int = 10
    ) -> list[CompetitorPost]:
        """Get the top performing posts from a competitor."""
        profile = self._profiles.get(handle)
        if not profile:
            profile = await self._build_profile(handle)
            self._profiles[handle] = profile

        return sorted(
            profile.recent_posts,
            key=lambda p: p.engagement_rate,
            reverse=True,
        )[:count]

    async def _build_profile(self, handle: str) -> CompetitorProfile:
        """
        Build a comprehensive profile for a competitor.
        In production, fetches data via X API v2 user timeline endpoint.
        """
        # Interface for X API: GET /2/users/:id/tweets
        return CompetitorProfile(handle=handle)

    def _find_content_gaps(self) -> list[CompetitorInsight]:
        """Identify topics/formats that competitors aren't covering well."""
        insights = []
        niche_keywords = set(self.config.niche_keywords)

        # Find niche keywords not well covered by competitors
        competitor_topics: set[str] = set()
        for profile in self._profiles.values():
            competitor_topics.update(t.lower() for t in profile.top_topics)

        uncovered = niche_keywords - competitor_topics
        if uncovered:
            insights.append(
                CompetitorInsight(
                    insight_type="content_gap",
                    description=f"Topics not well covered by competitors: {', '.join(uncovered)}",
                    confidence=0.7,
                    source_competitors=list(self._profiles.keys()),
                    suggested_action="Create authoritative content on these underserved topics",
                )
            )

        return insights

    def _find_trending_formats(self) -> list[CompetitorInsight]:
        """Identify content formats performing well for competitors."""
        insights = []
        format_engagement: dict[str, list[float]] = {}

        for profile in self._profiles.values():
            for fmt, rate in profile.content_type_distribution.items():
                format_engagement.setdefault(fmt, []).append(rate)

        for fmt, rates in format_engagement.items():
            avg = sum(rates) / len(rates) if rates else 0
            if avg > 0.5:
                insights.append(
                    CompetitorInsight(
                        insight_type="trending_format",
                        description=f"Format '{fmt}' has high engagement across competitors (avg: {avg:.2f})",
                        confidence=min(avg, 1.0),
                        source_competitors=list(self._profiles.keys()),
                        suggested_action=f"Increase usage of '{fmt}' format in content mix",
                    )
                )

        return insights

    def _find_timing_opportunities(self) -> list[CompetitorInsight]:
        """Find posting times where competitors are less active."""
        all_peak_hours: list[int] = []
        for profile in self._profiles.values():
            all_peak_hours.extend(profile.peak_posting_hours)

        if not all_peak_hours:
            return []

        # Find hours with less competition
        hour_counts: dict[int, int] = {}
        for h in all_peak_hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1

        active_start, active_end = self.config.strategy.active_hours_utc
        quiet_hours = [
            h for h in range(active_start, active_end) if hour_counts.get(h, 0) == 0
        ]

        if quiet_hours:
            return [
                CompetitorInsight(
                    insight_type="timing",
                    description=f"Low competition hours: {quiet_hours}",
                    confidence=0.6,
                    source_competitors=list(self._profiles.keys()),
                    suggested_action="Schedule some posts during these low-competition hours",
                )
            ]
        return []

    def _find_underserved_topics(self) -> list[CompetitorInsight]:
        """Find topics with high engagement but low coverage."""
        topic_engagement: dict[str, list[float]] = {}

        for profile in self._profiles.values():
            for post in profile.recent_posts:
                for tag in post.hashtags:
                    topic_engagement.setdefault(tag.lower(), []).append(post.engagement_rate)

        insights = []
        for topic, rates in topic_engagement.items():
            avg_rate = sum(rates) / len(rates) if rates else 0
            if avg_rate > 0.05 and len(rates) < 3:  # High engagement, low volume
                insights.append(
                    CompetitorInsight(
                        insight_type="topic",
                        description=f"Topic '{topic}' has high engagement ({avg_rate:.3f}) but low coverage",
                        confidence=min(avg_rate * 10, 1.0),
                        suggested_action=f"Create more content about '{topic}'",
                    )
                )

        return insights
