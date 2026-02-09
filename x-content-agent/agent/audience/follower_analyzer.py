"""
Follower analysis module.

Analyzes the account's followers to build audience profiles, understand
demographics, and identify the most engaged segments. This data feeds
into the engagement predictor and topic selector.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from agent.config import AgentConfig
from agent.types import AudienceProfile

logger = logging.getLogger(__name__)


@dataclass
class FollowerSegment:
    """A segment of followers with shared characteristics."""

    name: str
    size: int
    percentage: float
    interests: list[str] = field(default_factory=list)
    avg_engagement_rate: float = 0.0
    peak_active_hours: list[int] = field(default_factory=list)
    language: str = "en"


@dataclass
class FollowerGrowthMetrics:
    """Tracks follower growth over time."""

    current_count: int = 0
    growth_rate_daily: float = 0.0
    growth_rate_weekly: float = 0.0
    growth_rate_monthly: float = 0.0
    top_growth_sources: list[str] = field(default_factory=list)  # Where new followers come from
    churn_rate: float = 0.0  # Unfollow rate


class FollowerAnalyzer:
    """
    Analyzes follower base to build audience understanding.

    Analysis dimensions:
    1. Demographics    - Interests, bios, locations, languages
    2. Engagement      - Who engages most, what they engage with
    3. Activity        - When followers are most active
    4. Growth          - Where new followers come from
    5. Segmentation    - Group followers into actionable segments

    Pipeline role: RequestHydrator - enriches ContentRequest with audience data.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._cached_profile: AudienceProfile | None = None
        self._cached_at: datetime | None = None
        self._segments: list[FollowerSegment] = []

    @property
    def name(self) -> str:
        return "FollowerAnalyzer"

    def enabled(self, _) -> bool:
        return True

    async def build_profile(self) -> AudienceProfile:
        """
        Build a comprehensive audience profile.

        In production, this fetches follower data via X API:
        - GET /2/users/:id/followers (with user.fields)
        - Analyze bios, locations, interests from profile data
        """
        # Return cached profile if recent (avoid excessive API calls)
        if self._cached_profile and self._cached_at:
            from datetime import timedelta

            if datetime.now() - self._cached_at < timedelta(hours=6):
                return self._cached_profile

        # Fetch and analyze followers
        raw_followers = await self._fetch_followers()
        profile = self._analyze_followers(raw_followers)

        # Cache the profile
        self._cached_profile = profile
        self._cached_at = datetime.now()

        logger.info(
            "Built audience profile: %d followers, %.2f%% avg engagement",
            profile.total_followers,
            profile.avg_engagement_rate * 100,
        )
        return profile

    def get_segments(self) -> list[FollowerSegment]:
        """Get follower segments from last analysis."""
        return self._segments

    def get_most_engaged_segment(self) -> FollowerSegment | None:
        """Get the segment with highest engagement rate."""
        if not self._segments:
            return None
        return max(self._segments, key=lambda s: s.avg_engagement_rate)

    async def get_growth_metrics(self) -> FollowerGrowthMetrics:
        """Compute follower growth metrics."""
        # Interface for X API analytics data
        return FollowerGrowthMetrics()

    async def _fetch_followers(self) -> list[dict]:
        """
        Fetch follower data from X API.
        Returns list of follower profile dictionaries.
        """
        # X API v2: GET /2/users/:id/followers
        # Fields: description, location, public_metrics, created_at
        return []

    def _analyze_followers(self, followers: list[dict]) -> AudienceProfile:
        """Analyze raw follower data into an audience profile."""
        if not followers:
            return AudienceProfile(
                language=self.config.default_language,
                interests=self.config.niche_keywords[:5],
            )

        # Extract interests from bios
        all_interests: Counter[str] = Counter()
        languages: Counter[str] = Counter()

        for follower in followers:
            bio = follower.get("description", "")
            # Simple keyword extraction from bios
            words = bio.lower().split()
            for keyword in self.config.niche_keywords:
                if keyword.lower() in words:
                    all_interests[keyword] += 1

            lang = follower.get("lang", "en")
            languages[lang] += 1

        top_interests = [interest for interest, _ in all_interests.most_common(10)]
        primary_language = languages.most_common(1)[0][0] if languages else "en"

        profile = AudienceProfile(
            total_followers=len(followers),
            active_followers=int(len(followers) * 0.3),  # Estimate 30% active
            interests=top_interests,
            language=primary_language,
        )

        # Build segments
        self._segments = self._segment_followers(followers)

        return profile

    def _segment_followers(self, followers: list[dict]) -> list[FollowerSegment]:
        """Segment followers into groups based on characteristics."""
        # Simple segmentation by engagement level
        segments = [
            FollowerSegment(
                name="highly_engaged",
                size=int(len(followers) * 0.1),
                percentage=0.10,
                avg_engagement_rate=0.15,
            ),
            FollowerSegment(
                name="moderately_engaged",
                size=int(len(followers) * 0.3),
                percentage=0.30,
                avg_engagement_rate=0.05,
            ),
            FollowerSegment(
                name="passive",
                size=int(len(followers) * 0.6),
                percentage=0.60,
                avg_engagement_rate=0.01,
            ),
        ]
        return segments
