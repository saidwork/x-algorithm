"""
Topic selection module.

Selects the best topics for content generation based on trend signals,
audience interests, historical performance, and content calendar gaps.
Implements the ContentSelector protocol from the pipeline.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.types import (
    AudienceProfile,
    ContentGoal,
    ContentRequest,
    ContentType,
    Topic,
    TrendSignal,
)

logger = logging.getLogger(__name__)


class TopicSelector:
    """
    Selects optimal topics for content creation.

    Selection strategy considers:
    1. Trend momentum   - Is the topic rising or declining?
    2. Niche relevance  - Does it match the account's domain?
    3. Audience fit      - Will the audience care about this?
    4. Content gap       - Have we already covered this recently?
    5. Goal alignment    - Does the topic serve our current goal?
    6. Competition       - How saturated is this topic already?

    Uses a weighted scoring model similar to home-mixer's WeightedScorer
    but applied to topic selection instead of post ranking.
    """

    # Topic scoring weights
    WEIGHT_TREND = 0.25
    WEIGHT_RELEVANCE = 0.25
    WEIGHT_AUDIENCE = 0.20
    WEIGHT_FRESHNESS = 0.15
    WEIGHT_GOAL = 0.15

    def __init__(self, config: AgentConfig):
        self.config = config
        self._recent_topics: list[tuple[str, datetime]] = []

    @property
    def name(self) -> str:
        return "TopicSelector"

    def select_topics(
        self,
        trend_signals: list[TrendSignal],
        audience: AudienceProfile | None = None,
        goal: ContentGoal = ContentGoal.ENGAGEMENT,
        count: int = 5,
    ) -> list[Topic]:
        """
        Select the best topics from available trend signals and niche keywords.

        Args:
            trend_signals: Current trend signals from TrendAnalyzer
            audience: Target audience profile
            goal: Current content goal
            count: Number of topics to select

        Returns:
            Ranked list of Topic objects ready for content generation.
        """
        candidates: list[Topic] = []

        # Convert trend signals to topic candidates
        for signal in trend_signals:
            topic = Topic(
                name=signal.topic,
                keywords=signal.related_topics + [signal.topic],
                category=signal.category,
                trending_score=min(signal.velocity, 1.0),
                metadata={"signal": signal, "source": signal.source},
            )
            candidates.append(topic)

        # Add evergreen niche topics (always-relevant topics)
        evergreen = self._get_evergreen_topics()
        candidates.extend(evergreen)

        # Score all candidates
        scored = []
        for topic in candidates:
            score = self._score_topic(topic, audience, goal)
            topic.relevance_score = score
            scored.append(topic)

        # Sort by score descending
        scored.sort(key=lambda t: t.relevance_score, reverse=True)

        # Apply diversity: don't select too-similar topics
        selected = self._apply_diversity(scored, count)

        # Track selected topics
        now = datetime.now()
        for topic in selected:
            self._recent_topics.append((topic.name.lower(), now))
        self._trim_recent()

        logger.info(
            "TopicSelector selected %d topics from %d candidates",
            len(selected),
            len(candidates),
        )
        return selected

    def suggest_content_type(self, topic: Topic, goal: ContentGoal) -> ContentType:
        """Suggest the best content type for a given topic and goal."""
        # Complex/educational topics -> threads
        if topic.category in ("tutorial", "analysis", "deep_dive"):
            return ContentType.THREAD

        # Engagement goals with trending topics -> short posts (quick, timely)
        if goal == ContentGoal.ENGAGEMENT and topic.trending_score > 0.7:
            return ContentType.SHORT_POST

        # Community building -> polls
        if goal == ContentGoal.COMMUNITY:
            return ContentType.POLL

        # Authority building -> long posts or threads
        if goal == ContentGoal.AUTHORITY:
            return random.choice([ContentType.THREAD, ContentType.LONG_POST])

        return ContentType.SHORT_POST

    def _score_topic(
        self,
        topic: Topic,
        audience: AudienceProfile | None,
        goal: ContentGoal,
    ) -> float:
        """
        Compute a weighted topic score.

        Similar to Phoenix WeightedScorer but for topic selection:
        Score = w_trend * trend_score + w_relevance * relevance + ...
        """
        # 1. Trend score
        trend_score = topic.trending_score

        # 2. Niche relevance
        relevance = self._compute_niche_relevance(topic)

        # 3. Audience fit
        audience_score = self._compute_audience_fit(topic, audience) if audience else 0.5

        # 4. Freshness (penalize recently covered topics)
        freshness = self._compute_freshness(topic)

        # 5. Goal alignment
        goal_score = self._compute_goal_alignment(topic, goal)

        final_score = (
            self.WEIGHT_TREND * trend_score
            + self.WEIGHT_RELEVANCE * relevance
            + self.WEIGHT_AUDIENCE * audience_score
            + self.WEIGHT_FRESHNESS * freshness
            + self.WEIGHT_GOAL * goal_score
        )

        return max(0.0, min(1.0, final_score))

    def _compute_niche_relevance(self, topic: Topic) -> float:
        """How relevant is this topic to the account's niche?"""
        if not self.config.niche_keywords:
            return 0.5
        niche_set = {kw.lower() for kw in self.config.niche_keywords}
        topic_words = set(topic.name.lower().split()) | {kw.lower() for kw in topic.keywords}
        overlap = topic_words & niche_set
        return min(len(overlap) / max(len(niche_set), 1) * 2, 1.0)

    def _compute_audience_fit(self, topic: Topic, audience: AudienceProfile) -> float:
        """How well does this topic match audience interests?"""
        if not audience.interests:
            return 0.5
        interest_set = {i.lower() for i in audience.interests}
        topic_words = set(topic.name.lower().split()) | {kw.lower() for kw in topic.keywords}
        overlap = topic_words & interest_set
        return min(len(overlap) / max(len(interest_set), 1) * 3, 1.0)

    def _compute_freshness(self, topic: Topic) -> float:
        """Penalize topics we've recently covered. 1.0 = fresh, 0.0 = just covered."""
        cutoff = datetime.now() - timedelta(days=3)
        recent = [
            t for t, dt in self._recent_topics if t == topic.name.lower() and dt >= cutoff
        ]
        if not recent:
            return 1.0
        # More recent coverage = lower freshness
        return max(0.0, 1.0 - len(recent) * 0.3)

    def _compute_goal_alignment(self, topic: Topic, goal: ContentGoal) -> float:
        """How well does this topic serve the current content goal?"""
        # Trending topics are good for reach and engagement
        if goal in (ContentGoal.REACH, ContentGoal.ENGAGEMENT):
            return topic.trending_score

        # Niche topics are good for authority and community
        if goal in (ContentGoal.AUTHORITY, ContentGoal.COMMUNITY):
            return self._compute_niche_relevance(topic)

        # Follower growth benefits from both trending and niche
        if goal == ContentGoal.FOLLOWER_GROWTH:
            return (topic.trending_score + self._compute_niche_relevance(topic)) / 2

        return 0.5

    def _get_evergreen_topics(self) -> list[Topic]:
        """Generate evergreen (always-relevant) topics from niche keywords."""
        return [
            Topic(
                name=keyword,
                keywords=[keyword],
                category="evergreen",
                trending_score=0.2,  # Low trending score but always available
                metadata={"source": "evergreen"},
            )
            for keyword in self.config.niche_keywords[:5]
        ]

    def _apply_diversity(self, topics: list[Topic], count: int) -> list[Topic]:
        """Ensure selected topics are diverse (not too similar to each other)."""
        selected: list[Topic] = []
        selected_words: set[str] = set()

        for topic in topics:
            if len(selected) >= count:
                break
            topic_words = set(topic.name.lower().split())
            # Skip if too much overlap with already-selected topics
            overlap = topic_words & selected_words
            if len(overlap) > len(topic_words) * 0.5 and selected:
                continue
            selected.append(topic)
            selected_words |= topic_words

        return selected

    def _trim_recent(self, max_entries: int = 200):
        cutoff = datetime.now() - timedelta(days=7)
        self._recent_topics = [
            (t, dt) for t, dt in self._recent_topics if dt >= cutoff
        ][-max_entries:]
