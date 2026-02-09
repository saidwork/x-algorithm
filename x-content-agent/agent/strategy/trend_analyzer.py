"""
Trend analysis module.

Monitors X platform trends, news feeds, and niche-specific signals to identify
content opportunities. Acts as a TopicSource in the pipeline - analogous to how
Thunder and Phoenix act as candidate sources in the recommendation system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.types import ContentRequest, TrendSignal

logger = logging.getLogger(__name__)


@dataclass
class TrendSnapshot:
    """Point-in-time snapshot of trending topics."""

    signals: list[TrendSignal] = field(default_factory=list)
    captured_at: datetime = field(default_factory=datetime.now)
    source: str = ""


class TrendAnalyzer:
    """
    Analyzes trends across multiple sources to identify content opportunities.

    Sources:
    1. X Trending Topics - Platform-native trending hashtags/topics
    2. Niche Keywords   - Monitoring specific keywords relevant to the account
    3. News Feeds       - External news that may be relevant
    4. Competitor Posts  - What competitors are talking about

    The analyzer scores each trend by:
    - Volume: How many people are discussing it
    - Velocity: How fast the discussion is growing
    - Relevance: How relevant it is to the account's niche
    - Sentiment: Overall sentiment (positive/negative/neutral)
    - Timing: Is this trend peaking, rising, or declining?
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._history: list[TrendSnapshot] = []
        self._niche_keywords = config.niche_keywords

    @property
    def name(self) -> str:
        return "TrendAnalyzer"

    def enabled(self, request: ContentRequest) -> bool:
        return self.config.enable_trend_monitoring

    async def analyze(self) -> list[TrendSignal]:
        """
        Fetch and analyze trends from all sources.
        Returns scored and ranked trend signals.
        """
        all_signals: list[TrendSignal] = []

        # Source 1: Platform trending topics
        platform_trends = await self._fetch_platform_trends()
        all_signals.extend(platform_trends)

        # Source 2: Niche keyword monitoring
        niche_signals = await self._monitor_niche_keywords()
        all_signals.extend(niche_signals)

        # Source 3: News-based trends
        news_signals = await self._fetch_news_trends()
        all_signals.extend(news_signals)

        # Score and rank all signals
        scored = self._score_signals(all_signals)

        # Deduplicate overlapping trends
        deduped = self._deduplicate(scored)

        # Store snapshot for historical analysis
        snapshot = TrendSnapshot(signals=deduped, source="combined")
        self._history.append(snapshot)
        self._trim_history()

        logger.info("TrendAnalyzer found %d trend signals", len(deduped))
        return deduped

    async def get_topics(self, request: ContentRequest) -> list[dict]:
        """Pipeline-compatible topic source interface."""
        signals = await self.analyze()
        return [
            {
                "topic": s.topic,
                "score": s.velocity * 0.4 + (s.volume / max(s.volume, 1)) * 0.3 + abs(s.sentiment) * 0.3,
                "signal": s,
            }
            for s in signals
        ]

    def get_velocity(self, topic: str, window_hours: int = 6) -> float:
        """Calculate how fast a topic is growing over a time window."""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [
            s
            for snap in self._history
            if snap.captured_at >= cutoff
            for s in snap.signals
            if s.topic.lower() == topic.lower()
        ]
        if len(recent) < 2:
            return 0.0
        volumes = [s.volume for s in recent]
        return (volumes[-1] - volumes[0]) / max(volumes[0], 1)

    def is_rising(self, topic: str) -> bool:
        """Check if a topic trend is still rising (not peaked yet)."""
        return self.get_velocity(topic) > 0.1

    async def _fetch_platform_trends(self) -> list[TrendSignal]:
        """
        Fetch trending topics from X platform.

        In production, this would call the X API v2 /trends endpoint.
        The implementation provides the interface; actual API integration
        is handled by the orchestrator's HTTP client.
        """
        # Interface for X API integration
        # GET /2/trends/by/woeid/:woeid
        return []

    async def _monitor_niche_keywords(self) -> list[TrendSignal]:
        """
        Monitor specific keywords relevant to the account's niche.

        Uses X API search/recent endpoint to track volume of niche keywords
        and detect spikes in conversation.
        """
        signals = []
        for keyword in self._niche_keywords:
            signal = TrendSignal(
                topic=keyword,
                source="niche_monitor",
                category=self.config.niche,
            )
            signals.append(signal)
        return signals

    async def _fetch_news_trends(self) -> list[TrendSignal]:
        """
        Fetch trending news topics from external sources.
        Can integrate with news APIs (Google News, NewsAPI, etc.)
        """
        return []

    def _score_signals(self, signals: list[TrendSignal]) -> list[TrendSignal]:
        """
        Score each signal by relevance to the account's niche.
        Higher scores = more relevant and timely.
        """
        niche_set = {kw.lower() for kw in self._niche_keywords}

        for signal in signals:
            # Relevance boost: if topic matches niche keywords
            topic_words = set(signal.topic.lower().split())
            overlap = topic_words & niche_set
            niche_boost = len(overlap) / max(len(niche_set), 1)

            # Recency boost: trends detected more recently get higher scores
            signal.velocity = signal.velocity + niche_boost * 0.5

            # Add related niche keywords
            signal.related_topics = list(
                set(signal.related_topics) | {kw for kw in self._niche_keywords if kw.lower() in signal.topic.lower()}
            )

        return sorted(signals, key=lambda s: s.velocity, reverse=True)

    def _deduplicate(self, signals: list[TrendSignal]) -> list[TrendSignal]:
        """Remove duplicate/overlapping trend signals."""
        seen_topics: set[str] = set()
        unique: list[TrendSignal] = []
        for signal in signals:
            normalized = signal.topic.lower().strip()
            if normalized not in seen_topics:
                seen_topics.add(normalized)
                unique.append(signal)
        return unique

    def _trim_history(self, max_snapshots: int = 500):
        """Keep history bounded to prevent memory growth."""
        if len(self._history) > max_snapshots:
            self._history = self._history[-max_snapshots:]
