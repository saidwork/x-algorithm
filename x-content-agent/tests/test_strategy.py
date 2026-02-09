"""Tests for the strategy module."""

from datetime import datetime

import pytest

from agent.config import AgentConfig
from agent.strategy.content_calendar import ContentCalendar
from agent.strategy.topic_selector import TopicSelector
from agent.strategy.trend_analyzer import TrendAnalyzer
from agent.types import AudienceProfile, ContentGoal, ContentType, TrendSignal


class TestTrendAnalyzer:
    def setup_method(self):
        self.config = AgentConfig(
            niche="AI",
            niche_keywords=["machine learning", "deep learning", "AI", "LLM"],
        )
        self.analyzer = TrendAnalyzer(self.config)

    @pytest.mark.asyncio
    async def test_analyze_returns_list(self):
        signals = await self.analyzer.analyze()
        assert isinstance(signals, list)

    def test_velocity_no_history(self):
        v = self.analyzer.get_velocity("some topic")
        assert v == 0.0

    def test_is_rising_no_data(self):
        assert not self.analyzer.is_rising("unknown topic")


class TestTopicSelector:
    def setup_method(self):
        self.config = AgentConfig(
            niche="AI",
            niche_keywords=["machine learning", "deep learning", "AI", "LLM", "GPT"],
        )
        self.selector = TopicSelector(self.config)

    def test_select_from_signals(self):
        signals = [
            TrendSignal(topic="GPT-5 release", velocity=0.8, volume=50000, category="AI"),
            TrendSignal(topic="Climate summit", velocity=0.6, volume=30000, category="politics"),
            TrendSignal(topic="New LLM benchmark", velocity=0.5, volume=10000, category="AI"),
        ]
        topics = self.selector.select_topics(signals, count=2)
        assert len(topics) <= 2
        # AI topics should rank higher due to niche relevance
        assert any("GPT" in t.name or "LLM" in t.name for t in topics)

    def test_select_with_audience(self):
        signals = [
            TrendSignal(topic="machine learning tips", velocity=0.5),
        ]
        audience = AudienceProfile(
            interests=["machine learning", "python"],
            avg_engagement_rate=0.05,
        )
        topics = self.selector.select_topics(signals, audience=audience, count=3)
        assert len(topics) > 0

    def test_suggest_content_type_thread(self):
        from agent.types import Topic

        topic = Topic(name="Tutorial", category="tutorial")
        ct = self.selector.suggest_content_type(topic, ContentGoal.AUTHORITY)
        assert ct in (ContentType.THREAD, ContentType.LONG_POST)

    def test_suggest_content_type_trending(self):
        from agent.types import Topic

        topic = Topic(name="Breaking AI news", trending_score=0.9)
        ct = self.selector.suggest_content_type(topic, ContentGoal.ENGAGEMENT)
        assert ct == ContentType.SHORT_POST

    def test_diversity_no_duplicates(self):
        signals = [
            TrendSignal(topic="AI topic", velocity=0.8),
            TrendSignal(topic="AI topic again", velocity=0.7),
            TrendSignal(topic="Different topic", velocity=0.5),
        ]
        topics = self.selector.select_topics(signals, count=5)
        names = [t.name for t in topics]
        # Should include diverse topics
        assert len(set(names)) == len(names)  # No exact duplicates


class TestContentCalendar:
    def setup_method(self):
        self.config = AgentConfig()
        self.config.strategy.posts_per_day = 3
        self.calendar = ContentCalendar(self.config)

    def test_generate_week_plan(self):
        plans = self.calendar.generate_week_plan()
        assert len(plans) == 7

        for plan in plans:
            assert len(plan.slots) > 0
            assert len(plan.slots) <= self.config.strategy.posts_per_day

    def test_get_next_empty_slot(self):
        self.calendar.generate_week_plan()
        slot = self.calendar.get_next_empty_slot()
        assert slot is not None
        assert not slot.is_filled

    def test_assign_draft(self):
        from agent.types import ContentPiece, PostDraft, PostStatus

        self.calendar.generate_week_plan()
        slot = self.calendar.get_next_empty_slot()
        assert slot is not None

        draft = PostDraft(id="test-draft", pieces=[ContentPiece(text="test")])
        self.calendar.assign_draft(slot, draft)

        assert slot.is_filled
        assert draft.status == PostStatus.SCHEDULED
        assert draft.scheduled_time is not None

    def test_get_gaps(self):
        self.calendar.generate_week_plan()
        gaps = self.calendar.get_gaps(days_ahead=7)
        assert len(gaps) > 0  # All slots should be empty initially

    def test_content_type_distribution(self):
        self.calendar.generate_week_plan()
        dist = self.calendar.get_content_type_distribution()
        # Initially empty since nothing is filled
        assert isinstance(dist, dict)
