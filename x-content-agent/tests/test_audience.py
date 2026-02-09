"""Tests for the audience analysis module."""

import pytest

from agent.audience.engagement_predictor import EngagementPredictor
from agent.audience.persona_builder import PersonaBuilder
from agent.audience.timing_optimizer import TimingOptimizer
from agent.config import AgentConfig
from agent.types import (
    AudienceProfile,
    ContentPiece,
    ContentRequest,
    ContentType,
    PerformanceMetrics,
    PostDraft,
    Topic,
)
from datetime import datetime


class TestEngagementPredictor:
    def setup_method(self):
        self.config = AgentConfig(niche="AI", niche_keywords=["AI", "ML"])
        self.predictor = EngagementPredictor(self.config)

    def test_predict_returns_prediction(self):
        draft = PostDraft(
            id="t1",
            pieces=[ContentPiece(text="AI is transforming everything. What do you think?")],
            topic=Topic(name="AI", trending_score=0.8),
        )
        prediction = self.predictor.predict(draft)
        assert prediction.p_like > 0
        assert prediction.p_reply > 0
        assert prediction.predicted_engagement_rate > 0

    def test_question_boosts_reply(self):
        no_q = PostDraft(id="nq", pieces=[ContentPiece(text="AI is great.")])
        with_q = PostDraft(id="wq", pieces=[ContentPiece(text="Is AI great?")])

        pred_no_q = self.predictor.predict(no_q)
        pred_with_q = self.predictor.predict(with_q)

        assert pred_with_q.p_reply > pred_no_q.p_reply

    def test_thread_boosts_repost(self):
        single = PostDraft(id="s", pieces=[ContentPiece(text="Single post")])
        thread = PostDraft(
            id="t",
            pieces=[ContentPiece(text="Thread 1"), ContentPiece(text="Thread 2")],
        )

        pred_single = self.predictor.predict(single)
        pred_thread = self.predictor.predict(thread)

        assert pred_thread.p_repost > pred_single.p_repost

    def test_audience_multiplier(self):
        draft = PostDraft(
            id="am",
            pieces=[ContentPiece(text="Content about AI")],
            topic=Topic(name="AI"),
        )
        audience = AudienceProfile(
            total_followers=100000,
            avg_engagement_rate=0.08,
            top_performing_topics=["AI"],
        )

        pred_no_audience = self.predictor.predict(draft)
        pred_with_audience = self.predictor.predict(draft, audience)

        assert pred_with_audience.predicted_impressions > pred_no_audience.predicted_impressions

    def test_negative_signals_low(self):
        good_draft = PostDraft(
            id="good",
            pieces=[ContentPiece(text="Helpful AI tips for everyone. What's your favorite?")],
            topic=Topic(name="AI", trending_score=0.5),
        )
        prediction = self.predictor.predict(good_draft)

        # Negative signals should be very low for good content
        assert prediction.p_mute < 0.01
        assert prediction.p_unfollow < 0.01
        assert prediction.p_report < 0.01

    @pytest.mark.asyncio
    async def test_score_interface(self):
        drafts = [
            PostDraft(id="d1", pieces=[ContentPiece(text="Post 1")]),
            PostDraft(id="d2", pieces=[ContentPiece(text="Post 2")]),
        ]
        request = ContentRequest(
            request_id="r1", account_id="a1", topic=Topic(name="test")
        )
        predictions = await self.predictor.score(request, drafts)
        assert len(predictions) == 2


class TestTimingOptimizer:
    def setup_method(self):
        self.config = AgentConfig()
        self.optimizer = TimingOptimizer(self.config)

    def test_get_optimal_times(self):
        times = self.optimizer.get_optimal_times(count=3)
        assert len(times) == 3

        # Should be in score order
        scores = [t.score for t in times]
        assert scores == sorted(scores, reverse=True)

    def test_minimum_gap(self):
        times = self.optimizer.get_optimal_times(count=5)
        for i in range(1, len(times)):
            gap = (times[i].datetime_utc - times[i - 1].datetime_utc).total_seconds()
            min_gap = self.config.strategy.min_hours_between_posts * 3600
            assert abs(gap) >= min_gap

    def test_update_from_metrics(self):
        posted_at = datetime(2025, 1, 15, 10, 0, 0)  # Wednesday 10:00
        metrics = PerformanceMetrics(
            post_id="p1",
            impressions=10000,
            likes=500,
            engagement_rate=0.08,
        )
        self.optimizer.update_from_metrics(metrics, posted_at)

        # The slot should now have data
        # Wednesday = day 2, hour 10
        slot = self.optimizer._score_matrix.get((2, 10))
        assert slot is not None
        assert slot.sample_size >= 1

    def test_content_type_affects_timing(self):
        thread_times = self.optimizer.get_optimal_times(
            content_type=ContentType.THREAD, count=1
        )
        post_times = self.optimizer.get_optimal_times(
            content_type=ContentType.SHORT_POST, count=1
        )
        # Both should return results
        assert len(thread_times) == 1
        assert len(post_times) == 1


class TestPersonaBuilder:
    def setup_method(self):
        self.config = AgentConfig(niche="AI", niche_keywords=["AI", "ML"])
        self.builder = PersonaBuilder(self.config)

    def test_build_personas(self):
        audience = AudienceProfile(
            total_followers=5000,
            interests=["AI", "Python", "startups"],
        )
        personas = self.builder.build_personas(audience)
        assert len(personas) == 4  # Default 4 personas
        assert all(p.name for p in personas)
        assert all(p.description for p in personas)

    def test_personas_customized_with_niche(self):
        audience = AudienceProfile(interests=["AI"])
        personas = self.builder.build_personas(audience)
        assert any("AI" in p.description for p in personas)

    def test_format_for_prompt(self):
        audience = AudienceProfile(interests=["AI", "coding"])
        personas = self.builder.build_personas(audience)
        formatted = self.builder.format_for_prompt(personas[0])
        assert "Target Reader" in formatted
        assert personas[0].name in formatted
