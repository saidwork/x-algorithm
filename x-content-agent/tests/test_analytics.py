"""Tests for the analytics module."""

from datetime import datetime, timedelta

import pytest

from agent.analytics.ab_tester import ABTester
from agent.analytics.feedback_loop import FeedbackLoop
from agent.analytics.metrics_tracker import MetricsTracker
from agent.analytics.report_generator import ReportGenerator
from agent.config import AgentConfig
from agent.types import (
    ContentPiece,
    ContentTone,
    ContentType,
    PerformanceMetrics,
    PostDraft,
)


class TestMetricsTracker:
    def setup_method(self):
        self.tracker = MetricsTracker()

    def test_record_and_update(self):
        draft = PostDraft(id="p1", pieces=[ContentPiece(text="Test post")])
        self.tracker.record_publication(draft)
        self.tracker.update_metrics(
            "p1",
            PerformanceMetrics(
                post_id="p1",
                impressions=10000,
                likes=500,
                replies=50,
                reposts=100,
                engagement_rate=0.065,
            ),
        )

        agg = self.tracker.get_aggregate("week")
        assert agg.total_posts == 1
        assert agg.total_impressions == 10000
        assert agg.avg_likes == 500

    def test_top_posts(self):
        for i in range(5):
            draft = PostDraft(id=f"p{i}", pieces=[ContentPiece(text=f"Post {i}")])
            self.tracker.record_publication(draft)
            self.tracker.update_metrics(
                f"p{i}",
                PerformanceMetrics(
                    post_id=f"p{i}",
                    likes=i * 100,
                    replies=i * 10,
                    reposts=i * 20,
                    engagement_rate=i * 0.01,
                ),
            )

        top = self.tracker.get_top_posts(count=2)
        assert len(top) == 2
        assert top[0].metrics.likes >= top[1].metrics.likes

    def test_performance_by_type(self):
        d1 = PostDraft(
            id="t1",
            pieces=[ContentPiece(text="Short", content_type=ContentType.SHORT_POST)],
        )
        d2 = PostDraft(
            id="t2",
            pieces=[ContentPiece(text="Thread 1", content_type=ContentType.THREAD),
                    ContentPiece(text="Thread 2", content_type=ContentType.THREAD)],
        )

        self.tracker.record_publication(d1)
        self.tracker.record_publication(d2)
        self.tracker.update_metrics(
            "t1", PerformanceMetrics(post_id="t1", engagement_rate=0.03)
        )
        self.tracker.update_metrics(
            "t2", PerformanceMetrics(post_id="t2", engagement_rate=0.08)
        )

        by_type = self.tracker.get_performance_by_type()
        assert ContentType.SHORT_POST.value in by_type
        assert ContentType.THREAD.value in by_type
        assert by_type[ContentType.THREAD.value] > by_type[ContentType.SHORT_POST.value]

    def test_performance_by_tone(self):
        d1 = PostDraft(id="tn1", pieces=[ContentPiece(text="Casual")], tone=ContentTone.CASUAL)
        d2 = PostDraft(id="tn2", pieces=[ContentPiece(text="Pro")], tone=ContentTone.PROFESSIONAL)

        self.tracker.record_publication(d1)
        self.tracker.record_publication(d2)
        self.tracker.update_metrics(
            "tn1", PerformanceMetrics(post_id="tn1", engagement_rate=0.05)
        )
        self.tracker.update_metrics(
            "tn2", PerformanceMetrics(post_id="tn2", engagement_rate=0.03)
        )

        by_tone = self.tracker.get_performance_by_tone()
        assert ContentTone.CASUAL.value in by_tone


class TestABTester:
    def setup_method(self):
        self.config = AgentConfig()
        self.tester = ABTester(self.config)

    def test_create_test(self):
        test = self.tester.create_test(
            name="Tone Test",
            hypothesis="Casual tone gets more engagement than professional",
            variable="tone",
            variant_names=["casual", "professional"],
        )
        assert test.test_id.startswith("ab-")
        assert len(test.variants) == 2
        assert test.is_active

    def test_assign_variant(self):
        test = self.tester.create_test(
            name="Hook Test",
            hypothesis="Questions get more clicks",
            variable="hook_type",
            variant_names=["question", "statement"],
        )

        draft = PostDraft(id="d1", pieces=[ContentPiece(text="Test")])
        variant = self.tester.assign_variant(test.test_id, draft)

        assert variant is not None
        assert draft.variant_group == variant.variant_id

    def test_analyze_insufficient_data(self):
        test = self.tester.create_test(
            name="Test",
            hypothesis="Test",
            variable="tone",
            variant_names=["a", "b"],
        )
        result = self.tester.analyze_test(test.test_id)
        assert result is None  # Not enough data

    def test_get_active_tests(self):
        self.tester.create_test("T1", "H1", "tone", ["a", "b"])
        self.tester.create_test("T2", "H2", "hook", ["x", "y"])

        active = self.tester.get_active_tests()
        assert len(active) == 2


class TestFeedbackLoop:
    def setup_method(self):
        self.config = AgentConfig(niche="AI", niche_keywords=["AI"])
        self.tracker = MetricsTracker()
        self.loop = FeedbackLoop(self.config, self.tracker)

    @pytest.mark.asyncio
    async def test_run_analysis(self):
        report = await self.loop.run_analysis()
        assert report is not None
        assert report.analyzed_at is not None

    @pytest.mark.asyncio
    async def test_analysis_with_data(self):
        # Add some performance data
        for i in range(5):
            ct = ContentType.SHORT_POST if i % 2 == 0 else ContentType.THREAD
            draft = PostDraft(
                id=f"fb{i}",
                pieces=[ContentPiece(text=f"Post {i}", content_type=ct)],
                tone=ContentTone.CASUAL if i % 2 == 0 else ContentTone.PROFESSIONAL,
            )
            self.tracker.record_publication(draft)
            self.tracker.update_metrics(
                f"fb{i}",
                PerformanceMetrics(post_id=f"fb{i}", engagement_rate=0.01 * (i + 1)),
            )

        report = await self.loop.run_analysis()
        # Should have some insights since we have data
        assert isinstance(report.key_insights, list)


class TestReportGenerator:
    def setup_method(self):
        self.config = AgentConfig(account_handle="testaccount")
        self.tracker = MetricsTracker()
        self.reporter = ReportGenerator(self.config, self.tracker)

    def test_daily_report_empty(self):
        report = self.reporter.generate_daily_report()
        assert "Daily Performance Report" in report
        assert "@testaccount" in report

    def test_weekly_report_with_data(self):
        draft = PostDraft(id="rp1", pieces=[ContentPiece(text="Weekly post")])
        self.tracker.record_publication(draft)
        self.tracker.update_metrics(
            "rp1",
            PerformanceMetrics(
                post_id="rp1",
                impressions=5000,
                likes=200,
                replies=30,
                reposts=50,
                engagement_rate=0.056,
            ),
        )

        report = self.reporter.generate_weekly_report()
        assert "Weekly Performance Report" in report

    def test_monthly_report(self):
        report = self.reporter.generate_monthly_report()
        assert "Monthly Performance Report" in report
        assert "Recommendations" in report
