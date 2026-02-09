"""
Feedback loop module.

Closes the loop between content performance and strategy adjustment.
Analyzes what worked and what didn't, then adjusts the agent's behavior:
- Content type mix ratios
- Tone preferences
- Topic selection weights
- Posting time preferences
- Template effectiveness

This is the content agent's equivalent of Phoenix model retraining -
continuously improving based on real engagement data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.analytics.metrics_tracker import MetricsTracker, PostRecord
from agent.config import AgentConfig
from agent.types import ContentGoal, ContentTone, ContentType

logger = logging.getLogger(__name__)


@dataclass
class StrategyAdjustment:
    """A recommended adjustment to the content strategy."""

    adjustment_type: str  # "content_mix", "tone_mix", "posting_time", "topic_weight"
    parameter: str  # Which specific parameter to adjust
    current_value: float
    recommended_value: float
    reason: str
    confidence: float  # 0.0 - 1.0
    data_points: int


@dataclass
class FeedbackReport:
    """Report from a feedback analysis cycle."""

    analyzed_at: datetime
    posts_analyzed: int
    period_days: int
    adjustments: list[StrategyAdjustment] = field(default_factory=list)
    key_insights: list[str] = field(default_factory=list)
    overall_trend: str = ""  # "improving", "stable", "declining"


class FeedbackLoop:
    """
    Continuous improvement system that adjusts strategy based on performance.

    The feedback loop runs periodically and:
    1. Analyzes recent post performance by various dimensions
    2. Compares actual vs predicted engagement
    3. Identifies patterns in high/low performing content
    4. Generates strategy adjustments
    5. Applies adjustments to the agent config

    This creates a self-improving system where the agent gets better
    at content creation over time, similar to how the recommendation
    system improves its ranking model through retraining.
    """

    def __init__(self, config: AgentConfig, metrics_tracker: MetricsTracker):
        self.config = config
        self._tracker = metrics_tracker
        self._history: list[FeedbackReport] = []

    async def run_analysis(self, period_days: int = 7) -> FeedbackReport:
        """
        Run a full feedback analysis cycle.

        Analyzes the last N days of performance and generates
        strategy adjustment recommendations.
        """
        report = FeedbackReport(
            analyzed_at=datetime.now(),
            posts_analyzed=0,
            period_days=period_days,
        )

        # Analyze by content type
        type_adjustments = self._analyze_content_types(period_days)
        report.adjustments.extend(type_adjustments)

        # Analyze by tone
        tone_adjustments = self._analyze_tones(period_days)
        report.adjustments.extend(tone_adjustments)

        # Analyze prediction accuracy
        prediction_insights = self._analyze_prediction_accuracy()
        report.key_insights.extend(prediction_insights)

        # Determine overall trend
        report.overall_trend = self._compute_trend(period_days)

        # Generate key insights
        report.key_insights.extend(self._generate_insights(period_days))

        self._history.append(report)

        logger.info(
            "Feedback analysis complete: %d adjustments, trend=%s",
            len(report.adjustments),
            report.overall_trend,
        )
        return report

    def apply_adjustments(self, report: FeedbackReport, auto_apply: bool = False) -> list[str]:
        """
        Apply strategy adjustments from a feedback report.

        Args:
            report: The feedback report with adjustments
            auto_apply: If True, apply automatically. If False, return descriptions only.

        Returns:
            List of applied/recommended adjustment descriptions.
        """
        descriptions = []
        significant = [a for a in report.adjustments if a.confidence >= 0.7]

        for adj in significant:
            desc = (
                f"[{adj.adjustment_type}] {adj.parameter}: "
                f"{adj.current_value:.2f} -> {adj.recommended_value:.2f} "
                f"({adj.reason})"
            )
            descriptions.append(desc)

            if auto_apply:
                self._apply_single_adjustment(adj)

        return descriptions

    def _analyze_content_types(self, period_days: int) -> list[StrategyAdjustment]:
        """Analyze performance by content type and suggest mix adjustments."""
        adjustments = []
        perf_by_type = self._tracker.get_performance_by_type()

        if not perf_by_type:
            return adjustments

        avg_rate = sum(perf_by_type.values()) / len(perf_by_type)
        current_mix = self.config.strategy.content_mix

        for type_str, rate in perf_by_type.items():
            current_ratio = current_mix.get(type_str, 0.1)

            # If this type performs significantly above average, increase its share
            if rate > avg_rate * 1.3:
                new_ratio = min(current_ratio * 1.2, 0.5)
                adjustments.append(
                    StrategyAdjustment(
                        adjustment_type="content_mix",
                        parameter=type_str,
                        current_value=current_ratio,
                        recommended_value=new_ratio,
                        reason=f"Engagement rate ({rate:.3f}) is {rate / max(avg_rate, 0.001):.1f}x above average",
                        confidence=0.7,
                        data_points=10,
                    )
                )
            # If significantly below average, decrease its share
            elif rate < avg_rate * 0.7:
                new_ratio = max(current_ratio * 0.8, 0.05)
                adjustments.append(
                    StrategyAdjustment(
                        adjustment_type="content_mix",
                        parameter=type_str,
                        current_value=current_ratio,
                        recommended_value=new_ratio,
                        reason=f"Engagement rate ({rate:.3f}) is below average ({avg_rate:.3f})",
                        confidence=0.6,
                        data_points=10,
                    )
                )

        return adjustments

    def _analyze_tones(self, period_days: int) -> list[StrategyAdjustment]:
        """Analyze performance by tone and suggest adjustments."""
        adjustments = []
        perf_by_tone = self._tracker.get_performance_by_tone()

        if not perf_by_tone:
            return adjustments

        avg_rate = sum(perf_by_tone.values()) / len(perf_by_tone)
        current_mix = self.config.strategy.tone_mix

        for tone_str, rate in perf_by_tone.items():
            current_ratio = current_mix.get(tone_str, 0.1)

            if rate > avg_rate * 1.3:
                new_ratio = min(current_ratio * 1.15, 0.4)
                adjustments.append(
                    StrategyAdjustment(
                        adjustment_type="tone_mix",
                        parameter=tone_str,
                        current_value=current_ratio,
                        recommended_value=new_ratio,
                        reason=f"Tone '{tone_str}' engagement ({rate:.3f}) exceeds average by {((rate / max(avg_rate, 0.001)) - 1) * 100:.0f}%",
                        confidence=0.65,
                        data_points=10,
                    )
                )

        return adjustments

    def _analyze_prediction_accuracy(self) -> list[str]:
        """Compare predicted vs actual engagement to calibrate the predictor."""
        insights = []

        top_posts = self._tracker.get_top_posts(count=5)
        for record in top_posts:
            if record.metrics and record.draft.engagement_prediction:
                predicted = record.draft.engagement_prediction.predicted_engagement_rate
                actual = record.metrics.engagement_rate
                if actual > 0 and predicted > 0:
                    ratio = actual / predicted
                    if ratio > 2:
                        insights.append(
                            f"Post '{record.draft.id}' performed {ratio:.1f}x better than predicted "
                            f"- model may be underestimating this content type"
                        )
                    elif ratio < 0.5:
                        insights.append(
                            f"Post '{record.draft.id}' performed {ratio:.1f}x vs prediction "
                            f"- model may be overestimating this content type"
                        )

        return insights

    def _compute_trend(self, period_days: int) -> str:
        """Determine if overall performance is improving, stable, or declining."""
        current = self._tracker.get_aggregate("week")
        # Compare with previous period (rough approximation)
        if current.avg_engagement_rate > 0.05:
            return "improving"
        elif current.avg_engagement_rate > 0.02:
            return "stable"
        else:
            return "needs_data"  # Not enough data to determine

    def _generate_insights(self, period_days: int) -> list[str]:
        """Generate human-readable insights from the data."""
        insights = []

        # Best performing content type
        perf_by_type = self._tracker.get_performance_by_type()
        if perf_by_type:
            best_type = max(perf_by_type, key=perf_by_type.get)
            insights.append(
                f"Best performing content type: {best_type} "
                f"({perf_by_type[best_type]:.2%} avg engagement)"
            )

        # Best performing tone
        perf_by_tone = self._tracker.get_performance_by_tone()
        if perf_by_tone:
            best_tone = max(perf_by_tone, key=perf_by_tone.get)
            insights.append(
                f"Best performing tone: {best_tone} "
                f"({perf_by_tone[best_tone]:.2%} avg engagement)"
            )

        return insights

    def _apply_single_adjustment(self, adj: StrategyAdjustment) -> None:
        """Apply a single strategy adjustment to the config."""
        if adj.adjustment_type == "content_mix":
            self.config.strategy.content_mix[adj.parameter] = adj.recommended_value
        elif adj.adjustment_type == "tone_mix":
            self.config.strategy.tone_mix[adj.parameter] = adj.recommended_value
