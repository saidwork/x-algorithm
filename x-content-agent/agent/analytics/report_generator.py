"""
Analytics report generation module.

Generates comprehensive performance reports for the content agent.
Reports can be daily, weekly, or monthly summaries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from agent.analytics.metrics_tracker import AggregateMetrics, MetricsTracker
from agent.config import AgentConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates formatted analytics reports.

    Report types:
    1. Daily Summary   - Quick overview of today's performance
    2. Weekly Report   - Comprehensive weekly analysis with trends
    3. Monthly Report  - Strategic monthly review with recommendations
    """

    def __init__(self, config: AgentConfig, metrics_tracker: MetricsTracker):
        self.config = config
        self._tracker = metrics_tracker

    def generate_daily_report(self) -> str:
        """Generate a daily performance summary."""
        metrics = self._tracker.get_aggregate("day")
        return self._format_report("Daily", metrics)

    def generate_weekly_report(self) -> str:
        """Generate a weekly performance report."""
        metrics = self._tracker.get_aggregate("week")
        perf_by_type = self._tracker.get_performance_by_type()
        perf_by_tone = self._tracker.get_performance_by_tone()
        top_posts = self._tracker.get_top_posts(count=3, days=7)

        lines = [self._format_report("Weekly", metrics)]

        if perf_by_type:
            lines.append("\n## Performance by Content Type")
            for ct, rate in sorted(perf_by_type.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {ct}: {rate:.2%} avg engagement")

        if perf_by_tone:
            lines.append("\n## Performance by Tone")
            for tone, rate in sorted(perf_by_tone.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {tone}: {rate:.2%} avg engagement")

        if top_posts:
            lines.append("\n## Top Performing Posts")
            for i, record in enumerate(top_posts, 1):
                m = record.metrics
                if m:
                    preview = record.draft.full_text[:80].replace("\n", " ")
                    lines.append(
                        f"  {i}. \"{preview}...\" - {m.total_engagements} engagements ({m.engagement_rate:.2%})"
                    )

        return "\n".join(lines)

    def generate_monthly_report(self) -> str:
        """Generate a monthly strategic report."""
        metrics = self._tracker.get_aggregate("month")
        lines = [self._format_report("Monthly", metrics)]

        # Add weekly comparison
        lines.append("\n## Weekly Trends")
        lines.append("  (Weekly breakdown of engagement rates over the month)")

        # Add recommendations
        lines.append("\n## Recommendations")
        perf_by_type = self._tracker.get_performance_by_type()
        if perf_by_type:
            best = max(perf_by_type, key=perf_by_type.get)
            worst = min(perf_by_type, key=perf_by_type.get)
            lines.append(f"  - Increase '{best}' content (top performer)")
            lines.append(f"  - Review '{worst}' content strategy (lowest performer)")

        return "\n".join(lines)

    def _format_report(self, period: str, metrics: AggregateMetrics) -> str:
        """Format a metrics aggregate into a readable report section."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# {period} Performance Report\n"
            f"Generated: {now}\n"
            f"Account: @{self.config.account_handle}\n"
            f"\n"
            f"## Summary\n"
            f"  - Total Posts: {metrics.total_posts}\n"
            f"  - Total Impressions: {metrics.total_impressions:,}\n"
            f"  - Total Engagements: {metrics.total_engagements:,}\n"
            f"  - Avg Engagement Rate: {metrics.avg_engagement_rate:.2%}\n"
            f"  - Avg Likes/Post: {metrics.avg_likes:.1f}\n"
            f"  - Avg Replies/Post: {metrics.avg_replies:.1f}\n"
            f"  - Avg Reposts/Post: {metrics.avg_reposts:.1f}\n"
            f"  - New Followers: {metrics.total_new_followers}"
        )
