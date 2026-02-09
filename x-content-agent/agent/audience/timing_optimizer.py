"""
Posting time optimization module.

Analyzes historical engagement data to determine the optimal times to post
on X for maximum reach and engagement. Considers audience activity patterns,
competitor posting schedules, and content type.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.types import AudienceProfile, ContentType, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class TimeSlotScore:
    """Engagement score for a specific time slot."""

    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int  # 0-23 UTC
    score: float = 0.0
    sample_size: int = 0
    avg_engagement_rate: float = 0.0
    avg_impressions: int = 0


@dataclass
class OptimalPostingTime:
    """Recommended posting time with confidence."""

    datetime_utc: datetime
    score: float
    confidence: float  # 0.0 - 1.0 based on data quality
    reason: str


class TimingOptimizer:
    """
    Optimizes posting times for maximum engagement.

    Uses a time-slot scoring matrix (7 days x 24 hours) populated from:
    1. Historical performance data (own posts)
    2. Audience activity patterns (when followers are online)
    3. Competitor dead zones (when competitors aren't posting)
    4. Content-type adjustments (threads do better at different times)

    The matrix is continuously updated as new performance data arrives,
    creating a feedback loop similar to how Phoenix retrains on new
    engagement data.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # 7 days x 24 hours scoring matrix
        self._score_matrix: dict[tuple[int, int], TimeSlotScore] = {}
        self._initialize_matrix()

    def get_optimal_times(
        self,
        content_type: ContentType = ContentType.SHORT_POST,
        count: int = 3,
        after: datetime | None = None,
    ) -> list[OptimalPostingTime]:
        """
        Get the best times to post within the next 7 days.

        Args:
            content_type: Type of content to post
            count: Number of time suggestions to return
            after: Only return times after this datetime
        """
        start = after or datetime.now()
        candidates: list[OptimalPostingTime] = []

        for days_ahead in range(7):
            target_date = start + timedelta(days=days_ahead)
            day_of_week = target_date.weekday()

            for hour in range(24):
                slot = self._score_matrix.get((day_of_week, hour))
                if not slot:
                    continue

                post_time = target_date.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )
                if post_time <= start:
                    continue

                # Apply content type adjustment
                adjusted_score = slot.score * self._content_type_time_modifier(
                    content_type, hour, day_of_week
                )

                confidence = min(slot.sample_size / 20, 1.0)  # More data = higher confidence

                candidates.append(
                    OptimalPostingTime(
                        datetime_utc=post_time,
                        score=adjusted_score,
                        confidence=confidence,
                        reason=self._explain_score(slot, content_type),
                    )
                )

        # Sort by score, return top N
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Ensure minimum gap between recommendations
        min_gap = timedelta(hours=self.config.strategy.min_hours_between_posts)
        filtered: list[OptimalPostingTime] = []
        for candidate in candidates:
            if all(
                abs((candidate.datetime_utc - existing.datetime_utc).total_seconds())
                >= min_gap.total_seconds()
                for existing in filtered
            ):
                filtered.append(candidate)
                if len(filtered) >= count:
                    break

        return filtered

    def update_from_metrics(self, metrics: PerformanceMetrics, posted_at: datetime) -> None:
        """
        Update the scoring matrix with actual performance data.
        This creates the feedback loop for continuous improvement.
        """
        day = posted_at.weekday()
        hour = posted_at.hour
        key = (day, hour)

        slot = self._score_matrix.get(key)
        if not slot:
            slot = TimeSlotScore(day_of_week=day, hour=hour)
            self._score_matrix[key] = slot

        # Running average update
        n = slot.sample_size
        slot.avg_engagement_rate = (
            slot.avg_engagement_rate * n + metrics.engagement_rate
        ) / (n + 1)
        slot.avg_impressions = int(
            (slot.avg_impressions * n + metrics.impressions) / (n + 1)
        )
        slot.sample_size = n + 1
        slot.score = slot.avg_engagement_rate * 0.6 + (slot.avg_impressions / 10000) * 0.4

        logger.debug(
            "Updated time slot (%s, %02d:00) with new metrics. Score: %.4f (n=%d)",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day],
            hour,
            slot.score,
            slot.sample_size,
        )

    def _initialize_matrix(self) -> None:
        """Initialize the scoring matrix with baseline scores."""
        # Baseline scores based on general X engagement patterns
        # Peak hours: 9-12 and 17-21 UTC on weekdays
        for day in range(7):
            for hour in range(24):
                score = 0.3  # Base score

                is_weekday = day < 5

                # Morning peak (9-12 UTC)
                if 9 <= hour <= 12:
                    score += 0.3 if is_weekday else 0.15

                # Evening peak (17-21 UTC)
                if 17 <= hour <= 21:
                    score += 0.25 if is_weekday else 0.2

                # Lunch peak (12-14 UTC)
                if 12 <= hour <= 14:
                    score += 0.15

                # Night penalty (0-6 UTC)
                if 0 <= hour <= 6:
                    score *= 0.3

                # Weekend adjustment
                if not is_weekday:
                    # Weekends: late morning is best
                    if 10 <= hour <= 14:
                        score += 0.1

                self._score_matrix[(day, hour)] = TimeSlotScore(
                    day_of_week=day,
                    hour=hour,
                    score=max(0.0, min(1.0, score)),
                )

    def _content_type_time_modifier(
        self, content_type: ContentType, hour: int, day: int
    ) -> float:
        """Adjust time score based on content type."""
        # Threads perform better in morning (people have time to read)
        if content_type == ContentType.THREAD:
            if 8 <= hour <= 11:
                return 1.3
            if 22 <= hour or hour <= 5:
                return 0.6

        # Polls do well in afternoon (decision fatigue leads to quick engagement)
        if content_type == ContentType.POLL:
            if 14 <= hour <= 18:
                return 1.2

        # News/timely content should go out ASAP, no time adjustment
        if content_type == ContentType.QUOTE_POST:
            return 1.0

        return 1.0

    def _explain_score(self, slot: TimeSlotScore, content_type: ContentType) -> str:
        """Generate human-readable explanation for a time recommendation."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = day_names[slot.day_of_week]
        reasons = []

        if 9 <= slot.hour <= 12:
            reasons.append("morning peak activity window")
        elif 17 <= slot.hour <= 21:
            reasons.append("evening peak activity window")
        elif 12 <= slot.hour <= 14:
            reasons.append("lunch-break browsing window")

        if slot.day_of_week < 5:
            reasons.append("weekday (higher professional engagement)")
        else:
            reasons.append("weekend (more casual browsing)")

        if slot.sample_size > 10:
            reasons.append(f"backed by {slot.sample_size} data points")

        reason = f"{day_name} at {slot.hour:02d}:00 UTC"
        if reasons:
            reason += f" - {', '.join(reasons)}"
        return reason
