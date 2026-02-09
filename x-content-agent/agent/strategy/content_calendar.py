"""
Content calendar module.

Manages the scheduling of content across time, ensuring optimal posting
cadence, content type diversity, and alignment with the content strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.types import ContentGoal, ContentTone, ContentType, PostDraft, PostStatus

logger = logging.getLogger(__name__)


@dataclass
class CalendarSlot:
    """A scheduled slot in the content calendar."""

    scheduled_time: datetime
    content_type: ContentType | None = None
    tone: ContentTone | None = None
    goal: ContentGoal | None = None
    topic_hint: str = ""
    draft: PostDraft | None = None
    is_filled: bool = False


@dataclass
class DayPlan:
    """Planned content for a single day."""

    date: datetime
    slots: list[CalendarSlot] = field(default_factory=list)

    @property
    def filled_count(self) -> int:
        return sum(1 for s in self.slots if s.is_filled)

    @property
    def empty_count(self) -> int:
        return sum(1 for s in self.slots if not s.is_filled)


class ContentCalendar:
    """
    Manages content scheduling and calendar planning.

    Responsibilities:
    - Generate optimal time slots based on audience activity patterns
    - Ensure content type and tone diversity across the week
    - Track scheduled vs published content
    - Identify gaps in the content calendar
    - Prevent posting too frequently or too infrequently
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._calendar: dict[str, DayPlan] = {}  # date_str -> DayPlan

    def generate_week_plan(
        self,
        start_date: datetime | None = None,
        peak_hours: list[int] | None = None,
    ) -> list[DayPlan]:
        """
        Generate a week-long content plan with optimal time slots.

        Args:
            start_date: Start of the week (defaults to today)
            peak_hours: Audience peak activity hours in UTC

        Returns:
            List of DayPlan objects for 7 days.
        """
        start = start_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if peak_hours is None:
            peak_hours = list(range(
                self.config.strategy.active_hours_utc[0],
                self.config.strategy.active_hours_utc[1],
            ))

        plans: list[DayPlan] = []
        content_type_pool = self._build_content_type_pool()
        tone_pool = self._build_tone_pool()

        type_idx = 0
        tone_idx = 0

        for day_offset in range(7):
            day = start + timedelta(days=day_offset)
            slots = self._generate_day_slots(day, peak_hours)

            # Assign content types and tones to slots
            for slot in slots:
                slot.content_type = content_type_pool[type_idx % len(content_type_pool)]
                slot.tone = tone_pool[tone_idx % len(tone_pool)]
                type_idx += 1
                tone_idx += 1

            plan = DayPlan(date=day, slots=slots)
            self._calendar[day.strftime("%Y-%m-%d")] = plan
            plans.append(plan)

        logger.info(
            "Generated week plan: %d days, %d total slots",
            len(plans),
            sum(len(p.slots) for p in plans),
        )
        return plans

    def get_next_empty_slot(self) -> CalendarSlot | None:
        """Get the next unfilled slot in the calendar."""
        now = datetime.now()
        for date_str in sorted(self._calendar.keys()):
            plan = self._calendar[date_str]
            for slot in plan.slots:
                if not slot.is_filled and slot.scheduled_time > now:
                    return slot
        return None

    def assign_draft(self, slot: CalendarSlot, draft: PostDraft) -> None:
        """Assign a post draft to a calendar slot."""
        slot.draft = draft
        slot.is_filled = True
        draft.scheduled_time = slot.scheduled_time
        draft.status = PostStatus.SCHEDULED
        logger.info(
            "Assigned draft %s to slot at %s",
            draft.id,
            slot.scheduled_time.isoformat(),
        )

    def get_gaps(self, days_ahead: int = 7) -> list[CalendarSlot]:
        """Find all empty slots within the specified number of days."""
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)
        gaps = []
        for date_str in sorted(self._calendar.keys()):
            plan = self._calendar[date_str]
            for slot in plan.slots:
                if not slot.is_filled and now < slot.scheduled_time < cutoff:
                    gaps.append(slot)
        return gaps

    def get_content_type_distribution(self, days: int = 7) -> dict[str, int]:
        """Get distribution of content types in the calendar."""
        now = datetime.now()
        cutoff = now + timedelta(days=days)
        dist: dict[str, int] = {}
        for plan in self._calendar.values():
            for slot in plan.slots:
                if slot.is_filled and slot.scheduled_time < cutoff and slot.content_type:
                    key = slot.content_type.value
                    dist[key] = dist.get(key, 0) + 1
        return dist

    def _generate_day_slots(
        self, day: datetime, peak_hours: list[int]
    ) -> list[CalendarSlot]:
        """Generate time slots for a single day, weighted toward peak hours."""
        posts_per_day = self.config.strategy.posts_per_day
        min_gap = self.config.strategy.min_hours_between_posts

        # Prefer peak hours but space evenly
        available_hours = [h for h in peak_hours if self.config.strategy.active_hours_utc[0] <= h < self.config.strategy.active_hours_utc[1]]
        if not available_hours:
            available_hours = list(range(9, 22))

        # Distribute posts across available hours with minimum gap
        slots: list[CalendarSlot] = []
        step = max(len(available_hours) // max(posts_per_day, 1), 1)
        for i in range(0, len(available_hours), step):
            if len(slots) >= posts_per_day:
                break
            hour = available_hours[i]
            slot_time = day.replace(hour=hour, minute=0, second=0, microsecond=0)

            # Enforce minimum gap
            if slots and (slot_time - slots[-1].scheduled_time).total_seconds() < min_gap * 3600:
                continue

            slots.append(CalendarSlot(scheduled_time=slot_time))

        return slots

    def _build_content_type_pool(self) -> list[ContentType]:
        """Build a weighted pool of content types based on config mix percentages."""
        pool: list[ContentType] = []
        mix = self.config.strategy.content_mix
        total_posts = self.config.strategy.posts_per_day * 7

        for type_str, ratio in mix.items():
            ct = ContentType(type_str)
            count = max(1, round(ratio * total_posts))
            pool.extend([ct] * count)

        return pool

    def _build_tone_pool(self) -> list[ContentTone]:
        """Build a weighted pool of tones based on config mix percentages."""
        pool: list[ContentTone] = []
        mix = self.config.strategy.tone_mix
        total_posts = self.config.strategy.posts_per_day * 7

        for tone_str, ratio in mix.items():
            tone = ContentTone(tone_str)
            count = max(1, round(ratio * total_posts))
            pool.extend([tone] * count)

        return pool
