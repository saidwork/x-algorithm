"""
Agent scheduler module.

Manages the periodic execution of agent tasks:
- Trend monitoring (every N minutes)
- Content generation (daily, based on calendar)
- Performance metrics collection (hourly)
- Feedback analysis (weekly)
- Report generation (daily/weekly)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from agent.config import AgentConfig

logger = logging.getLogger(__name__)


class TaskFrequency(Enum):
    """How often a scheduled task should run."""

    EVERY_15_MIN = "every_15_min"
    HOURLY = "hourly"
    EVERY_4_HOURS = "every_4_hours"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class ScheduledTask:
    """A task that runs on a schedule."""

    task_id: str
    name: str
    frequency: TaskFrequency
    handler: str  # Name of the method to call
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""


class AgentScheduler:
    """
    Schedules and manages periodic agent tasks.

    Task schedule:
    - Trend Monitor:      Every 15 min - Check for trending topics
    - Content Generator:  Every 4 hours - Fill calendar gaps
    - Metrics Collector:  Hourly - Fetch post performance data
    - Content Publisher:  Hourly - Publish scheduled posts
    - Feedback Analysis:  Weekly - Analyze and adjust strategy
    - Daily Report:       Daily - Generate performance summary
    """

    FREQUENCY_INTERVALS: dict[TaskFrequency, timedelta] = {
        TaskFrequency.EVERY_15_MIN: timedelta(minutes=15),
        TaskFrequency.HOURLY: timedelta(hours=1),
        TaskFrequency.EVERY_4_HOURS: timedelta(hours=4),
        TaskFrequency.DAILY: timedelta(days=1),
        TaskFrequency.WEEKLY: timedelta(weeks=1),
    }

    def __init__(self, config: AgentConfig):
        self.config = config
        self._tasks: dict[str, ScheduledTask] = {}
        self._initialize_default_tasks()

    def _initialize_default_tasks(self) -> None:
        """Set up the default task schedule."""
        default_tasks = [
            ScheduledTask(
                task_id="trend_monitor",
                name="Trend Monitoring",
                frequency=TaskFrequency.EVERY_15_MIN,
                handler="monitor_trends",
                enabled=self.config.enable_trend_monitoring,
            ),
            ScheduledTask(
                task_id="content_generator",
                name="Content Generation",
                frequency=TaskFrequency.EVERY_4_HOURS,
                handler="generate_scheduled_content",
                enabled=True,
            ),
            ScheduledTask(
                task_id="metrics_collector",
                name="Metrics Collection",
                frequency=TaskFrequency.HOURLY,
                handler="collect_metrics",
                enabled=True,
            ),
            ScheduledTask(
                task_id="content_publisher",
                name="Content Publisher",
                frequency=TaskFrequency.HOURLY,
                handler="publish_scheduled",
                enabled=self.config.auto_publish,
            ),
            ScheduledTask(
                task_id="feedback_analysis",
                name="Feedback Analysis",
                frequency=TaskFrequency.WEEKLY,
                handler="run_feedback_analysis",
                enabled=self.config.analytics.feedback_loop_enabled,
            ),
            ScheduledTask(
                task_id="daily_report",
                name="Daily Report",
                frequency=TaskFrequency.DAILY,
                handler="generate_daily_report",
                enabled=True,
            ),
        ]

        for task in default_tasks:
            task.next_run = datetime.now()  # All run immediately on first tick
            self._tasks[task.task_id] = task

    def get_due_tasks(self) -> list[ScheduledTask]:
        """Get all tasks that are due to run now."""
        now = datetime.now()
        due = []
        for task in self._tasks.values():
            if task.enabled and task.next_run and task.next_run <= now:
                due.append(task)
        return due

    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed and schedule its next run."""
        task = self._tasks.get(task_id)
        if not task:
            return

        now = datetime.now()
        task.last_run = now
        task.run_count += 1
        interval = self.FREQUENCY_INTERVALS.get(task.frequency, timedelta(hours=1))
        task.next_run = now + interval

        logger.debug(
            "Task '%s' completed (run #%d). Next run: %s",
            task.name,
            task.run_count,
            task.next_run.isoformat(),
        )

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed and schedule retry."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.error_count += 1
        task.last_error = error

        # Exponential backoff for retries (max 1 hour)
        backoff_minutes = min(2 ** task.error_count, 60)
        task.next_run = datetime.now() + timedelta(minutes=backoff_minutes)

        logger.warning(
            "Task '%s' failed (error #%d): %s. Retry in %d min",
            task.name,
            task.error_count,
            error,
            backoff_minutes,
        )

    def enable_task(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True

    def disable_task(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False

    def get_status(self) -> dict[str, dict]:
        """Get status of all scheduled tasks."""
        return {
            task_id: {
                "name": task.name,
                "enabled": task.enabled,
                "frequency": task.frequency.value,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "next_run": task.next_run.isoformat() if task.next_run else None,
                "run_count": task.run_count,
                "error_count": task.error_count,
            }
            for task_id, task in self._tasks.items()
        }
