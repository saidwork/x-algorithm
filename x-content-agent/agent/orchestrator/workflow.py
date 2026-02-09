"""
Content workflow management.

Defines the end-to-end workflows for content generation, from trend analysis
to publishing. Each workflow is a sequence of pipeline stages that can be
executed manually or automatically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agent.config import AgentConfig
from agent.types import (
    ContentGoal,
    ContentRequest,
    ContentTone,
    ContentType,
    PostDraft,
    PostStatus,
    Topic,
)

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in a content workflow."""

    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str = ""
    result: dict = field(default_factory=dict)


@dataclass
class WorkflowRun:
    """A single execution of a workflow."""

    run_id: str
    workflow_name: str
    steps: list[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    drafts: list[PostDraft] = field(default_factory=list)


class ContentWorkflow:
    """
    Manages content creation workflows.

    Workflows:
    1. FULL PIPELINE    - Trend analysis -> Topic selection -> Generation -> Optimization -> Scheduling
    2. QUICK GENERATE   - Direct topic -> Generation -> Optimization
    3. REACT TO TREND   - Trend detected -> Rapid content generation
    4. THREAD BUILDER   - Topic -> Research -> Thread composition
    5. ENGAGEMENT REPLY - Trending post -> Generate reply/quote

    Each workflow is a configurable sequence of steps that coordinates
    the various agent modules.
    """

    WORKFLOW_FULL = "full_pipeline"
    WORKFLOW_QUICK = "quick_generate"
    WORKFLOW_TREND_REACT = "trend_react"
    WORKFLOW_THREAD = "thread_builder"
    WORKFLOW_REPLY = "engagement_reply"

    def __init__(self, config: AgentConfig):
        self.config = config
        self._runs: dict[str, WorkflowRun] = {}

    def create_full_pipeline_run(self) -> WorkflowRun:
        """Create a full content pipeline workflow run."""
        run = WorkflowRun(
            run_id=f"wf-{uuid.uuid4().hex[:8]}",
            workflow_name=self.WORKFLOW_FULL,
            steps=[
                WorkflowStep(name="analyze_trends"),
                WorkflowStep(name="select_topics"),
                WorkflowStep(name="build_audience_profile"),
                WorkflowStep(name="generate_content"),
                WorkflowStep(name="optimize_content"),
                WorkflowStep(name="predict_engagement"),
                WorkflowStep(name="select_best"),
                WorkflowStep(name="schedule_posts"),
            ],
        )
        self._runs[run.run_id] = run
        return run

    def create_quick_generate_run(
        self,
        topic: Topic,
        content_type: ContentType = ContentType.SHORT_POST,
        tone: ContentTone = ContentTone.PROFESSIONAL,
        goal: ContentGoal = ContentGoal.ENGAGEMENT,
    ) -> WorkflowRun:
        """Create a quick content generation workflow."""
        run = WorkflowRun(
            run_id=f"wf-{uuid.uuid4().hex[:8]}",
            workflow_name=self.WORKFLOW_QUICK,
            steps=[
                WorkflowStep(name="generate_content"),
                WorkflowStep(name="optimize_content"),
                WorkflowStep(name="predict_engagement"),
                WorkflowStep(name="select_best"),
            ],
        )
        run.drafts = []  # Will be populated during execution
        self._runs[run.run_id] = run
        return run

    def create_trend_reaction_run(self) -> WorkflowRun:
        """Create a rapid trend reaction workflow."""
        run = WorkflowRun(
            run_id=f"wf-{uuid.uuid4().hex[:8]}",
            workflow_name=self.WORKFLOW_TREND_REACT,
            steps=[
                WorkflowStep(name="detect_trend"),
                WorkflowStep(name="assess_relevance"),
                WorkflowStep(name="rapid_generate"),
                WorkflowStep(name="quality_check"),
                WorkflowStep(name="immediate_publish"),
            ],
        )
        self._runs[run.run_id] = run
        return run

    def create_thread_workflow_run(self, topic: Topic) -> WorkflowRun:
        """Create a thread building workflow."""
        run = WorkflowRun(
            run_id=f"wf-{uuid.uuid4().hex[:8]}",
            workflow_name=self.WORKFLOW_THREAD,
            steps=[
                WorkflowStep(name="research_topic"),
                WorkflowStep(name="create_outline"),
                WorkflowStep(name="generate_segments"),
                WorkflowStep(name="optimize_thread"),
                WorkflowStep(name="review_and_schedule"),
            ],
        )
        self._runs[run.run_id] = run
        return run

    def advance_step(self, run_id: str, step_result: dict | None = None) -> WorkflowStep | None:
        """
        Advance a workflow to the next step.
        Returns the next step to execute, or None if complete.
        """
        run = self._runs.get(run_id)
        if not run:
            return None

        # Find current step
        current_idx = -1
        for i, step in enumerate(run.steps):
            if step.status == WorkflowStatus.RUNNING:
                step.status = WorkflowStatus.COMPLETED
                step.completed_at = datetime.now()
                if step_result:
                    step.result = step_result
                current_idx = i
                break
            elif step.status == WorkflowStatus.PENDING:
                current_idx = i - 1
                break

        # Start next step
        next_idx = current_idx + 1
        if next_idx < len(run.steps):
            next_step = run.steps[next_idx]
            next_step.status = WorkflowStatus.RUNNING
            next_step.started_at = datetime.now()
            run.status = WorkflowStatus.RUNNING
            return next_step
        else:
            run.status = WorkflowStatus.COMPLETED
            run.completed_at = datetime.now()
            return None

    def fail_step(self, run_id: str, error: str) -> None:
        """Mark the current step as failed."""
        run = self._runs.get(run_id)
        if not run:
            return

        for step in run.steps:
            if step.status == WorkflowStatus.RUNNING:
                step.status = WorkflowStatus.FAILED
                step.error = error
                step.completed_at = datetime.now()
                break

        run.status = WorkflowStatus.FAILED

    def get_run(self, run_id: str) -> WorkflowRun | None:
        return self._runs.get(run_id)

    def get_active_runs(self) -> list[WorkflowRun]:
        return [r for r in self._runs.values() if r.status == WorkflowStatus.RUNNING]

    def build_content_request(
        self,
        topic: Topic | None = None,
        content_type: ContentType = ContentType.SHORT_POST,
        tone: ContentTone = ContentTone.PROFESSIONAL,
        goal: ContentGoal = ContentGoal.ENGAGEMENT,
        num_variants: int = 3,
    ) -> ContentRequest:
        """Build a ContentRequest from workflow parameters."""
        return ContentRequest(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            account_id=self.config.account_id,
            topic=topic,
            content_type=content_type,
            tone=tone,
            goal=goal,
            num_variants=num_variants,
        )
