"""Tests for the orchestrator module."""

import pytest

from agent.config import AgentConfig
from agent.orchestrator.agent import XContentAgent
from agent.orchestrator.scheduler import AgentScheduler, TaskFrequency
from agent.orchestrator.state_manager import StateManager
from agent.orchestrator.workflow import ContentWorkflow, WorkflowStatus
from agent.types import (
    ContentGoal,
    ContentPiece,
    ContentRequest,
    ContentTone,
    ContentType,
    PostDraft,
    PostStatus,
    Topic,
)


class TestWorkflow:
    def setup_method(self):
        self.config = AgentConfig()
        self.workflow = ContentWorkflow(self.config)

    def test_create_full_pipeline_run(self):
        run = self.workflow.create_full_pipeline_run()
        assert run.workflow_name == "full_pipeline"
        assert len(run.steps) == 8
        assert run.status == WorkflowStatus.PENDING

    def test_create_quick_generate_run(self):
        topic = Topic(name="AI")
        run = self.workflow.create_quick_generate_run(topic=topic)
        assert run.workflow_name == "quick_generate"
        assert len(run.steps) == 4

    def test_advance_step(self):
        run = self.workflow.create_full_pipeline_run()

        # Advance to first step
        step = self.workflow.advance_step(run.run_id)
        assert step is not None
        assert step.name == "analyze_trends"
        assert step.status == WorkflowStatus.RUNNING

        # Complete first step and advance to next
        step = self.workflow.advance_step(run.run_id, {"trends": 5})
        assert step is not None
        assert step.name == "select_topics"

    def test_advance_to_completion(self):
        run = self.workflow.create_quick_generate_run(topic=Topic(name="test"))

        for _ in range(10):  # More than enough steps
            step = self.workflow.advance_step(run.run_id)
            if step is None:
                break

        assert run.status == WorkflowStatus.COMPLETED

    def test_fail_step(self):
        run = self.workflow.create_full_pipeline_run()
        self.workflow.advance_step(run.run_id)  # Start first step
        self.workflow.fail_step(run.run_id, "API error")

        assert run.status == WorkflowStatus.FAILED

    def test_build_content_request(self):
        request = self.workflow.build_content_request(
            topic=Topic(name="AI"),
            content_type=ContentType.THREAD,
            tone=ContentTone.EDUCATIONAL,
            goal=ContentGoal.AUTHORITY,
            num_variants=5,
        )
        assert request.topic.name == "AI"
        assert request.content_type == ContentType.THREAD
        assert request.num_variants == 5


class TestScheduler:
    def setup_method(self):
        self.config = AgentConfig()
        self.scheduler = AgentScheduler(self.config)

    def test_default_tasks_initialized(self):
        status = self.scheduler.get_status()
        assert "trend_monitor" in status
        assert "content_generator" in status
        assert "metrics_collector" in status
        assert "daily_report" in status

    def test_get_due_tasks(self):
        due = self.scheduler.get_due_tasks()
        # All tasks should be due initially
        assert len(due) > 0

    def test_mark_completed(self):
        self.scheduler.mark_completed("trend_monitor")
        status = self.scheduler.get_status()
        assert status["trend_monitor"]["run_count"] == 1
        assert status["trend_monitor"]["last_run"] is not None

    def test_mark_failed_with_backoff(self):
        self.scheduler.mark_failed("trend_monitor", "API timeout")
        status = self.scheduler.get_status()
        assert status["trend_monitor"]["error_count"] == 1

    def test_enable_disable_task(self):
        self.scheduler.disable_task("content_publisher")
        status = self.scheduler.get_status()
        assert not status["content_publisher"]["enabled"]

        self.scheduler.enable_task("content_publisher")
        status = self.scheduler.get_status()
        assert status["content_publisher"]["enabled"]


class TestStateManager:
    def setup_method(self):
        self.manager = StateManager(state_dir="/tmp/test_agent_state")

    def test_add_to_draft_queue(self):
        draft = PostDraft(id="d1", pieces=[ContentPiece(text="Test")])
        self.manager.add_to_draft_queue(draft)

        assert draft.status == PostStatus.REVIEW
        pending = self.manager.get_pending_drafts()
        assert len(pending) == 1

    def test_approve_draft(self):
        draft = PostDraft(id="d2", pieces=[ContentPiece(text="Test")])
        self.manager.add_to_draft_queue(draft)
        approved = self.manager.approve_draft("d2")

        assert approved is not None
        assert approved.status == PostStatus.SCHEDULED
        assert len(self.manager.get_pending_drafts()) == 0
        assert len(self.manager.get_scheduled_drafts()) == 1

    def test_reject_draft(self):
        draft = PostDraft(id="d3", pieces=[ContentPiece(text="Test")])
        self.manager.add_to_draft_queue(draft)
        self.manager.reject_draft("d3", reason="Off-brand")

        assert draft.status == PostStatus.FAILED
        assert len(self.manager.get_pending_drafts()) == 0

    def test_mark_published(self):
        draft = PostDraft(id="d4", pieces=[ContentPiece(text="Test")])
        self.manager.add_to_draft_queue(draft)
        self.manager.approve_draft("d4")
        self.manager.mark_published("d4")

        assert draft.status == PostStatus.PUBLISHED
        stats = self.manager.get_stats()
        assert stats["published"] == 1

    def test_get_stats(self):
        stats = self.manager.get_stats()
        assert "drafts_in_queue" in stats
        assert "scheduled" in stats
        assert "published" in stats
        assert "total_generated" in stats

    def test_save_and_load_state(self):
        draft = PostDraft(id="d5", pieces=[ContentPiece(text="Test")])
        self.manager.add_to_draft_queue(draft)
        self.manager.save_state()

        # Create new manager and load state
        new_manager = StateManager(state_dir="/tmp/test_agent_state")
        new_manager.load_state()
        assert new_manager.get_stats()["total_generated"] == 1


class TestXContentAgent:
    def setup_method(self):
        self.config = AgentConfig(
            account_handle="testbot",
            niche="AI/ML",
            niche_keywords=["AI", "machine learning", "deep learning"],
        )
        self.agent = XContentAgent(self.config)

    def test_initialization(self):
        assert self.agent.config.account_handle == "testbot"
        assert self.agent.trend_analyzer is not None
        assert self.agent.generator is not None
        assert self.agent.engagement_predictor is not None
        assert self.agent.metrics_tracker is not None

    def test_get_status(self):
        status = self.agent.get_status()
        assert status["account"] == "@testbot"
        assert status["niche"] == "AI/ML"
        assert "queues" in status
        assert "scheduler" in status

    @pytest.mark.asyncio
    async def test_generate_content(self):
        drafts = await self.agent.generate_content(
            topic="GPT-5",
            content_type=ContentType.SHORT_POST,
            num_variants=2,
        )
        # With mock LLM (returns []), should get empty list
        assert isinstance(drafts, list)

    def test_get_pending_drafts_empty(self):
        drafts = self.agent.get_pending_drafts()
        assert drafts == []

    def test_daily_report(self):
        report = self.agent.get_daily_report()
        assert "Daily Performance Report" in report
        assert "@testbot" in report
