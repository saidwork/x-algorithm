"""
Main AI Agent orchestrator.

The XContentAgent is the top-level class that coordinates all modules.
It's the entry point for users and the scheduler - all operations go through here.

Architecture mirrors home-mixer's server.rs: a single entry point that
assembles and executes the content pipeline with all configured components.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from agent.analytics.ab_tester import ABTester
from agent.analytics.feedback_loop import FeedbackLoop
from agent.analytics.metrics_tracker import MetricsTracker
from agent.analytics.report_generator import ReportGenerator
from agent.audience.engagement_predictor import EngagementPredictor
from agent.audience.follower_analyzer import FollowerAnalyzer
from agent.audience.persona_builder import PersonaBuilder
from agent.audience.timing_optimizer import TimingOptimizer
from agent.config import AgentConfig
from agent.content.generator import LLMContentGenerator
from agent.content.media_suggester import MediaSuggester
from agent.content.optimizer import ContentOptimizer
from agent.content.templates import TemplateEngine
from agent.content.thread_composer import ThreadComposer
from agent.orchestrator.scheduler import AgentScheduler
from agent.orchestrator.state_manager import StateManager
from agent.orchestrator.workflow import ContentWorkflow
from agent.pipeline import ContentPipeline, PipelineResult
from agent.strategy.competitor_analyzer import CompetitorAnalyzer
from agent.strategy.content_calendar import ContentCalendar
from agent.strategy.topic_selector import TopicSelector
from agent.strategy.trend_analyzer import TrendAnalyzer
from agent.types import (
    ContentGoal,
    ContentRequest,
    ContentTone,
    ContentType,
    PerformanceMetrics,
    PostDraft,
    Topic,
)

logger = logging.getLogger(__name__)


class XContentAgent:
    """
    Main AI Agent for X content generation.

    Coordinates all subsystems:
    ┌──────────────────────────────────────────────────────────────┐
    │                     XContentAgent                            │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              ContentPipeline                          │   │
    │  │  RequestHydrators -> Generators -> DraftHydrators     │   │
    │  │  -> Filters -> Scorers -> Selector -> PostProcessors  │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                                                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
    │  │  Strategy    │  │  Audience   │  │  Analytics       │     │
    │  │  - Trends    │  │  - Predict  │  │  - Metrics       │     │
    │  │  - Topics    │  │  - Analyze  │  │  - A/B Test      │     │
    │  │  - Calendar  │  │  - Personas │  │  - Feedback      │     │
    │  │  - Compete   │  │  - Timing   │  │  - Reports       │     │
    │  └─────────────┘  └─────────────┘  └─────────────────┘     │
    │                                                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
    │  │  Workflow    │  │  Scheduler  │  │  State Manager   │     │
    │  └─────────────┘  └─────────────┘  └─────────────────┘     │
    └──────────────────────────────────────────────────────────────┘

    Usage:
        config = AgentConfig(
            account_handle="myaccount",
            niche="AI/ML",
            niche_keywords=["machine learning", "AI", "deep learning"],
            llm=LLMConfig(api_key="sk-..."),
        )
        agent = XContentAgent(config)

        # Generate content for a specific topic
        drafts = await agent.generate_content(
            topic="GPT-5 release",
            content_type=ContentType.SHORT_POST,
            tone=ContentTone.PROFESSIONAL,
        )

        # Or run the full automated pipeline
        result = await agent.run_full_pipeline()
    """

    def __init__(self, config: AgentConfig):
        self.config = config

        # --- Strategy Layer ---
        self.trend_analyzer = TrendAnalyzer(config)
        self.topic_selector = TopicSelector(config)
        self.content_calendar = ContentCalendar(config)
        self.competitor_analyzer = CompetitorAnalyzer(config)

        # --- Content Layer ---
        self.generator = LLMContentGenerator(config)
        self.optimizer = ContentOptimizer(config)
        self.template_engine = TemplateEngine()
        self.thread_composer = ThreadComposer(config)
        self.media_suggester = MediaSuggester(config)

        # --- Audience Layer ---
        self.engagement_predictor = EngagementPredictor(config)
        self.follower_analyzer = FollowerAnalyzer(config)
        self.persona_builder = PersonaBuilder(config)
        self.timing_optimizer = TimingOptimizer(config)

        # --- Analytics Layer ---
        self.metrics_tracker = MetricsTracker()
        self.ab_tester = ABTester(config)
        self.feedback_loop = FeedbackLoop(config, self.metrics_tracker)
        self.report_generator = ReportGenerator(config, self.metrics_tracker)

        # --- Orchestration Layer ---
        self.workflow = ContentWorkflow(config)
        self.scheduler = AgentScheduler(config)
        self.state_manager = StateManager()

        # --- Assembled Pipeline ---
        self._pipeline = self._build_pipeline()

        logger.info(
            "XContentAgent initialized for @%s (niche: %s)",
            config.account_handle,
            config.niche,
        )

    def _build_pipeline(self) -> ContentPipeline:
        """
        Assemble the content pipeline with all components.
        Mirrors how PhoenixCandidatePipeline assembles its pipeline.
        """
        return ContentPipeline(
            request_hydrators=[],  # Will be populated with audience hydrator
            generators=[self.generator],
            draft_hydrators=[self.optimizer, self.media_suggester],
            filters=[],  # Content quality filters
            scorers=[self.engagement_predictor],
            selector=None,  # Default: sort by score
            post_processors=[],  # Scheduling, caching
            result_size=self.config.strategy.posts_per_day,
        )

    # ------------------------------------------------------------------
    # Public API: Content Generation
    # ------------------------------------------------------------------

    async def generate_content(
        self,
        topic: str | Topic | None = None,
        content_type: ContentType = ContentType.SHORT_POST,
        tone: ContentTone | None = None,
        goal: ContentGoal | None = None,
        num_variants: int = 3,
    ) -> list[PostDraft]:
        """
        Generate content for a given topic.

        This is the main entry point for content generation.
        Runs the full pipeline: generate -> optimize -> score -> select.
        """
        # Normalize topic
        if isinstance(topic, str):
            topic_obj = Topic(name=topic, keywords=[topic])
        elif topic is None:
            topic_obj = Topic(name=self.config.niche or "general")
        else:
            topic_obj = topic

        request = ContentRequest(
            request_id=f"gen-{uuid.uuid4().hex[:8]}",
            account_id=self.config.account_id,
            topic=topic_obj,
            content_type=content_type,
            tone=tone or self.config.default_tone,
            goal=goal or self.config.default_goal,
            num_variants=num_variants,
        )

        result = await self._pipeline.execute(request)

        # Add selected drafts to the state manager
        for draft in result.selected_drafts:
            self.state_manager.add_to_draft_queue(draft)

        logger.info(
            "Generated %d drafts for topic '%s' (selected from %d candidates)",
            len(result.selected_drafts),
            topic_obj.name,
            len(result.generated_drafts),
        )
        return result.selected_drafts

    async def generate_thread(
        self,
        topic: str | Topic,
        key_points: list[str],
        tone: ContentTone | None = None,
    ) -> PostDraft | None:
        """Generate a thread from a topic and key points."""
        topic_obj = Topic(name=topic) if isinstance(topic, str) else topic

        request = ContentRequest(
            request_id=f"thread-{uuid.uuid4().hex[:8]}",
            account_id=self.config.account_id,
            topic=topic_obj,
            content_type=ContentType.THREAD,
            tone=tone or self.config.default_tone,
        )

        outline = self.thread_composer.create_outline(
            topic=topic_obj,
            key_points=key_points,
        )
        draft = self.thread_composer.compose_from_outline(outline, request)

        # Validate
        issues = self.thread_composer.validate_thread(draft)
        if issues:
            logger.warning("Thread validation issues: %s", issues)

        self.state_manager.add_to_draft_queue(draft)
        return draft

    async def quick_post(
        self,
        template_id: str,
        variables: dict,
    ) -> PostDraft | None:
        """Generate a quick post from a template."""
        rendered = self.template_engine.render(template_id, variables)
        if not rendered:
            return None

        from agent.types import ContentPiece

        piece = ContentPiece(text=rendered, content_type=ContentType.SHORT_POST)
        draft = PostDraft(
            id=f"quick-{uuid.uuid4().hex[:8]}",
            pieces=[piece],
            tone=self.config.default_tone,
            goal=self.config.default_goal,
        )

        # Score the draft
        prediction = self.engagement_predictor.predict(draft)
        draft.engagement_prediction = prediction

        self.state_manager.add_to_draft_queue(draft)
        return draft

    # ------------------------------------------------------------------
    # Public API: Full Pipeline
    # ------------------------------------------------------------------

    async def run_full_pipeline(self) -> PipelineResult | None:
        """
        Run the complete content pipeline:
        1. Analyze trends
        2. Select topics
        3. Generate content for each topic
        4. Optimize and score
        5. Schedule the best content
        """
        logger.info("Starting full pipeline run...")

        # Step 1: Analyze trends
        trends = await self.trend_analyzer.analyze()

        # Step 2: Build audience profile
        audience = await self.follower_analyzer.build_profile()

        # Step 3: Select topics
        topics = self.topic_selector.select_topics(
            trend_signals=trends,
            audience=audience,
            goal=self.config.default_goal,
            count=self.config.strategy.posts_per_day,
        )

        if not topics:
            logger.warning("No topics selected, using niche keywords as fallback")
            topics = [Topic(name=kw) for kw in self.config.niche_keywords[:3]]

        # Step 4: Generate content for each topic
        all_drafts: list[PostDraft] = []
        for topic in topics:
            content_type = self.topic_selector.suggest_content_type(
                topic, self.config.default_goal
            )
            drafts = await self.generate_content(
                topic=topic,
                content_type=content_type,
                num_variants=2,
            )
            all_drafts.extend(drafts)

        # Step 5: Schedule into calendar
        for draft in all_drafts:
            slot = self.content_calendar.get_next_empty_slot()
            if slot:
                optimal_times = self.timing_optimizer.get_optimal_times(count=1)
                if optimal_times:
                    slot.scheduled_time = optimal_times[0].datetime_utc
                self.content_calendar.assign_draft(slot, draft)

        logger.info(
            "Full pipeline complete: %d topics -> %d drafts scheduled",
            len(topics),
            len(all_drafts),
        )
        return None  # Full pipeline result is in state_manager

    # ------------------------------------------------------------------
    # Public API: Analytics & Management
    # ------------------------------------------------------------------

    def record_metrics(self, post_id: str, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for a published post."""
        self.metrics_tracker.update_metrics(post_id, metrics)

        # Update timing optimizer
        draft = self.state_manager.get_draft(post_id)
        if draft and draft.scheduled_time:
            self.timing_optimizer.update_from_metrics(metrics, draft.scheduled_time)

    async def run_feedback_analysis(self) -> list[str]:
        """Run feedback analysis and get strategy adjustments."""
        report = await self.feedback_loop.run_analysis()
        adjustments = self.feedback_loop.apply_adjustments(report)
        return adjustments

    def get_daily_report(self) -> str:
        """Generate a daily performance report."""
        return self.report_generator.generate_daily_report()

    def get_weekly_report(self) -> str:
        """Generate a weekly performance report."""
        return self.report_generator.generate_weekly_report()

    def approve_draft(self, draft_id: str) -> PostDraft | None:
        """Approve a draft for publishing."""
        return self.state_manager.approve_draft(draft_id)

    def reject_draft(self, draft_id: str, reason: str = "") -> None:
        """Reject a draft."""
        self.state_manager.reject_draft(draft_id, reason)

    def get_pending_drafts(self) -> list[PostDraft]:
        """Get all drafts awaiting review."""
        return self.state_manager.get_pending_drafts()

    def get_status(self) -> dict:
        """Get full agent status."""
        return {
            "agent": self.config.agent_name,
            "account": f"@{self.config.account_handle}",
            "niche": self.config.niche,
            "queues": self.state_manager.get_stats(),
            "scheduler": self.scheduler.get_status(),
            "calendar_gaps": len(self.content_calendar.get_gaps()),
        }
