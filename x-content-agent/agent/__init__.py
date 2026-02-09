"""
X Content Agent - AI-powered content generation and strategy for X platform.

Architecture follows the same pipeline-based, modular design as the X recommendation
algorithm (home-mixer), but inverted: instead of ranking existing content for consumption,
this agent generates, optimizes, and schedules new content for publication.

Core modules:
    - agent.types: Data models and type definitions
    - agent.config: Agent configuration
    - agent.pipeline: Content generation pipeline framework
    - agent.strategy: Trend analysis, topic selection, content calendar
    - agent.content: Content generation, optimization, templates
    - agent.audience: Engagement prediction, follower analysis
    - agent.analytics: Metrics tracking, A/B testing, feedback loops
    - agent.orchestrator: Workflow management, scheduling, state
"""

from agent.config import AgentConfig
from agent.types import (
    ContentPiece,
    ContentType,
    PostDraft,
    PostStatus,
)

__all__ = [
    "AgentConfig",
    "ContentPiece",
    "ContentType",
    "PostDraft",
    "PostStatus",
]
