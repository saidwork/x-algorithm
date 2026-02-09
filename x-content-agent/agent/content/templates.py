"""
Content template engine.

Provides pre-built templates for common high-performing X post formats.
Templates are parameterized and can be filled with topic-specific content
to quickly generate drafts without full LLM calls.

Templates are organized by:
- Content type (short post, thread, poll, etc.)
- Hook type (question, statistic, contrarian, story, etc.)
- Goal (engagement, reach, authority, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, BaseLoader

from agent.types import ContentGoal, ContentTone, ContentType


@dataclass
class ContentTemplate:
    """A parameterized content template."""

    id: str
    name: str
    content_type: ContentType
    tone: ContentTone
    goal: ContentGoal
    template: str  # Jinja2 template string
    required_vars: list[str] = field(default_factory=list)
    optional_vars: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    avg_engagement_rate: float = 0.0  # Historical performance


# ---------------------------------------------------------------------------
# Pre-built templates for common high-performing formats
# ---------------------------------------------------------------------------

TEMPLATES: list[ContentTemplate] = [
    # --- ENGAGEMENT: Questions & Debates ---
    ContentTemplate(
        id="hot_take",
        name="Hot Take / Contrarian Opinion",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.PROVOCATIVE,
        goal=ContentGoal.ENGAGEMENT,
        template="Unpopular opinion: {{ opinion }}\n\n{{ supporting_point }}\n\nAgree or disagree?",
        required_vars=["opinion", "supporting_point"],
        tags=["debate", "engagement", "contrarian"],
    ),
    ContentTemplate(
        id="this_or_that",
        name="This or That Debate",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.CASUAL,
        goal=ContentGoal.ENGAGEMENT,
        template="{{ option_a }} or {{ option_b }}?\n\n{{ context }}\n\nDrop your answer below 👇",
        required_vars=["option_a", "option_b"],
        optional_vars=["context"],
        tags=["engagement", "poll", "debate"],
    ),
    ContentTemplate(
        id="fill_blank",
        name="Fill in the Blank",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.CASUAL,
        goal=ContentGoal.ENGAGEMENT,
        template="{{ setup }} ______.\n\nWrong answers only 😂",
        required_vars=["setup"],
        tags=["engagement", "fun", "interactive"],
    ),

    # --- EDUCATIONAL: Tips & How-To ---
    ContentTemplate(
        id="quick_tip",
        name="Quick Tip",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.EDUCATIONAL,
        goal=ContentGoal.AUTHORITY,
        template="💡 {{ topic }} tip:\n\n{{ tip }}\n\nMost people don't know this, but it can {{ benefit }}.",
        required_vars=["topic", "tip", "benefit"],
        tags=["tip", "educational", "value"],
    ),
    ContentTemplate(
        id="mistake_list",
        name="Common Mistakes List",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.EDUCATIONAL,
        goal=ContentGoal.ENGAGEMENT,
        template="{{ count }} {{ topic }} mistakes that are costing you {{ cost }}:\n\n{% for mistake in mistakes %}{{ loop.index }}. {{ mistake }}\n{% endfor %}\nWhich one are you guilty of?",
        required_vars=["count", "topic", "cost", "mistakes"],
        tags=["list", "educational", "mistakes"],
    ),
    ContentTemplate(
        id="how_to_thread",
        name="How-To Thread",
        content_type=ContentType.THREAD,
        tone=ContentTone.EDUCATIONAL,
        goal=ContentGoal.AUTHORITY,
        template="🧵 How to {{ goal }} (step by step):\n\nA thread 👇",
        required_vars=["goal"],
        tags=["thread", "how-to", "educational"],
    ),

    # --- REACH: Viral Formats ---
    ContentTemplate(
        id="stat_hook",
        name="Statistic Hook",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.PROFESSIONAL,
        goal=ContentGoal.REACH,
        template="{{ statistic }}\n\n{{ explanation }}\n\nHere's why this matters 👇",
        required_vars=["statistic", "explanation"],
        tags=["statistic", "hook", "reach"],
    ),
    ContentTemplate(
        id="before_after",
        name="Before/After Transformation",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.INSPIRATIONAL,
        goal=ContentGoal.REACH,
        template="{{ timeframe }} ago: {{ before }}\n\nToday: {{ after }}\n\n{{ lesson }}",
        required_vars=["timeframe", "before", "after", "lesson"],
        tags=["transformation", "story", "inspirational"],
    ),

    # --- COMMUNITY: Conversation Starters ---
    ContentTemplate(
        id="community_question",
        name="Community Question",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.CASUAL,
        goal=ContentGoal.COMMUNITY,
        template="Question for {{ audience }}:\n\n{{ question }}\n\nI'll start: {{ own_answer }}",
        required_vars=["audience", "question", "own_answer"],
        tags=["community", "question", "conversation"],
    ),
    ContentTemplate(
        id="poll_template",
        name="Poll Post",
        content_type=ContentType.POLL,
        tone=ContentTone.CASUAL,
        goal=ContentGoal.COMMUNITY,
        template="{{ question }}",
        required_vars=["question"],
        tags=["poll", "community", "engagement"],
    ),

    # --- FOLLOWER GROWTH: Value Posts ---
    ContentTemplate(
        id="resource_share",
        name="Resource Share",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.PROFESSIONAL,
        goal=ContentGoal.FOLLOWER_GROWTH,
        template="{{ count }} {{ resource_type }} that will {{ benefit }}:\n\n{% for item in resources %}{{ loop.index }}. {{ item }}\n{% endfor %}\nBookmark this for later 🔖\n\nFollow for more {{ topic }} content.",
        required_vars=["count", "resource_type", "benefit", "resources", "topic"],
        tags=["resources", "value", "growth"],
    ),
    ContentTemplate(
        id="lesson_learned",
        name="Lesson Learned",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.PROFESSIONAL,
        goal=ContentGoal.AUTHORITY,
        template="After {{ experience }}:\n\nThe biggest lesson I learned:\n\n{{ lesson }}\n\n{{ actionable_takeaway }}",
        required_vars=["experience", "lesson", "actionable_takeaway"],
        tags=["lesson", "experience", "authority"],
    ),

    # --- NEWS: Timely Commentary ---
    ContentTemplate(
        id="news_take",
        name="News Commentary",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.NEWS,
        goal=ContentGoal.REACH,
        template="{{ news_hook }}\n\n{{ analysis }}\n\nWhat this means for {{ audience }}: {{ implication }}",
        required_vars=["news_hook", "analysis", "audience", "implication"],
        tags=["news", "commentary", "timely"],
    ),
]


class TemplateEngine:
    """
    Template engine for rapid content generation using pre-built formats.

    Uses Jinja2 for template rendering. Templates can be used:
    1. Standalone for quick content without LLM calls
    2. As seeds/structures that the LLM fills in with topic content
    3. As formatting guides for post-processing LLM output
    """

    def __init__(self):
        self._env = Environment(loader=BaseLoader())
        self._templates = {t.id: t for t in TEMPLATES}

    def get_template(self, template_id: str) -> ContentTemplate | None:
        return self._templates.get(template_id)

    def list_templates(
        self,
        content_type: ContentType | None = None,
        goal: ContentGoal | None = None,
        tone: ContentTone | None = None,
    ) -> list[ContentTemplate]:
        """List templates filtered by criteria."""
        result = list(self._templates.values())
        if content_type:
            result = [t for t in result if t.content_type == content_type]
        if goal:
            result = [t for t in result if t.goal == goal]
        if tone:
            result = [t for t in result if t.tone == tone]
        return result

    def render(self, template_id: str, variables: dict[str, Any]) -> str | None:
        """
        Render a template with the provided variables.

        Returns None if template not found or required variables missing.
        """
        tmpl = self._templates.get(template_id)
        if not tmpl:
            return None

        # Check required variables
        missing = [v for v in tmpl.required_vars if v not in variables]
        if missing:
            return None

        try:
            jinja_tmpl = self._env.from_string(tmpl.template)
            return jinja_tmpl.render(**variables)
        except Exception:
            return None

    def suggest_templates(
        self,
        topic: str,
        goal: ContentGoal = ContentGoal.ENGAGEMENT,
        count: int = 3,
    ) -> list[ContentTemplate]:
        """Suggest the best templates for a given topic and goal."""
        candidates = self.list_templates(goal=goal)
        if not candidates:
            candidates = list(self._templates.values())

        # Sort by historical performance
        candidates.sort(key=lambda t: t.avg_engagement_rate, reverse=True)
        return candidates[:count]

    def add_template(self, template: ContentTemplate) -> None:
        """Register a custom template."""
        self._templates[template.id] = template
