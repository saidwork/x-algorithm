"""
Audience persona builder.

Creates detailed fictional personas representing key audience segments.
Personas are used by the content generator to tailor voice, examples,
and references to resonate with specific audience groups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.config import AgentConfig
from agent.types import AudienceProfile

logger = logging.getLogger(__name__)


@dataclass
class AudiencePersona:
    """A fictional persona representing an audience segment."""

    name: str
    description: str
    age_range: str = ""
    occupation: str = ""
    interests: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    preferred_content_types: list[str] = field(default_factory=list)
    preferred_tone: str = ""
    x_usage_pattern: str = ""  # e.g., "scrolls during commute", "active in evenings"
    engagement_style: str = ""  # e.g., "likes silently", "replies often", "reposts valuable content"
    segment_size_pct: float = 0.0  # % of total audience


class PersonaBuilder:
    """
    Builds audience personas from follower analysis data.

    Personas help the content generator write for specific people rather than
    abstract demographics. Each persona captures:
    - Who they are (demographics, occupation)
    - What they care about (interests, pain points, goals)
    - How they use X (usage patterns, engagement style)
    - What content they respond to (preferred types, tone)
    """

    # Default persona templates for common audience types
    DEFAULT_PERSONA_TEMPLATES = [
        {
            "name": "The Learner",
            "description": "Follows you to learn and grow professionally",
            "engagement_style": "Bookmarks educational content, replies with questions",
            "preferred_content_types": ["threads", "how-to", "tips"],
            "preferred_tone": "educational",
        },
        {
            "name": "The Networker",
            "description": "Uses X to build professional connections",
            "engagement_style": "Quote posts to add their take, active in replies",
            "preferred_content_types": ["discussions", "debates", "community questions"],
            "preferred_tone": "professional",
        },
        {
            "name": "The Lurker",
            "description": "Reads content but rarely engages publicly",
            "engagement_style": "Likes occasionally, bookmarks often, rarely replies",
            "preferred_content_types": ["short posts", "visual content"],
            "preferred_tone": "casual",
        },
        {
            "name": "The Advocate",
            "description": "Your most loyal follower who shares your content",
            "engagement_style": "Reposts frequently, defends your takes in replies",
            "preferred_content_types": ["hot takes", "valuable resources"],
            "preferred_tone": "provocative",
        },
    ]

    def __init__(self, config: AgentConfig):
        self.config = config
        self._personas: list[AudiencePersona] = []

    def build_personas(
        self,
        audience: AudienceProfile,
        max_personas: int = 4,
    ) -> list[AudiencePersona]:
        """
        Build audience personas from audience profile data.

        Combines follower analysis with default persona templates,
        customized for the account's specific niche.
        """
        personas = []

        for i, template in enumerate(self.DEFAULT_PERSONA_TEMPLATES[:max_personas]):
            persona = AudiencePersona(
                name=template["name"],
                description=template["description"],
                interests=audience.interests[:3] if audience.interests else [],
                preferred_content_types=template["preferred_content_types"],
                preferred_tone=template["preferred_tone"],
                engagement_style=template["engagement_style"],
                segment_size_pct=1.0 / max_personas,
            )

            # Customize with niche-specific data
            if self.config.niche:
                persona.description += f" in the {self.config.niche} space"

            if audience.interests:
                persona.pain_points = [
                    f"Keeping up with {audience.interests[0]} trends"
                    if audience.interests
                    else "Staying informed"
                ]
                persona.goals = [
                    f"Become better at {audience.interests[0]}"
                    if audience.interests
                    else "Professional growth"
                ]

            personas.append(persona)

        self._personas = personas
        logger.info("Built %d audience personas", len(personas))
        return personas

    def get_persona_for_content(self, content_type: str, tone: str) -> AudiencePersona | None:
        """Find the persona most likely to engage with a given content type and tone."""
        for persona in self._personas:
            if (
                content_type in persona.preferred_content_types
                or tone == persona.preferred_tone
            ):
                return persona
        return self._personas[0] if self._personas else None

    def get_all_personas(self) -> list[AudiencePersona]:
        return list(self._personas)

    def format_for_prompt(self, persona: AudiencePersona) -> str:
        """Format a persona for inclusion in LLM prompts."""
        return (
            f"Target Reader: {persona.name}\n"
            f"- {persona.description}\n"
            f"- Interests: {', '.join(persona.interests)}\n"
            f"- They engage by: {persona.engagement_style}\n"
            f"- Preferred content: {', '.join(persona.preferred_content_types)}\n"
            f"- Pain points: {', '.join(persona.pain_points)}"
        )
