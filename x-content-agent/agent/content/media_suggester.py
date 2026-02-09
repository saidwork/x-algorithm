"""
Media suggestion module.

Analyzes content and suggests appropriate media attachments to boost engagement.
Posts with media consistently outperform text-only posts on X.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agent.config import AgentConfig
from agent.types import (
    ContentRequest,
    ContentType,
    MediaSuggestion,
    MediaType,
    PostDraft,
)

logger = logging.getLogger(__name__)


@dataclass
class MediaAnalysis:
    """Analysis result for media suggestion."""

    should_add_media: bool
    suggested_type: MediaType
    confidence: float  # 0.0 - 1.0
    reason: str


class MediaSuggester:
    """
    Suggests media attachments for content drafts.

    Pipeline role: DraftHydrator - enriches drafts with media suggestions.

    Decision logic:
    1. Analyze content text for media-worthy elements
    2. Determine best media type based on content type and goal
    3. Generate description/prompt for media creation
    4. Add alt text suggestion for accessibility
    """

    # Content patterns that benefit from specific media types
    STAT_PATTERNS = [r"\d+%", r"\$[\d,]+", r"\d+x", r"#\d+ ", r"\d+\s+(million|billion|thousand)"]
    LIST_PATTERNS = [r"^\d+[./)]", r"^[-•]"]
    COMPARISON_PATTERNS = [r"vs\.?", r"before.*after", r"old.*new"]

    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "MediaSuggester"

    def enabled(self, request: ContentRequest) -> bool:
        return self.config.enable_media_suggestions

    async def hydrate(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> list[PostDraft]:
        """Add media suggestions to all drafts (pipeline DraftHydrator interface)."""
        for draft in drafts:
            for piece in draft.pieces:
                if not piece.media:  # Don't override existing media
                    suggestion = self.suggest(piece.text, request)
                    if suggestion:
                        piece.media.append(suggestion)
        return drafts

    def suggest(self, text: str, request: ContentRequest) -> MediaSuggestion | None:
        """Suggest media for a piece of content."""
        analysis = self._analyze_content(text, request)

        if not analysis.should_add_media:
            return None

        return self._create_suggestion(text, analysis, request)

    def _analyze_content(self, text: str, request: ContentRequest) -> MediaAnalysis:
        """Analyze content to determine if and what media to suggest."""
        # Statistics/data -> infographic/chart image
        has_stats = any(re.search(p, text) for p in self.STAT_PATTERNS)
        if has_stats:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.IMAGE,
                confidence=0.8,
                reason="Content contains statistics - visual representation would boost engagement",
            )

        # Lists -> carousel or infographic
        lines = text.split("\n")
        list_lines = sum(1 for line in lines if any(re.match(p, line.strip()) for p in self.LIST_PATTERNS))
        if list_lines >= 3:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.IMAGE,
                confidence=0.7,
                reason="List-style content works well as a visual infographic",
            )

        # Comparisons -> side-by-side image
        has_comparison = any(re.search(p, text, re.IGNORECASE) for p in self.COMPARISON_PATTERNS)
        if has_comparison:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.IMAGE,
                confidence=0.7,
                reason="Comparison content benefits from visual side-by-side presentation",
            )

        # Short engagement posts -> GIF
        if len(text) < 100 and request.content_type == ContentType.SHORT_POST:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.GIF,
                confidence=0.5,
                reason="Short posts can benefit from an engaging GIF",
            )

        # Thread first posts -> eye-catching image
        if request.content_type == ContentType.THREAD:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.IMAGE,
                confidence=0.6,
                reason="Thread opener benefits from an attention-grabbing image",
            )

        # Default: suggest image for longer content
        if len(text) > 200:
            return MediaAnalysis(
                should_add_media=True,
                suggested_type=MediaType.IMAGE,
                confidence=0.4,
                reason="Longer posts generally perform better with visual media",
            )

        return MediaAnalysis(
            should_add_media=False,
            suggested_type=MediaType.IMAGE,
            confidence=0.0,
            reason="Content works well as text-only",
        )

    def _create_suggestion(
        self, text: str, analysis: MediaAnalysis, request: ContentRequest
    ) -> MediaSuggestion:
        """Create a detailed media suggestion with generation prompt."""
        topic = request.topic.name if request.topic else "the topic"

        if analysis.suggested_type == MediaType.IMAGE:
            description = f"Visual graphic related to: {topic}"
            generation_prompt = (
                f"Create a clean, professional infographic or visual for a social media post about: "
                f"{text[:200]}. Style: modern, minimal, high contrast, suitable for X/Twitter."
            )
            alt_text = f"Infographic about {topic}"
        elif analysis.suggested_type == MediaType.GIF:
            description = f"Reaction GIF matching the tone of the post"
            generation_prompt = ""  # GIFs are typically sourced, not generated
            alt_text = f"Animated GIF reaction"
        else:
            description = f"Media content related to {topic}"
            generation_prompt = ""
            alt_text = f"Media about {topic}"

        return MediaSuggestion(
            media_type=analysis.suggested_type,
            description=description,
            alt_text=alt_text,
            generation_prompt=generation_prompt,
        )
