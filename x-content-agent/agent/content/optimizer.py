"""
Content optimization module.

Applies rule-based and heuristic optimizations to generated content drafts.
Acts as a DraftHydrator in the pipeline - enriching drafts with improvements.

Optimization areas:
- Character count optimization
- Hashtag placement and relevance
- Hook strength analysis
- Readability scoring
- Engagement trigger detection
- Formatting improvements
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agent.config import AgentConfig
from agent.types import ContentConstraints, ContentPiece, ContentRequest, ContentType, PostDraft

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimizing a single piece of content."""

    original_text: str
    optimized_text: str
    changes_made: list[str]
    readability_score: float
    hook_score: float
    engagement_trigger_score: float


class ContentOptimizer:
    """
    Optimizes content drafts for maximum engagement on X.

    Pipeline role: DraftHydrator - takes generated drafts and improves them.

    Optimization pipeline (applied sequentially):
    1. Character count check and trimming
    2. Hook analysis and strengthening suggestions
    3. Hashtag optimization (placement, count, relevance)
    4. Engagement trigger insertion (questions, CTAs)
    5. Readability scoring
    6. Formatting cleanup
    """

    # Hook patterns that tend to perform well on X
    STRONG_HOOK_PATTERNS = [
        r"^\d+\s",  # Starts with a number
        r"^(How|Why|What|When|Where)\s",  # Question openers
        r"^(Stop|Don't|Never|Always)\s",  # Command openers
        r"^(The truth|Here's what|Most people|Unpopular opinion)",  # Contrarian openers
        r"^(I just|I've been|After \d+)",  # Personal story openers
        r"^(Breaking|JUST IN|NEW)",  # News openers
    ]

    # Engagement triggers
    ENGAGEMENT_TRIGGERS = [
        r"\?$",  # Ends with question
        r"(agree|disagree|thoughts)\??$",  # Asks for opinion
        r"(share|repost|like if)\s",  # Direct CTAs
        r"(comment|reply|tell me)\s",  # Reply triggers
        r"(thread|🧵)",  # Thread indicator
    ]

    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ContentOptimizer"

    def enabled(self, request: ContentRequest) -> bool:
        return True

    async def hydrate(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> list[PostDraft]:
        """Optimize all drafts (pipeline DraftHydrator interface)."""
        for draft in drafts:
            for i, piece in enumerate(draft.pieces):
                result = self.optimize_piece(piece, request)
                draft.pieces[i] = ContentPiece(
                    text=result.optimized_text,
                    content_type=piece.content_type,
                    hashtags=piece.hashtags,
                    mentions=piece.mentions,
                    url=piece.url,
                    poll_options=piece.poll_options,
                )
                draft.metadata["readability_score"] = result.readability_score
                draft.metadata["hook_score"] = result.hook_score
                draft.metadata["engagement_trigger_score"] = result.engagement_trigger_score
        return drafts

    def optimize_piece(
        self, piece: ContentPiece, request: ContentRequest
    ) -> OptimizationResult:
        """Apply all optimizations to a single content piece."""
        text = piece.text
        changes: list[str] = []
        constraints = request.constraints or ContentConstraints()

        # 1. Clean up formatting
        optimized, formatting_changes = self._clean_formatting(text)
        changes.extend(formatting_changes)

        # 2. Character count optimization
        max_chars = constraints.max_characters
        if piece.content_type != ContentType.LONG_POST and len(optimized) > max_chars:
            optimized, trim_changes = self._trim_to_limit(optimized, max_chars)
            changes.extend(trim_changes)

        # 3. Hashtag optimization
        optimized, hashtag_changes = self._optimize_hashtags(
            optimized, piece.hashtags, constraints
        )
        changes.extend(hashtag_changes)

        # 4. Score the content
        hook_score = self._score_hook(optimized)
        engagement_score = self._score_engagement_triggers(optimized)
        readability = self._score_readability(optimized)

        return OptimizationResult(
            original_text=text,
            optimized_text=optimized,
            changes_made=changes,
            readability_score=readability,
            hook_score=hook_score,
            engagement_trigger_score=engagement_score,
        )

    def _clean_formatting(self, text: str) -> tuple[str, list[str]]:
        """Clean up whitespace and formatting issues."""
        changes = []
        original = text

        # Remove excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Trim leading/trailing whitespace
        text = text.strip()

        if text != original:
            changes.append("cleaned_formatting")
        return text, changes

    def _trim_to_limit(self, text: str, max_chars: int) -> tuple[str, list[str]]:
        """Trim text to character limit while preserving meaning."""
        if len(text) <= max_chars:
            return text, []

        # Try trimming at sentence boundary
        sentences = re.split(r"(?<=[.!?])\s+", text)
        trimmed = ""
        for sentence in sentences:
            if len(trimmed) + len(sentence) + 1 <= max_chars:
                trimmed = f"{trimmed} {sentence}".strip() if trimmed else sentence
            else:
                break

        if trimmed and len(trimmed) >= max_chars * 0.5:
            return trimmed, ["trimmed_at_sentence_boundary"]

        # Fallback: trim at word boundary
        words = text.split()
        trimmed = ""
        for word in words:
            if len(trimmed) + len(word) + 1 <= max_chars - 3:  # Leave room for "..."
                trimmed = f"{trimmed} {word}".strip() if trimmed else word
            else:
                break

        return trimmed + "...", ["trimmed_at_word_boundary"]

    def _optimize_hashtags(
        self, text: str, hashtags: list[str], constraints: ContentConstraints
    ) -> tuple[str, list[str]]:
        """Optimize hashtag placement and count."""
        changes = []
        max_tags = self.config.filters.max_hashtags

        # Count existing hashtags in text
        existing_tags = re.findall(r"#\w+", text)

        # Remove excessive hashtags
        if len(existing_tags) > max_tags:
            # Keep only the first max_tags hashtags
            tags_to_remove = existing_tags[max_tags:]
            for tag in tags_to_remove:
                text = text.replace(tag, "").strip()
            changes.append(f"removed_{len(tags_to_remove)}_excess_hashtags")

        # Add required hashtags if missing
        for required in constraints.required_hashtags:
            tag = f"#{required}" if not required.startswith("#") else required
            if tag.lower() not in text.lower():
                text = f"{text}\n\n{tag}"
                changes.append(f"added_required_hashtag_{required}")

        return text.strip(), changes

    def _score_hook(self, text: str) -> float:
        """Score the opening hook strength (0.0 - 1.0)."""
        first_line = text.split("\n")[0].strip()

        score = 0.0
        for pattern in self.STRONG_HOOK_PATTERNS:
            if re.search(pattern, first_line, re.IGNORECASE):
                score += 0.3

        # Short, punchy first lines score higher
        if len(first_line) < 80:
            score += 0.2
        elif len(first_line) < 140:
            score += 0.1

        # Emoji in first line can boost engagement
        if re.search(r"[\U0001F300-\U0001F9FF]", first_line):
            score += 0.1

        return min(score, 1.0)

    def _score_engagement_triggers(self, text: str) -> float:
        """Score how many engagement triggers the content has (0.0 - 1.0)."""
        score = 0.0
        for pattern in self.ENGAGEMENT_TRIGGERS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25

        # Line breaks improve readability and engagement
        if "\n" in text:
            score += 0.1

        return min(score, 1.0)

    def _score_readability(self, text: str) -> float:
        """
        Score readability for X platform (0.0 - 1.0).
        Optimized for social media, not academic text.
        """
        words = text.split()
        if not words:
            return 0.0

        avg_word_length = sum(len(w) for w in words) / len(words)
        sentences = re.split(r"[.!?]+", text)
        avg_sentence_length = len(words) / max(len([s for s in sentences if s.strip()]), 1)

        score = 1.0

        # Penalize long average word length (prefer simple words)
        if avg_word_length > 6:
            score -= 0.2
        elif avg_word_length > 8:
            score -= 0.4

        # Penalize long sentences
        if avg_sentence_length > 20:
            score -= 0.3
        elif avg_sentence_length > 30:
            score -= 0.5

        # Bonus for using line breaks (easier to scan)
        if "\n" in text:
            score += 0.1

        return max(0.0, min(score, 1.0))
