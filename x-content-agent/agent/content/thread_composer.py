"""
Thread composition module.

Specialized generator for multi-post threads on X. Handles thread-specific
concerns: narrative arc, segment sizing, numbering, hook-to-CTA flow.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from agent.config import AgentConfig
from agent.types import ContentPiece, ContentRequest, ContentType, PostDraft, Topic

logger = logging.getLogger(__name__)

MAX_SEGMENT_CHARS = 280
DEFAULT_SEGMENT_COUNT = 7


@dataclass
class ThreadSegment:
    """A single segment in a thread with its role."""

    text: str
    role: str  # "hook", "context", "point", "example", "transition", "cta"
    index: int = 0
    character_count: int = 0

    def __post_init__(self):
        self.character_count = len(self.text)


@dataclass
class ThreadOutline:
    """Structured outline for a thread before generation."""

    topic: str
    hook: str  # Opening hook for first post
    key_points: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    cta: str = ""  # Call to action for final post
    target_segments: int = DEFAULT_SEGMENT_COUNT


class ThreadComposer:
    """
    Composes structured X threads with proper narrative flow.

    Thread structure follows a proven format:
    1. HOOK      - Attention-grabbing first post (🧵 indicator)
    2. CONTEXT   - Set up the problem/question
    3. POINTS    - Key insights/tips (numbered, one per post)
    4. EXAMPLES  - Supporting evidence/stories
    5. SUMMARY   - Recap the key takeaway
    6. CTA       - Call to action (follow, bookmark, repost)

    Each segment is optimized for standalone readability (some users
    only see individual posts in their feed, not the full thread).
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ThreadComposer"

    def enabled(self, request: ContentRequest) -> bool:
        return (
            self.config.enable_thread_generation
            and request.content_type == ContentType.THREAD
        )

    def create_outline(
        self,
        topic: Topic,
        key_points: list[str],
        target_segments: int = DEFAULT_SEGMENT_COUNT,
    ) -> ThreadOutline:
        """Create a structured thread outline from key points."""
        hook = f"🧵 Everything you need to know about {topic.name}:\n\nA thread 👇"

        cta = (
            f"That's a wrap!\n\n"
            f"If you found this helpful:\n"
            f"• Repost to share with your audience\n"
            f"• Follow for more {topic.category or topic.name} content\n"
            f"• Bookmark to save for later 🔖"
        )

        return ThreadOutline(
            topic=topic.name,
            hook=hook,
            key_points=key_points,
            cta=cta,
            target_segments=target_segments,
        )

    def compose_from_outline(
        self, outline: ThreadOutline, request: ContentRequest
    ) -> PostDraft:
        """
        Compose a complete thread from an outline.
        This generates a rule-based thread structure that can be
        enhanced by the LLM generator.
        """
        segments: list[ThreadSegment] = []

        # 1. Hook
        segments.append(
            ThreadSegment(text=outline.hook, role="hook", index=1)
        )

        # 2. Context / Setup (optional)
        if len(outline.key_points) > 3:
            context = f"Let me break down {len(outline.key_points)} key points about {outline.topic}:"
            segments.append(
                ThreadSegment(text=context, role="context", index=2)
            )

        # 3. Key points
        start_idx = len(segments) + 1
        for i, point in enumerate(outline.key_points):
            numbered = f"{start_idx + i}/ {point}"
            # Trim if too long
            if len(numbered) > MAX_SEGMENT_CHARS:
                numbered = numbered[: MAX_SEGMENT_CHARS - 3] + "..."
            segments.append(
                ThreadSegment(text=numbered, role="point", index=start_idx + i)
            )

        # 4. Examples (interleaved or at end)
        for i, example in enumerate(outline.examples):
            idx = len(segments) + 1
            text = f"Example: {example}"
            if len(text) > MAX_SEGMENT_CHARS:
                text = text[: MAX_SEGMENT_CHARS - 3] + "..."
            segments.append(
                ThreadSegment(text=text, role="example", index=idx)
            )

        # 5. CTA (final post)
        cta_text = outline.cta or self._default_cta(outline.topic)
        if len(cta_text) > MAX_SEGMENT_CHARS:
            cta_text = cta_text[: MAX_SEGMENT_CHARS - 3] + "..."
        segments.append(
            ThreadSegment(text=cta_text, role="cta", index=len(segments) + 1)
        )

        # Convert to PostDraft
        pieces = [
            ContentPiece(text=seg.text, content_type=ContentType.THREAD)
            for seg in segments
        ]

        topic_obj = request.topic or Topic(name=outline.topic)

        return PostDraft(
            id=f"thread-{uuid.uuid4().hex[:8]}",
            pieces=pieces,
            topic=topic_obj,
            tone=request.tone,
            goal=request.goal,
            tags=["thread", f"segments_{len(segments)}"],
            metadata={
                "generator": self.name,
                "segment_count": len(segments),
                "segment_roles": [s.role for s in segments],
            },
        )

    def validate_thread(self, draft: PostDraft) -> list[str]:
        """
        Validate a thread draft and return any issues found.
        Returns empty list if thread is valid.
        """
        issues = []
        max_segments = self.config.filters.thread_max_segments

        if not draft.is_thread:
            issues.append("Draft is not a thread (only 1 segment)")
            return issues

        if len(draft.pieces) > max_segments:
            issues.append(
                f"Thread has {len(draft.pieces)} segments, max is {max_segments}"
            )

        for i, piece in enumerate(draft.pieces):
            if piece.character_count > MAX_SEGMENT_CHARS:
                issues.append(
                    f"Segment {i + 1} is {piece.character_count} chars "
                    f"(max {MAX_SEGMENT_CHARS})"
                )

        # Check that first post has a hook indicator
        first_text = draft.pieces[0].text.lower()
        if "🧵" not in first_text and "thread" not in first_text:
            issues.append("First segment should indicate it's a thread (🧵 or 'thread')")

        return issues

    @staticmethod
    def _default_cta(topic: str) -> str:
        return (
            f"That's it for this thread!\n\n"
            f"If you learned something about {topic}:\n"
            f"→ Repost the first post\n"
            f"→ Follow for more insights"
        )
