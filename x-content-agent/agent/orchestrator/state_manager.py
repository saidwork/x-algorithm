"""
Agent state management module.

Manages the persistent state of the agent including:
- Published post history
- Draft queue
- Strategy configuration state
- Feedback loop learnings
- A/B test state
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from agent.types import PostDraft, PostStatus

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Serializable agent state."""

    # Queues
    draft_queue: list[str] = field(default_factory=list)  # Draft IDs awaiting review
    scheduled_queue: list[str] = field(default_factory=list)  # Draft IDs scheduled
    published_ids: list[str] = field(default_factory=list)  # Published draft IDs

    # Counters
    total_generated: int = 0
    total_published: int = 0
    total_filtered: int = 0

    # State
    last_generation_time: str = ""
    last_publish_time: str = ""
    last_feedback_analysis: str = ""

    # Learnings from feedback loop
    learned_preferences: dict[str, float] = field(default_factory=dict)


class StateManager:
    """
    Manages persistent agent state.

    Provides:
    - Draft queue management (add, review, approve, reject)
    - Publishing queue management
    - State persistence (save/load to JSON)
    - State snapshots for debugging
    """

    def __init__(self, state_dir: str = ".agent_state"):
        self._state_dir = Path(state_dir)
        self._state = AgentState()
        self._drafts: dict[str, PostDraft] = {}  # In-memory draft storage

    def add_to_draft_queue(self, draft: PostDraft) -> None:
        """Add a draft to the review queue."""
        self._drafts[draft.id] = draft
        draft.status = PostStatus.REVIEW
        self._state.draft_queue.append(draft.id)
        self._state.total_generated += 1
        logger.info("Added draft %s to review queue", draft.id)

    def approve_draft(self, draft_id: str) -> PostDraft | None:
        """Approve a draft and move it to the scheduled queue."""
        draft = self._drafts.get(draft_id)
        if not draft:
            return None

        if draft_id in self._state.draft_queue:
            self._state.draft_queue.remove(draft_id)
        self._state.scheduled_queue.append(draft_id)
        draft.status = PostStatus.SCHEDULED
        logger.info("Approved draft %s -> scheduled", draft_id)
        return draft

    def reject_draft(self, draft_id: str, reason: str = "") -> None:
        """Reject a draft from the queue."""
        if draft_id in self._state.draft_queue:
            self._state.draft_queue.remove(draft_id)
        self._state.total_filtered += 1
        draft = self._drafts.get(draft_id)
        if draft:
            draft.status = PostStatus.FAILED
            draft.metadata["rejection_reason"] = reason
        logger.info("Rejected draft %s: %s", draft_id, reason)

    def mark_published(self, draft_id: str) -> None:
        """Mark a draft as published."""
        if draft_id in self._state.scheduled_queue:
            self._state.scheduled_queue.remove(draft_id)
        self._state.published_ids.append(draft_id)
        self._state.total_published += 1
        self._state.last_publish_time = datetime.now().isoformat()

        draft = self._drafts.get(draft_id)
        if draft:
            draft.status = PostStatus.PUBLISHED

        logger.info("Draft %s marked as published", draft_id)

    def get_draft(self, draft_id: str) -> PostDraft | None:
        return self._drafts.get(draft_id)

    def get_pending_drafts(self) -> list[PostDraft]:
        """Get all drafts awaiting review."""
        return [
            self._drafts[did]
            for did in self._state.draft_queue
            if did in self._drafts
        ]

    def get_scheduled_drafts(self) -> list[PostDraft]:
        """Get all scheduled drafts."""
        return [
            self._drafts[did]
            for did in self._state.scheduled_queue
            if did in self._drafts
        ]

    def get_stats(self) -> dict[str, int]:
        """Get agent statistics."""
        return {
            "drafts_in_queue": len(self._state.draft_queue),
            "scheduled": len(self._state.scheduled_queue),
            "published": self._state.total_published,
            "total_generated": self._state.total_generated,
            "total_filtered": self._state.total_filtered,
        }

    def save_state(self) -> None:
        """Persist state to disk."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        state_file = self._state_dir / "state.json"

        state_dict = asdict(self._state)
        state_file.write_text(json.dumps(state_dict, indent=2, default=str))
        logger.info("Saved agent state to %s", state_file)

    def load_state(self) -> None:
        """Load state from disk."""
        state_file = self._state_dir / "state.json"
        if not state_file.exists():
            logger.info("No existing state found, starting fresh")
            return

        try:
            data = json.loads(state_file.read_text())
            self._state = AgentState(**data)
            logger.info("Loaded agent state from %s", state_file)
        except Exception:
            logger.exception("Failed to load state, starting fresh")
            self._state = AgentState()
