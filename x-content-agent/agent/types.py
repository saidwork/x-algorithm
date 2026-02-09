"""
Core type definitions for the X Content Agent.

Mirrors the candidate-pipeline's type system but for content generation:
- Candidate -> ContentPiece (generated content instead of retrieved posts)
- Query -> ContentRequest (generation parameters instead of user feed request)
- Scores -> EngagementPrediction (predicted engagement for generated content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContentType(Enum):
    """Type of content to generate."""

    SHORT_POST = "short_post"  # Standard post (<=280 chars)
    LONG_POST = "long_post"  # Long-form post (X Premium)
    THREAD = "thread"  # Multi-post thread
    QUOTE_POST = "quote_post"  # Quote post with commentary
    REPLY = "reply"  # Reply to another post
    POLL = "poll"  # Poll post


class PostStatus(Enum):
    """Lifecycle status of a post draft."""

    DRAFT = "draft"
    REVIEW = "review"  # Awaiting human review
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"


class ContentTone(Enum):
    """Tone of the generated content."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    EDUCATIONAL = "educational"
    PROVOCATIVE = "provocative"  # Thought-provoking / debate-starting
    INSPIRATIONAL = "inspirational"
    NEWS = "news"  # Breaking news / commentary


class ContentGoal(Enum):
    """Strategic goal for the content."""

    ENGAGEMENT = "engagement"  # Maximize likes, replies, reposts
    REACH = "reach"  # Maximize impressions
    FOLLOWER_GROWTH = "follower_growth"  # Attract new followers
    BRAND_AWARENESS = "brand_awareness"  # Build brand recognition
    TRAFFIC = "traffic"  # Drive clicks to external links
    COMMUNITY = "community"  # Build community / start conversations
    AUTHORITY = "authority"  # Establish thought leadership


class MediaType(Enum):
    """Type of media attachment."""

    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    POLL = "poll"
    LINK_CARD = "link_card"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Topic:
    """A topic/subject for content generation."""

    name: str
    keywords: list[str] = field(default_factory=list)
    category: str = ""
    trending_score: float = 0.0  # 0.0 - 1.0, how trending the topic is
    relevance_score: float = 0.0  # 0.0 - 1.0, relevance to audience
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendSignal:
    """A detected trend signal from the platform."""

    topic: str
    volume: int = 0  # Number of posts about this topic
    velocity: float = 0.0  # Rate of change in volume
    sentiment: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    geographic_scope: str = "global"  # global, country, city
    category: str = ""
    peak_time: datetime | None = None
    related_topics: list[str] = field(default_factory=list)
    source: str = ""  # Where the trend was detected


@dataclass
class AudienceProfile:
    """Profile of the target audience."""

    total_followers: int = 0
    active_followers: int = 0
    demographics: dict[str, float] = field(default_factory=dict)  # e.g. {"tech": 0.6}
    interests: list[str] = field(default_factory=list)
    peak_active_hours: list[int] = field(default_factory=list)  # Hours in UTC
    avg_engagement_rate: float = 0.0
    top_performing_topics: list[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class MediaSuggestion:
    """Suggested media to accompany a post."""

    media_type: MediaType
    description: str  # Description of what the media should be
    alt_text: str = ""  # Accessibility text
    source_url: str = ""  # URL if referencing existing media
    generation_prompt: str = ""  # Prompt if media needs to be generated


@dataclass
class EngagementPrediction:
    """
    Predicted engagement metrics for a content piece.

    Mirrors Phoenix's multi-action prediction but for content we're creating:
    instead of P(user engages with existing post), this predicts
    P(audience engages with our generated content).
    """

    p_like: float = 0.0
    p_reply: float = 0.0
    p_repost: float = 0.0
    p_quote: float = 0.0
    p_click: float = 0.0
    p_bookmark: float = 0.0
    p_follow: float = 0.0  # P(viewer follows after seeing post)
    p_profile_visit: float = 0.0

    # Negative signals
    p_mute: float = 0.0
    p_unfollow: float = 0.0
    p_report: float = 0.0

    # Aggregate scores
    predicted_impressions: int = 0
    predicted_engagement_rate: float = 0.0
    virality_score: float = 0.0  # 0.0 - 1.0

    @property
    def weighted_score(self) -> float:
        """
        Weighted engagement score, similar to home-mixer's WeightedScorer.
        Positive engagement weighted positively, negative signals weighted negatively.
        """
        return (
            1.0 * self.p_like
            + 3.0 * self.p_reply
            + 5.0 * self.p_repost
            + 4.0 * self.p_quote
            + 2.0 * self.p_click
            + 2.5 * self.p_bookmark
            + 10.0 * self.p_follow
            + 1.5 * self.p_profile_visit
            - 20.0 * self.p_mute
            - 30.0 * self.p_unfollow
            - 50.0 * self.p_report
        )


@dataclass
class ContentPiece:
    """
    A single piece of content (post or thread segment).

    This is the "candidate" in our content pipeline, analogous to the
    recommendation system's scored post candidate.
    """

    text: str
    content_type: ContentType = ContentType.SHORT_POST
    media: list[MediaSuggestion] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    url: str = ""  # External link
    poll_options: list[str] = field(default_factory=list)  # For poll type
    character_count: int = 0

    def __post_init__(self):
        self.character_count = len(self.text)


@dataclass
class PostDraft:
    """
    A complete post draft ready for review/publishing.

    Aggregates content pieces (single for posts, multiple for threads),
    engagement predictions, scheduling info, and metadata.
    """

    id: str
    pieces: list[ContentPiece] = field(default_factory=list)
    topic: Topic | None = None
    tone: ContentTone = ContentTone.PROFESSIONAL
    goal: ContentGoal = ContentGoal.ENGAGEMENT
    engagement_prediction: EngagementPrediction | None = None
    status: PostStatus = PostStatus.DRAFT
    scheduled_time: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)  # Internal tags
    variant_group: str = ""  # A/B testing group
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n---\n".join(p.text for p in self.pieces)

    @property
    def total_characters(self) -> int:
        return sum(p.character_count for p in self.pieces)

    @property
    def is_thread(self) -> bool:
        return len(self.pieces) > 1


@dataclass
class ContentRequest:
    """
    Request to generate content - analogous to home-mixer's Query.
    Carries all context needed for the content generation pipeline.
    """

    request_id: str
    account_id: str
    topic: Topic | None = None
    content_type: ContentType = ContentType.SHORT_POST
    tone: ContentTone = ContentTone.PROFESSIONAL
    goal: ContentGoal = ContentGoal.ENGAGEMENT
    audience: AudienceProfile | None = None
    num_variants: int = 1  # Number of content variants to generate
    constraints: ContentConstraints | None = None
    context: dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class ContentConstraints:
    """Constraints for content generation."""

    max_characters: int = 280
    required_hashtags: list[str] = field(default_factory=list)
    forbidden_words: list[str] = field(default_factory=list)
    required_mentions: list[str] = field(default_factory=list)
    include_media: bool = False
    include_url: bool = False
    language: str = "en"
    brand_voice_keywords: list[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Actual performance metrics of a published post."""

    post_id: str
    impressions: int = 0
    likes: int = 0
    replies: int = 0
    reposts: int = 0
    quotes: int = 0
    clicks: int = 0
    bookmarks: int = 0
    profile_visits: int = 0
    new_followers: int = 0
    engagement_rate: float = 0.0
    measured_at: datetime = field(default_factory=datetime.now)

    @property
    def total_engagements(self) -> int:
        return self.likes + self.replies + self.reposts + self.quotes + self.clicks + self.bookmarks
