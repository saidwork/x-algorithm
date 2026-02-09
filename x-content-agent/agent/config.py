"""
Agent configuration module.

Centralized configuration for all agent components. Follows the same pattern
as home-mixer's pipeline configuration where each component can be individually
tuned via weights, thresholds, and feature flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.types import ContentGoal, ContentTone, ContentType


@dataclass
class LLMConfig:
    """Configuration for the LLM provider used for content generation."""

    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.8
    max_tokens: int = 1024
    top_p: float = 0.95


@dataclass
class EngagementWeights:
    """
    Weights for engagement scoring, mirroring home-mixer's WeightedScorer config.
    Used by the engagement predictor to compute final content quality scores.
    """

    like: float = 1.0
    reply: float = 3.0
    repost: float = 5.0
    quote: float = 4.0
    click: float = 2.0
    bookmark: float = 2.5
    follow: float = 10.0
    profile_visit: float = 1.5
    # Negative weights
    mute: float = -20.0
    unfollow: float = -30.0
    report: float = -50.0


@dataclass
class ContentStrategyConfig:
    """Configuration for the content strategy engine."""

    # Topic selection
    max_topics_per_day: int = 10
    trend_relevance_threshold: float = 0.3
    min_topic_score: float = 0.2

    # Content mix - target percentages for each content type per week
    content_mix: dict[str, float] = field(
        default_factory=lambda: {
            ContentType.SHORT_POST.value: 0.50,
            ContentType.THREAD.value: 0.15,
            ContentType.QUOTE_POST.value: 0.15,
            ContentType.REPLY.value: 0.10,
            ContentType.POLL.value: 0.05,
            ContentType.LONG_POST.value: 0.05,
        }
    )

    # Tone mix - target percentages for each tone
    tone_mix: dict[str, float] = field(
        default_factory=lambda: {
            ContentTone.EDUCATIONAL.value: 0.30,
            ContentTone.PROFESSIONAL.value: 0.25,
            ContentTone.CASUAL.value: 0.20,
            ContentTone.PROVOCATIVE.value: 0.10,
            ContentTone.HUMOROUS.value: 0.10,
            ContentTone.INSPIRATIONAL.value: 0.05,
        }
    )

    # Posting schedule
    posts_per_day: int = 3
    min_hours_between_posts: float = 2.0
    active_hours_utc: tuple[int, int] = (8, 23)  # Start, end hour


@dataclass
class FilterConfig:
    """
    Content filter configuration.
    Analogous to home-mixer's filter chain but for generated content quality.
    """

    # Quality thresholds
    min_engagement_score: float = 0.1
    max_similarity_to_recent: float = 0.85  # Prevent repetitive content
    min_readability_score: float = 0.5

    # Content safety
    forbidden_words: list[str] = field(default_factory=list)
    sensitive_topics: list[str] = field(default_factory=list)

    # Character limits
    short_post_max_chars: int = 280
    long_post_max_chars: int = 25000
    thread_max_segments: int = 15

    # Hashtag limits
    max_hashtags: int = 3
    min_hashtags: int = 0


@dataclass
class AnalyticsConfig:
    """Configuration for analytics and feedback loop."""

    # Tracking
    track_impressions: bool = True
    track_engagement: bool = True
    metrics_retention_days: int = 90

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_sample_size: int = 100
    ab_test_confidence_level: float = 0.95

    # Feedback loop
    feedback_loop_enabled: bool = True
    retrain_interval_hours: int = 168  # Weekly
    min_posts_for_feedback: int = 20


@dataclass
class AgentConfig:
    """
    Master configuration for the X Content Agent.
    All component configs are nested here for centralized management.
    """

    # Agent identity
    agent_name: str = "x-content-agent"
    account_id: str = ""
    account_handle: str = ""

    # Component configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    engagement_weights: EngagementWeights = field(default_factory=EngagementWeights)
    strategy: ContentStrategyConfig = field(default_factory=ContentStrategyConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    # Default content preferences
    default_tone: ContentTone = ContentTone.PROFESSIONAL
    default_goal: ContentGoal = ContentGoal.ENGAGEMENT
    default_language: str = "en"

    # Niche / domain of the account
    niche: str = ""
    niche_keywords: list[str] = field(default_factory=list)
    brand_description: str = ""

    # Feature flags
    auto_publish: bool = False  # If False, all posts go to review queue
    enable_media_suggestions: bool = True
    enable_thread_generation: bool = True
    enable_reply_suggestions: bool = True
    enable_trend_monitoring: bool = True
