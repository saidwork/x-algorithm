"""
Engagement prediction module.

Predicts how well a content draft will perform based on historical data,
content features, and audience characteristics. This is the content-side
analog of Phoenix's ranking model - instead of predicting P(user engages
with existing post), we predict P(audience engages with our generated content).

The predictor uses a feature-based scoring model that can be enhanced with
actual ML models trained on account-specific performance data.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime

from agent.config import AgentConfig, EngagementWeights
from agent.types import (
    AudienceProfile,
    ContentGoal,
    ContentRequest,
    ContentType,
    EngagementPrediction,
    PostDraft,
)

logger = logging.getLogger(__name__)


# Feature importance weights (learned from X engagement patterns)
FEATURE_WEIGHTS = {
    "has_media": 0.15,
    "has_question": 0.12,
    "has_numbers": 0.08,
    "has_emoji": 0.05,
    "is_thread": 0.10,
    "is_timely": 0.12,  # Matches trending topic
    "char_length_optimal": 0.08,  # ~70-140 chars tend to perform best
    "has_hook": 0.15,
    "has_cta": 0.10,
    "readability": 0.05,
}


class EngagementPredictor:
    """
    Predicts engagement metrics for content drafts.

    Pipeline role: EngagementScorer - scores drafts by predicted performance.

    Prediction approach (multi-signal, like Phoenix):
    1. Extract content features (has_media, length, question, etc.)
    2. Compute feature-based base engagement probability
    3. Apply audience-specific multipliers
    4. Apply temporal multipliers (time of day, day of week)
    5. Apply goal-specific adjustments
    6. Output per-action predictions (P(like), P(reply), P(repost), etc.)

    The model can be replaced with a trained ML model using the same
    EngagementPrediction output format.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._weights = config.engagement_weights
        self._historical_rates: dict[str, float] = {}  # content_feature -> avg engagement rate

    @property
    def name(self) -> str:
        return "EngagementPredictor"

    def enabled(self, request: ContentRequest) -> bool:
        return True

    async def score(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> list[EngagementPrediction]:
        """Score all drafts (pipeline EngagementScorer interface)."""
        predictions = []
        for draft in drafts:
            prediction = self.predict(draft, request.audience)
            predictions.append(prediction)
        return predictions

    def predict(
        self,
        draft: PostDraft,
        audience: AudienceProfile | None = None,
    ) -> EngagementPrediction:
        """
        Predict engagement for a single draft.

        Returns an EngagementPrediction with probabilities for each action type,
        mirroring Phoenix's multi-action prediction output.
        """
        text = draft.full_text
        features = self._extract_features(draft)
        base_score = self._compute_base_score(features)

        # Audience multiplier
        audience_mult = self._audience_multiplier(draft, audience)

        # Temporal multiplier
        temporal_mult = self._temporal_multiplier(draft)

        # Content type multiplier
        type_mult = self._content_type_multiplier(draft)

        # Combined base engagement probability
        engagement_prob = base_score * audience_mult * temporal_mult * type_mult
        engagement_prob = max(0.001, min(0.5, engagement_prob))  # Clamp to realistic range

        # Distribute across action types
        # These ratios are based on typical X engagement distributions
        prediction = EngagementPrediction(
            p_like=engagement_prob * 0.60,
            p_reply=engagement_prob * 0.08 * (1.5 if features["has_question"] else 1.0),
            p_repost=engagement_prob * 0.12 * (1.3 if features["is_thread"] else 1.0),
            p_quote=engagement_prob * 0.04,
            p_click=engagement_prob * 0.10,
            p_bookmark=engagement_prob * 0.05 * (1.5 if features["is_thread"] else 1.0),
            p_follow=engagement_prob * 0.01,
            p_profile_visit=engagement_prob * 0.03,
            # Negative signals (should be very low for good content)
            p_mute=max(0.0, 0.001 - engagement_prob * 0.01),
            p_unfollow=max(0.0, 0.0005 - engagement_prob * 0.005),
            p_report=0.0001,
            # Aggregate estimates
            predicted_impressions=self._estimate_impressions(engagement_prob, audience),
            predicted_engagement_rate=engagement_prob,
            virality_score=self._compute_virality_score(features, engagement_prob),
        )

        return prediction

    def _extract_features(self, draft: PostDraft) -> dict[str, float]:
        """Extract content features for scoring."""
        text = draft.full_text
        char_count = len(text)

        # Optimal length analysis (70-140 chars perform best for short posts)
        if draft.pieces and draft.pieces[0].content_type != ContentType.LONG_POST:
            length_score = 1.0 - abs(char_count - 105) / 200
            length_score = max(0.0, min(1.0, length_score))
        else:
            length_score = 0.7  # Long posts have different dynamics

        has_media = any(bool(p.media) for p in draft.pieces)
        has_question = "?" in text
        has_numbers = bool(re.search(r"\d+", text))
        has_emoji = bool(re.search(r"[\U0001F300-\U0001F9FF]", text))
        has_hook = bool(re.search(
            r"^(How|Why|What|Stop|Don\'t|Most people|\d+\s)", text, re.IGNORECASE
        ))
        has_cta = bool(re.search(
            r"(follow|repost|share|bookmark|comment|reply|thoughts\??)", text, re.IGNORECASE
        ))

        return {
            "has_media": 1.0 if has_media else 0.0,
            "has_question": 1.0 if has_question else 0.0,
            "has_numbers": 1.0 if has_numbers else 0.0,
            "has_emoji": 1.0 if has_emoji else 0.0,
            "is_thread": 1.0 if draft.is_thread else 0.0,
            "is_timely": draft.topic.trending_score if draft.topic else 0.0,
            "char_length_optimal": length_score,
            "has_hook": 1.0 if has_hook else 0.0,
            "has_cta": 1.0 if has_cta else 0.0,
            "readability": 0.7,  # Placeholder, computed by optimizer
        }

    def _compute_base_score(self, features: dict[str, float]) -> float:
        """Compute base engagement score from content features."""
        score = 0.0
        for feature_name, weight in FEATURE_WEIGHTS.items():
            feature_val = features.get(feature_name, 0.0)
            score += weight * feature_val

        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))

    def _audience_multiplier(
        self, draft: PostDraft, audience: AudienceProfile | None
    ) -> float:
        """Audience-specific engagement multiplier."""
        if not audience:
            return 1.0

        mult = 1.0

        # Larger, more engaged audiences get higher base predictions
        if audience.avg_engagement_rate > 0:
            mult *= 1.0 + audience.avg_engagement_rate * 5

        # Topic match boost
        if draft.topic and audience.top_performing_topics:
            topic_lower = draft.topic.name.lower()
            if any(t.lower() in topic_lower for t in audience.top_performing_topics):
                mult *= 1.3

        return mult

    def _temporal_multiplier(self, draft: PostDraft) -> float:
        """Time-based engagement multiplier."""
        if not draft.scheduled_time:
            return 1.0

        hour = draft.scheduled_time.hour
        day = draft.scheduled_time.weekday()

        # Peak hours (9-12, 17-21 UTC) get boost
        peak_hours = {9, 10, 11, 12, 17, 18, 19, 20, 21}
        hour_mult = 1.2 if hour in peak_hours else 0.8

        # Weekdays slightly better for professional content
        day_mult = 1.1 if day < 5 else 0.9

        return hour_mult * day_mult

    def _content_type_multiplier(self, draft: PostDraft) -> float:
        """Content type engagement multiplier based on platform norms."""
        type_multipliers = {
            ContentType.SHORT_POST: 1.0,
            ContentType.LONG_POST: 0.8,
            ContentType.THREAD: 1.3,  # Threads tend to get higher engagement
            ContentType.QUOTE_POST: 1.1,
            ContentType.REPLY: 0.6,  # Replies get less visibility
            ContentType.POLL: 1.4,  # Polls drive high engagement
        }
        ct = draft.pieces[0].content_type if draft.pieces else ContentType.SHORT_POST
        return type_multipliers.get(ct, 1.0)

    def _estimate_impressions(
        self, engagement_prob: float, audience: AudienceProfile | None
    ) -> int:
        """Estimate impression count based on engagement and audience size."""
        follower_count = audience.total_followers if audience else 1000
        # X typically shows posts to 5-15% of followers, more for engaging content
        base_reach = follower_count * 0.10
        engagement_boost = 1.0 + engagement_prob * 10  # High engagement -> algorithmic boost
        return int(base_reach * engagement_boost)

    def _compute_virality_score(
        self, features: dict[str, float], engagement_prob: float
    ) -> float:
        """
        Estimate virality potential (0.0 - 1.0).
        Viral content = high repost + quote probability.
        """
        virality = (
            engagement_prob * 0.3
            + features.get("is_timely", 0) * 0.3
            + features.get("has_hook", 0) * 0.2
            + features.get("has_media", 0) * 0.1
            + features.get("is_thread", 0) * 0.1
        )
        return max(0.0, min(1.0, virality))
