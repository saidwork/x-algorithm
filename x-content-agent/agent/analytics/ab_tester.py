"""
A/B testing framework for content variants.

Enables systematic testing of different content approaches (tones, hooks,
formats, posting times) to continuously improve content performance.
Similar to how the recommendation system uses multi-action predictions
to optimize ranking, this module optimizes content creation decisions.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.types import PerformanceMetrics, PostDraft

logger = logging.getLogger(__name__)


@dataclass
class ABTestVariant:
    """A variant in an A/B test."""

    variant_id: str
    name: str
    description: str
    draft: PostDraft | None = None
    metrics: PerformanceMetrics | None = None
    sample_size: int = 0


@dataclass
class ABTest:
    """An A/B test comparing content variants."""

    test_id: str
    name: str
    hypothesis: str
    variable: str  # What we're testing: "tone", "hook_type", "posting_time", "format"
    variants: list[ABTestVariant] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    concluded_at: datetime | None = None
    winner: str | None = None  # variant_id of the winner
    is_active: bool = True
    confidence_level: float = 0.95

    @property
    def is_concluded(self) -> bool:
        return self.winner is not None


@dataclass
class ABTestResult:
    """Result of analyzing an A/B test."""

    test_id: str
    winner_variant: str
    loser_variant: str
    improvement_pct: float  # % improvement of winner over loser
    confidence: float  # Statistical confidence
    is_significant: bool
    recommendation: str


class ABTester:
    """
    A/B testing framework for content strategy optimization.

    Test types:
    1. Tone Test      - Same topic, different tones (casual vs professional)
    2. Hook Test      - Same content, different opening hooks
    3. Format Test    - Same topic, different formats (post vs thread vs poll)
    4. Time Test      - Same content, posted at different times
    5. CTA Test       - Same content, different calls to action
    6. Hashtag Test   - Same content, with/without hashtags

    Each test generates variants, tracks their performance, and uses
    statistical analysis to determine the winner.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._tests: dict[str, ABTest] = {}

    def create_test(
        self,
        name: str,
        hypothesis: str,
        variable: str,
        variant_names: list[str],
    ) -> ABTest:
        """Create a new A/B test."""
        test_id = f"ab-{uuid.uuid4().hex[:8]}"

        variants = [
            ABTestVariant(
                variant_id=f"{test_id}-{i}",
                name=vname,
                description=f"Variant {i}: {vname}",
            )
            for i, vname in enumerate(variant_names)
        ]

        test = ABTest(
            test_id=test_id,
            name=name,
            hypothesis=hypothesis,
            variable=variable,
            variants=variants,
        )

        self._tests[test_id] = test
        logger.info(
            "Created A/B test '%s' (%s) with %d variants",
            name,
            test_id,
            len(variants),
        )
        return test

    def assign_variant(self, test_id: str, draft: PostDraft) -> ABTestVariant | None:
        """
        Assign a draft to a test variant.
        Uses random assignment for unbiased testing.
        """
        test = self._tests.get(test_id)
        if not test or not test.is_active:
            return None

        # Find variant with fewest samples (balanced assignment)
        variant = min(test.variants, key=lambda v: v.sample_size)
        variant.draft = draft
        variant.sample_size += 1
        draft.variant_group = variant.variant_id

        return variant

    def record_result(
        self, test_id: str, variant_id: str, metrics: PerformanceMetrics
    ) -> None:
        """Record performance metrics for a test variant."""
        test = self._tests.get(test_id)
        if not test:
            return

        for variant in test.variants:
            if variant.variant_id == variant_id:
                variant.metrics = metrics
                variant.sample_size += 1
                break

    def analyze_test(self, test_id: str) -> ABTestResult | None:
        """
        Analyze an A/B test and determine if there's a statistically
        significant winner.
        """
        test = self._tests.get(test_id)
        if not test or len(test.variants) < 2:
            return None

        # Need metrics for all variants
        variants_with_metrics = [v for v in test.variants if v.metrics]
        if len(variants_with_metrics) < 2:
            return None

        # Sort by engagement rate
        sorted_variants = sorted(
            variants_with_metrics,
            key=lambda v: v.metrics.engagement_rate if v.metrics else 0,
            reverse=True,
        )

        best = sorted_variants[0]
        worst = sorted_variants[-1]

        best_rate = best.metrics.engagement_rate if best.metrics else 0
        worst_rate = worst.metrics.engagement_rate if worst.metrics else 0

        # Calculate improvement
        improvement = (best_rate - worst_rate) / max(worst_rate, 0.001) * 100

        # Simple statistical significance check
        # Using Z-test approximation for proportions
        confidence = self._compute_confidence(best, worst)
        is_significant = confidence >= test.confidence_level

        if is_significant:
            test.winner = best.variant_id
            test.concluded_at = datetime.now()
            test.is_active = False

        recommendation = (
            f"Variant '{best.name}' outperforms '{worst.name}' by {improvement:.1f}%"
        )
        if is_significant:
            recommendation += f" (statistically significant at {confidence:.1%} confidence)"
        else:
            recommendation += " (not yet statistically significant, need more data)"

        result = ABTestResult(
            test_id=test_id,
            winner_variant=best.variant_id,
            loser_variant=worst.variant_id,
            improvement_pct=improvement,
            confidence=confidence,
            is_significant=is_significant,
            recommendation=recommendation,
        )

        logger.info("A/B test %s analysis: %s", test_id, recommendation)
        return result

    def get_active_tests(self) -> list[ABTest]:
        return [t for t in self._tests.values() if t.is_active]

    def get_concluded_tests(self) -> list[ABTest]:
        return [t for t in self._tests.values() if t.is_concluded]

    def get_learnings(self) -> list[str]:
        """Extract key learnings from all concluded tests."""
        learnings = []
        for test in self.get_concluded_tests():
            result = self.analyze_test(test.test_id)
            if result and result.is_significant:
                learnings.append(
                    f"[{test.variable}] {result.recommendation}"
                )
        return learnings

    @staticmethod
    def _compute_confidence(best: ABTestVariant, worst: ABTestVariant) -> float:
        """
        Compute statistical confidence using Z-test for proportions.
        Returns confidence level (0.0 - 1.0).
        """
        if not best.metrics or not worst.metrics:
            return 0.0

        p1 = best.metrics.engagement_rate
        p2 = worst.metrics.engagement_rate
        n1 = max(best.sample_size, 1)
        n2 = max(worst.sample_size, 1)

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        if p_pool <= 0 or p_pool >= 1:
            return 0.0

        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return 0.0

        # Z-score
        z = abs(p1 - p2) / se

        # Approximate confidence from Z-score (using lookup)
        # z=1.645 -> 90%, z=1.96 -> 95%, z=2.576 -> 99%
        if z >= 2.576:
            return 0.99
        elif z >= 1.96:
            return 0.95
        elif z >= 1.645:
            return 0.90
        elif z >= 1.28:
            return 0.80
        else:
            return min(0.5 + z * 0.2, 0.79)
