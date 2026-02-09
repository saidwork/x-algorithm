"""Tests for the content generation pipeline."""

import pytest

from agent.pipeline import ContentPipeline, FilterResult, PipelineResult
from agent.types import (
    ContentPiece,
    ContentRequest,
    ContentType,
    EngagementPrediction,
    PostDraft,
    PostStatus,
    Topic,
)


# ---------------------------------------------------------------------------
# Mock pipeline components
# ---------------------------------------------------------------------------


class MockGenerator:
    @property
    def name(self) -> str:
        return "MockGenerator"

    def enabled(self, request):
        return True

    async def generate(self, request):
        return [
            PostDraft(
                id=f"{request.request_id}-v{i}",
                pieces=[ContentPiece(text=f"Generated content variant {i} about {request.topic.name if request.topic else 'general'}")],
                topic=request.topic,
                tone=request.tone,
                goal=request.goal,
            )
            for i in range(request.num_variants)
        ]


class MockScorer:
    @property
    def name(self) -> str:
        return "MockScorer"

    def enabled(self, request):
        return True

    async def score(self, request, drafts):
        predictions = []
        for i, draft in enumerate(drafts):
            # Give different scores to different variants
            base = 0.1 * (len(drafts) - i)
            predictions.append(
                EngagementPrediction(
                    p_like=base,
                    p_reply=base * 0.2,
                    p_repost=base * 0.3,
                    predicted_engagement_rate=base,
                )
            )
        return predictions


class MockFilter:
    def __init__(self, keep_count=None):
        self._keep_count = keep_count

    @property
    def name(self) -> str:
        return "MockFilter"

    def enabled(self, request):
        return True

    async def filter(self, request, drafts):
        if self._keep_count is not None:
            return FilterResult(
                kept=drafts[: self._keep_count],
                removed=drafts[self._keep_count:],
                removal_reasons={d.id: "filtered" for d in drafts[self._keep_count:]},
            )
        return FilterResult(kept=drafts, removed=[])


class MockHydrator:
    @property
    def name(self) -> str:
        return "MockHydrator"

    def enabled(self, request):
        return True

    async def hydrate(self, request, drafts):
        for draft in drafts:
            draft.metadata["hydrated"] = True
        return drafts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContentPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        pipeline = ContentPipeline()
        request = ContentRequest(
            request_id="test-1",
            account_id="acc-1",
            topic=Topic(name="test"),
        )
        result = await pipeline.execute(request)
        assert isinstance(result, PipelineResult)
        assert result.selected_drafts == []
        assert result.generated_drafts == []

    @pytest.mark.asyncio
    async def test_generate_only(self):
        pipeline = ContentPipeline(generators=[MockGenerator()])
        request = ContentRequest(
            request_id="test-2",
            account_id="acc-1",
            topic=Topic(name="AI"),
            num_variants=3,
        )
        result = await pipeline.execute(request)
        assert len(result.generated_drafts) == 3
        assert len(result.selected_drafts) == 3

    @pytest.mark.asyncio
    async def test_generate_and_score(self):
        pipeline = ContentPipeline(
            generators=[MockGenerator()],
            scorers=[MockScorer()],
        )
        request = ContentRequest(
            request_id="test-3",
            account_id="acc-1",
            topic=Topic(name="ML"),
            num_variants=5,
        )
        result = await pipeline.execute(request)

        # All drafts should have predictions
        for draft in result.selected_drafts:
            assert draft.engagement_prediction is not None

        # Should be sorted by score (first = highest)
        scores = [d.engagement_prediction.weighted_score for d in result.selected_drafts]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_filter_removes_drafts(self):
        pipeline = ContentPipeline(
            generators=[MockGenerator()],
            filters=[MockFilter(keep_count=2)],
        )
        request = ContentRequest(
            request_id="test-4",
            account_id="acc-1",
            topic=Topic(name="test"),
            num_variants=5,
        )
        result = await pipeline.execute(request)
        assert len(result.selected_drafts) == 2
        assert len(result.filtered_drafts) == 3

    @pytest.mark.asyncio
    async def test_result_size_limit(self):
        pipeline = ContentPipeline(
            generators=[MockGenerator()],
            result_size=2,
        )
        request = ContentRequest(
            request_id="test-5",
            account_id="acc-1",
            topic=Topic(name="test"),
            num_variants=10,
        )
        result = await pipeline.execute(request)
        assert len(result.selected_drafts) <= 2

    @pytest.mark.asyncio
    async def test_hydrator_enriches_drafts(self):
        pipeline = ContentPipeline(
            generators=[MockGenerator()],
            draft_hydrators=[MockHydrator()],
        )
        request = ContentRequest(
            request_id="test-6",
            account_id="acc-1",
            topic=Topic(name="test"),
            num_variants=2,
        )
        result = await pipeline.execute(request)
        for draft in result.selected_drafts:
            assert draft.metadata.get("hydrated") is True

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test with all components assembled."""
        pipeline = ContentPipeline(
            generators=[MockGenerator()],
            draft_hydrators=[MockHydrator()],
            filters=[MockFilter(keep_count=3)],
            scorers=[MockScorer()],
            result_size=2,
        )
        request = ContentRequest(
            request_id="test-full",
            account_id="acc-1",
            topic=Topic(name="full pipeline test"),
            num_variants=5,
        )
        result = await pipeline.execute(request)

        assert len(result.generated_drafts) == 5
        assert len(result.filtered_drafts) == 2
        assert len(result.selected_drafts) == 2

        for draft in result.selected_drafts:
            assert draft.metadata.get("hydrated") is True
            assert draft.engagement_prediction is not None
