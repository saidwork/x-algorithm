"""
Content generation pipeline framework.

Mirrors the candidate-pipeline architecture from the recommendation system:
  Source -> Hydrator -> Filter -> Scorer -> Selector -> SideEffect

But adapted for content generation:
  TopicSource -> ContentGenerator -> ContentFilter -> EngagementScorer -> ContentSelector -> Publisher

Each stage is defined as a Protocol (similar to Rust traits) that can be
independently implemented and composed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent.types import ContentRequest, EngagementPrediction, PostDraft

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Stage Protocols (analogous to candidate-pipeline traits)
# ---------------------------------------------------------------------------


@runtime_checkable
class TopicSource(Protocol):
    """
    Fetches candidate topics for content generation.
    Analogous to candidate-pipeline's Source trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def get_topics(self, request: ContentRequest) -> list[dict]: ...


@runtime_checkable
class RequestHydrator(Protocol):
    """
    Enriches a ContentRequest with additional context.
    Analogous to candidate-pipeline's QueryHydrator trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def hydrate(self, request: ContentRequest) -> ContentRequest: ...


@runtime_checkable
class ContentGenerator(Protocol):
    """
    Generates content drafts from a request.
    This is the core creative stage - no direct analog in the recommendation system.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def generate(self, request: ContentRequest) -> list[PostDraft]: ...


@runtime_checkable
class DraftHydrator(Protocol):
    """
    Enriches drafts with additional data (media suggestions, hashtags, etc.).
    Analogous to candidate-pipeline's Hydrator trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def hydrate(self, request: ContentRequest, drafts: list[PostDraft]) -> list[PostDraft]: ...


@runtime_checkable
class ContentFilter(Protocol):
    """
    Filters out low-quality or inappropriate drafts.
    Analogous to candidate-pipeline's Filter trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def filter(self, request: ContentRequest, drafts: list[PostDraft]) -> FilterResult: ...


@runtime_checkable
class EngagementScorer(Protocol):
    """
    Scores drafts by predicted engagement.
    Analogous to candidate-pipeline's Scorer trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def score(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> list[EngagementPrediction]: ...


@runtime_checkable
class ContentSelector(Protocol):
    """
    Selects the best drafts from scored candidates.
    Analogous to candidate-pipeline's Selector trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    def select(self, request: ContentRequest, drafts: list[PostDraft]) -> list[PostDraft]: ...


@runtime_checkable
class PostProcessor(Protocol):
    """
    Post-selection processing (scheduling, formatting, etc.).
    Analogous to candidate-pipeline's SideEffect trait.
    """

    @property
    def name(self) -> str: ...

    def enabled(self, request: ContentRequest) -> bool: ...

    async def process(self, request: ContentRequest, drafts: list[PostDraft]) -> None: ...


# ---------------------------------------------------------------------------
# Pipeline Data Structures
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Result of a filter operation - kept and removed drafts."""

    kept: list[PostDraft]
    removed: list[PostDraft]
    removal_reasons: dict[str, str] = field(default_factory=dict)  # draft_id -> reason


@dataclass
class PipelineResult:
    """Complete result of the content generation pipeline."""

    request: ContentRequest
    generated_drafts: list[PostDraft]
    filtered_drafts: list[PostDraft]
    selected_drafts: list[PostDraft]
    filter_reasons: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Content Pipeline (analogous to CandidatePipeline)
# ---------------------------------------------------------------------------


class ContentPipeline:
    """
    Orchestrates the content generation pipeline.

    Mirrors the CandidatePipeline from the recommendation system:
    1. Hydrate request with context (audience data, trends, history)
    2. Generate content drafts via LLM
    3. Hydrate drafts (add media, hashtags, formatting)
    4. Filter drafts (quality, safety, deduplication)
    5. Score drafts (predicted engagement)
    6. Select best drafts
    7. Run post-processors (scheduling, caching)
    """

    def __init__(
        self,
        request_hydrators: list[RequestHydrator] | None = None,
        generators: list[ContentGenerator] | None = None,
        draft_hydrators: list[DraftHydrator] | None = None,
        filters: list[ContentFilter] | None = None,
        scorers: list[EngagementScorer] | None = None,
        selector: ContentSelector | None = None,
        post_processors: list[PostProcessor] | None = None,
        result_size: int = 5,
    ):
        self._request_hydrators = request_hydrators or []
        self._generators = generators or []
        self._draft_hydrators = draft_hydrators or []
        self._filters = filters or []
        self._scorers = scorers or []
        self._selector = selector
        self._post_processors = post_processors or []
        self._result_size = result_size

    async def execute(self, request: ContentRequest) -> PipelineResult:
        """Execute the full content generation pipeline."""
        rid = request.request_id

        # Stage 1: Hydrate request
        hydrated_request = await self._hydrate_request(request)

        # Stage 2: Generate drafts
        all_drafts = await self._generate(hydrated_request)
        logger.info("request_id=%s stage=generate produced %d drafts", rid, len(all_drafts))

        # Stage 3: Hydrate drafts
        hydrated_drafts = await self._hydrate_drafts(hydrated_request, all_drafts)

        # Stage 4: Filter drafts
        kept_drafts, filtered_drafts, filter_reasons = await self._filter(
            hydrated_request, hydrated_drafts
        )
        logger.info(
            "request_id=%s stage=filter kept=%d removed=%d",
            rid,
            len(kept_drafts),
            len(filtered_drafts),
        )

        # Stage 5: Score drafts
        scored_drafts = await self._score(hydrated_request, kept_drafts)

        # Stage 6: Select best
        selected_drafts = self._select(hydrated_request, scored_drafts)
        selected_drafts = selected_drafts[: self._result_size]
        logger.info("request_id=%s stage=select selected %d drafts", rid, len(selected_drafts))

        # Stage 7: Post-processing
        await self._post_process(hydrated_request, selected_drafts)

        return PipelineResult(
            request=hydrated_request,
            generated_drafts=all_drafts,
            filtered_drafts=filtered_drafts,
            selected_drafts=selected_drafts,
            filter_reasons=filter_reasons,
        )

    async def _hydrate_request(self, request: ContentRequest) -> ContentRequest:
        for hydrator in self._request_hydrators:
            if hydrator.enabled(request):
                try:
                    request = await hydrator.hydrate(request)
                except Exception:
                    logger.exception(
                        "request_id=%s stage=request_hydrate component=%s failed",
                        request.request_id,
                        hydrator.name,
                    )
        return request

    async def _generate(self, request: ContentRequest) -> list[PostDraft]:
        all_drafts: list[PostDraft] = []
        for generator in self._generators:
            if generator.enabled(request):
                try:
                    drafts = await generator.generate(request)
                    all_drafts.extend(drafts)
                    logger.info(
                        "request_id=%s stage=generate component=%s produced %d drafts",
                        request.request_id,
                        generator.name,
                        len(drafts),
                    )
                except Exception:
                    logger.exception(
                        "request_id=%s stage=generate component=%s failed",
                        request.request_id,
                        generator.name,
                    )
        return all_drafts

    async def _hydrate_drafts(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> list[PostDraft]:
        for hydrator in self._draft_hydrators:
            if hydrator.enabled(request):
                try:
                    drafts = await hydrator.hydrate(request, drafts)
                except Exception:
                    logger.exception(
                        "request_id=%s stage=draft_hydrate component=%s failed",
                        request.request_id,
                        hydrator.name,
                    )
        return drafts

    async def _filter(
        self, request: ContentRequest, drafts: list[PostDraft]
    ) -> tuple[list[PostDraft], list[PostDraft], dict[str, str]]:
        all_removed: list[PostDraft] = []
        all_reasons: dict[str, str] = {}

        for content_filter in self._filters:
            if content_filter.enabled(request):
                try:
                    result = await content_filter.filter(request, drafts)
                    drafts = result.kept
                    all_removed.extend(result.removed)
                    all_reasons.update(result.removal_reasons)
                except Exception:
                    logger.exception(
                        "request_id=%s stage=filter component=%s failed",
                        request.request_id,
                        content_filter.name,
                    )

        return drafts, all_removed, all_reasons

    async def _score(self, request: ContentRequest, drafts: list[PostDraft]) -> list[PostDraft]:
        for scorer in self._scorers:
            if scorer.enabled(request):
                try:
                    predictions = await scorer.score(request, drafts)
                    if len(predictions) == len(drafts):
                        for draft, pred in zip(drafts, predictions):
                            draft.engagement_prediction = pred
                    else:
                        logger.warning(
                            "request_id=%s stage=score component=%s length_mismatch "
                            "expected=%d got=%d",
                            request.request_id,
                            scorer.name,
                            len(drafts),
                            len(predictions),
                        )
                except Exception:
                    logger.exception(
                        "request_id=%s stage=score component=%s failed",
                        request.request_id,
                        scorer.name,
                    )
        return drafts

    def _select(self, request: ContentRequest, drafts: list[PostDraft]) -> list[PostDraft]:
        if self._selector and self._selector.enabled(request):
            return self._selector.select(request, drafts)
        # Default: sort by weighted engagement score descending
        return sorted(
            drafts,
            key=lambda d: d.engagement_prediction.weighted_score
            if d.engagement_prediction
            else 0.0,
            reverse=True,
        )

    async def _post_process(self, request: ContentRequest, drafts: list[PostDraft]) -> None:
        for processor in self._post_processors:
            if processor.enabled(request):
                try:
                    await processor.process(request, drafts)
                except Exception:
                    logger.exception(
                        "request_id=%s stage=post_process component=%s failed",
                        request.request_id,
                        processor.name,
                    )
