"""
LLM-powered content generation module.

Core creative engine that generates X posts using large language models.
Implements the ContentGenerator protocol from the pipeline framework.

The generator builds structured prompts from the ContentRequest context
(topic, tone, goal, audience, constraints) and produces multiple content
variants for downstream scoring and selection.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from agent.config import AgentConfig, LLMConfig
from agent.types import (
    ContentConstraints,
    ContentGoal,
    ContentPiece,
    ContentRequest,
    ContentTone,
    ContentType,
    PostDraft,
    Topic,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert X (Twitter) content creator. You write engaging posts
that drive meaningful interactions. You understand the platform's culture, algorithms,
and what makes content perform well.

Key principles:
- Hook readers in the first line
- Use clear, concise language
- Create value (educate, entertain, or inspire)
- End with engagement triggers (questions, CTAs, bold statements)
- Adapt tone and style to the target audience
- Never use cringe hashtag stuffing - max 1-2 natural hashtags
- Understand that shorter posts often outperform longer ones"""

GENERATION_PROMPT = """Generate {num_variants} variant(s) of an X post with these parameters:

**Topic:** {topic_name}
{topic_keywords}
**Content Type:** {content_type}
**Tone:** {tone}
**Goal:** {goal}
**Language:** {language}
{audience_context}
{constraints_text}
{brand_context}

IMPORTANT RULES:
- Each variant must be distinctly different in approach/angle
- {char_limit}
- Output valid JSON array of objects with fields: "text", "hashtags", "hook_type"
- hook_type is one of: "question", "statistic", "bold_claim", "story", "contrarian", "how_to"

Output ONLY the JSON array, no other text."""

THREAD_PROMPT = """Generate an X thread ({segment_count} posts) on this topic:

**Topic:** {topic_name}
{topic_keywords}
**Tone:** {tone}
**Goal:** {goal}
**Language:** {language}
{audience_context}
{brand_context}

THREAD STRUCTURE:
1. First post: Strong hook that makes people want to read more
2. Middle posts: Core value/information, one key point per post
3. Last post: Summary + CTA (call to action)

RULES:
- Each post must be under 280 characters
- Use "🧵" in the first post to signal it's a thread
- Number each post (1/, 2/, etc.)
- Each post should work standalone but flow as a narrative
- Output valid JSON array of objects with field "text" for each post in order

Output ONLY the JSON array, no other text."""


class LLMContentGenerator:
    """
    Generates content using LLM providers (OpenAI, Anthropic, etc.).

    Pipeline role: ContentGenerator - takes a hydrated ContentRequest and
    produces multiple PostDraft variants for scoring and selection.

    Generation strategy:
    1. Build context-rich prompt from request parameters
    2. Generate multiple variants in a single LLM call
    3. Parse structured output into PostDraft objects
    4. Tag each draft with metadata for downstream processing
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._llm_config = config.llm

    @property
    def name(self) -> str:
        return "LLMContentGenerator"

    def enabled(self, request: ContentRequest) -> bool:
        return True

    async def generate(self, request: ContentRequest) -> list[PostDraft]:
        """
        Generate content variants for the given request.

        Returns multiple PostDraft objects, each with a different creative angle.
        """
        if request.content_type == ContentType.THREAD:
            return await self._generate_thread(request)
        return await self._generate_posts(request)

    async def _generate_posts(self, request: ContentRequest) -> list[PostDraft]:
        """Generate single-post variants."""
        prompt = self._build_generation_prompt(request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response_text = await self._call_llm(messages)
        return self._parse_post_response(response_text, request)

    async def _generate_thread(self, request: ContentRequest) -> list[PostDraft]:
        """Generate a thread (multi-post sequence)."""
        prompt = self._build_thread_prompt(request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response_text = await self._call_llm(messages)
        return self._parse_thread_response(response_text, request)

    def _build_generation_prompt(self, request: ContentRequest) -> str:
        """Build the generation prompt with all context."""
        topic = request.topic or Topic(name="general")
        constraints = request.constraints or ContentConstraints()

        topic_keywords = ""
        if topic.keywords:
            topic_keywords = f"**Related Keywords:** {', '.join(topic.keywords)}"

        audience_context = ""
        if request.audience:
            audience_context = (
                f"**Target Audience:**\n"
                f"- Interests: {', '.join(request.audience.interests[:5])}\n"
                f"- Engagement rate: {request.audience.avg_engagement_rate:.2%}\n"
                f"- Language: {request.audience.language}"
            )

        constraints_text = ""
        if constraints.forbidden_words:
            constraints_text += f"\n- AVOID these words: {', '.join(constraints.forbidden_words)}"
        if constraints.required_hashtags:
            constraints_text += f"\n- MUST include hashtags: {', '.join(constraints.required_hashtags)}"
        if constraints.required_mentions:
            constraints_text += f"\n- MUST mention: {', '.join(constraints.required_mentions)}"

        brand_context = ""
        if self.config.brand_description:
            brand_context = f"**Brand Voice:** {self.config.brand_description}"
        if constraints.brand_voice_keywords:
            brand_context += f"\n**Brand Keywords:** {', '.join(constraints.brand_voice_keywords)}"

        char_limit = f"Each post must be under {constraints.max_characters} characters"
        if request.content_type == ContentType.LONG_POST:
            char_limit = "Long-form post, can be up to 25000 characters"

        return GENERATION_PROMPT.format(
            num_variants=request.num_variants,
            topic_name=topic.name,
            topic_keywords=topic_keywords,
            content_type=request.content_type.value,
            tone=request.tone.value,
            goal=request.goal.value,
            language=constraints.language,
            audience_context=audience_context,
            constraints_text=constraints_text,
            brand_context=brand_context,
            char_limit=char_limit,
        )

    def _build_thread_prompt(self, request: ContentRequest) -> str:
        """Build the thread generation prompt."""
        topic = request.topic or Topic(name="general")
        constraints = request.constraints or ContentConstraints()

        topic_keywords = ""
        if topic.keywords:
            topic_keywords = f"**Related Keywords:** {', '.join(topic.keywords)}"

        audience_context = ""
        if request.audience:
            audience_context = f"**Target Audience Interests:** {', '.join(request.audience.interests[:5])}"

        brand_context = ""
        if self.config.brand_description:
            brand_context = f"**Brand Voice:** {self.config.brand_description}"

        segment_count = min(constraints.max_characters // 280, 10) if constraints.max_characters > 280 else 5

        return THREAD_PROMPT.format(
            segment_count=segment_count,
            topic_name=topic.name,
            topic_keywords=topic_keywords,
            tone=request.tone.value,
            goal=request.goal.value,
            language=constraints.language,
            audience_context=audience_context,
            brand_context=brand_context,
        )

    def _parse_post_response(self, response_text: str, request: ContentRequest) -> list[PostDraft]:
        """Parse LLM response into PostDraft objects."""
        try:
            variants = json.loads(self._extract_json(response_text))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM response as JSON, creating single draft")
            variants = [{"text": response_text.strip(), "hashtags": [], "hook_type": "unknown"}]

        drafts = []
        for i, variant in enumerate(variants):
            text = variant.get("text", "")
            if not text:
                continue

            piece = ContentPiece(
                text=text,
                content_type=request.content_type,
                hashtags=variant.get("hashtags", []),
            )

            draft = PostDraft(
                id=f"{request.request_id}-v{i}-{uuid.uuid4().hex[:8]}",
                pieces=[piece],
                topic=request.topic,
                tone=request.tone,
                goal=request.goal,
                tags=[variant.get("hook_type", "unknown")],
                metadata={
                    "generator": self.name,
                    "variant_index": i,
                    "hook_type": variant.get("hook_type", "unknown"),
                },
            )
            drafts.append(draft)

        return drafts

    def _parse_thread_response(
        self, response_text: str, request: ContentRequest
    ) -> list[PostDraft]:
        """Parse LLM response into a thread PostDraft."""
        try:
            segments = json.loads(self._extract_json(response_text))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse thread response, returning empty")
            return []

        pieces = []
        for segment in segments:
            text = segment.get("text", "") if isinstance(segment, dict) else str(segment)
            if text:
                pieces.append(
                    ContentPiece(text=text, content_type=ContentType.THREAD)
                )

        if not pieces:
            return []

        draft = PostDraft(
            id=f"{request.request_id}-thread-{uuid.uuid4().hex[:8]}",
            pieces=pieces,
            topic=request.topic,
            tone=request.tone,
            goal=request.goal,
            tags=["thread"],
            metadata={"generator": self.name, "segment_count": len(pieces)},
        )
        return [draft]

    async def _call_llm(self, messages: list[dict]) -> str:
        """
        Call the LLM provider and return the response text.
        Supports OpenAI-compatible API format (works with OpenAI, Anthropic via proxy, local).
        """
        import httpx

        if not self._llm_config.api_key:
            logger.warning("No API key configured, returning empty response")
            return "[]"

        # Determine base URL
        base_url = self._llm_config.base_url
        if not base_url:
            if self._llm_config.provider == "openai":
                base_url = "https://api.openai.com/v1"
            elif self._llm_config.provider == "anthropic":
                base_url = "https://api.anthropic.com/v1"
            else:
                base_url = "https://api.openai.com/v1"

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._llm_config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._llm_config.model,
            "messages": messages,
            "temperature": self._llm_config.temperature,
            "max_tokens": self._llm_config.max_tokens,
        }

        logger.info(
            "LLM call: provider=%s model=%s messages=%d",
            self._llm_config.provider,
            self._llm_config.model,
            len(messages),
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("LLM response received: %d chars", len(content))
                return content
        except httpx.HTTPStatusError as e:
            logger.error("LLM API error %d: %s", e.response.status_code, e.response.text[:200])
            raise
        except Exception:
            logger.exception("LLM call failed")
            raise

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON array from LLM response text (handles markdown code blocks)."""
        text = text.strip()
        # Handle ```json ... ``` blocks
        if "```" in text:
            start = text.find("```")
            end = text.rfind("```")
            if start != end:
                inner = text[start + 3 : end].strip()
                if inner.startswith("json"):
                    inner = inner[4:].strip()
                return inner
        # Try to find array directly
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text
