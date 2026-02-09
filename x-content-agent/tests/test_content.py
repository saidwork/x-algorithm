"""Tests for the content generation module."""

import pytest

from agent.config import AgentConfig
from agent.content.generator import LLMContentGenerator
from agent.content.media_suggester import MediaSuggester
from agent.content.optimizer import ContentOptimizer
from agent.content.templates import TemplateEngine
from agent.content.thread_composer import ThreadComposer
from agent.types import (
    ContentConstraints,
    ContentGoal,
    ContentPiece,
    ContentRequest,
    ContentTone,
    ContentType,
    MediaType,
    PostDraft,
    Topic,
)


class TestContentOptimizer:
    def setup_method(self):
        self.config = AgentConfig()
        self.optimizer = ContentOptimizer(self.config)

    def test_clean_formatting(self):
        piece = ContentPiece(text="  Too   many   spaces  ")
        request = ContentRequest(
            request_id="t1", account_id="a1", topic=Topic(name="test")
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert "  " not in result.optimized_text.replace("\n\n", "  ")  # Allow double newlines

    def test_trim_to_limit(self):
        long_text = "This is a sentence. " * 20  # Very long
        piece = ContentPiece(text=long_text)
        request = ContentRequest(
            request_id="t2",
            account_id="a1",
            topic=Topic(name="test"),
            constraints=ContentConstraints(max_characters=100),
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert len(result.optimized_text) <= 100

    def test_hook_score_question(self):
        piece = ContentPiece(text="How do you build a startup in 2024?")
        request = ContentRequest(
            request_id="t3", account_id="a1", topic=Topic(name="test")
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert result.hook_score > 0

    def test_hook_score_number(self):
        piece = ContentPiece(text="5 ways to improve your code")
        request = ContentRequest(
            request_id="t4", account_id="a1", topic=Topic(name="test")
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert result.hook_score > 0

    def test_engagement_trigger_question(self):
        piece = ContentPiece(text="What are your thoughts?")
        request = ContentRequest(
            request_id="t5", account_id="a1", topic=Topic(name="test")
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert result.engagement_trigger_score > 0

    def test_readability_score(self):
        piece = ContentPiece(text="Simple words. Short sentences. Clear message.")
        request = ContentRequest(
            request_id="t6", account_id="a1", topic=Topic(name="test")
        )
        result = self.optimizer.optimize_piece(piece, request)
        assert result.readability_score > 0.5

    @pytest.mark.asyncio
    async def test_hydrate_drafts(self):
        drafts = [
            PostDraft(
                id="d1",
                pieces=[ContentPiece(text="Test post content here")],
            )
        ]
        request = ContentRequest(
            request_id="t7", account_id="a1", topic=Topic(name="test")
        )
        result = await self.optimizer.hydrate(request, drafts)
        assert len(result) == 1
        assert "readability_score" in result[0].metadata


class TestTemplateEngine:
    def setup_method(self):
        self.engine = TemplateEngine()

    def test_list_templates(self):
        all_templates = self.engine.list_templates()
        assert len(all_templates) > 0

    def test_list_by_goal(self):
        engagement = self.engine.list_templates(goal=ContentGoal.ENGAGEMENT)
        assert all(t.goal == ContentGoal.ENGAGEMENT for t in engagement)

    def test_render_hot_take(self):
        result = self.engine.render(
            "hot_take",
            {
                "opinion": "AI will replace 80% of coding jobs",
                "supporting_point": "GitHub Copilot already writes 40% of code",
            },
        )
        assert result is not None
        assert "Unpopular opinion" in result
        assert "AI will replace" in result

    def test_render_quick_tip(self):
        result = self.engine.render(
            "quick_tip",
            {
                "topic": "Python",
                "tip": "Use list comprehensions instead of loops",
                "benefit": "make your code 3x faster",
            },
        )
        assert result is not None
        assert "Python" in result

    def test_render_missing_vars(self):
        result = self.engine.render("hot_take", {"opinion": "test"})
        assert result is None  # Missing required var

    def test_render_nonexistent_template(self):
        result = self.engine.render("nonexistent", {})
        assert result is None

    def test_suggest_templates(self):
        suggestions = self.engine.suggest_templates("AI", goal=ContentGoal.ENGAGEMENT)
        assert len(suggestions) > 0
        assert all(s.goal == ContentGoal.ENGAGEMENT for s in suggestions)


class TestThreadComposer:
    def setup_method(self):
        self.config = AgentConfig()
        self.composer = ThreadComposer(self.config)

    def test_create_outline(self):
        topic = Topic(name="Machine Learning", category="tutorial")
        outline = self.composer.create_outline(
            topic=topic,
            key_points=["Point 1", "Point 2", "Point 3"],
        )
        assert outline.topic == "Machine Learning"
        assert len(outline.key_points) == 3
        assert "🧵" in outline.hook

    def test_compose_from_outline(self):
        topic = Topic(name="AI Tips")
        outline = self.composer.create_outline(
            topic=topic,
            key_points=["Tip 1: Use transformers", "Tip 2: Fine-tune models", "Tip 3: Monitor metrics"],
        )
        request = ContentRequest(
            request_id="t1",
            account_id="a1",
            content_type=ContentType.THREAD,
            topic=topic,
        )
        draft = self.composer.compose_from_outline(outline, request)

        assert draft.is_thread
        assert len(draft.pieces) >= 5  # hook + points + cta
        assert "🧵" in draft.pieces[0].text

    def test_validate_thread_valid(self):
        pieces = [ContentPiece(text=f"🧵 Thread post {i}", content_type=ContentType.THREAD) for i in range(5)]
        draft = PostDraft(id="v1", pieces=pieces)
        issues = self.composer.validate_thread(draft)
        assert len(issues) == 0

    def test_validate_thread_too_long_segment(self):
        pieces = [
            ContentPiece(text="🧵 " + "x" * 300, content_type=ContentType.THREAD),
            ContentPiece(text="Post 2", content_type=ContentType.THREAD),
        ]
        draft = PostDraft(id="v2", pieces=pieces)
        issues = self.composer.validate_thread(draft)
        assert any("chars" in issue for issue in issues)

    def test_validate_single_post_not_thread(self):
        draft = PostDraft(id="v3", pieces=[ContentPiece(text="Single")])
        issues = self.composer.validate_thread(draft)
        assert any("not a thread" in issue for issue in issues)


class TestMediaSuggester:
    def setup_method(self):
        self.config = AgentConfig()
        self.suggester = MediaSuggester(self.config)

    def test_suggest_for_stats(self):
        request = ContentRequest(
            request_id="t1", account_id="a1", topic=Topic(name="data")
        )
        suggestion = self.suggester.suggest("Revenue grew 150% this quarter", request)
        assert suggestion is not None
        assert suggestion.media_type == MediaType.IMAGE

    def test_suggest_for_list(self):
        text = "1. First item\n2. Second item\n3. Third item\n4. Fourth item"
        request = ContentRequest(
            request_id="t2", account_id="a1", topic=Topic(name="tips")
        )
        suggestion = self.suggester.suggest(text, request)
        assert suggestion is not None

    def test_suggest_for_short_post(self):
        request = ContentRequest(
            request_id="t3",
            account_id="a1",
            topic=Topic(name="fun"),
            content_type=ContentType.SHORT_POST,
        )
        suggestion = self.suggester.suggest("LOL", request)
        assert suggestion is not None
        assert suggestion.media_type == MediaType.GIF

    def test_no_override_existing_media(self):
        from agent.types import MediaSuggestion, MediaType

        existing_media = MediaSuggestion(media_type=MediaType.VIDEO, description="existing")
        drafts = [
            PostDraft(
                id="d1",
                pieces=[ContentPiece(text="Post with 50% growth stats", media=[existing_media])],
            )
        ]
        request = ContentRequest(
            request_id="t4", account_id="a1", topic=Topic(name="test")
        )
        # Hydrate should not add media to pieces that already have it
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self.suggester.hydrate(request, drafts)
        )
        assert len(result[0].pieces[0].media) == 1  # Still just the original


class TestLLMContentGenerator:
    def test_extract_json_from_markdown(self):
        text = '```json\n[{"text": "hello"}]\n```'
        result = LLMContentGenerator._extract_json(text)
        assert result == '[{"text": "hello"}]'

    def test_extract_json_direct(self):
        text = '[{"text": "hello"}]'
        result = LLMContentGenerator._extract_json(text)
        assert result == '[{"text": "hello"}]'

    def test_extract_json_with_surrounding_text(self):
        text = 'Here is the content:\n[{"text": "hello"}]\nDone!'
        result = LLMContentGenerator._extract_json(text)
        assert result == '[{"text": "hello"}]'
