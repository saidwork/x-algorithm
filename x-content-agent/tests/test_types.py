"""Tests for core type definitions."""

from agent.types import (
    ContentConstraints,
    ContentGoal,
    ContentPiece,
    ContentRequest,
    ContentTone,
    ContentType,
    EngagementPrediction,
    MediaSuggestion,
    MediaType,
    PerformanceMetrics,
    PostDraft,
    Topic,
    TrendSignal,
)


class TestContentPiece:
    def test_character_count(self):
        piece = ContentPiece(text="Hello world!")
        assert piece.character_count == 12

    def test_empty_piece(self):
        piece = ContentPiece(text="")
        assert piece.character_count == 0

    def test_default_type(self):
        piece = ContentPiece(text="test")
        assert piece.content_type == ContentType.SHORT_POST


class TestPostDraft:
    def test_full_text_single(self):
        piece = ContentPiece(text="Hello world")
        draft = PostDraft(id="test-1", pieces=[piece])
        assert draft.full_text == "Hello world"

    def test_full_text_thread(self):
        pieces = [
            ContentPiece(text="Part 1"),
            ContentPiece(text="Part 2"),
            ContentPiece(text="Part 3"),
        ]
        draft = PostDraft(id="test-2", pieces=pieces)
        assert draft.full_text == "Part 1\n---\nPart 2\n---\nPart 3"

    def test_is_thread(self):
        single = PostDraft(id="s", pieces=[ContentPiece(text="a")])
        thread = PostDraft(id="t", pieces=[ContentPiece(text="a"), ContentPiece(text="b")])
        assert not single.is_thread
        assert thread.is_thread

    def test_total_characters(self):
        pieces = [ContentPiece(text="12345"), ContentPiece(text="678")]
        draft = PostDraft(id="tc", pieces=pieces)
        assert draft.total_characters == 8


class TestEngagementPrediction:
    def test_weighted_score_positive(self):
        pred = EngagementPrediction(p_like=0.5, p_reply=0.1, p_repost=0.1)
        assert pred.weighted_score > 0

    def test_weighted_score_negative_signals(self):
        pred = EngagementPrediction(p_mute=0.5, p_unfollow=0.3, p_report=0.2)
        assert pred.weighted_score < 0

    def test_weighted_score_mixed(self):
        good = EngagementPrediction(p_like=0.8, p_repost=0.5)
        bad = EngagementPrediction(p_like=0.01, p_mute=0.3)
        assert good.weighted_score > bad.weighted_score


class TestTopic:
    def test_defaults(self):
        topic = Topic(name="AI")
        assert topic.trending_score == 0.0
        assert topic.relevance_score == 0.0
        assert topic.keywords == []

    def test_with_keywords(self):
        topic = Topic(name="AI", keywords=["machine learning", "deep learning"])
        assert len(topic.keywords) == 2


class TestPerformanceMetrics:
    def test_total_engagements(self):
        m = PerformanceMetrics(
            post_id="p1",
            likes=100,
            replies=20,
            reposts=30,
            quotes=5,
            clicks=50,
            bookmarks=10,
        )
        assert m.total_engagements == 215
