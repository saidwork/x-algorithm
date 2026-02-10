"""Tests for API generation helpers."""

from api.generate import infer_domain, normalize_domain, predict_engagement


def test_normalize_domain_aliases():
    assert normalize_domain("Yatırım") == "yatirim"
    assert normalize_domain("crypto") == "kripto"
    assert normalize_domain("real estate") == "emlak"


def test_infer_domain_from_topic_keywords():
    assert infer_domain("Bitcoin dominansı yükseliyor") == "kripto"
    assert infer_domain("Konut kredisi faizleri düştü") == "emlak"
    assert infer_domain("Borsa İstanbul'da temettü hisseleri") == "yatirim"


def test_predict_engagement_detects_turkish_hooks_and_cta():
    text = "Neden yatırımcılar bu hafta altına dönüyor? Fikrini yorumlara yaz ve kaydet."
    prediction = predict_engagement(text)
    assert prediction["weighted_score"] > 0
    assert prediction["engagement_trigger_score"] > 0
