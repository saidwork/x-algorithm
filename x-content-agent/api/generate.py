"""
POST /api/generate - Content generation endpoint.
Calls LLM API and returns generated post variants with engagement predictions.
"""

from http.server import BaseHTTPRequestHandler
import json
import re
import httpx

# ---------------------------------------------------------------------------
# LLM Prompt
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
- Understand that shorter posts often outperform longer ones
- Prioritize conversation quality signals (replies, bookmarks, meaningful shares)
- Keep claims specific and grounded for trust-sensitive topics (finance, crypto, real estate)"""

DOMAIN_CONTEXT = {
    "yatirim": "Focus on investing education, risk awareness, and actionable insights.",
    "kripto": "Focus on market structure, volatility context, on-chain or macro narratives, and risk framing.",
    "emlak": "Focus on local market dynamics, financing costs, rental yield, and practical buyer/seller guidance.",
}

GENERATION_PROMPT = """Generate {num_variants} variant(s) of an X post with these parameters:

**Topic:** {topic}
**Content Type:** {content_type}
**Tone:** {tone}
**Goal:** {goal}
**Language:** {language}
{brand_context}

IMPORTANT RULES:
- Each variant must be distinctly different in approach/angle
- Each post must be under {max_chars} characters
- Output valid JSON array of objects with fields: "text", "hashtags", "hook_type"
- hook_type is one of: "question", "statistic", "bold_claim", "story", "contrarian", "how_to"

Output ONLY the JSON array, no other text."""

THREAD_PROMPT = """Generate an X thread ({segment_count} posts) on this topic:

**Topic:** {topic}
**Tone:** {tone}
**Goal:** {goal}
**Language:** {language}
{brand_context}
{key_points_text}

THREAD STRUCTURE:
1. First post: Strong hook that makes people want to read more (include 🧵)
2. Middle posts: Core value/information, one key point per post
3. Last post: Summary + CTA (call to action)

RULES:
- Each post must be under 280 characters
- Number each post (1/, 2/, etc.)
- Each post should work standalone but flow as a narrative
- Output valid JSON array of objects with field "text" for each post in order

Output ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# Engagement Predictor (lightweight version for serverless)
# ---------------------------------------------------------------------------

FEATURE_WEIGHTS = {
    "has_media": 0.15, "has_question": 0.12, "has_numbers": 0.08,
    "has_emoji": 0.05, "is_thread": 0.10, "is_timely": 0.12,
    "char_length_optimal": 0.08, "has_hook": 0.15, "has_cta": 0.10,
    "readability": 0.05,
}

ENGAGEMENT_WEIGHTS = {
    "like": 1.0, "reply": 3.0, "repost": 5.0, "quote": 4.0,
    "click": 2.0, "bookmark": 2.5, "follow": 10.0, "profile_visit": 1.5,
    "mute": -20.0, "unfollow": -30.0, "report": -50.0,
}

HOOK_PATTERNS = [
    r"^\d+\s",
    r"^(How|Why|What|When|Stop|Don't|Most people|The truth|Here's what)",
    r"^(Nasıl|Neden|Ne zaman|Çoğu kişi|Gerçek şu|Dikkat)",
]

CTA_PATTERN = (
    r"(follow|repost|share|bookmark|comment|reply|thoughts\??|"
    r"takip et|yeniden paylaş|paylaş|kaydet|yorum|fikrin ne|katılıyor musun)"
)


def normalize_domain(value: str) -> str:
    val = (value or "").strip().lower()
    mapping = {
        "yatırım": "yatirim",
        "yatirim": "yatirim",
        "investment": "yatirim",
        "investing": "yatirim",
        "kripto": "kripto",
        "crypto": "kripto",
        "cryptocurrency": "kripto",
        "emlak": "emlak",
        "real_estate": "emlak",
        "real estate": "emlak",
        "property": "emlak",
    }
    return mapping.get(val, "")


def infer_domain(topic: str, hint: str = "") -> str:
    normalized_hint = normalize_domain(hint)
    if normalized_hint:
        return normalized_hint

    text = (topic or "").lower()
    if any(k in text for k in ["kripto", "crypto", "bitcoin", "ethereum", "altcoin", "defi"]):
        return "kripto"
    if any(k in text for k in ["emlak", "konut", "kira", "gayrimenkul", "real estate", "mortgage"]):
        return "emlak"
    if any(k in text for k in ["yatırım", "yatirim", "borsa", "hisse", "fon", "invest"]):
        return "yatirim"
    return ""


def predict_engagement(text, trending_score=0.3, is_thread=False):
    char_count = len(text)
    length_score = max(0.0, min(1.0, 1.0 - abs(char_count - 105) / 200))

    features = {
        "has_media": 0.0,
        "has_question": 1.0 if "?" in text else 0.0,
        "has_numbers": 1.0 if re.search(r"\d+", text) else 0.0,
        "has_emoji": 1.0 if re.search(r"[\U0001F300-\U0001F9FF]", text) else 0.0,
        "is_thread": 1.0 if is_thread else 0.0,
        "is_timely": trending_score,
        "char_length_optimal": length_score,
        "has_hook": 1.0 if any(re.search(p, text, re.IGNORECASE) for p in HOOK_PATTERNS) else 0.0,
        "has_cta": 1.0 if re.search(CTA_PATTERN, text, re.IGNORECASE) else 0.0,
        "readability": 0.7,
    }

    base = sum(FEATURE_WEIGHTS.get(k, 0) * v for k, v in features.items())
    base = max(0.001, min(0.5, base))

    type_mult = 1.3 if is_thread else 1.0

    ep = base * type_mult
    q_mult = 1.5 if features["has_question"] else 1.0
    t_mult = 1.5 if is_thread else 1.0

    pred = {
        "p_like": ep * 0.60,
        "p_reply": ep * 0.08 * q_mult,
        "p_repost": ep * 0.12 * t_mult,
        "p_quote": ep * 0.04,
        "p_click": ep * 0.10,
        "p_bookmark": ep * 0.05 * t_mult,
        "p_follow": ep * 0.01,
        "p_profile_visit": ep * 0.03,
        "p_mute": max(0.0, 0.001 - ep * 0.01),
        "p_unfollow": max(0.0, 0.0005 - ep * 0.005),
        "p_report": 0.0001,
        "predicted_impressions": int(1000 * 0.10 * (1.0 + ep * 10)),
        "predicted_engagement_rate": ep,
    }

    pred["weighted_score"] = (
        1.0 * pred["p_like"] + 3.0 * pred["p_reply"] + 5.0 * pred["p_repost"]
        + 4.0 * pred["p_quote"] + 2.0 * pred["p_click"] + 2.5 * pred["p_bookmark"]
        + 10.0 * pred["p_follow"] + 1.5 * pred["p_profile_visit"]
        - 20.0 * pred["p_mute"] - 30.0 * pred["p_unfollow"] - 50.0 * pred["p_report"]
    )

    virality = ep * 0.3 + features["is_timely"] * 0.3 + features["has_hook"] * 0.2 + features["is_thread"] * 0.1
    pred["virality_score"] = max(0.0, min(1.0, virality))

    # Hook score
    hook_score = 0.0
    first_line = text.split("\n")[0]
    for p in [r"^\d+\s", r"^(How|Why|What|When|Nasıl|Neden)\s", r"^(Stop|Don't|Never|Dikkat)\s",
              r"^(The truth|Here's what|Most people|Unpopular opinion|Gerçek şu|Çoğu kişi)", r"^(I just|After \d+)"]:
        if re.search(p, first_line, re.IGNORECASE):
            hook_score += 0.3
    if len(first_line) < 80:
        hook_score += 0.2
    pred["hook_score"] = min(hook_score, 1.0)

    # Engagement trigger score
    trigger = 0.0
    for p in [r"\?$", r"(agree|disagree|thoughts|katılıyor musun|fikrin ne)\??$", r"(share|repost|like if|paylaş|yeniden paylaş)\s", r"(comment|reply|tell me|yorum yaz|yorumlara yaz|cevap ver)\s"]:
        if re.search(p, text, re.IGNORECASE):
            trigger += 0.25
    if re.search(CTA_PATTERN, text, re.IGNORECASE):
        trigger += 0.2
    if "\n" in text:
        trigger += 0.1
    pred["engagement_trigger_score"] = min(trigger, 1.0)

    pred["readability_score"] = 0.7  # simplified

    return pred


def extract_json(text):
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            inner = text[start + 3: end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            return inner
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start: end + 1]
    return text


async def call_llm(api_key, provider, model, messages, temperature=0.8, max_tokens=1024):
    if provider == "openai":
        base_url = "https://api.openai.com/v1"
    elif provider == "anthropic":
        base_url = "https://api.anthropic.com/v1"
    else:
        base_url = "https://api.openai.com/v1"

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=55.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            api_key = body.get("api_key", "")
            provider = body.get("provider", "openai")
            model = body.get("model", "gpt-4o")
            topic = body.get("topic", "")
            content_type = body.get("content_type", "short_post")
            tone = body.get("tone", "professional")
            goal = body.get("goal", "engagement")
            language = body.get("language", "tr")
            num_variants = body.get("num_variants", 3)
            brand_context = body.get("brand_description", "")
            max_chars = max(40, min(280, int(body.get("max_chars", 280))))
            is_thread = content_type == "thread"
            key_points = body.get("key_points", [])
            domain = infer_domain(topic, hint=body.get("domain", body.get("niche", "")))

            if not api_key:
                self._respond(400, {"error": "API key gerekli"})
                return
            if not topic:
                self._respond(400, {"error": "Konu gerekli"})
                return

            domain_text = DOMAIN_CONTEXT.get(domain, "")
            brand_bits = []
            if brand_context:
                brand_bits.append(f"**Brand Voice:** {brand_context}")
            if domain_text:
                brand_bits.append(f"**Domain Context:** {domain_text}")
            brand_text = "\n".join(brand_bits)

            if is_thread:
                kp_text = ""
                if key_points:
                    kp_text = "**Key Points to cover:**\n" + "\n".join(f"- {p}" for p in key_points)
                prompt = THREAD_PROMPT.format(
                    segment_count=body.get("segment_count", 7),
                    topic=topic, tone=tone, goal=goal, language=language,
                    brand_context=brand_text, key_points_text=kp_text,
                )
            else:
                prompt = GENERATION_PROMPT.format(
                    num_variants=num_variants, topic=topic, content_type=content_type,
                    tone=tone, goal=goal, language=language,
                    brand_context=brand_text, max_chars=max_chars,
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            import asyncio
            loop = asyncio.new_event_loop()
            raw = loop.run_until_complete(call_llm(api_key, provider, model, messages))
            loop.close()

            # Parse response
            try:
                variants = json.loads(extract_json(raw))
            except (json.JSONDecodeError, ValueError):
                variants = [{"text": raw.strip(), "hashtags": [], "hook_type": "unknown"}]

            # Add predictions
            results = []
            for i, v in enumerate(variants):
                text = v.get("text", v) if isinstance(v, dict) else str(v)
                pred = predict_engagement(text, trending_score=0.3, is_thread=is_thread)
                results.append({
                    "index": i,
                    "text": text,
                    "hashtags": v.get("hashtags", []) if isinstance(v, dict) else [],
                    "hook_type": v.get("hook_type", "unknown") if isinstance(v, dict) else "unknown",
                    "prediction": pred,
                    "char_count": len(text),
                })

            # Sort by weighted score
            results.sort(key=lambda x: x["prediction"]["weighted_score"], reverse=True)

            self._respond(200, {"drafts": results, "count": len(results), "is_thread": is_thread})

        except httpx.HTTPStatusError as e:
            self._respond(e.response.status_code, {
                "error": f"LLM API hatasi: {e.response.status_code}",
                "detail": e.response.text[:300],
            })
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
