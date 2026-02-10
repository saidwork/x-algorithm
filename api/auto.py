"""
POST /api/auto - Fully autonomous content generation.
Only requires API key. Agent discovers topic, selects strategy, generates content.

Pipeline:
  1. Ask LLM for trending/high-potential topics
  2. Score topics using algorithm engagement model
  3. Generate optimized content for best topic
  4. Return multiple variants with engagement predictions
"""

from http.server import BaseHTTPRequestHandler
import json
import re
import httpx

# ---------------------------------------------------------------------------
# X Algorithm Knowledge - baked from candidate-pipeline analysis
# ---------------------------------------------------------------------------

ALGORITHM_KNOWLEDGE = """You have deep knowledge of X (Twitter)'s recommendation algorithm:

## How X's For-You Feed Works
- **Multi-stage pipeline**: Candidate Sources → Hydration → Filtering → Scoring → Selection
- **Engagement signals ranked by weight**:
  - Reply (weighted 3x) > Repost (5x) > Quote Tweet (4x) > Like (1x)
  - Bookmark (2.5x), Follow from post (10x), Profile visit (1.5x)
  - NEGATIVE: Mute (-20x), Unfollow (-30x), Report (-50x)
- **What the algorithm prioritizes**:
  - Posts that generate REPLIES are heavily boosted (conversation starters)
  - Quote tweets with commentary signal high-quality content
  - Bookmarks indicate "save-worthy" value content
  - New follows from a post are the strongest positive signal
  - Time-on-post (dwell time) is a hidden ranking factor
  - Posts with media get ~2x distribution vs text-only
  - Threads get algorithmic boost for sustained engagement

## Content Patterns That Win
- Strong first-line hooks (pattern interrupt, curiosity gap, bold claim)
- Optimal length: 70-140 chars for short posts, 5-9 tweets for threads
- Questions at the end drive replies (algorithm's highest-weighted action)
- Contrarian takes generate quote tweets (4x weight)
- Lists and frameworks get bookmarked (2.5x weight)
- Personal stories drive follows (10x weight)

## What Gets Suppressed
- External links (algorithm deprioritizes off-platform clicks)
- Hashtag stuffing (>2 hashtags = spam signal)
- Engagement bait without substance ("like if you agree")
- Repetitive/similar content patterns (novelty matters)
- Posts that get muted/unfollowed destroy your reach for weeks"""

# ---------------------------------------------------------------------------
# Step 1: Topic Discovery Prompt
# ---------------------------------------------------------------------------

TOPIC_DISCOVERY_PROMPT = """You are an expert X (Twitter) content strategist.

{algorithm_knowledge}

## Your Task
Based on your knowledge of what performs well on X right now, suggest exactly 5 content ideas.

For each idea, provide:
1. **topic**: The specific topic/angle (not generic)
2. **hook**: A compelling first line that would stop the scroll
3. **content_type**: "short_post" or "thread"
4. **tone**: One of: provocative, educational, inspirational, casual, storytelling
5. **goal**: Primary goal - "engagement" (replies), "reach" (reposts/quotes), "authority" (bookmarks/follows)
6. **why**: Why this would perform well algorithmically (1 sentence)
7. **score**: Your confidence score 1-10 that this would get high engagement

Focus on topics that are:
- Currently relevant and timely (tech, AI, startups, productivity, self-improvement, business)
- Have high reply potential (the algorithm values replies 3x)
- Can generate quote tweets (4x) or bookmarks (2.5x)
- Fresh angles, NOT overused clichés

{niche_context}
{language_instruction}

Output ONLY a valid JSON array of 5 objects. No other text."""

# ---------------------------------------------------------------------------
# Step 2: Content Generation Prompt
# ---------------------------------------------------------------------------

CONTENT_GENERATION_PROMPT = """You are an elite X (Twitter) ghostwriter who understands the algorithm deeply.

{algorithm_knowledge}

## Your Task
Generate {num_variants} different post variants for this topic:

**Topic/Angle:** {topic}
**Hook idea:** {hook}
**Content Type:** {content_type}
**Tone:** {tone}
**Goal:** {goal}

## Rules Based on Algorithm
- First line MUST be a scroll-stopping hook (this determines 80% of performance)
- If goal is "engagement": end with a question or controversial take to drive REPLIES (3x weight)
- If goal is "reach": make it highly quotable/repostable (4-5x weight)
- If goal is "authority": pack actionable value that gets BOOKMARKED (2.5x weight)
- Keep under {max_chars} characters per post
- NO hashtag stuffing (max 1 natural hashtag or none)
- NO external links (algorithm suppresses them)
- Make each variant take a DIFFERENT angle/approach on the same topic
- Use line breaks strategically for readability (dwell time = hidden ranking factor)

{thread_instruction}
{language_instruction}

Output ONLY a valid JSON array of objects with fields: "text", "hook_type", "expected_action"
- hook_type: "question", "bold_claim", "statistic", "story", "contrarian", "how_to"
- expected_action: primary action this post is designed to trigger ("reply", "repost", "quote", "bookmark", "follow")

No other text, just the JSON array."""

THREAD_INSTRUCTION = """## Thread-Specific Rules
- Generate a {segment_count}-post thread
- Post 1: Irresistible hook + "🧵" (must make people click "Show more")
- Posts 2-{middle_end}: One clear insight per post, each valuable standalone
- Last post: Strong CTA (ask for follow/repost to amplify algorithmic reach)
- Number each: 1/, 2/, etc.
- Output JSON array of objects with field "text" for each post in order, plus "hook_type" and "expected_action" on the first object"""

# ---------------------------------------------------------------------------
# Engagement Prediction (from algorithm weights)
# ---------------------------------------------------------------------------

def predict_engagement(text, is_thread=False, goal="engagement"):
    char_count = len(text)
    length_score = max(0.0, min(1.0, 1.0 - abs(char_count - 105) / 200))

    features = {
        "has_question": 1.0 if "?" in text else 0.0,
        "has_numbers": 1.0 if re.search(r"\d+", text) else 0.0,
        "has_emoji": 1.0 if re.search(r"[\U0001F300-\U0001F9FF]", text) else 0.0,
        "is_thread": 1.0 if is_thread else 0.0,
        "char_length_optimal": length_score,
        "has_hook": 1.0 if re.search(r"^(How|Why|What|Stop|Don't|Most people|\d+\s|I\s)", text, re.IGNORECASE) else 0.0,
        "has_cta": 1.0 if re.search(r"(follow|repost|share|bookmark|comment|reply|thoughts\??|agree)", text, re.IGNORECASE) else 0.0,
        "has_linebreaks": 1.0 if "\n\n" in text else 0.0,
        "no_links": 0.0 if re.search(r"https?://", text) else 1.0,
        "low_hashtags": 0.0 if text.count("#") > 2 else 1.0,
    }

    weights = {
        "has_question": 0.14, "has_numbers": 0.06, "has_emoji": 0.03,
        "is_thread": 0.12, "char_length_optimal": 0.08, "has_hook": 0.18,
        "has_cta": 0.10, "has_linebreaks": 0.06, "no_links": 0.08,
        "low_hashtags": 0.05,
    }

    base = sum(weights.get(k, 0) * v for k, v in features.items())
    base = max(0.01, min(0.55, base))
    thread_mult = 1.3 if is_thread else 1.0
    ep = base * thread_mult

    q_mult = 1.5 if features["has_question"] else 1.0
    t_mult = 1.5 if is_thread else 1.0

    pred = {
        "p_like": round(ep * 0.60, 4),
        "p_reply": round(ep * 0.08 * q_mult, 4),
        "p_repost": round(ep * 0.12 * t_mult, 4),
        "p_quote": round(ep * 0.04, 4),
        "p_bookmark": round(ep * 0.05 * t_mult, 4),
        "p_follow": round(ep * 0.01, 4),
        "engagement_rate": round(ep, 4),
    }

    pred["weighted_score"] = round(
        1.0 * pred["p_like"] + 3.0 * pred["p_reply"] + 5.0 * pred["p_repost"]
        + 4.0 * pred["p_quote"] + 2.5 * pred["p_bookmark"] + 10.0 * pred["p_follow"], 4
    )

    virality = ep * 0.3 + features["has_hook"] * 0.25 + features["has_question"] * 0.15 + features["is_thread"] * 0.1 + features["has_cta"] * 0.1
    pred["virality_score"] = round(max(0.0, min(1.0, virality)), 4)

    hook = 0.0
    first_line = text.split("\n")[0]
    for p in [r"^\d+\s", r"^(How|Why|What|When)\s", r"^(Stop|Don't|Never)\s",
              r"^(The truth|Here's what|Most people|Unpopular opinion)", r"^(I just|After \d+)"]:
        if re.search(p, first_line, re.IGNORECASE):
            hook += 0.3
    if len(first_line) < 80:
        hook += 0.2
    pred["hook_score"] = round(min(hook, 1.0), 2)

    return pred


def extract_json(text):
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            inner = text[start + 3:end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            return inner
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


async def call_llm(api_key, provider, model, messages, temperature=0.85, max_tokens=2048):
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


def _scrub_secrets(text, api_key=""):
    """Remove any API key fragments from error messages."""
    if api_key and len(api_key) > 8:
        text = text.replace(api_key, "sk-***REDACTED***")
        # Also scrub partial key (first/last 4 chars pattern)
        text = text.replace(api_key[:8], "sk-***")
    # Scrub common key patterns that may leak
    text = re.sub(r'sk-[A-Za-z0-9_-]{10,}', 'sk-***REDACTED***', text)
    text = re.sub(r'sk-ant-[A-Za-z0-9_-]{10,}', 'sk-ant-***REDACTED***', text)
    text = re.sub(r'Bearer\s+[A-Za-z0-9_-]{10,}', 'Bearer ***REDACTED***', text)
    return text


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default stderr logging to prevent key leaks in logs."""
        pass
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            api_key = body.get("api_key", "")
            provider = body.get("provider", "openai")
            model = body.get("model", "gpt-4o")
            language = body.get("language", "tr")
            niche = body.get("niche", "")

            if not api_key:
                self._respond(400, {"error": "API key gerekli"})
                return

            import asyncio
            loop = asyncio.new_event_loop()

            # ---- STEP 1: Discover Topics ----
            niche_ctx = f"Focus on this niche/industry: {niche}" if niche else "Cover a mix of tech, business, productivity, self-improvement, AI topics"
            lang_instr = "Generate everything in Turkish language." if language == "tr" else f"Generate everything in {language}."

            discovery_prompt = TOPIC_DISCOVERY_PROMPT.format(
                algorithm_knowledge=ALGORITHM_KNOWLEDGE,
                niche_context=niche_ctx,
                language_instruction=lang_instr,
            )

            discovery_messages = [
                {"role": "system", "content": "You are an expert X/Twitter content strategist. Output only valid JSON."},
                {"role": "user", "content": discovery_prompt},
            ]

            raw_topics = loop.run_until_complete(
                call_llm(api_key, provider, model, discovery_messages, temperature=0.9, max_tokens=2048)
            )

            try:
                topics = json.loads(extract_json(raw_topics))
            except (json.JSONDecodeError, ValueError):
                topics = [{"topic": "AI and productivity", "hook": "Most people use AI wrong.", "content_type": "short_post", "tone": "provocative", "goal": "engagement", "why": "Contrarian AI takes drive replies", "score": 8}]

            # Sort by score, pick top
            if isinstance(topics, list) and len(topics) > 0:
                topics.sort(key=lambda t: t.get("score", 5), reverse=True)

            best = topics[0] if topics else topics

            # ---- STEP 2: Generate Content ----
            is_thread = best.get("content_type", "short_post") == "thread"
            num_variants = 3 if not is_thread else 1
            max_chars = 280

            thread_instr = ""
            if is_thread:
                seg = 7
                thread_instr = THREAD_INSTRUCTION.format(segment_count=seg, middle_end=seg - 1)

            gen_prompt = CONTENT_GENERATION_PROMPT.format(
                algorithm_knowledge=ALGORITHM_KNOWLEDGE,
                num_variants=num_variants,
                topic=best.get("topic", ""),
                hook=best.get("hook", ""),
                content_type=best.get("content_type", "short_post"),
                tone=best.get("tone", "provocative"),
                goal=best.get("goal", "engagement"),
                max_chars=max_chars,
                thread_instruction=thread_instr,
                language_instruction=lang_instr,
            )

            gen_messages = [
                {"role": "system", "content": "You are an elite X/Twitter ghostwriter. Output only valid JSON."},
                {"role": "user", "content": gen_prompt},
            ]

            raw_content = loop.run_until_complete(
                call_llm(api_key, provider, model, gen_messages, temperature=0.8, max_tokens=2048)
            )
            loop.close()

            try:
                variants = json.loads(extract_json(raw_content))
            except (json.JSONDecodeError, ValueError):
                variants = [{"text": raw_content.strip(), "hook_type": "unknown", "expected_action": "like"}]

            # ---- STEP 3: Score & Rank ----
            results = []
            for i, v in enumerate(variants):
                text = v.get("text", str(v)) if isinstance(v, dict) else str(v)
                pred = predict_engagement(text, is_thread=is_thread, goal=best.get("goal", "engagement"))
                results.append({
                    "index": i,
                    "text": text,
                    "hook_type": v.get("hook_type", "unknown") if isinstance(v, dict) else "unknown",
                    "expected_action": v.get("expected_action", "like") if isinstance(v, dict) else "like",
                    "prediction": pred,
                    "char_count": len(text),
                })

            results.sort(key=lambda x: x["prediction"]["weighted_score"], reverse=True)

            self._respond(200, {
                "topic_selected": best,
                "all_topics": topics[:5],
                "drafts": results,
                "count": len(results),
                "is_thread": is_thread,
                "pipeline": "auto",
            })

        except httpx.HTTPStatusError as e:
            # Scrub: never leak API key or auth headers in error responses
            detail = e.response.text[:300] if e.response else ""
            detail = _scrub_secrets(detail, body.get("api_key", ""))
            self._respond(e.response.status_code, {
                "error": f"LLM API hatasi: {e.response.status_code}",
                "detail": detail,
            })
        except Exception as e:
            msg = _scrub_secrets(str(e), body.get("api_key", "") if 'body' in dir() else "")
            self._respond(500, {"error": msg})

    def do_OPTIONS(self):
        self.send_response(200)
        origin = self.headers.get("Origin", "")
        self._set_cors(origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _set_cors(self, origin):
        """Only allow same-origin requests (Vercel deployment domain)."""
        host = self.headers.get("Host", "")
        if origin and host and (host in origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            # Fallback: same origin only, no wildcard
            self.send_header("Access-Control-Allow-Origin", f"https://{host}" if host else "")
        self.send_header("Vary", "Origin")

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        origin = self.headers.get("Origin", "")
        self._set_cors(origin)
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
