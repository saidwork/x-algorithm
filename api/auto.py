"""
POST /api/auto - Fully autonomous content generation.
Only requires API key. Agent discovers topic, selects strategy, generates
long-form engaging content with image prompts.

Pipeline:
  1. Discover trending topics with high algorithmic potential
  2. Generate long-form, informative, argumentative content
  3. Generate DALL-E image prompt for visual
  4. Score with algorithm engagement model
"""

from http.server import BaseHTTPRequestHandler
import json
import re
import httpx

# ---------------------------------------------------------------------------
# X Algorithm Knowledge - baked from candidate-pipeline analysis
# ---------------------------------------------------------------------------

ALGORITHM_KNOWLEDGE = """## X (Twitter) Recommendation Algorithm - Deep Analysis

### Engagement Signal Weights (from candidate_pipeline.rs WeightedScorer)
| Signal | Weight | Meaning |
|--------|--------|---------|
| Reply | 3.0x | Conversations = highest organic boost |
| Repost | 5.0x | Content worth sharing = massive reach |
| Quote Tweet | 4.0x | Content that sparks opinion = viral potential |
| Like | 1.0x | Baseline approval signal |
| Bookmark | 2.5x | "Save for later" = high-value content signal |
| Follow from post | 10.0x | STRONGEST positive signal - you gained a fan |
| Profile visit | 1.5x | Curiosity driver |
| Mute | -20.0x | Content annoyed someone |
| Unfollow | -30.0x | Content was bad enough to lose a follower |
| Report | -50.0x | Content violated expectations |

### Hidden Ranking Factors
1. **Dwell Time**: Time spent reading your post. LONGER posts that hold attention = massive boost. This is why multi-line, well-structured posts outperform one-liners.
2. **Full Read Rate**: If users expand "Show more" AND read the full post, the algorithm treats this as extremely high quality.
3. **Reply Depth**: A post that generates reply chains (back-and-forth conversations) gets exponentially more distribution.
4. **Early Velocity**: Engagement in the first 15-30 minutes determines reach ceiling.
5. **Media Boost**: Posts with images get ~2x distribution. Posts with original/unique images get even more.
6. **Cluster Affinity**: The algorithm groups users by interest clusters. Your post is first shown to your core cluster, and if it performs well there, it expands to adjacent clusters.

### What Gets SUPPRESSED
- External links (-50% reach)
- Hashtag stuffing (>2 = spam signal)
- "Like if you agree" style engagement bait
- Very short low-effort posts (<30 chars)
- Repetitive content patterns
- Posts that only get likes but no replies/reposts (low-quality engagement)"""

# ---------------------------------------------------------------------------
# Step 1: Topic Discovery
# ---------------------------------------------------------------------------

TOPIC_DISCOVERY_PROMPT = """You are an expert X content strategist who has studied the algorithm deeply.

{algorithm_knowledge}

## Your Task
Suggest exactly 5 HIGH-IMPACT content ideas optimized for maximum algorithmic distribution.

For each idea:
1. **topic**: Specific angle (NOT generic like "AI is cool" - be precise like "Why RAG is replacing fine-tuning and most engineers don't realize it yet")
2. **hook**: A powerful first line (pattern interrupt, curiosity gap, or bold claim)
3. **content_type**: "long_post" (500-1500 chars) or "thread" (5-8 posts)
4. **tone**: provocative | educational | storytelling | contrarian | data_driven
5. **goal**: "engagement" | "reach" | "authority"
6. **argument**: The central argument/thesis readers will debate (this is what drives replies and quotes)
7. **data_points**: 2-3 specific facts/statistics/examples to include
8. **why**: Why this performs well algorithmically
9. **image_concept**: A visual concept that would make this post stand out in the feed
10. **score**: Confidence 1-10

CRITICAL RULES:
- Every topic MUST have a debatable argument (drives replies = 3x weight)
- Every topic MUST provide real information/value (drives bookmarks = 2.5x weight)
- Every topic MUST be quotable (drives quote tweets = 4x weight)
- Prefer "long_post" over "short_post" - longer content = more dwell time = more algorithmic boost
- NO generic motivational fluff. Specificity wins.

{niche_context}
{language_instruction}

Output ONLY a valid JSON array of 5 objects. No other text."""

# ---------------------------------------------------------------------------
# Step 2: Content Generation
# ---------------------------------------------------------------------------

CONTENT_GENERATION_PROMPT = """You are an elite X ghostwriter. You write posts that go viral because they are GENUINELY valuable and thought-provoking.

{algorithm_knowledge}

## Your Task
Generate {num_variants} post variant(s) for:

**Topic:** {topic}
**Hook:** {hook}
**Argument:** {argument}
**Data Points to Include:** {data_points}
**Content Type:** {content_type}
**Tone:** {tone}
**Goal:** {goal}

## MANDATORY STRUCTURE FOR EVERY POST

### 1. HOOK (First 1-2 lines) - 80% of your post's success
- Pattern interrupt: Say something unexpected that breaks the scroll
- Create a curiosity gap or make a bold claim
- NEVER start generic. Start with a number, a contrarian take, or a "most people don't know" angle

### 2. BODY (Core content - 60-70% of post)
- Provide REAL information: specific data, examples, frameworks, or insights
- Include at least 2 concrete facts or examples
- Use line breaks after every 1-2 sentences (increases dwell time dramatically)
- Build your argument logically - lead the reader through your reasoning
- Include a "plot twist" or counterintuitive insight in the middle

### 3. ARGUMENT / HOT TAKE (Engagement trigger)
- Present a debatable position that smart people could disagree on
- This is what drives replies (3x algorithmic weight) and quote tweets (4x weight)
- Make it specific enough that people HAVE to share their opinion

### 4. CLOSER (Last 1-2 lines)
- End with a question that invites replies OR a bold statement that invites quote tweets
- Optionally: Ask people to bookmark/repost if it resonated

## FORMAT RULES
- Target length: {min_chars}-{max_chars} characters (LONGER = more dwell time = more reach)
- Use line breaks generously: every 1-2 sentences should have a blank line
- NO external links (algorithm suppresses them by ~50%)
- Maximum 1 hashtag (or zero - organic reach > hashtag reach)
- NO cringe engagement bait like "like if you agree"
- Write in a human, authentic voice - not corporate or AI-sounding

{thread_instruction}
{language_instruction}

Output ONLY valid JSON array with fields per variant:
- "text": the full post text
- "hook_type": "statistic" | "bold_claim" | "question" | "story" | "contrarian" | "framework"
- "expected_action": primary action designed to trigger ("reply" | "repost" | "quote" | "bookmark")
- "image_prompt": A DALL-E prompt to generate a complementary visual (dark theme, modern, eye-catching, no text in image)

No other text, just the JSON array."""

THREAD_INSTRUCTION = """## THREAD STRUCTURE ({segment_count} posts)
- Post 1: Irresistible hook + "🧵" - this MUST make people click through
- Post 2: Context/background - why this matters NOW
- Posts 3-{middle_end}: Core insights - ONE powerful idea per post, each with a specific example or data point
- Post {before_last}: The contrarian take / hot argument
- Post {segment_count}: Summary + CTA (follow for more + repost to share)

THREAD-SPECIFIC RULES:
- Each post MUST work standalone AND as part of the narrative
- Number them: 1/, 2/, etc.
- 200-280 characters per post is ideal for threads
- End at least 2 posts with a mini-hook that makes people want the next post
- Output JSON array of objects with "text" field per post. "hook_type", "expected_action", "image_prompt" on the first object only."""

# ---------------------------------------------------------------------------
# Engagement Prediction (calibrated for long-form content)
# ---------------------------------------------------------------------------

def predict_engagement(text, is_thread=False, goal="engagement"):
    char_count = len(text)

    # Long-form sweet spot: 500-1200 chars for posts, 200-280 per thread post
    if is_thread:
        length_score = max(0.0, min(1.0, 1.0 - abs(char_count - 240) / 200))
    else:
        # Favor longer content: peak at ~800 chars
        if char_count < 100:
            length_score = char_count / 200  # Very short = low score
        elif char_count <= 1500:
            length_score = 0.5 + min(0.5, (char_count - 100) / 1400 * 0.5)
        else:
            length_score = max(0.3, 1.0 - (char_count - 1500) / 2000)

    line_count = text.count("\n")
    word_count = len(text.split())

    features = {
        "has_question": 1.0 if "?" in text else 0.0,
        "has_numbers": 1.0 if re.search(r"\d+", text) else 0.0,
        "has_emoji": min(1.0, len(re.findall(r"[\U0001F300-\U0001F9FF]", text)) * 0.3),
        "is_thread": 1.0 if is_thread else 0.0,
        "char_length_optimal": length_score,
        "has_hook": 1.0 if re.search(r"^(How|Why|What|Stop|Don't|Most people|\d+\s|I\s|The truth|Here's|Nobody|Everyone)", text, re.IGNORECASE) else 0.0,
        "has_cta": 1.0 if re.search(r"(follow|repost|share|bookmark|comment|reply|thoughts\??|agree|disagree|what do you think)", text, re.IGNORECASE) else 0.0,
        "has_linebreaks": min(1.0, line_count / 6),  # More line breaks = more dwell time
        "no_links": 0.0 if re.search(r"https?://", text) else 1.0,
        "low_hashtags": 0.0 if text.count("#") > 2 else 1.0,
        "has_argument": 1.0 if re.search(r"(but|however|actually|the real|problem is|unpopular|most people|disagree|controversial|hot take)", text, re.IGNORECASE) else 0.0,
        "has_data": 1.0 if re.search(r"\d+[%xX]|\$\d|billion|million|\d+\s*(percent|times|years|months)", text, re.IGNORECASE) else 0.0,
        "word_density": min(1.0, word_count / 100),  # More words = more substance
    }

    weights = {
        "has_question": 0.12,
        "has_numbers": 0.05,
        "has_emoji": 0.02,
        "is_thread": 0.10,
        "char_length_optimal": 0.10,
        "has_hook": 0.16,
        "has_cta": 0.08,
        "has_linebreaks": 0.08,
        "no_links": 0.06,
        "low_hashtags": 0.03,
        "has_argument": 0.10,
        "has_data": 0.05,
        "word_density": 0.05,
    }

    base = sum(weights.get(k, 0) * v for k, v in features.items())
    base = max(0.01, min(0.65, base))
    thread_mult = 1.3 if is_thread else 1.0
    ep = base * thread_mult

    q_mult = 1.5 if features["has_question"] else 1.0
    t_mult = 1.5 if is_thread else 1.0
    arg_mult = 1.3 if features["has_argument"] else 1.0

    pred = {
        "p_like": round(ep * 0.55, 4),
        "p_reply": round(ep * 0.10 * q_mult * arg_mult, 4),
        "p_repost": round(ep * 0.12 * t_mult, 4),
        "p_quote": round(ep * 0.06 * arg_mult, 4),
        "p_bookmark": round(ep * 0.07 * t_mult, 4),
        "p_follow": round(ep * 0.015, 4),
        "engagement_rate": round(ep, 4),
        "dwell_time_score": round(min(1.0, (char_count / 800) * features["has_linebreaks"]), 2),
    }

    pred["weighted_score"] = round(
        1.0 * pred["p_like"] + 3.0 * pred["p_reply"] + 5.0 * pred["p_repost"]
        + 4.0 * pred["p_quote"] + 2.5 * pred["p_bookmark"] + 10.0 * pred["p_follow"], 4
    )

    virality = (
        ep * 0.20
        + features["has_hook"] * 0.20
        + features["has_question"] * 0.10
        + features["has_argument"] * 0.20
        + features["is_thread"] * 0.10
        + features["has_cta"] * 0.05
        + features["has_data"] * 0.10
        + features["word_density"] * 0.05
    )
    pred["virality_score"] = round(max(0.0, min(1.0, virality)), 4)

    hook = 0.0
    first_line = text.split("\n")[0]
    for p in [r"^\d+", r"^(How|Why|What|When)\s", r"^(Stop|Don't|Never)\s",
              r"^(The truth|Here's|Most people|Unpopular|Nobody|Everyone)",
              r"^(I just|After \d+|In \d+)", r"^(Hot take|Controversial)"]:
        if re.search(p, first_line, re.IGNORECASE):
            hook += 0.25
    if len(first_line) < 100:
        hook += 0.15
    if ":" in first_line or "—" in first_line:
        hook += 0.1
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


async def call_llm(api_key, provider, model, messages, temperature=0.85, max_tokens=4096):
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


async def generate_image(api_key, prompt):
    """Call DALL-E 3 to generate an image. Returns the image URL."""
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1792x1024",
        "quality": "standard",
    }

    async with httpx.AsyncClient(timeout=55.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0].get("url", "")


def _scrub_secrets(text, api_key=""):
    """Remove any API key fragments from error messages."""
    if api_key and len(api_key) > 8:
        text = text.replace(api_key, "sk-***REDACTED***")
        text = text.replace(api_key[:8], "sk-***")
    text = re.sub(r'sk-[A-Za-z0-9_-]{10,}', 'sk-***REDACTED***', text)
    text = re.sub(r'sk-ant-[A-Za-z0-9_-]{10,}', 'sk-ant-***REDACTED***', text)
    text = re.sub(r'Bearer\s+[A-Za-z0-9_-]{10,}', 'Bearer ***REDACTED***', text)
    return text


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = {}
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            api_key = body.get("api_key", "")
            provider = body.get("provider", "openai")
            model = body.get("model", "gpt-4o")
            language = body.get("language", "tr")
            niche = body.get("niche", "")
            generate_visual = body.get("generate_image", True)

            if not api_key:
                self._respond(400, {"error": "API key gerekli"})
                return

            import asyncio
            loop = asyncio.new_event_loop()

            # ---- STEP 1: Discover Topics ----
            niche_ctx = f"Focus specifically on this niche/industry: {niche}" if niche else "Cover a mix of: tech, AI, startups, productivity, career growth, business strategy"
            lang_instr = "IMPORTANT: Generate ALL content in Turkish language. Write naturally in Turkish, not translated." if language == "tr" else f"Generate everything in {language}."

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
                call_llm(api_key, provider, model, discovery_messages, temperature=0.9, max_tokens=3000)
            )

            try:
                topics = json.loads(extract_json(raw_topics))
            except (json.JSONDecodeError, ValueError):
                topics = [{
                    "topic": "AI agents are replacing SaaS products faster than anyone expected",
                    "hook": "In 2025, 40% of Y Combinator startups are AI agent companies.",
                    "content_type": "long_post",
                    "tone": "data_driven",
                    "goal": "engagement",
                    "argument": "Traditional SaaS is dying because AI agents can do the same work at 1/10th the cost",
                    "data_points": ["40% of YC W25 are AI agents", "$4.5B invested in AI agents in Q1 2025", "Salesforce lost 8% stock after agent announcements"],
                    "why": "Contrarian + data-driven + affects many people = high reply + quote potential",
                    "image_concept": "Dark futuristic visualization showing AI agents replacing software icons",
                    "score": 9,
                }]

            if isinstance(topics, list) and len(topics) > 0:
                topics.sort(key=lambda t: t.get("score", 5), reverse=True)

            best = topics[0] if topics else topics

            # ---- STEP 2: Generate Content ----
            is_thread = best.get("content_type", "long_post") == "thread"
            num_variants = 3 if not is_thread else 1

            thread_instr = ""
            if is_thread:
                seg = best.get("segment_count", 7)
                seg = max(5, min(10, seg))
                thread_instr = THREAD_INSTRUCTION.format(
                    segment_count=seg, middle_end=seg - 2,
                    before_last=seg - 1,
                )
                min_chars = 200
                max_chars = 280
            else:
                min_chars = 500
                max_chars = 1500

            gen_prompt = CONTENT_GENERATION_PROMPT.format(
                algorithm_knowledge=ALGORITHM_KNOWLEDGE,
                num_variants=num_variants,
                topic=best.get("topic", ""),
                hook=best.get("hook", ""),
                argument=best.get("argument", "A debatable position on this topic"),
                data_points=json.dumps(best.get("data_points", []), ensure_ascii=False),
                content_type=best.get("content_type", "long_post"),
                tone=best.get("tone", "provocative"),
                goal=best.get("goal", "engagement"),
                min_chars=min_chars,
                max_chars=max_chars,
                thread_instruction=thread_instr,
                language_instruction=lang_instr,
            )

            gen_messages = [
                {"role": "system", "content": "You are an elite X/Twitter ghostwriter. You write viral posts that are LONG, informative, and argumentative. Output only valid JSON."},
                {"role": "user", "content": gen_prompt},
            ]

            raw_content = loop.run_until_complete(
                call_llm(api_key, provider, model, gen_messages, temperature=0.8, max_tokens=4096)
            )

            try:
                variants = json.loads(extract_json(raw_content))
            except (json.JSONDecodeError, ValueError):
                variants = [{"text": raw_content.strip(), "hook_type": "unknown", "expected_action": "like", "image_prompt": ""}]

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
                    "image_prompt": v.get("image_prompt", "") if isinstance(v, dict) else "",
                    "prediction": pred,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                })

            results.sort(key=lambda x: x["prediction"]["weighted_score"], reverse=True)

            # ---- STEP 4: Generate Image for best variant ----
            image_url = ""
            if generate_visual and provider == "openai" and results:
                best_image_prompt = results[0].get("image_prompt", "")
                if best_image_prompt:
                    try:
                        image_url = loop.run_until_complete(
                            generate_image(api_key, best_image_prompt)
                        )
                    except Exception:
                        image_url = ""

            loop.close()

            self._respond(200, {
                "topic_selected": best,
                "all_topics": topics[:5],
                "drafts": results,
                "count": len(results),
                "is_thread": is_thread,
                "image_url": image_url,
                "pipeline": "auto",
            })

        except httpx.HTTPStatusError as e:
            detail = e.response.text[:300] if e.response else ""
            detail = _scrub_secrets(detail, body.get("api_key", ""))
            self._respond(e.response.status_code, {
                "error": f"LLM API hatasi: {e.response.status_code}",
                "detail": detail,
            })
        except Exception as e:
            msg = _scrub_secrets(str(e), body.get("api_key", ""))
            self._respond(500, {"error": msg})

    def do_OPTIONS(self):
        self.send_response(200)
        origin = self.headers.get("Origin", "")
        self._set_cors(origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _set_cors(self, origin):
        host = self.headers.get("Host", "")
        if origin and host and (host in origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", f"https://{host}" if host else "")
        self.send_header("Vary", "Origin")

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        origin = self.headers.get("Origin", "")
        self._set_cors(origin)
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
