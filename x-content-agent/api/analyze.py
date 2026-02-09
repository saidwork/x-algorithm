"""
POST /api/analyze - Engagement analysis endpoint.
Analyzes text and returns predicted engagement metrics.
"""

from http.server import BaseHTTPRequestHandler
import json
import re


def predict_engagement(text, trending_score=0.3, is_thread=False):
    char_count = len(text)
    length_score = max(0.0, min(1.0, 1.0 - abs(char_count - 105) / 200))

    features = {
        "has_question": 1.0 if "?" in text else 0.0,
        "has_numbers": 1.0 if re.search(r"\d+", text) else 0.0,
        "has_emoji": 1.0 if re.search(r"[\U0001F300-\U0001F9FF]", text) else 0.0,
        "is_thread": 1.0 if is_thread else 0.0,
        "is_timely": trending_score,
        "char_length_optimal": length_score,
        "has_hook": 1.0 if re.search(r"^(How|Why|What|Stop|Don't|Most people|\d+\s)", text, re.IGNORECASE) else 0.0,
        "has_cta": 1.0 if re.search(r"(follow|repost|share|bookmark|comment|reply|thoughts\??)", text, re.IGNORECASE) else 0.0,
        "readability": 0.7,
    }

    weights = {"has_question": 0.12, "has_numbers": 0.08, "has_emoji": 0.05,
               "is_thread": 0.10, "is_timely": 0.12, "char_length_optimal": 0.08,
               "has_hook": 0.15, "has_cta": 0.10, "readability": 0.05}

    base = sum(weights.get(k, 0) * v for k, v in features.items())
    base = max(0.001, min(0.5, base))
    type_mult = 1.3 if is_thread else 1.0
    ep = base * type_mult
    q_mult = 1.5 if features["has_question"] else 1.0
    t_mult = 1.5 if is_thread else 1.0

    pred = {
        "p_like": round(ep * 0.60, 4), "p_reply": round(ep * 0.08 * q_mult, 4),
        "p_repost": round(ep * 0.12 * t_mult, 4), "p_quote": round(ep * 0.04, 4),
        "p_click": round(ep * 0.10, 4), "p_bookmark": round(ep * 0.05 * t_mult, 4),
        "p_follow": round(ep * 0.01, 4), "p_profile_visit": round(ep * 0.03, 4),
        "p_mute": round(max(0.0, 0.001 - ep * 0.01), 4),
        "p_unfollow": round(max(0.0, 0.0005 - ep * 0.005), 4),
        "p_report": 0.0001,
        "predicted_impressions": int(1000 * 0.10 * (1.0 + ep * 10)),
        "engagement_rate": round(ep, 4),
    }

    pred["weighted_score"] = round(
        1.0 * pred["p_like"] + 3.0 * pred["p_reply"] + 5.0 * pred["p_repost"]
        + 4.0 * pred["p_quote"] + 2.0 * pred["p_click"] + 2.5 * pred["p_bookmark"]
        + 10.0 * pred["p_follow"] + 1.5 * pred["p_profile_visit"]
        - 20.0 * pred["p_mute"] - 30.0 * pred["p_unfollow"] - 50.0 * pred["p_report"], 4
    )

    virality = ep * 0.3 + features["is_timely"] * 0.3 + features["has_hook"] * 0.2 + features["is_thread"] * 0.1
    pred["virality_score"] = round(max(0.0, min(1.0, virality)), 4)

    # Hook score
    hook = 0.0
    first_line = text.split("\n")[0]
    for p in [r"^\d+\s", r"^(How|Why|What|When)\s", r"^(Stop|Don't|Never)\s",
              r"^(The truth|Here's what|Most people|Unpopular opinion)", r"^(I just|After \d+)"]:
        if re.search(p, first_line, re.IGNORECASE):
            hook += 0.3
    if len(first_line) < 80:
        hook += 0.2
    pred["hook_score"] = round(min(hook, 1.0), 2)

    trigger = 0.0
    for p in [r"\?$", r"(agree|disagree|thoughts)\??$", r"(share|repost)\s", r"(comment|reply)\s"]:
        if re.search(p, text, re.IGNORECASE):
            trigger += 0.25
    if "\n" in text:
        trigger += 0.1
    pred["engagement_trigger_score"] = round(min(trigger, 1.0), 2)
    pred["char_count"] = char_count
    pred["features"] = {k: round(v, 2) for k, v in features.items()}

    return pred


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            text = body.get("text", "")
            trending = body.get("trending_score", 0.3)
            is_thread = body.get("is_thread", False)

            if not text:
                self._respond(400, {"error": "Metin gerekli"})
                return

            pred = predict_engagement(text, trending, is_thread)
            self._respond(200, pred)
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
