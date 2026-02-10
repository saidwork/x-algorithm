"""
GET /api/templates - List available content templates.
POST /api/templates - Render a template with variables.
"""

from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import parse_qs, urlparse
from jinja2 import Environment, BaseLoader

TEMPLATES = [
    {"id": "hot_take", "name": "Hot Take / Karsi Gorus", "type": "short_post", "tone": "provocative", "goal": "engagement",
     "template": "Unpopular opinion: {{ opinion }}\n\n{{ supporting_point }}\n\nAgree or disagree?",
     "required_vars": ["opinion", "supporting_point"], "tags": ["tartisma", "etkilesim"]},

    {"id": "this_or_that", "name": "Bu mu O mu", "type": "short_post", "tone": "casual", "goal": "engagement",
     "template": "{{ option_a }} or {{ option_b }}?\n\n{{ context }}\n\nDrop your answer below 👇",
     "required_vars": ["option_a", "option_b"], "optional_vars": ["context"], "tags": ["etkilesim", "tartisma"]},

    {"id": "quick_tip", "name": "Hizli Ipucu", "type": "short_post", "tone": "educational", "goal": "authority",
     "template": "💡 {{ topic }} tip:\n\n{{ tip }}\n\nMost people don't know this, but it can {{ benefit }}.",
     "required_vars": ["topic", "tip", "benefit"], "tags": ["ipucu", "egitici"]},

    {"id": "mistake_list", "name": "Hatalar Listesi", "type": "short_post", "tone": "educational", "goal": "engagement",
     "template": "{{ count }} {{ topic }} mistakes that are costing you {{ cost }}:\n\n{% for m in mistakes %}{{ loop.index }}. {{ m }}\n{% endfor %}\nWhich one are you guilty of?",
     "required_vars": ["count", "topic", "cost", "mistakes"], "tags": ["liste", "egitici"]},

    {"id": "stat_hook", "name": "Istatistik Hook", "type": "short_post", "tone": "professional", "goal": "reach",
     "template": "{{ statistic }}\n\n{{ explanation }}\n\nHere's why this matters 👇",
     "required_vars": ["statistic", "explanation"], "tags": ["istatistik", "erisim"]},

    {"id": "before_after", "name": "Once / Sonra", "type": "short_post", "tone": "inspirational", "goal": "reach",
     "template": "{{ timeframe }} ago: {{ before }}\n\nToday: {{ after }}\n\n{{ lesson }}",
     "required_vars": ["timeframe", "before", "after", "lesson"], "tags": ["donusum", "ilham"]},

    {"id": "community_question", "name": "Topluluk Sorusu", "type": "short_post", "tone": "casual", "goal": "community",
     "template": "Question for {{ audience }}:\n\n{{ question }}\n\nI'll start: {{ own_answer }}",
     "required_vars": ["audience", "question", "own_answer"], "tags": ["topluluk", "soru"]},

    {"id": "resource_share", "name": "Kaynak Paylasimi", "type": "short_post", "tone": "professional", "goal": "follower_growth",
     "template": "{{ count }} {{ resource_type }} that will {{ benefit }}:\n\n{% for r in resources %}{{ loop.index }}. {{ r }}\n{% endfor %}\nBookmark this for later 🔖\n\nFollow for more {{ topic }} content.",
     "required_vars": ["count", "resource_type", "benefit", "resources", "topic"], "tags": ["kaynak", "takipci"]},

    {"id": "lesson_learned", "name": "Ogrenilenler", "type": "short_post", "tone": "professional", "goal": "authority",
     "template": "After {{ experience }}:\n\nThe biggest lesson I learned:\n\n{{ lesson }}\n\n{{ actionable_takeaway }}",
     "required_vars": ["experience", "lesson", "actionable_takeaway"], "tags": ["ders", "otorite"]},

    {"id": "news_take", "name": "Haber Yorumu", "type": "short_post", "tone": "news", "goal": "reach",
     "template": "{{ news_hook }}\n\n{{ analysis }}\n\nWhat this means for {{ audience }}: {{ implication }}",
     "required_vars": ["news_hook", "analysis", "audience", "implication"], "tags": ["haber", "erisim"]},

    {"id": "how_to_thread", "name": "Nasil Yapilir Thread", "type": "thread", "tone": "educational", "goal": "authority",
     "template": "🧵 How to {{ goal }} (step by step):\n\nA thread 👇",
     "required_vars": ["goal"], "tags": ["thread", "egitici"]},

    {"id": "fill_blank", "name": "Bosluk Doldur", "type": "short_post", "tone": "casual", "goal": "engagement",
     "template": "{{ setup }} ______.\n\nWrong answers only 😂",
     "required_vars": ["setup"], "tags": ["eglence", "etkilesim"]},
]

env = Environment(loader=BaseLoader())


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        goal = params.get("goal", [None])[0]
        ttype = params.get("type", [None])[0]

        result = TEMPLATES
        if goal:
            result = [t for t in result if t["goal"] == goal]
        if ttype:
            result = [t for t in result if t["type"] == ttype]

        self._respond(200, {"templates": result, "count": len(result)})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            template_id = body.get("template_id", "")
            variables = body.get("variables", {})

            tmpl = next((t for t in TEMPLATES if t["id"] == template_id), None)
            if not tmpl:
                self._respond(404, {"error": f"Sablon bulunamadi: {template_id}"})
                return

            missing = [v for v in tmpl["required_vars"] if v not in variables]
            if missing:
                self._respond(400, {"error": f"Eksik degiskenler: {missing}"})
                return

            jinja = env.from_string(tmpl["template"])
            rendered = jinja.render(**variables)
            self._respond(200, {"rendered": rendered, "char_count": len(rendered), "template": tmpl})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        origin = self.headers.get("Origin", "")
        self._set_cors(origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
