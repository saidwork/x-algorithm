"""
X Content Agent - Streamlit Dashboard

Run with: streamlit run app.py
"""

import asyncio
import sys
from pathlib import Path

import streamlit as st

# Ensure agent package is importable
sys.path.insert(0, str(Path(__file__).parent))

from agent.config import AgentConfig, LLMConfig
from agent.content.optimizer import ContentOptimizer
from agent.content.templates import TemplateEngine
from agent.content.thread_composer import ThreadComposer
from agent.audience.engagement_predictor import EngagementPredictor
from agent.audience.timing_optimizer import TimingOptimizer
from agent.analytics.metrics_tracker import MetricsTracker
from agent.analytics.report_generator import ReportGenerator
from agent.orchestrator.agent import XContentAgent
from agent.types import (
    ContentConstraints,
    ContentGoal,
    ContentPiece,
    ContentRequest,
    ContentTone,
    ContentType,
    PostDraft,
    PostStatus,
    Topic,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="X Content Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Dark theme override */
    .stApp {
        background-color: #0a0a0a;
    }

    /* Card style */
    .draft-card {
        background: #16181c;
        border: 1px solid #2f3336;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }

    .draft-card:hover {
        border-color: #1d9bf0;
    }

    .metric-card {
        background: #16181c;
        border: 1px solid #2f3336;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }

    .score-high { background: #00ba7c22; color: #00ba7c; }
    .score-mid { background: #ffd40022; color: #ffd400; }
    .score-low { background: #f9197f22; color: #f9197f; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "agent": None,
        "config": None,
        "api_key": "",
        "provider": "openai",
        "model": "gpt-4o",
        "account_handle": "",
        "niche": "",
        "niche_keywords": "",
        "brand_description": "",
        "language": "tr",
        "generated_drafts": [],
        "setup_complete": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# Helper to run async functions
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_agent() -> XContentAgent | None:
    """Get or create the agent from session state."""
    return st.session_state.get("agent")


def create_agent():
    """Create agent from current session state config."""
    keywords = [k.strip() for k in st.session_state.niche_keywords.split(",") if k.strip()]

    config = AgentConfig(
        account_handle=st.session_state.account_handle,
        niche=st.session_state.niche,
        niche_keywords=keywords,
        brand_description=st.session_state.brand_description,
        default_language=st.session_state.language,
        llm=LLMConfig(
            provider=st.session_state.provider,
            model=st.session_state.model,
            api_key=st.session_state.api_key,
            temperature=0.8,
            max_tokens=1024,
        ),
    )

    agent = XContentAgent(config)
    st.session_state.agent = agent
    st.session_state.config = config
    st.session_state.setup_complete = True
    return agent


# ---------------------------------------------------------------------------
# Sidebar - Settings & Configuration
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 X Content Agent")
        st.markdown("---")

        # --- API Configuration ---
        st.markdown("### API Ayarlari")

        st.session_state.provider = st.selectbox(
            "LLM Provider",
            ["openai", "anthropic", "local"],
            index=["openai", "anthropic", "local"].index(st.session_state.provider),
            help="OpenAI, Anthropic veya lokal bir API secin",
        )

        model_options = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
            "local": ["local-model"],
        }
        models = model_options.get(st.session_state.provider, ["gpt-4o"])
        st.session_state.model = st.selectbox("Model", models)

        st.session_state.api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="OpenAI veya Anthropic API anahtarinizi girin",
        )

        st.markdown("---")

        # --- Account Configuration ---
        st.markdown("### Hesap Ayarlari")

        st.session_state.account_handle = st.text_input(
            "X Kullanici Adi",
            value=st.session_state.account_handle,
            placeholder="@kullaniciadi",
        )

        st.session_state.niche = st.text_input(
            "Nis / Alan",
            value=st.session_state.niche,
            placeholder="ornek: Yapay Zeka, Teknoloji",
        )

        st.session_state.niche_keywords = st.text_area(
            "Nis Anahtar Kelimeler",
            value=st.session_state.niche_keywords,
            placeholder="AI, yapay zeka, machine learning, deep learning",
            help="Virgul ile ayirin",
            height=80,
        )

        st.session_state.brand_description = st.text_area(
            "Marka Sesi Tanimi",
            value=st.session_state.brand_description,
            placeholder="Gelistiriciler icin pratik AI icgoruleri paylasiyoruz",
            height=80,
        )

        st.session_state.language = st.selectbox(
            "Icerik Dili",
            ["tr", "en", "de", "fr", "es"],
            index=["tr", "en", "de", "fr", "es"].index(st.session_state.language),
        )

        st.markdown("---")

        # --- Initialize Agent ---
        if st.button("Agent'i Baslat / Guncelle", use_container_width=True, type="primary"):
            if not st.session_state.api_key:
                st.error("Lutfen bir API Key girin!")
            elif not st.session_state.account_handle:
                st.error("Lutfen X kullanici adinizi girin!")
            else:
                create_agent()
                st.success("Agent basariyla baslatildi!")

        if st.session_state.setup_complete:
            st.markdown("---")
            st.markdown("### Durum")
            st.markdown(f"**Hesap:** @{st.session_state.account_handle}")
            st.markdown(f"**Model:** {st.session_state.model}")
            st.markdown(f"**Nis:** {st.session_state.niche}")
            st.success("Agent aktif")


# ---------------------------------------------------------------------------
# Tab 1: Content Generation
# ---------------------------------------------------------------------------

def render_content_generation():
    st.markdown("## Icerik Uret")
    st.markdown("X icin optimize edilmis icerik uretmek icin asagidaki parametreleri ayarlayin.")

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.text_input(
            "Konu",
            placeholder="ornek: AI agentlerin gelecegi, Python ipuclari, startup dersleri",
            help="Icerigin ana konusu",
        )

        col_type, col_tone, col_goal = st.columns(3)

        with col_type:
            content_type = st.selectbox(
                "Icerik Turu",
                [
                    ("Kisa Post (280 karakter)", ContentType.SHORT_POST),
                    ("Thread (Coklu post)", ContentType.THREAD),
                    ("Uzun Post", ContentType.LONG_POST),
                    ("Alinti Post", ContentType.QUOTE_POST),
                    ("Anket", ContentType.POLL),
                ],
                format_func=lambda x: x[0],
            )

        with col_tone:
            tone = st.selectbox(
                "Ton",
                [
                    ("Profesyonel", ContentTone.PROFESSIONAL),
                    ("Egitici", ContentTone.EDUCATIONAL),
                    ("Gundelik", ContentTone.CASUAL),
                    ("Dusunduren", ContentTone.PROVOCATIVE),
                    ("Esprili", ContentTone.HUMOROUS),
                    ("Ilham Verici", ContentTone.INSPIRATIONAL),
                    ("Haber", ContentTone.NEWS),
                ],
                format_func=lambda x: x[0],
            )

        with col_goal:
            goal = st.selectbox(
                "Hedef",
                [
                    ("Etkilesim", ContentGoal.ENGAGEMENT),
                    ("Erisim", ContentGoal.REACH),
                    ("Takipci Artisi", ContentGoal.FOLLOWER_GROWTH),
                    ("Otorite", ContentGoal.AUTHORITY),
                    ("Topluluk", ContentGoal.COMMUNITY),
                    ("Trafik", ContentGoal.TRAFFIC),
                ],
                format_func=lambda x: x[0],
            )

        num_variants = st.slider("Varyant Sayisi", 1, 5, 3, help="Kac farkli icerik uretilsin")

        # Additional constraints
        with st.expander("Gelismis Ayarlar"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                required_hashtags = st.text_input(
                    "Zorunlu Hashtag'ler",
                    placeholder="#AI, #tech",
                )
                forbidden_words = st.text_input(
                    "Yasakli Kelimeler",
                    placeholder="spam, clickbait",
                )
            with col_c2:
                max_chars = st.number_input("Maks Karakter", value=280, min_value=50, max_value=25000)
                include_media = st.checkbox("Medya Onerisi Ekle", value=True)

    with col2:
        st.markdown("### Hizli Bilgi")
        st.info(
            "**En iyi sonuc icin:**\n"
            "- Spesifik bir konu girin\n"
            "- Hedef kitlenize uygun ton secin\n"
            "- 3 varyant uretin, en iyisini secin\n"
            "- Thread'ler genelde daha cok erisim alir"
        )

        if st.session_state.setup_complete:
            st.markdown("### Tahmin")
            st.markdown(f"**Model:** `{st.session_state.model}`")
            st.markdown(f"**Dil:** `{st.session_state.language}`")
            st.markdown(f"**Nis:** `{st.session_state.niche}`")

    st.markdown("---")

    # Generate button
    generate_clicked = st.button(
        "Icerik Uret",
        use_container_width=True,
        type="primary",
        disabled=not st.session_state.setup_complete,
    )

    if not st.session_state.setup_complete and generate_clicked:
        st.warning("Lutfen once sol panelden Agent'i baslatin!")

    if generate_clicked and st.session_state.setup_complete and topic:
        agent = get_agent()
        if agent:
            with st.spinner("Icerik uretiliyor... (LLM cagiriliyor)"):
                try:
                    drafts = run_async(
                        agent.generate_content(
                            topic=topic,
                            content_type=content_type[1],
                            tone=tone[1],
                            goal=goal[1],
                            num_variants=num_variants,
                        )
                    )
                    st.session_state.generated_drafts = drafts

                    if drafts:
                        st.success(f"{len(drafts)} icerik varyanti uretildi!")
                    else:
                        st.warning("Icerik uretilemedi. API baglantinizi kontrol edin.")
                except Exception as e:
                    st.error(f"Hata: {e}")

    # Display generated drafts
    if st.session_state.generated_drafts:
        render_drafts(st.session_state.generated_drafts)


def render_drafts(drafts: list[PostDraft]):
    """Render generated drafts with scores."""
    st.markdown("### Uretilen Icerikler")

    for i, draft in enumerate(drafts):
        with st.container():
            st.markdown(f"""
            <div class="draft-card">
            """, unsafe_allow_html=True)

            col_content, col_metrics = st.columns([3, 1])

            with col_content:
                st.markdown(f"**Varyant {i + 1}** | `{draft.tags[0] if draft.tags else 'standart'}`")

                if draft.is_thread:
                    for j, piece in enumerate(draft.pieces):
                        st.markdown(f"**{j + 1}/{len(draft.pieces)}** {piece.text}")
                else:
                    for piece in draft.pieces:
                        st.markdown(f"> {piece.text}")
                        st.caption(f"{piece.character_count} karakter")

                if draft.pieces and draft.pieces[0].hashtags:
                    st.markdown(
                        " ".join(f"`#{tag}`" for tag in draft.pieces[0].hashtags)
                    )

            with col_metrics:
                pred = draft.engagement_prediction
                if pred:
                    score = pred.weighted_score
                    if score > 1.0:
                        badge = "score-high"
                        label = "Yuksek"
                    elif score > 0.3:
                        badge = "score-mid"
                        label = "Orta"
                    else:
                        badge = "score-low"
                        label = "Dusuk"

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="score-badge {badge}">{label}</div>
                        <br><br>
                        <small>Skor: {score:.2f}</small><br>
                        <small>Beg: {pred.p_like:.1%}</small><br>
                        <small>Ynt: {pred.p_reply:.1%}</small><br>
                        <small>RT: {pred.p_repost:.1%}</small><br>
                        <small>Erisim: ~{pred.predicted_impressions:,}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # Copy button
                full_text = draft.full_text
                st.code(full_text, language=None)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")


# ---------------------------------------------------------------------------
# Tab 2: Templates
# ---------------------------------------------------------------------------

def render_templates():
    st.markdown("## Sablon Kutuphanesi")
    st.markdown("Yuksek performansli X post formatlari icin hazir sablonlar.")

    engine = TemplateEngine()
    templates = engine.list_templates()

    # Filter
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        goal_filter = st.selectbox(
            "Hedefe Gore Filtrele",
            [("Tumu", None)] + [
                ("Etkilesim", ContentGoal.ENGAGEMENT),
                ("Erisim", ContentGoal.REACH),
                ("Otorite", ContentGoal.AUTHORITY),
                ("Topluluk", ContentGoal.COMMUNITY),
                ("Takipci Artisi", ContentGoal.FOLLOWER_GROWTH),
            ],
            format_func=lambda x: x[0],
        )
    with col_f2:
        type_filter = st.selectbox(
            "Ture Gore Filtrele",
            [("Tumu", None)] + [
                ("Kisa Post", ContentType.SHORT_POST),
                ("Thread", ContentType.THREAD),
                ("Anket", ContentType.POLL),
            ],
            format_func=lambda x: x[0],
        )

    filtered = engine.list_templates(
        content_type=type_filter[1] if type_filter[1] else None,
        goal=goal_filter[1] if goal_filter[1] else None,
    )

    st.markdown("---")

    for tmpl in filtered:
        with st.expander(f"**{tmpl.name}** | `{tmpl.content_type.value}` | Hedef: `{tmpl.goal.value}`"):
            st.markdown(f"**ID:** `{tmpl.id}`")
            st.markdown(f"**Ton:** `{tmpl.tone.value}`")
            st.markdown(f"**Etiketler:** {', '.join(f'`{t}`' for t in tmpl.tags)}")

            st.markdown("**Sablon:**")
            st.code(tmpl.template, language="jinja2")

            st.markdown("**Gerekli Degiskenler:**")
            st.markdown(", ".join(f"`{v}`" for v in tmpl.required_vars))

            # Quick fill form
            st.markdown("---")
            st.markdown("**Hizli Kullan:**")
            variables = {}
            for var in tmpl.required_vars:
                if var == "mistakes" or var == "resources":
                    val = st.text_area(
                        f"{var} (her satira bir tane)",
                        key=f"tmpl_{tmpl.id}_{var}",
                        height=80,
                    )
                    variables[var] = [line.strip() for line in val.split("\n") if line.strip()]
                    if var == "mistakes":
                        variables["count"] = len(variables[var])
                    elif var == "resources":
                        variables["count"] = len(variables[var])
                else:
                    variables[var] = st.text_input(
                        var,
                        key=f"tmpl_{tmpl.id}_{var}",
                    )

            if st.button("Uygula", key=f"apply_{tmpl.id}"):
                result = engine.render(tmpl.id, variables)
                if result:
                    st.success("Sablon uyguland!")
                    st.code(result, language=None)
                    st.metric("Karakter", len(result))
                else:
                    st.error("Tum zorunlu alanlari doldurun.")


# ---------------------------------------------------------------------------
# Tab 3: Thread Builder
# ---------------------------------------------------------------------------

def render_thread_builder():
    st.markdown("## Thread Olusturucu")
    st.markdown("Yapilandirilmis, anlatim akisina sahip X thread'leri olusturun.")

    agent = get_agent()

    topic = st.text_input("Thread Konusu", placeholder="ornek: AI Agent Gelistirme Rehberi")

    key_points = st.text_area(
        "Ana Noktalar (her satira bir tane)",
        placeholder="1. Agentlerin temel mimarisi\n2. Prompt muhendisligi\n3. Arac entegrasyonu\n4. Test stratejileri",
        height=150,
    )

    col1, col2 = st.columns(2)
    with col1:
        thread_tone = st.selectbox(
            "Ton",
            [
                ("Egitici", ContentTone.EDUCATIONAL),
                ("Profesyonel", ContentTone.PROFESSIONAL),
                ("Gundelik", ContentTone.CASUAL),
            ],
            format_func=lambda x: x[0],
            key="thread_tone",
        )
    with col2:
        thread_goal = st.selectbox(
            "Hedef",
            [
                ("Otorite", ContentGoal.AUTHORITY),
                ("Etkilesim", ContentGoal.ENGAGEMENT),
                ("Takipci Artisi", ContentGoal.FOLLOWER_GROWTH),
            ],
            format_func=lambda x: x[0],
            key="thread_goal",
        )

    if st.button("Thread Olustur", type="primary", use_container_width=True):
        if not topic or not key_points:
            st.warning("Konu ve ana noktalari girin.")
        elif not st.session_state.setup_complete:
            st.warning("Lutfen once Agent'i baslatin!")
        elif agent:
            points = [p.strip() for p in key_points.split("\n") if p.strip()]
            with st.spinner("Thread olusturuluyor..."):
                try:
                    draft = run_async(agent.generate_thread(
                        topic=topic,
                        key_points=points,
                        tone=thread_tone[1],
                    ))

                    if draft:
                        st.success(f"Thread olusturuldu! ({len(draft.pieces)} segment)")
                        st.markdown("---")

                        for j, piece in enumerate(draft.pieces):
                            col_num, col_text = st.columns([0.1, 0.9])
                            with col_num:
                                st.markdown(f"### {j + 1}")
                            with col_text:
                                st.info(piece.text)
                                st.caption(f"{piece.character_count} karakter")

                        # Full thread copy
                        st.markdown("---")
                        st.markdown("### Tam Thread (Kopyalamak icin)")
                        full = "\n\n---\n\n".join(
                            f"**{i+1}/{len(draft.pieces)}**\n{p.text}"
                            for i, p in enumerate(draft.pieces)
                        )
                        st.code(draft.full_text, language=None)
                except Exception as e:
                    st.error(f"Hata: {e}")


# ---------------------------------------------------------------------------
# Tab 4: Analytics
# ---------------------------------------------------------------------------

def render_analytics():
    st.markdown("## Analitik & Performans")

    agent = get_agent()

    if not agent:
        st.info("Agent henuz baslatilmadi. Sol panelden baslatin.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    stats = agent.state_manager.get_stats()

    with col1:
        st.metric("Uretilen Icerik", stats["total_generated"])
    with col2:
        st.metric("Kuyrukta", stats["drafts_in_queue"])
    with col3:
        st.metric("Planlanmis", stats["scheduled"])
    with col4:
        st.metric("Yayinlanmis", stats["published"])

    st.markdown("---")

    # Timing optimizer
    st.markdown("### Optimal Paylasim Zamanlari")

    timing = agent.timing_optimizer
    optimal_times = timing.get_optimal_times(count=5)

    if optimal_times:
        for ot in optimal_times:
            col_time, col_score, col_reason = st.columns([1, 1, 3])
            with col_time:
                st.markdown(f"**{ot.datetime_utc.strftime('%a %H:%M')} UTC**")
            with col_score:
                st.progress(ot.score, text=f"Skor: {ot.score:.2f}")
            with col_reason:
                st.caption(ot.reason)

    st.markdown("---")

    # Reports
    st.markdown("### Raporlar")

    report_type = st.selectbox("Rapor Turu", ["Gunluk", "Haftalik", "Aylik"])

    if st.button("Rapor Olustur"):
        if report_type == "Gunluk":
            report = agent.get_daily_report()
        elif report_type == "Haftalik":
            report = agent.get_weekly_report()
        else:
            report = agent.report_generator.generate_monthly_report()

        st.code(report, language="markdown")


# ---------------------------------------------------------------------------
# Tab 5: Draft Queue
# ---------------------------------------------------------------------------

def render_draft_queue():
    st.markdown("## Icerik Kuyrugu")

    agent = get_agent()

    if not agent:
        st.info("Agent henuz baslatilmadi.")
        return

    pending = agent.get_pending_drafts()

    if not pending:
        st.info("Kuyrukta bekleyen icerik yok. 'Icerik Uret' sekmesinden icerik uretin.")
        return

    st.markdown(f"**{len(pending)} icerik inceleme bekliyor**")

    for draft in pending:
        with st.container():
            st.markdown(f"""
            <div class="draft-card">
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                st.markdown(f"**Draft ID:** `{draft.id}`")
                st.markdown(f"**Konu:** {draft.topic.name if draft.topic else 'Belirtilmemis'}")
                st.markdown(f"**Ton:** `{draft.tone.value}` | **Hedef:** `{draft.goal.value}`")

                for piece in draft.pieces:
                    st.info(piece.text)

                if draft.engagement_prediction:
                    pred = draft.engagement_prediction
                    st.caption(
                        f"Tahmini: Beg {pred.p_like:.1%} | Ynt {pred.p_reply:.1%} | "
                        f"RT {pred.p_repost:.1%} | Skor: {pred.weighted_score:.2f}"
                    )

            with col2:
                if st.button("Onayla", key=f"approve_{draft.id}", type="primary"):
                    agent.approve_draft(draft.id)
                    st.success("Onaylandi!")
                    st.rerun()

            with col3:
                if st.button("Reddet", key=f"reject_{draft.id}"):
                    agent.reject_draft(draft.id, "Manuel red")
                    st.warning("Reddedildi")
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 6: Engagement Calculator
# ---------------------------------------------------------------------------

def render_engagement_calculator():
    st.markdown("## Etkilesim Hesaplayici")
    st.markdown("Bir icerigin tahmini performansini hesaplayin.")

    agent = get_agent()

    text = st.text_area(
        "Post Metni",
        placeholder="Buraya test etmek istediginiz icerigi yapin...",
        height=150,
    )

    col1, col2 = st.columns(2)
    with col1:
        calc_type = st.selectbox(
            "Icerik Turu",
            [
                ("Kisa Post", ContentType.SHORT_POST),
                ("Thread", ContentType.THREAD),
                ("Anket", ContentType.POLL),
            ],
            format_func=lambda x: x[0],
            key="calc_type",
        )
    with col2:
        trending_score = st.slider("Trend Skoru", 0.0, 1.0, 0.3, 0.1)

    if st.button("Analiz Et", type="primary") and text:
        config = st.session_state.config or AgentConfig()
        predictor = EngagementPredictor(config)
        optimizer = ContentOptimizer(config)

        piece = ContentPiece(text=text, content_type=calc_type[1])
        draft = PostDraft(
            id="calc-1",
            pieces=[piece],
            topic=Topic(name="custom", trending_score=trending_score),
        )

        prediction = predictor.predict(draft)

        # Optimization analysis
        request = ContentRequest(request_id="calc", account_id="a1", topic=Topic(name="test"))
        opt_result = optimizer.optimize_piece(piece, request)

        st.markdown("---")

        # Results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Etkilesim Tahmini")
            st.metric("Toplam Skor", f"{prediction.weighted_score:.2f}")
            st.metric("Beg. Olasil.", f"{prediction.p_like:.1%}")
            st.metric("Yanit Olasil.", f"{prediction.p_reply:.1%}")
            st.metric("RT Olasil.", f"{prediction.p_repost:.1%}")
            st.metric("Yer Imi Olasil.", f"{prediction.p_bookmark:.1%}")
            st.metric("Tahmini Erisim", f"~{prediction.predicted_impressions:,}")

        with col2:
            st.markdown("### Icerik Analizi")
            st.metric("Hook Gucu", f"{opt_result.hook_score:.0%}")
            st.metric("Etkilesim Tetikleyici", f"{opt_result.engagement_trigger_score:.0%}")
            st.metric("Okunabilirlik", f"{opt_result.readability_score:.0%}")
            st.metric("Karakter Sayisi", len(text))

        with col3:
            st.markdown("### Virallik")
            st.metric("Virallik Skoru", f"{prediction.virality_score:.0%}")
            st.metric("Takip Olasil.", f"{prediction.p_follow:.2%}")
            st.metric("Profil Ziyareti", f"{prediction.p_profile_visit:.1%}")
            st.markdown("---")
            st.markdown("**Negatif Sinyaller**")
            st.metric("Susturma Risk", f"{prediction.p_mute:.3%}")
            st.metric("Takipten Cikma Risk", f"{prediction.p_unfollow:.3%}")

        if opt_result.changes_made:
            st.markdown("### Optimizasyon Onerileri")
            for change in opt_result.changes_made:
                st.markdown(f"- {change}")


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Icerik Uret",
        "Sablonlar",
        "Thread Olusturucu",
        "Analitik",
        "Icerik Kuyrugu",
        "Etkilesim Hesaplayici",
    ])

    with tab1:
        render_content_generation()

    with tab2:
        render_templates()

    with tab3:
        render_thread_builder()

    with tab4:
        render_analytics()

    with tab5:
        render_draft_queue()

    with tab6:
        render_engagement_calculator()


if __name__ == "__main__":
    main()
