# X Content Agent

AI-powered content generation and strategy engine for the X platform. Built with the same pipeline-based, modular architecture as the X "For You" recommendation algorithm, but inverted: instead of ranking existing content for consumption, this agent **generates, optimizes, and schedules new content** for publication.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              XContentAgent                                       │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                         ContentPipeline                                    │  │
│  │                                                                            │  │
│  │  RequestHydrators → Generators → DraftHydrators → Filters → Scorers       │  │
│  │  → Selector → PostProcessors                                              │  │
│  │                                                                            │  │
│  │  (Mirrors home-mixer's CandidatePipeline: Sources → Hydrators → Filters   │  │
│  │   → Scorers → Selector → SideEffects)                                     │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐     │
│  │    Strategy       │  │    Audience      │  │    Analytics               │     │
│  │  ┌──────────────┐ │  │  ┌────────────┐ │  │  ┌──────────────────────┐ │     │
│  │  │TrendAnalyzer │ │  │  │Engagement  │ │  │  │MetricsTracker       │ │     │
│  │  │TopicSelector │ │  │  │Predictor   │ │  │  │ABTester             │ │     │
│  │  │ContentCalendr│ │  │  │FollowerAnlz│ │  │  │FeedbackLoop         │ │     │
│  │  │CompetitorAnlz│ │  │  │PersonaBlder│ │  │  │ReportGenerator      │ │     │
│  │  └──────────────┘ │  │  │TimingOptmzr│ │  │  └──────────────────────┘ │     │
│  └──────────────────┘  │  └────────────┘ │  └────────────────────────────┘     │
│                         └──────────────────┘                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐     │
│  │    Workflow       │  │    Scheduler     │  │    State Manager           │     │
│  └──────────────────┘  └──────────────────┘  └────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## How It Relates to the Recommendation Algorithm

This agent is the **creation-side mirror** of the recommendation system:

| Recommendation System (home-mixer) | Content Agent (x-content-agent) |
|-----------------------------------|---------------------------------|
| **Source** → Fetch candidate posts | **TopicSource** → Find content opportunities |
| **QueryHydrator** → Enrich user context | **RequestHydrator** → Enrich with audience data |
| **Hydrator** → Add post metadata | **DraftHydrator** → Optimize content, add media |
| **Filter** → Remove bad candidates | **ContentFilter** → Remove low-quality drafts |
| **Scorer** → Predict P(engagement) | **EngagementScorer** → Predict content performance |
| **Selector** → Pick top K posts | **ContentSelector** → Pick best drafts |
| **SideEffect** → Cache/log results | **PostProcessor** → Schedule/publish |
| Phoenix ML Model → Rank content | LLM (GPT-4/Claude) → Generate content |
| WeightedScorer → Combine action predictions | EngagementPredictor → Predict multi-action engagement |

## Components

### Strategy Engine (`agent/strategy/`)

| Module | Purpose |
|--------|---------|
| `TrendAnalyzer` | Monitors X trending topics, niche keywords, and news for content opportunities |
| `TopicSelector` | Scores and selects the best topics using weighted multi-signal scoring |
| `ContentCalendar` | Manages posting schedule with content type and tone diversity |
| `CompetitorAnalyzer` | Analyzes competitor accounts for content gaps and format insights |

### Content Generation (`agent/content/`)

| Module | Purpose |
|--------|---------|
| `LLMContentGenerator` | Generates content variants using LLMs with structured prompts |
| `ContentOptimizer` | Optimizes drafts for hooks, readability, hashtags, character limits |
| `TemplateEngine` | Pre-built templates for proven high-performing post formats |
| `ThreadComposer` | Specialized thread generation with proper narrative structure |
| `MediaSuggester` | Analyzes content and suggests appropriate media attachments |

### Audience Analysis (`agent/audience/`)

| Module | Purpose |
|--------|---------|
| `EngagementPredictor` | Multi-action engagement prediction (P(like), P(reply), P(repost), ...) |
| `FollowerAnalyzer` | Follower demographics, interests, and segmentation |
| `PersonaBuilder` | Creates audience personas for targeted content creation |
| `TimingOptimizer` | Optimizes posting times using a 7×24 time-slot scoring matrix |

### Analytics (`agent/analytics/`)

| Module | Purpose |
|--------|---------|
| `MetricsTracker` | Tracks all post performance with aggregation by type, tone, time |
| `ABTester` | A/B testing framework with statistical significance analysis |
| `FeedbackLoop` | Closes the loop: analyzes performance and adjusts strategy |
| `ReportGenerator` | Daily/weekly/monthly performance reports |

### Orchestrator (`agent/orchestrator/`)

| Module | Purpose |
|--------|---------|
| `XContentAgent` | Main agent class - coordinates all subsystems |
| `ContentWorkflow` | Defines end-to-end workflows (full pipeline, quick generate, trend react) |
| `AgentScheduler` | Periodic task execution (trend monitoring, content generation, metrics) |
| `StateManager` | Draft queue management, persistence, publication tracking |

## Pipeline Flow

```
┌─ CONTENT GENERATION PIPELINE ──────────────────────────────────────────────────┐
│                                                                                 │
│  1. REQUEST HYDRATION                                                          │
│     Enrich request with audience data, engagement history                      │
│                                                                                 │
│  2. CONTENT GENERATION (LLM)                                                  │
│     Generate N content variants using structured prompts                       │
│     ┌─────────────────────────────────────────────────────────┐               │
│     │  System Prompt (X content expertise)                     │               │
│     │  + Topic, Tone, Goal, Audience Context, Constraints     │               │
│     │  → Multiple variant drafts with different angles        │               │
│     └─────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  3. DRAFT HYDRATION                                                            │
│     ├─ ContentOptimizer → Hook analysis, readability, formatting              │
│     └─ MediaSuggester → Appropriate image/gif/video suggestions              │
│                                                                                 │
│  4. CONTENT FILTERING                                                          │
│     Remove drafts that fail quality, safety, or dedup checks                  │
│                                                                                 │
│  5. ENGAGEMENT SCORING                                                         │
│     ┌─────────────────────────────────────────────────────────┐               │
│     │  Feature Extraction:                                     │               │
│     │  has_media, has_question, has_numbers, is_thread,       │               │
│     │  is_timely, char_length_optimal, has_hook, has_cta      │               │
│     │                                                          │               │
│     │  Multi-Action Predictions:                               │               │
│     │  P(like), P(reply), P(repost), P(quote), P(bookmark),  │               │
│     │  P(click), P(follow), P(mute), P(unfollow), P(report)  │               │
│     │                                                          │               │
│     │  Weighted Score = Σ (weight_i × P(action_i))            │               │
│     │  (Same formula as home-mixer's WeightedScorer)          │               │
│     └─────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  6. SELECTION                                                                  │
│     Sort by weighted score, select top K drafts                               │
│                                                                                 │
│  7. POST-PROCESSING                                                            │
│     Schedule into content calendar at optimal times                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
import asyncio
from agent.config import AgentConfig, LLMConfig
from agent.orchestrator.agent import XContentAgent
from agent.types import ContentType, ContentTone

# Configure the agent
config = AgentConfig(
    account_handle="youraccount",
    niche="AI/ML",
    niche_keywords=["machine learning", "AI", "deep learning", "LLM"],
    brand_description="We share practical AI insights for developers",
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-...",
    ),
)

# Initialize the agent
agent = XContentAgent(config)

async def main():
    # Generate content for a specific topic
    drafts = await agent.generate_content(
        topic="The future of AI agents",
        content_type=ContentType.SHORT_POST,
        tone=ContentTone.PROFESSIONAL,
        num_variants=3,
    )

    # Generate a thread
    thread = await agent.generate_thread(
        topic="Building AI Agents",
        key_points=[
            "Define clear objectives for your agent",
            "Use structured prompts for consistency",
            "Implement feedback loops for improvement",
            "Monitor and measure everything",
        ],
    )

    # Quick post from template
    post = await agent.quick_post(
        template_id="hot_take",
        variables={
            "opinion": "AI agents will replace 90% of social media managers",
            "supporting_point": "They can generate, optimize, and schedule 24/7",
        },
    )

    # Run full automated pipeline
    await agent.run_full_pipeline()

    # Check status
    print(agent.get_status())

    # Get performance reports
    print(agent.get_weekly_report())

asyncio.run(main())
```

## Content Templates

Pre-built templates for high-performing post formats:

| Template | Type | Goal | Example |
|----------|------|------|---------|
| `hot_take` | Short Post | Engagement | "Unpopular opinion: {opinion}..." |
| `this_or_that` | Short Post | Engagement | "{A} or {B}? Drop your answer below" |
| `quick_tip` | Short Post | Authority | "💡 {topic} tip: {tip}" |
| `mistake_list` | Short Post | Engagement | "N mistakes costing you {cost}" |
| `how_to_thread` | Thread | Authority | "🧵 How to {goal} (step by step)" |
| `stat_hook` | Short Post | Reach | "{statistic} - Here's why this matters" |
| `before_after` | Short Post | Reach | "N months ago: {before}. Today: {after}" |
| `community_question` | Short Post | Community | "Question for {audience}: {question}" |
| `resource_share` | Short Post | Growth | "N {resources} that will {benefit}" |
| `news_take` | Short Post | Reach | "{news} - What this means for {audience}" |

## Engagement Prediction

The engagement predictor uses feature-based scoring similar to Phoenix's multi-action model:

```
Content Features:
├── has_media         (weight: 0.15)  → Images/videos boost engagement
├── has_question      (weight: 0.12)  → Questions drive replies
├── has_numbers       (weight: 0.08)  → Stats/numbers catch attention
├── is_thread         (weight: 0.10)  → Threads get higher reach
├── is_timely         (weight: 0.12)  → Trending topics get algorithmic boost
├── char_length       (weight: 0.08)  → Optimal: 70-140 chars
├── has_hook          (weight: 0.15)  → Strong opening line
├── has_cta           (weight: 0.10)  → Call to action drives interaction
└── readability       (weight: 0.05)  → Simple language performs better

Engagement Weights (WeightedScorer):
├── like:           +1.0
├── reply:          +3.0
├── repost:         +5.0
├── quote:          +4.0
├── click:          +2.0
├── bookmark:       +2.5
├── follow:        +10.0
├── profile_visit:  +1.5
├── mute:          -20.0
├── unfollow:      -30.0
└── report:        -50.0
```

## Scheduled Tasks

The agent scheduler runs these periodic tasks:

| Task | Frequency | Purpose |
|------|-----------|---------|
| Trend Monitor | Every 15 min | Check for trending topics |
| Content Generator | Every 4 hours | Fill calendar gaps with content |
| Metrics Collector | Hourly | Fetch post performance data |
| Content Publisher | Hourly | Publish scheduled posts |
| Feedback Analysis | Weekly | Analyze performance and adjust strategy |
| Daily Report | Daily | Generate performance summary |

## Feedback Loop

The agent continuously improves through a feedback loop:

```
Generate Content → Publish → Collect Metrics → Analyze Performance
      ↑                                              │
      │                                              ▼
      └──── Adjust Strategy ← Generate Insights ────┘

Adjustments include:
- Content type mix ratios (more threads if threads perform better)
- Tone preferences (shift toward tones that resonate)
- Topic selection weights (prioritize topics that get engagement)
- Posting time optimization (learn when audience is most active)
- Template effectiveness (promote templates with high engagement)
```

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
