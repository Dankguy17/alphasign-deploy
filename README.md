# AlphaSign — Project Skeleton

## Directory structure

```
alphasign/
├── .gitignore
├── README.md
├── backend/
│   ├── .env.example                # copy -> .env, fill in real keys
│   ├── agent_config.yaml.example   # copy -> agent_config.yaml, fill in Band agent creds
│   ├── pyproject.toml
│   ├── scripts/
│   │   └── test_connections.py     # <- run this first
│   ├── shared/
│   │   ├── schemas.py              # findings_packet / request_packet definitions (Pydantic)
│   │   ├── band_client.py          # thin wrapper around band-sdk for sending packets to a room
│   │   └── llm_client.py           # shared Featherless / AI-ML API client factory
│   └── agents/
│       ├── signal_processing/
│       │   └── agent.py
│       ├── narrative_analyst/
│       │   └── agent.py
│       ├── latent_state/
│       │   └── agent.py
│       └── executive/
│           └── agent.py
└── frontend/                       # Next.js app (separate setup, scaffold with `npx create-next-app`)
```

## The three API sources, and what each is for

**Band** (`app.band.ai`) — the multi-agent collaboration platform. Each agent
is created on the Band dashboard as a "Remote Agent," which gives you an
`agent_id` (UUID) and `api_key`. Agents connect via the `band-sdk` Python
package, join shared chat rooms, and communicate by @mentioning each other.
This is the layer that satisfies the hackathon's "agents collaborate through
Band" requirement — Band does NOT run the LLM itself; it's the
communication/coordination layer between agents that each run their own LLM
calls.

**Featherless** (`api.featherless.ai`) — serverless inference for
open-source models (Llama, Qwen, Mistral, etc.) via an OpenAI-compatible API.
This is likely where most of your agents' LLM calls go, since it's the
allotted-credits provider for the hackathon. Model names are typically
HuggingFace repo paths, e.g. `Qwen/Qwen2.5-7B-Instruct`.

**AI/ML API** (`api.aimlapi.com`) — a broader hosted-model catalog (GPT-4o,
Gemini, Claude, DeepSeek, and 300+ others), also OpenAI-compatible. Your $10
budget here is best treated as a *reserve* — e.g. for the Executive Agent's
final synthesis pass if you want a stronger model than what's available on
Featherless, or as a fallback if a Featherless model is rate-limited during
the demo. Both are drop-in replacements for the `openai` Python client; you
just swap `base_url` and `api_key`.

In short: **Band routes messages between agents. Featherless and AI/ML API
are where each agent's actual "thinking" (LLM calls) happens.** An agent's
code will typically do both — receive a message via Band, call Featherless
or AI/ML API to reason about it, then send a response back via Band.

## Setup order

1. `cd backend`
2. `cp .env.example .env` and fill in `FEATHERLESS_API_KEY`, `AIML_API_KEY`,
   `FRED_API_KEY`, `NEWS_API_KEY`.
3. Create 4 agents on [app.band.ai/agents](https://app.band.ai/agents)
   (Remote Agent type): `signal_processing`, `narrative_analyst`,
   `latent_state`, `executive`. Copy each `agent_id` + `api_key`.
4. `cp agent_config.yaml.example agent_config.yaml` and fill in those
   credentials.
5. Install dependencies: `uv init` (if not already) then
   `uv add openai python-dotenv pyyaml httpx numpy pandas scipy yfinance pykalman fastapi "uvicorn[standard]"`,
   plus the Band SDK with an adapter extra, e.g. `uv add "band-sdk[pydantic-ai]"`.
6. Run `uv run python scripts/test_connections.py` — this checks all three
   API sources independently and tells you exactly which credentials (if
   any) are wrong before you write any agent logic.
7. Once all three pass, create a chat room on Band, add all 4 agents as
   participants, and start building each agent's `agent.py` using the
   pattern from the Anthropic/Pydantic AI adapter tutorials
   (`docs.band.ai/integrations/sdks/tutorials/`).

## Note on Band's collaboration model

Band agents communicate as participants in shared chat rooms via
`@mention` routing — not via an arbitrary shared key-value store. The
`findings_packet` / `request_packet` JSON structures from the project
proposal are realized as **structured JSON embedded in chat messages**
sent to a shared room, with agents @mentioning the agent they're
addressing. `shared/schemas.py` defines these structures as Pydantic
models so every agent serializes/parses them the same way.
