# AlphaSign

AlphaSign is a multi-agent financial intelligence system that turns a single ticker symbol into:

- live narrative research
- quantitative market analysis
- latent-state / regime detection
- a final executive PDF report
- a browser-based dashboard with streamed agent activity

The repo contains two coordinated applications:

- `backend/` runs the Band-connected agent system and the local adapter that feeds the UI.
- `frontend/` is a Next.js dashboard that proxies the backend adapter, streams live updates, and renders the report.

## What AlphaSign Does

At a high level, the system works like this:

1. A user enters a ticker in the frontend.
2. The frontend starts a session through the backend adapter.
3. The backend creates or reuses a Band room and routes the ticker to the Narrative Analyst agent.
4. The Narrative Analyst researches news, source quality, catalysts, and risk flags, then asks the other specialist agents for follow-up analysis.
5. The Signal Processing agent computes market statistics such as returns, volatility, beta, idiosyncratic volatility, and market-adjusted performance.
6. The Latent State agent estimates trend and regime behavior with Kalman-style prediction tools.
7. The Executive report is generated once the configured turn limit is reached.
8. The frontend streams the live transcript, normalized protocol cards, market snapshot, and the final PDF report.

## Repository Layout

```text
AlphaSign/
├── backend/                  # Python agent runtime, Band integration, adapter, report generation
├── frontend/                 # Next.js UI and proxy route to the backend adapter
├── output/pdf/               # Generated PDF artifacts
└── README.md
```

## Backend Overview

The backend is a Python 3.11 project defined in [`backend/pyproject.toml`](backend/pyproject.toml). It contains four main pieces:

- `main.py` orchestrates the session lifecycle and agent runtime.
- `adapter.py` exposes the local HTTP/SSE API used by the frontend.
- `start_agent.py` handles Band room creation, participant management, and ticker kickoff.
- `agents/` contains the specialist agents and the executive report logic.

### Agent roles

- Narrative Analyst
  - Researches company/news context.
  - Builds a narrative radar with bullish and bearish theses.
  - Produces source reliability tiers, catalysts, risks, and follow-up requests.

- Signal Processing
  - Computes ticker-level market statistics.
  - Chooses time windows appropriate to the narrative lens.
  - Produces quantitative findings and a signal opinion.

- Latent State
  - Estimates hidden trend / regime behavior.
  - Produces Kalman-style latent level, slope, prediction, and regime-shift indicators.

- Executive
  - Synthesizes all agent output.
  - Renders a structured PDF report.
  - Produces a direct strategy recommendation with supporting evidence.

## Frontend Overview

The frontend is a Next.js app in [`frontend/`](frontend/). It provides:

- a ticker entry flow
- a live Band transcript viewer
- an agent workflow graph
- per-agent lanes and filtering
- live market snapshot cards
- interactive stock charts with technical overlays
- news / narrative panels
- parsed executive report sections
- raw report toggling
- PDF download support

The main dashboard is built from the components in `frontend/src/components/` and the data hooks in `frontend/src/hooks/`.

### UI features

- Live streaming transcript over Server-Sent Events.
- Agent filtering by Narrative Analyst, Signal Processing, and Latent State.
- Collapsed/expanded raw transcript for long messages.
- Protocol card rendering for normalized agent output.
- Market snapshot with price, change, market cap, volume, and 52-week range.
- Interactive chart modes:
  - line/area
  - candlesticks
  - volume overlay
  - SMA 20
  - EMA 20
  - Bollinger Bands
- Executive report viewer with parsed sections and raw fallback.
- Final report download link once the PDF is ready.

## Data Flow

### Band collaboration loop

The Band room is used as the collaboration layer between agents, not as the LLM provider. The specialist agents exchange structured research through messages and `@mentions`.

The default loop is:

`Narrative Analyst -> Signal Processing -> Latent State -> Narrative Analyst`

That loop continues until the configured turn limit is reached.

### Adapter API

The backend adapter serves the UI and maintains the live message bus. The frontend proxies requests to it through `frontend/src/app/api/alphasign/[...path]/route.ts`.

Relevant endpoints:

- `GET /stream`
  - SSE stream of live agent messages and report-ready events.
- `GET /messages`
  - Full message history for replay and initial page load.
- `GET /report`
  - Downloads the generated PDF report when available.
- `POST /reset`
  - Clears the current session history.
- `GET /config`
  - Returns the active turn limit.
- `POST /config`
  - Updates the active turn limit, capped at 3.
- `POST /api/sessions`
  - Starts a new analysis session from the UI.
- `POST /api/rooms`
  - Creates a Band room and populates participants.
- `POST /api/rooms/close`
  - Removes runtime agents from a Band room.
- `GET /api/market/{ticker}`
  - Returns live Yahoo Finance market snapshot data.

## Report Generation

The Executive agent produces a structured report and renders it to PDF. The report includes:

- strategy recommendation
- executive summary
- asset overview
- key quantitative findings
- narrative signals
- regime and trend assessment
- hypothesis verdict
- risk factors
- recommended next steps

The PDF is written to `alphasign_report.pdf` by default, with the output path configurable through environment variables.

The frontend also parses the raw report into sections for easier reading, while preserving the raw text for auditability.

## External Services

AlphaSign depends on a few external services:

- Band / Thenvoi
  - Used for room creation, agent authentication, and message routing.
- Featherless
  - Open-source model inference for some agent reasoning paths.
- AI/ML API
  - Additional OpenAI-compatible hosted models.
- Groq
  - Used for some LLM calls, including report synthesis and protocol normalization when configured.
- Yahoo Finance
  - Used for live market snapshots and chart history.
- FRED
  - Used by the Signal Processing agent for macro series when relevant.
- News sources
  - Used by the Narrative Analyst for research and source aggregation.

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- `uv` for the backend Python environment
- Band account and remote agent credentials
- API keys for the providers you plan to use

### Backend setup

1. Change into the backend directory:

   ```bash
   cd backend
   ```

2. Create the environment file:

   ```bash
   cp .env.example .env
   ```

3. Fill in the required API keys and provider settings in `.env`.

4. Create and populate `agent_config.yaml` from the example:

   ```bash
   cp agent_config.yaml.example agent_config.yaml
   ```

5. Install dependencies:

   ```bash
   uv sync
   ```

6. Run the connection test before agent development:

   ```bash
   uv run python scripts/test_connections.py
   ```

### Frontend setup

1. Change into the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the dev server:

   ```bash
   npm run dev
   ```

## Running the System

### Recommended local workflow

1. Start the backend runtime from `backend/`.
2. Start the frontend from `frontend/`.
3. Open `http://localhost:3000`.
4. Enter a ticker symbol.
5. Watch the narrative, quant, and latent-state loop progress in the UI.
6. Download the final PDF report when it becomes available.

### Backend runtime

The backend entrypoint starts the agent loop and adapter together. From `backend/`:

```bash
uv run python main.py
```

If you only want to exercise specific agents, the individual modules under `backend/agents/` can also be run directly.

## Environment Variables

The most important backend variables are defined in [`backend/.env.example`](backend/.env.example).

### Core runtime

- `THENVOI_WS_URL`
- `THENVOI_REST_URL`
- `ADAPTER_PORT`
- `ADAPTER_ALLOWED_ORIGIN`
- `ADAPTER_MAX_HISTORY`
- `MAX_TURNS_PER_SESSION`
- `CONVERSATION_LOG_PATH`
- `PDF_OUTPUT_PATH`

### Providers

- `FEATHERLESS_API_KEY`
- `FEATHERLESS_BASE_URL`
- `AIML_API_KEY`
- `AIML_BASE_URL`
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `GROQ_MODEL`
- `DEEPSEEK_MODEL`
- `LATENT_STATE_MODEL`
- `GEMINI_API_KEY`

### Market / research data

- `FRED_API_KEY`
- `NEWS_API_KEY`
- `SIGNAL_PROCESSING_PROVIDER`
- `SIGNAL_MAX_TOOL_ROUNDS`
- `SIGNAL_OPINION_PROVIDER`
- `NARRATIVE_LLM_PROVIDER`

### Frontend

- `ALPHASIGN_API_URL`
- `NEXT_PUBLIC_ALPHASIGN_API_URL`

## Testing And Verification

Useful validation commands in the repo:

- `backend/scripts/test_connections.py`
  - checks Featherless, AI/ML API, and Band authentication.
- `backend/tests/test_signal_tool_control.py`
  - validates signal-agent tool control behavior.
- `backend/tests/test_narrative_tool_repair.py`
  - validates narrative tool repair / inference behavior.
- `backend/scripts/test_narrative_agent_local.py`
  - local narrative-agent smoke test.
- `backend/scripts/verify_narrative_gui.py`
  - GUI-oriented narrative verification.
- `backend/scripts/verify_narrative_band_connection.py`
  - Band connection verification.

Frontend validation:

- `npm run lint`
- `npm run build`

## Configuration Notes

- The session turn limit is hard-capped at 3 in the backend.
- The backend normalizes live agent messages into protocol cards for stable UI rendering.
- The frontend can run against a local adapter at `http://localhost:8765` by default.
- If the adapter is unavailable, the UI surfaces a degraded/offline state and can fall back to fixtures in some flows.
- The report and protocol logs are persisted locally for replay and inspection.

## Design Notes

The UI is intentionally styled as a dark, high-contrast analytical dashboard rather than a generic admin panel. It uses:

- layered surfaces and borders
- live status indicators
- motion for workflow activity
- interactive market charts
- distinct treatment for the transcript, report, and snapshot panels

## Important Caveats

- AlphaSign is an analysis tool, not financial advice.
- The quality of output depends on external APIs, live market data, and current Band credentials.
- Band agent IDs and API keys must be real, correctly configured values.
- The executive report can only be generated once the configured session turn limit is reached.

## Related Documentation

- [`backend/agents/narrative_analyst/README.md`](backend/agents/narrative_analyst/README.md)
- [`frontend/README.md`](frontend/README.md)
- [`backend/pyproject.toml`](backend/pyproject.toml)
- [`backend/.env.example`](backend/.env.example)

