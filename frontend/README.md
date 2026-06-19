# AlphaSign Frontend

This is the Next.js dashboard for AlphaSign. It connects to the local backend adapter, streams live Band agent activity, and renders the market snapshot, narrative cards, executive report, and PDF download.

## What This App Does

The frontend is responsible for:

- starting and monitoring an analysis session
- proxying API requests to the backend adapter
- streaming live agent messages over SSE
- rendering the Band collaboration loop
- showing live market data for the selected ticker
- parsing and displaying the final executive report
- providing a PDF download for the generated report

## Main Screens And Components

The page is assembled from the components under `src/components/`:

- `app-shell.tsx`
  - top-level dashboard shell and ticker entry flow
- `agent-graph.tsx`
  - visual Band workflow diagram
- `agent-lanes.tsx`
  - per-agent message lanes and filters
- `message-stream.tsx`
  - live transcript and protocol card viewer
- `market-snapshot.tsx`
  - live Yahoo Finance snapshot and chart area
- `advanced-stock-chart.tsx`
  - interactive chart with ranges and studies
- `news-panel.tsx`
  - report-provided news and narrative items
- `recommendation-panel.tsx`
  - executive recommendation summary
- `report-viewer.tsx`
  - parsed report sections and raw report toggle
- `report-panel.tsx`
  - final PDF download panel

## Data Flow

The frontend talks to the backend adapter through the Next.js route handler in:

`src/app/api/alphasign/[...path]/route.ts`

That proxy forwards requests to the adapter at:

`https://server.alphasign.trade`

unless `ALPHASIGN_API_URL` or `NEXT_PUBLIC_ALPHASIGN_API_URL` is set.

For a Cloudflare Tunnel deployment, set both variables to the public tunnel URL
instead of `localhost`, for example:

```text
ALPHASIGN_API_URL=https://alphasign-api.example.com
NEXT_PUBLIC_ALPHASIGN_API_URL=https://alphasign-api.example.com
```

That works on Vercel because the Next.js route handler proxies server-side
requests to the same public origin that the browser uses.

The client-side API helpers live in `src/lib/api.ts`, while the live stream and protocol parsing logic live in `src/lib/alphasign.ts`.

## Runtime Behavior

When you enter a ticker:

1. The app validates the symbol.
2. It starts a session through the backend adapter.
3. The adapter creates or reuses a Band room.
4. Agent messages stream into the transcript.
5. Market snapshot data loads from the backend market endpoint.
6. When the session finishes, the executive report becomes available for download.

## Features

- Live session status and reconnect handling
- SSE-based transcript replay and streaming
- Agent selection and filtering
- Raw transcript expansion for long messages
- Protocol card normalization and rendering
- Live market chart with multiple ranges
- Technical overlays:
  - SMA 20
  - EMA 20
  - Bollinger Bands
  - volume
- Report parsing into readable sections
- Raw report fallback when parsing is incomplete
- PDF download link once the backend report is ready

## Environment Variables

The frontend understands the following environment variables:

- `ALPHASIGN_API_URL`
  - backend adapter base URL for server-side and proxy requests
- `NEXT_PUBLIC_ALPHASIGN_API_URL`
  - client-side base URL for browser fetches and EventSource

If neither is set, the app defaults to `https://server.alphasign.trade`.
For production, both values should usually point at the same Cloudflare Tunnel
hostname or other public backend URL.

## Local Development

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

Recommended workflow:

1. Start the backend adapter and agent runtime first.
2. Start this frontend.
3. Enter a ticker symbol on the landing screen.
4. Watch the transcript and market panels update.
5. Download the final report when it appears.

## Available Scripts

Defined in `package.json`:

- `npm run dev`
  - start the dev server with webpack
- `npm run dev:turbo`
  - start the dev server with Turbopack
- `npm run build`
  - build the production bundle
- `npm run start`
  - start the production server
- `npm run lint`
  - run ESLint

## Notes

- The app is designed to run against the AlphaSign backend adapter, not as a standalone stock app.
- If the adapter is offline, the UI shows a disconnected or degraded state.
- The transcript can contain both raw agent messages and normalized protocol cards.
- The executive report is informational only and should not be treated as financial advice.
