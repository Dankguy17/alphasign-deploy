"use client";

import { useCallback } from "react";
import { AgentActivity } from "@/components/agent-activity";
import { AgentGraph } from "@/components/agent-graph";
import { MarketSnapshot } from "@/components/market-snapshot";
import { NewsPanel } from "@/components/news-panel";
import { RecommendationPanel } from "@/components/recommendation-panel";
import { ReportViewer } from "@/components/report-viewer";
import { SignalCharts } from "@/components/signal-charts";
import { TickerCommand } from "@/components/ticker-command";
import { useAnalysisSession } from "@/hooks/use-analysis-session";
import { useEventStream } from "@/hooks/use-event-stream";
import { ADAPTER_BASE_URL } from "@/lib/api";

export function AppShell() {
  const {
    appendEvent,
    beginSession,
    canStartSession,
    connectionLabel,
    error,
    events,
    fixtureMode,
    health,
    loadFixturePreview,
    market,
    refreshHealth,
    report,
    session,
    startDisabledReason,
  } = useAnalysisSession();

  const onEvent = useCallback((event: Parameters<typeof appendEvent>[0]) => {
    appendEvent(event);
  }, [appendEvent]);
  const stream = useEventStream(
    fixtureMode ? null : (session?.session_id ?? null),
    onEvent,
  );

  const isConnected = health?.status === "ok";
  const bandConfigured = health?.band_configured === true;

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div>
            <h1 className="text-xl font-bold tracking-normal text-slate-950">AlphaSign</h1>
            <p className="text-sm text-slate-500">
              Financial intelligence dashboard for Band agent analysis.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill label="Backend" value={connectionLabel} active={isConnected} />
            <StatusPill
              label="Band"
              value={bandConfigured ? "Configured" : "Not confirmed"}
              active={bandConfigured}
            />
            {fixtureMode ? (
              <span className="rounded-md bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-800">
                Fixture preview
              </span>
            ) : null}
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_24rem] lg:px-8">
        <div className="space-y-4">
          <TickerCommand
            disabled={!canStartSession}
            disabledReason={startDisabledReason}
            onSubmit={beginSession}
          />
          {error || health?.message ? (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <div className="font-semibold">Adapter status</div>
              <p className="mt-1 leading-6">{error ?? health?.message}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => void refreshHealth()}
                  className="rounded-md border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900"
                >
                  Retry health
                </button>
                <button
                  type="button"
                  onClick={loadFixturePreview}
                  className="rounded-md bg-amber-700 px-3 py-1.5 text-xs font-semibold text-white"
                >
                  Load isolated fixture
                </button>
              </div>
            </div>
          ) : null}
          <MarketSnapshot market={market} />
          <div className="grid gap-4 xl:grid-cols-2">
            <SignalCharts market={market} report={report} />
            <AgentGraph events={events} session={session} />
          </div>
          <ReportViewer report={report} />
        </div>

        <aside className="space-y-4">
          <section className="rounded-lg border border-[var(--border)] bg-white p-4 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-950">Adapter target</h2>
            <p className="mt-2 break-all font-mono text-xs leading-5 text-slate-600">
              {ADAPTER_BASE_URL}
            </p>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-md bg-slate-50 p-3">
                <dt className="text-slate-500">Session</dt>
                <dd className="mt-1 font-mono font-semibold text-slate-900">
                  {session?.session_id ?? "none"}
                </dd>
              </div>
              <div className="rounded-md bg-slate-50 p-3">
                <dt className="text-slate-500">Ticker</dt>
                <dd className="mt-1 font-mono font-semibold text-slate-900">
                  {session?.ticker ?? "idle"}
                </dd>
              </div>
            </dl>
          </section>
          <RecommendationPanel report={report} />
          <NewsPanel report={report} />
          <AgentActivity
            events={events}
            session={session}
            streamState={stream.state}
          />
          {stream.error ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
              {stream.error}
            </div>
          ) : null}
        </aside>
      </div>
    </main>
  );
}

function StatusPill({
  label,
  value,
  active,
}: {
  label: string;
  value: string;
  active: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-700">
      <span className={`h-2 w-2 rounded-full ${active ? "bg-green-600" : "bg-slate-400"}`} />
      <span>{label}:</span>
      <span className="font-semibold">{value}</span>
    </span>
  );
}
