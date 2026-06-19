"use client";

import { FormEvent, useState } from "react";
import { AgentGraph } from "@/components/agent-graph";
import { AgentLanes } from "@/components/agent-lanes";
import { MessageStream } from "@/components/message-stream";
import { ReportPanel } from "@/components/report-panel";
import { useAlphaSignStream, type StreamStatus } from "@/hooks/use-alphasign-stream";
import { ALPHASIGN_BASE_URL, AgentId, relativeTime } from "@/lib/alphasign";

export function AppShell() {
  const { messages, status, reportReady, reportTs, lastEventTs, error, reset, reload } =
    useAlphaSignStream();
  const [selected, setSelected] = useState<AgentId | "all">("all");
  const [resetting, setResetting] = useState(false);
  const [tickerInput, setTickerInput] = useState("");
  const [ticker, setTicker] = useState<string | null>(null);

  const activeAgent: AgentId | null =
    messages.length > 0 ? messages[messages.length - 1].agent : null;

  async function handleReset() {
    if (!window.confirm("Clear the current session history on the adapter?")) return;
    setResetting(true);
    try {
      await reset();
    } catch {
      /* surfaced via error banner */
    } finally {
      setResetting(false);
    }
  }

  function handleTickerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = tickerInput.trim().toUpperCase();
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(normalized)) return;
    setTicker(normalized);
    setTickerInput(normalized);
  }

  return (
    <main className="relative z-10 min-h-screen">
      <header className="sticky top-0 z-20 border-b border-[var(--hairline)] bg-[color-mix(in_srgb,var(--canvas)_82%,transparent)] backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-3.5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center gap-3">
            <BrandMark />
            <div>
              <h1 className="display text-[17px] leading-none">AlphaSign</h1>
              <p className="mt-1.5 text-[12px] leading-none text-[var(--ink-subtle)]">
                Band agent observation deck
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StreamPill status={status} lastEventTs={lastEventTs} />
            {reportReady ? (
              <span className="chip border-[var(--primary-line)] text-[var(--primary-hover)]">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--primary)]" />
                Report ready
              </span>
            ) : null}
            <button
              type="button"
              onClick={handleReset}
              disabled={resetting}
              className="btn-secondary px-3 py-1.5 text-xs"
            >
              {resetting ? "Resetting…" : "Reset session"}
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-7 sm:px-6 lg:grid-cols-[minmax(0,1fr)_24rem] lg:px-8">
        <div className="space-y-6">
          <section className="panel p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="eyebrow">Research target</p>
                <h2 className="panel-title mt-1.5">
                  {ticker ? `Tracking ${ticker}` : "Enter a stock ticker"}
                </h2>
                <p className="panel-sub mt-1.5">
                  Set the ticker you want to follow in this observation session.
                </p>
              </div>
              <form
                className="flex w-full gap-2 sm:w-auto"
                onSubmit={handleTickerSubmit}
              >
                <label className="sr-only" htmlFor="ticker">
                  Stock ticker
                </label>
                <input
                  id="ticker"
                  name="ticker"
                  value={tickerInput}
                  onChange={(event) => setTickerInput(event.target.value.toUpperCase())}
                  placeholder="AAPL"
                  maxLength={10}
                  autoComplete="off"
                  spellCheck={false}
                  pattern="[A-Za-z][A-Za-z0-9.-]{0,9}"
                  title="Enter a ticker such as AAPL or BRK.B"
                  className="h-10 min-w-0 flex-1 rounded-md border border-[var(--hairline-strong)] bg-[var(--surface-2)] px-3 font-mono text-sm font-medium uppercase text-[var(--ink)] placeholder:text-[var(--ink-tertiary)] focus:border-[var(--primary-focus)] sm:w-40"
                />
                <button type="submit" className="btn-primary h-10 px-4 text-sm">
                  Set ticker
                </button>
              </form>
            </div>
          </section>

          {error ? (
            <div className="panel overflow-hidden">
              <div className="border-l-2 border-[var(--warning)] p-4">
                <div className="eyebrow text-[var(--warning)]">Adapter unreachable</div>
                <p className="mt-1.5 text-[13px] leading-6 text-[var(--ink-muted)]">{error}</p>
                <button
                  type="button"
                  onClick={reload}
                  className="btn-primary mt-3.5 px-3 py-1.5 text-xs"
                >
                  Reconnect
                </button>
              </div>
            </div>
          ) : null}

          <AgentGraph
            messages={messages}
            reportReady={reportReady}
            activeAgent={activeAgent}
            selected={selected}
            onSelect={setSelected}
          />
          <MessageStream
            messages={messages}
            selected={selected}
            onSelect={setSelected}
            status={status}
          />
        </div>

        <aside className="space-y-6">
          <AgentLanes
            messages={messages}
            activeAgent={activeAgent}
            selected={selected}
            onSelect={setSelected}
          />
          <ReportPanel reportReady={reportReady} reportTs={reportTs} />
          <section className="panel p-5">
            <h2 className="panel-title">Connection</h2>
            <p className="mt-2.5 break-all font-mono text-xs leading-5 text-[var(--ink-subtle)]">
              {ALPHASIGN_BASE_URL}
            </p>
            <dl className="mt-4 grid grid-cols-2 gap-2.5 text-xs">
              <div className="inset p-3">
                <dt className="text-[var(--ink-subtle)]">Stream</dt>
                <dd className="mt-1.5 font-mono font-medium capitalize text-[var(--ink)]">
                  {status}
                </dd>
              </div>
              <div className="inset p-3">
                <dt className="text-[var(--ink-subtle)]">Last event</dt>
                <dd className="mt-1.5 font-mono font-medium text-[var(--ink)]">
                  {relativeTime(lastEventTs)}
                </dd>
              </div>
            </dl>
            <button
              type="button"
              onClick={reload}
              className="btn-secondary mt-3 w-full px-3 py-1.5 text-xs"
            >
              Reconnect stream
            </button>
          </section>
        </aside>
      </div>
    </main>
  );
}

function BrandMark() {
  return (
    <span
      aria-hidden
      className="flex h-9 w-9 items-center justify-center rounded-[8px] bg-[var(--primary)]"
      style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.25)" }}
    >
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <path
          d="M2 13.5 L6.5 7 L9.5 10 L16 2.5"
          stroke="#fff"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="16" cy="2.5" r="1.6" fill="#fff" />
      </svg>
    </span>
  );
}

function StreamPill({
  status,
  lastEventTs,
}: {
  status: StreamStatus;
  lastEventTs: string | null;
}) {
  const dot =
    status === "live"
      ? "live-dot bg-[var(--positive)]"
      : status === "connecting"
        ? "live-dot bg-[var(--warning)]"
        : "bg-[var(--negative)]";
  const label =
    status === "live" ? "Live" : status === "connecting" ? "Connecting" : "Offline";
  return (
    <span className="chip">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="font-medium text-[var(--ink)]">{label}</span>
      {status === "live" && lastEventTs ? (
        <span className="text-[var(--ink-subtle)]">· {relativeTime(lastEventTs)}</span>
      ) : null}
    </span>
  );
}
