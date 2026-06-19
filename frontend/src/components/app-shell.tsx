"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { AgentGraph } from "@/components/agent-graph";
import { AgentLanes } from "@/components/agent-lanes";
import { MessageStream } from "@/components/message-stream";
import { MarketSnapshot } from "@/components/market-snapshot";
import { ReportPanel } from "@/components/report-panel";
import AnimatedContent from "@/components/animated-content";
import DarkVeil from "@/components/dark-veil";
import { useAlphaSignStream, type StreamStatus } from "@/hooks/use-alphasign-stream";
import { AgentId, getAlphaSignBaseUrl, relativeTime } from "@/lib/alphasign";
import { clearAdapterUrl, setAdapterUrl } from "@/lib/adapter-url";
import type { MarketSnapshot as MarketSnapshotData } from "@/lib/types";

export function AppShell() {
  const { messages, cards, status, reportReady, reportTs, lastEventTs, error, reset, reload } =
    useAlphaSignStream();
  const [selected, setSelected] = useState<AgentId | "all">("all");
  const [resetting, setResetting] = useState(false);
  const [tickerInput, setTickerInput] = useState("");
  const [ticker, setTicker] = useState<string | null>(null);
  const [marketTicker, setMarketTicker] = useState<string | null>(null);
  const previewTimer = useRef<number | null>(null);
  const workflowHideTimer = useRef<number | null>(null);
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [maxTurns, setMaxTurns] = useState(3);
  const [savingTurns, setSavingTurns] = useState(false);
  const [turnsStatus, setTurnsStatus] = useState<string | null>(null);
  const [tickerStatus, setTickerStatus] = useState<string | null>(null);
  const [sendingTicker, setSendingTicker] = useState(false);
  const [roomStatus, setRoomStatus] = useState<string | null>(null);
  const [roomId, setRoomId] = useState<string | null>(null);
  const [closingRoom, setClosingRoom] = useState(false);
  const [roomClosed, setRoomClosed] = useState(false);
  const [market, setMarket] = useState<MarketSnapshotData | null>(null);
  const [marketError, setMarketError] = useState<string | null>(null);
  const [marketLoading, setMarketLoading] = useState(false);
  const [introPhase, setIntroPhase] = useState<"idle" | "leaving" | "skeleton" | "ready">("idle");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [adapterUrlInput, setAdapterUrlInput] = useState("");

  useEffect(() => {
    setAdapterUrlInput(getAlphaSignBaseUrl());
    fetch("/api/alphasign/config", { cache: "no-store" })
      .then((response) => response.json())
      .then((value: { max_turns?: number }) => {
        if (value.max_turns) setMaxTurns(value.max_turns);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!marketTicker) return;
    const requestedTicker = marketTicker;
    const controller = new AbortController();
    let active = true;

    async function loadMarket() {
      setMarketLoading(true);
      try {
        const response = await fetch(`/api/alphasign/api/market/${encodeURIComponent(requestedTicker)}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const result = (await response.json()) as MarketSnapshotData & { error?: string };
        if (!response.ok) throw new Error(result.error ?? "Could not load Yahoo Finance data.");
        if (active) {
          setMarket(result);
          setMarketError(null);
        }
      } catch (loadError) {
        if (active && !(loadError instanceof DOMException && loadError.name === "AbortError")) {
          setMarketError(loadError instanceof Error ? loadError.message : "Could not load market data.");
        }
      } finally {
        if (active) setMarketLoading(false);
      }
    }

    void loadMarket();
    const refresh = window.setInterval(loadMarket, 60_000);
    return () => {
      active = false;
      controller.abort();
      window.clearInterval(refresh);
    };
  }, [marketTicker]);

  useEffect(() => () => {
    if (previewTimer.current) window.clearTimeout(previewTimer.current);
    if (workflowHideTimer.current) window.clearTimeout(workflowHideTimer.current);
  }, []);

  function handleTickerInput(value: string) {
    const normalized = value.toUpperCase();
    setTickerInput(normalized);
    if (previewTimer.current) window.clearTimeout(previewTimer.current);
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(normalized.trim())) {
      setMarketTicker(null);
      setMarket(null);
      setMarketError(null);
      setMarketLoading(false);
      return;
    }
    setMarketLoading(true);
    setMarket(null);
    setMarketError(null);
    previewTimer.current = window.setTimeout(() => {
      setMarketTicker(normalized.trim());
    }, 450);
  }

  const activeAgent: AgentId | null =
    messages.length > 0 ? messages[messages.length - 1].agent : null;
  const effectiveRoomId = roomId ?? messages.at(-1)?.room_id ?? null;
  const analysisRunning = sendingTicker || Boolean(roomId && !reportReady && !roomClosed);
  const sessionMessageCount = roomId
    ? messages.filter((message) => message.room_id === roomId).length
    : 0;
  const analysisProgress = Math.min(
    96,
    Math.max(sendingTicker ? 6 : 12, Math.round((sessionMessageCount / (maxTurns * 3)) * 100)),
  );
  const showTranscript = showWorkflow || analysisRunning || ticker !== null || marketTicker !== null;

  useEffect(() => {
    if (workflowHideTimer.current) {
      window.clearTimeout(workflowHideTimer.current);
      workflowHideTimer.current = null;
    }

    if (analysisRunning) {
      return;
    }

    if (showWorkflow) {
      workflowHideTimer.current = window.setTimeout(() => {
        setShowWorkflow(false);
        workflowHideTimer.current = null;
      }, 5_000);
    }
  }, [analysisRunning, showWorkflow]);

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

  async function handleTickerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = tickerInput.trim().toUpperCase();
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(normalized)) return;
    setShowWorkflow(true);
    setSendingTicker(true);
    setTickerStatus(null);
    setRoomStatus(null);
    try {
      const response = await fetch("/api/alphasign/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: normalized }),
      });
      const result = (await response.json()) as { session_id?: string; error?: string };
      if (!response.ok) throw new Error(result.error ?? "Could not send ticker to Band.");
      setRoomId(result.session_id ?? null);
      setRoomClosed(false);
      setTicker(normalized);
      setMarketTicker(normalized);
      setTickerInput(normalized);
      setTickerStatus("New Band room created and sent to Narrative Analyst.");
    } catch (sendError) {
      setTickerStatus(sendError instanceof Error ? sendError.message : "Could not send ticker.");
    } finally {
      setSendingTicker(false);
    }
  }

  function handleTickerOpen(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = tickerInput.trim().toUpperCase();
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(normalized)) return;
    setTicker(normalized);
    setTickerInput(normalized);
    setMarketTicker(normalized);
    setTickerStatus(null);
    setIntroPhase("leaving");
    window.setTimeout(() => setIntroPhase("skeleton"), 450);
    window.setTimeout(() => setIntroPhase("ready"), 1250);
  }

  function saveAdapterUrl() {
    const normalized = adapterUrlInput.trim().replace(/\/$/, "");
    if (!normalized) return;
    setAdapterUrl(normalized);
    setSettingsOpen(false);
    reload();
  }

  function resetAdapterUrl() {
    clearAdapterUrl();
    setAdapterUrlInput(getAlphaSignBaseUrl());
    setSettingsOpen(false);
    reload();
  }

  async function handleCloseRoom() {
    if (!effectiveRoomId || !window.confirm("Close this Band room and remove its runtime agents?")) return;
    setClosingRoom(true);
    setRoomStatus(null);
    try {
      const response = await fetch("/api/alphasign/api/rooms/close", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_id: effectiveRoomId }),
      });
      const result = (await response.json()) as { error?: string };
      if (!response.ok) throw new Error(result.error ?? "Could not close Band room.");
      setRoomClosed(true);
      setRoomStatus("Band room closed.");
    } catch (roomError) {
      setRoomStatus(roomError instanceof Error ? roomError.message : "Could not close room.");
    } finally {
      setClosingRoom(false);
    }
  }

  async function handleTurnsSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSavingTurns(true);
    setTurnsStatus(null);
    try {
      const response = await fetch("/api/alphasign/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_turns: maxTurns }),
      });
      const result = (await response.json()) as { max_turns?: number; error?: string };
      if (!response.ok) throw new Error(result.error ?? "Could not save turn limit.");
      setMaxTurns(result.max_turns ?? maxTurns);
      setTurnsStatus(`Limited to ${result.max_turns ?? maxTurns} turns.`);
    } catch (saveError) {
      setTurnsStatus(saveError instanceof Error ? saveError.message : "Could not save turn limit.");
    } finally {
      setSavingTurns(false);
    }
  }

  return (
    <>
      {introPhase === "idle" || introPhase === "leaving" ? (
        <section className={`ticker-intro ${introPhase === "leaving" ? "ticker-intro--leaving" : ""}`}>
          <div className="absolute inset-0">
            <DarkVeil speed={1} />
          </div>
          <div className="ticker-intro-shade" />
          <div className="ticker-intro-content">
            <p className="ticker-intro-kicker">Multi-agent market intelligence</p>
            <h1 className="ticker-intro-title">
              <span className="ticker-intro-mark">
                <Image
                  src="/logo.png"
                  alt="A"
                  width={118}
                  height={118}
                  priority
                  className="brand-logo ticker-intro-logo"
                />
              </span>
              <span className="ticker-intro-wordmark">lphaSign</span>
            </h1>
            <p className="ticker-intro-copy">Enter a market ticker to begin the signal.</p>
            <form className="ticker-intro-form" onSubmit={handleTickerOpen}>
              <label className="sr-only" htmlFor="intro-ticker">Stock ticker</label>
              <input
                id="intro-ticker"
                value={tickerInput}
                onChange={(event) => handleTickerInput(event.target.value)}
                placeholder="Input ticker"
                maxLength={10}
                autoComplete="off"
                spellCheck={false}
                pattern="[A-Za-z][A-Za-z0-9.-]{0,9}"
                aria-describedby={tickerStatus ? "intro-ticker-status" : undefined}
              />
              <button type="submit" disabled={!tickerInput.trim()}>
                Open
              </button>
            </form>
            {tickerStatus ? <p id="intro-ticker-status" className="ticker-intro-status">{tickerStatus}</p> : null}
          </div>
        </section>
      ) : null}
      {introPhase === "skeleton" ? <DashboardSkeleton /> : null}
    <main className={`relative z-10 min-h-screen ${introPhase === "ready" ? "dashboard-enter" : introPhase !== "idle" && introPhase !== "leaving" ? "invisible" : "hidden"}`}>
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
            {reportReady && effectiveRoomId && !roomClosed ? (
              <button
                type="button"
                onClick={handleCloseRoom}
                disabled={closingRoom}
                className="btn-secondary px-3 py-1.5 text-xs"
              >
                {closingRoom ? "Closing room…" : "Close Band room"}
              </button>
            ) : null}
            <div className="relative">
              <button
                type="button"
                onClick={() => setSettingsOpen((open) => !open)}
                className="btn-secondary px-3 py-1.5 text-xs"
              >
                Settings
              </button>
              {settingsOpen ? (
                <div className="absolute right-0 top-full z-30 mt-2 w-[min(92vw,360px)] rounded-lg border border-[var(--hairline-strong)] bg-[var(--canvas)] p-3 shadow-[0_24px_80px_rgba(0,0,0,.28)]">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-[var(--ink-tertiary)]">
                    Backend URL
                  </div>
                  <input
                    value={adapterUrlInput}
                    onChange={(event) => setAdapterUrlInput(event.target.value)}
                    placeholder="http://localhost:8765"
                    className="mt-2 h-10 w-full rounded-md border border-[var(--hairline-strong)] bg-[var(--surface-2)] px-3 font-mono text-xs text-[var(--ink)] placeholder:text-[var(--ink-tertiary)]"
                  />
                  <p className="mt-2 text-[11px] leading-5 text-[var(--ink-subtle)]">
                    Stored in a browser cookie and used by the proxy and live adapter calls.
                  </p>
                  <div className="mt-3 flex items-center justify-between gap-2">
                    <button type="button" onClick={resetAdapterUrl} className="btn-secondary px-3 py-1.5 text-xs">
                      Reset
                    </button>
                    <div className="flex gap-2">
                      <button type="button" onClick={() => setSettingsOpen(false)} className="btn-secondary px-3 py-1.5 text-xs">
                        Cancel
                      </button>
                      <button type="button" onClick={saveAdapterUrl} className="btn-primary px-3 py-1.5 text-xs">
                        Save
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-7 sm:px-6 lg:grid-cols-[minmax(0,1fr)_24rem] lg:px-8">
        <div className="space-y-6">
          <section className="panel p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="eyebrow">AlphaSign</p>
                <h2 className="panel-title mt-1.5">
                  {ticker ? `Tracking ${ticker}` : marketTicker ? `Previewing ${marketTicker}` : "Enter a stock ticker"}
                </h2>
                <p className="panel-sub mt-1.5">
                  Review the market data, then run AlphaSign when you are ready for agent analysis.
                </p>
              </div>
              <div className="flex w-full flex-col gap-2 sm:w-auto">
                <form className="flex gap-2" onSubmit={handleTickerSubmit}>
                <label className="sr-only" htmlFor="ticker">
                  Stock ticker
                </label>
                <input
                  id="ticker"
                  name="ticker"
                  value={tickerInput}
                  onChange={(event) => handleTickerInput(event.target.value)}
                  placeholder="AAPL"
                  maxLength={10}
                  autoComplete="off"
                  spellCheck={false}
                  pattern="[A-Za-z][A-Za-z0-9.-]{0,9}"
                  title="Enter a ticker such as AAPL or BRK.B"
                  className="h-10 min-w-0 flex-1 rounded-md border border-[var(--hairline-strong)] bg-[var(--surface-2)] px-3 font-mono text-sm font-medium uppercase text-[var(--ink)] placeholder:text-[var(--ink-tertiary)] focus:border-[var(--primary-focus)] sm:w-40"
                />
                {analysisRunning ? (
                  <div className="analysis-progress" role="progressbar" aria-label="AlphaSign analysis in progress">
                    <span />
                  </div>
                ) : (
                  <button type="submit" className="btn-primary analyze-button h-10 px-5 text-sm">
                    Analyze
                  </button>
                )}
                </form>
                {tickerStatus ? (
                  <span className="text-xs text-[var(--ink-subtle)]">{tickerStatus}</span>
                ) : null}
                {roomStatus ? (
                  <span className="text-xs text-[var(--ink-subtle)]">{roomStatus}</span>
                ) : null}
                {analysisRunning ? (
                  <div
                    className="analysis-run-status"
                    role="progressbar"
                    aria-label="AlphaSign agent analysis in progress"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={analysisProgress}
                  >
                    <div className="analysis-run-copy">
                      <span>Running AlphaSign agents</span>
                      <span>{analysisProgress}%</span>
                    </div>
                    <div className="analysis-run-track">
                      <span style={{ width: `${analysisProgress}%` }} />
                    </div>
                  </div>
                ) : (
                <form className="flex items-center gap-2" onSubmit={handleTurnsSubmit}>
                <label htmlFor="max-turns" className="text-xs text-[var(--ink-subtle)]">
                  Turns
                </label>
                <input
                  id="max-turns"
                  type="number"
                  min={1}
                  max={3}
                  value={maxTurns}
                  onChange={(event) => setMaxTurns(Math.max(1, Math.min(3, Number(event.target.value))))}
                  className="h-9 w-16 rounded-md border border-[var(--hairline-strong)] bg-[var(--surface-2)] px-2 font-mono text-sm text-[var(--ink)]"
                />
                <button type="submit" disabled={savingTurns} className="btn-secondary h-9 px-3 text-xs">
                  {savingTurns ? "Saving…" : "Apply"}
                </button>
                <span className="text-xs text-[var(--ink-subtle)]">
                  {turnsStatus ?? "Maximum 3"}
                </span>
                </form>
                )}
              </div>
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

          {marketTicker || marketLoading ? (
            <MarketSnapshot
              market={market}
              loading={marketLoading}
              error={marketError}
              condensed={showWorkflow}
            />
          ) : null}

          {showWorkflow ? (
            <AgentGraph
              messages={messages}
              reportReady={reportReady}
              activeAgent={activeAgent ?? (analysisRunning ? "narrative_analyst" : null)}
              selected={selected}
              onSelect={setSelected}
            />
          ) : null}
          {showTranscript ? (
            <AnimatedContent
              distance={100}
              direction="vertical"
              reverse={false}
              duration={1.6}
              ease="power3.out"
              initialOpacity={0}
              animateOpacity
              scale={0.7}
              threshold={0.4}
              delay={0}
              animateLayout
            >
              <MessageStream
                messages={messages}
                cards={cards}
                selected={selected}
                onSelect={setSelected}
                status={status}
              />
            </AnimatedContent>
          ) : null}
        </div>

        <aside className="space-y-6">
          {reportReady ? (
            <div className="dashboard-fade-in dashboard-fade-in--late">
              <ReportPanel reportTs={reportTs} />
            </div>
          ) : null}
          {messages.length > 0 || reportReady || showTranscript ? (
            <AnimatedContent
              distance={100}
              direction="vertical"
              reverse={false}
              duration={1.6}
              ease="power3.out"
              initialOpacity={0}
              animateOpacity
              scale={0.7}
              threshold={0.4}
              delay={0}
              animateLayout
            >
              <AgentLanes
                messages={messages}
                activeAgent={activeAgent}
                selected={selected}
                onSelect={setSelected}
              />
            </AnimatedContent>
          ) : null}
          <section className="panel p-5">
            <h2 className="panel-title">Connection</h2>
            <p className="mt-2.5 break-all font-mono text-xs leading-5 text-[var(--ink-subtle)]">
              {getAlphaSignBaseUrl()}
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
      <button
        type="button"
        onClick={handleReset}
        disabled={resetting}
        className="btn-secondary fixed bottom-4 right-4 z-[70] px-3 py-1.5 text-xs opacity-40 transition-opacity hover:opacity-100 focus-visible:opacity-100 disabled:opacity-30"
      >
        {resetting ? "Resetting…" : "Reset session"}
      </button>
    </>
  );
}

function DashboardSkeleton() {
  return (
    <div className="dashboard-skeleton" aria-label="Loading AlphaSign dashboard" aria-busy="true">
      <div className="dashboard-skeleton-header"><span className="skeleton h-9 w-40 rounded-lg" /><span className="skeleton h-8 w-56 rounded-full" /></div>
      <div className="dashboard-skeleton-grid">
        <div className="space-y-6">
          <div className="skeleton h-36 rounded-[14px]" />
          <div className="skeleton h-72 rounded-[14px]" />
          <div className="skeleton h-96 rounded-[14px]" />
        </div>
        <div className="space-y-6">
          <div className="skeleton h-80 rounded-[14px]" />
          <div className="skeleton h-48 rounded-[14px]" />
        </div>
      </div>
    </div>
  );
}

function BrandMark() {
  return (
    <Image
      src="/logo.png"
      alt=""
      width={36}
      height={36}
      className="brand-logo h-9 w-9 object-contain"
    />
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
