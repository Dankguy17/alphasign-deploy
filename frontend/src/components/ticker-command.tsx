"use client";

import { FormEvent, useMemo, useState } from "react";

const defaultTickers = [
  "AAPL",
  "MSFT",
  "NVDA",
  "TSLA",
  "AMZN",
  "GOOGL",
  "META",
  "AMD",
  "NFLX",
  "SPY",
];

type TickerCommandProps = {
  disabled?: boolean;
  disabledReason?: string | null;
  onSubmit: (ticker: string) => void;
};

export function TickerCommand({
  disabled,
  disabledReason,
  onSubmit,
}: TickerCommandProps) {
  const [ticker, setTicker] = useState("NVDA");
  const [recent, setRecent] = useState<string[]>(() => {
    if (typeof window === "undefined") return [];
    const saved = window.localStorage.getItem("alphasign.recentTickers");
    return saved ? (JSON.parse(saved) as string[]) : [];
  });

  const suggestions = useMemo(() => {
    const query = ticker.trim().toUpperCase();
    return [...new Set([...recent, ...defaultTickers])]
      .filter((item) => !query || item.includes(query))
      .slice(0, 8);
  }, [recent, ticker]);

  function submit(nextTicker = ticker) {
    if (disabled) return;
    const normalized = nextTicker.trim().toUpperCase();
    if (!normalized) return;
    const nextRecent = [normalized, ...recent.filter((item) => item !== normalized)].slice(
      0,
      6,
    );
    setRecent(nextRecent);
    window.localStorage.setItem(
      "alphasign.recentTickers",
      JSON.stringify(nextRecent),
    );
    setTicker(normalized);
    onSubmit(normalized);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  return (
    <section className="rounded-lg border border-[var(--border)] bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-950">Start analysis</h2>
          <p className="text-xs text-slate-500">
            Submit a ticker to the future backend adapter.
          </p>
        </div>
      </div>
      <form className="flex flex-col gap-3 sm:flex-row" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="ticker">
          Ticker
        </label>
        <input
          id="ticker"
          value={ticker}
          maxLength={10}
          onChange={(event) => setTicker(event.target.value.toUpperCase())}
          placeholder="AAPL"
          className="h-11 min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 font-mono text-base font-semibold tracking-normal text-slate-950 shadow-inner"
        />
        <button
          type="submit"
          disabled={disabled}
          aria-describedby={disabledReason ? "ticker-submit-status" : undefined}
          className="h-11 rounded-md bg-[var(--alpha-700)] px-4 text-sm font-semibold text-white transition hover:bg-[var(--alpha-800)] disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {disabled ? "Start unavailable" : "Run Band analysis"}
        </button>
      </form>
      {disabledReason ? (
        <p id="ticker-submit-status" className="mt-2 text-xs text-slate-500">
          {disabledReason}
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap gap-2" aria-label="Ticker suggestions">
        {suggestions.map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => submit(item)}
            disabled={disabled}
            className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 font-mono text-xs font-semibold text-slate-700 hover:border-[var(--alpha-300)] hover:bg-[var(--alpha-50)] disabled:opacity-60"
          >
            {item}
          </button>
        ))}
      </div>
    </section>
  );
}
