"""
agents/signal_processing/agent.py

The Signal Processing agent for AlphaSign.

KEY DESIGN CHANGE vs. the original skeleton
--------------------------------------------
Rather than a single monolithic run_signal_analysis tool that always
fetches the same window and always computes every metric, this agent
exposes GRANULAR tools so the LLM can decide:

  • WHICH tickers to analyse (it may receive several from the
    Narrative Analyst, each needing different treatment)
  • WHICH time window to fetch (1M/3M/4M/6M/1Y/2Y) based on the
    narrative context — e.g. "earnings surprise last quarter" → 3M;
    "multi-year trend reversal" → 2Y
  • WHICH metrics are meaningful given that context — e.g. for a
    short-term momentum question it may only want log_return +
    volatility; for a macro-sensitivity question it needs beta +
    market_adjusted_return; for a "is this move idiosyncratic?"
    question it needs idiosyncratic_vol

The LLM's job (driven by SYSTEM_PROMPT below) is to:
  1. Parse the incoming request from the Narrative Analyst (tickers,
     window hints, lens/hypothesis).
  2. Call whatever combination of tools it decides is appropriate.
  3. Optionally call generate_opinion once it has the numbers it needs.
  4. Send a structured findings_packet back to the room — including
     a clear justification of WHY it chose those windows/metrics.

Individual tools:
  fetch_prices(ticker, window)         → raw close prices as JSON list
  compute_log_return(ticker, window)   → most-recent log return
  compute_volatility(ticker, window)   → annualised std-dev of log returns
  compute_beta_metrics(ticker, window) → beta, market-adj return, idio-vol
  compute_all_metrics(ticker, window)  → convenience: all of the above
  get_fred_series(series_id, window)   → macro series (optional context)
  generate_signal_opinion(findings_json, lens) → LLM opinion + confidence

Setup (in addition to backend/.env and agent_config.yaml):
  1. Add a 'signal_processing' block to agent_config.yaml:
       signal_processing:
         agent_id: "your-agent-id"
         api_key:  "your-api-key"
  2. Create the agent on app.band.ai/agents (Remote Agent type) and
     add it as a participant in your test chat room.
  3. Set GROQ_API_KEY in backend/.env. Optionally set
     SIGNAL_PROCESSING_MODEL or GROQ_MODEL to override the default Groq model.
  4. (Optional) set SIGNAL_OPINION_PROVIDER=groq|gemini|featherless|aimlapi
     in backend/.env — defaults to "groq".

Run:
    cd backend/
    python -m agents.signal_processing.agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.callbacks import BaseCallbackHandler

# Centralized workspace environment and configuration management
from shared.config import load_agent_credentials

# Package-relative imports for local agent submodules
from .calculations import compute_all, log_returns, rolling_volatility, beta_and_market_adjusted_return
from .data_fetch import fetch_price_series, fetch_market_series, fetch_fred_series
from .opinion import generate_opinion

logging.basicConfig(level=os.getenv("SIGNAL_LOG_LEVEL", "WARNING").upper())
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("SIGNAL_AGENT_LOG_LEVEL", "INFO").upper())

for noisy_logger in (
    "httpx",
    "thenvoi",
    "phoenix_channels_python_client",
    "langgraph",
    "langchain",
):
    logging.getLogger(noisy_logger).setLevel(os.getenv("SIGNAL_DEP_LOG_LEVEL", "WARNING").upper())

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
SIGNAL_DEBUG = os.getenv("SIGNAL_DEBUG", "").lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Signal Processing agent in AlphaSign, a \
multi-agent financial risk intelligence system that communicates through Band.

YOUR ROLE IN THE PIPELINE
──────────────────────────
You receive requests from the Narrative Analyst agent. Each request contains:
  • One or more ticker symbols to analyse
  • A "lens" — a hypothesis or narrative context the Narrative Analyst has
    developed from news research (e.g. "supply-chain disruption in Q1",
    "competitor hardware defect may boost relative demand", "possible
    multi-year trend reversal")
  • Optional hints about relevant time horizons

YOUR JOB: use the available tools to answer the Narrative Analyst's question
quantitatively. You decide:

1. WHICH TIME WINDOW to fetch for each ticker. Match the window to the lens:
   • Short-term event (earnings, product launch, shock): 1M or 3M
   • Medium-term trend / sector rotation: 4M or 6M
   • Macro or multi-year structural question: 1Y or 2Y
   Windows: "1M", "3M", "4M", "6M", "1Y", "2Y"

2. WHICH METRICS are meaningful for the lens:
   • Momentum / recent performance → log_return, volatility
   • Market sensitivity question → beta, market_adjusted_return
   • "Is this move idiosyncratic?" → idiosyncratic_vol
   • Full picture → compute_all_metrics (convenience wrapper)
   You don't have to compute everything every time. Pick what the lens calls for.

3. WHICH MACRO SERIES (if any) are relevant via get_fred_series:
   • Interest-rate sensitivity questions → "DGS10" (10Y Treasury yield)
   • Inflation context → "CPIAUCSL"
   • Credit stress → "BAMLH0A0HYM2" (HY spread)
   FRED data is optional — only fetch it if the lens makes it relevant.

4. AFTER you have the numbers, call generate_signal_opinion with the findings
   JSON and the original lens. This produces an interpretation
   you should include in your reply.

YOUR RESPONSE back to the Band room must contain:
  • A report title with the ticker(s), for example:
    "SIGNAL PROCESSING ANALYSIS — AAPL"
  • A one-paragraph opening that says what lens/hypothesis you tested.
  • "Time Windows Chosen": list each window with start/end dates and explain
    why that window is appropriate for the lens. If the request asks about a
    recent narrative shift, usually compare at least 1M and 3M.
  • "Metrics Chosen": name each metric and explain what it tests:
      - log_return: latest momentum / event reaction
      - volatility: realized uncertainty / abnormal movement benchmark
      - beta: market sensitivity
      - market_adjusted_return: relative performance after market exposure
      - idiosyncratic_vol: company-specific movement not explained by market
  • "Quantitative Findings": provide the actual computed numbers, grouped by
    ticker and window. Do not omit numbers. Use clear labels and 4-6 decimals
    for metrics.
  • "Addressing The Narrative Questions": explicitly answer the user's or
    Narrative Analyst's subquestions. If no subquestions were provided, create
    2-3 relevant questions from the lens and answer them using the numbers.
  • "Signal Opinion": include the generated opinion and confidence. Convert
    confidence to a percentage in the final report.
  • "Conclusion": give a direct assessment of whether the quantitative signal
    supports, weakly supports, or does not support the lens. Mention the key
    metric evidence driving that call.

DEPTH AND NUMBERS REQUIREMENTS
──────────────────────────────
Your final response should be analytical and substantive, not a terse summary.
Do not only say "the metrics are X." Explain what the values imply, compare
windows when multiple windows are used, and tie each interpretation back to
the narrative lens. Avoid inventing news or article dates that were not given.
If article-specific dates are requested but unavailable, say that limitation
clearly and use the available window metrics as supporting evidence.

Use this general format:

SIGNAL PROCESSING ANALYSIS — <TICKER>

I have completed the quantitative analysis for <ticker>, focusing on:
"<lens>"

Time Windows Chosen:
...

Metrics Chosen:
...

Quantitative Findings:
...

Addressing The Narrative Questions:
...

Signal Opinion:
...
Confidence: <percent>

Conclusion:
...

If a message doesn't contain enough information to identify a ticker or
hypothesis, ask for clarification rather than guessing.

DELIVERING YOUR RESPONSE TO THE ROOM
────────────────────────────────────
Do not send progress updates, status messages, or interim findings. Use tools
silently, then finish with one complete final answer. The runtime will deliver
that final answer to the Band room automatically.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Granular tools — the LLM picks which ones to call and with what args
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_prices(ticker: str, window: str = "6M") -> str:
    """
    Fetch the daily closing price series for a ticker over a given window.
    Returns a JSON object with 'ticker', 'window', 'start', 'end', and
    'prices' (list of {date, close} dicts).

    Use this if you need the raw price series for a custom calculation,
    or just want to inspect the data before deciding which metrics to run.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL", "NVDA", "^GSPC"
        window: one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    """
    df = fetch_price_series(ticker, window)
    prices = [
        {"date": str(idx.date()), "close": round(float(val), 4)}
        for idx, val in df["close"].items()
    ]
    return json.dumps({
        "ticker": ticker.upper(),
        "window": window,
        "start":  str(df.attrs["start"]),
        "end":    str(df.attrs["end"]),
        "prices": prices,
    })


@tool
def compute_log_return(ticker: str, window: str = "6M") -> str:
    """
    Compute the most-recent single-day log return for a ticker.
    R_t = ln(P_t / P_{t-1})

    Good for: quick momentum check, short-term event reaction.

    Args:
        ticker: stock ticker symbol
        window: one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    Returns JSON: {ticker, window, start, end, log_return}
    """
    df = fetch_price_series(ticker, window)
    ret = log_returns(df["close"])
    return json.dumps({
        "ticker":     ticker.upper(),
        "window":     window,
        "start":      str(df.attrs["start"]),
        "end":        str(df.attrs["end"]),
        "log_return": round(float(ret.iloc[-1]), 6),
    })


@tool
def compute_volatility(ticker: str, window: str = "6M") -> str:
    """
    Compute the rolling volatility (std dev of log returns) for a ticker
    over the given window.

    Good for: assessing risk level, confirming whether a price move is
    abnormal relative to the asset's typical behaviour.

    Args:
        ticker: stock ticker symbol
        window: one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    Returns JSON: {ticker, window, start, end, volatility, log_return}
    """
    df = fetch_price_series(ticker, window)
    ret = log_returns(df["close"])
    return json.dumps({
        "ticker":     ticker.upper(),
        "window":     window,
        "start":      str(df.attrs["start"]),
        "end":        str(df.attrs["end"]),
        "log_return": round(float(ret.iloc[-1]), 6),
        "volatility": round(rolling_volatility(ret), 6),
    })


@tool
def compute_beta_metrics(ticker: str, window: str = "6M") -> str:
    """
    Compute beta, market-adjusted return, and idiosyncratic volatility
    for a ticker vs. the S&P 500 (^GSPC) over the given window.

    beta                  — sensitivity to broad market moves
    market_adjusted_return — most-recent day's return MINUS beta * market return
                             (positive = outperformed market on a relative basis)
    idiosyncratic_vol     — volatility of the market-adjusted return series;
                             high values mean the stock moves for its OWN reasons

    Good for: "is this move market-driven or specific to this company?",
    macro-sensitivity questions, competitive-positioning lenses.

    Args:
        ticker: stock ticker symbol
        window: one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    Returns JSON: {ticker, window, start, end, beta, market_adjusted_return,
                   idiosyncratic_vol, r_value}
    """
    import pandas as pd
    asset_df  = fetch_price_series(ticker, window)
    market_df = fetch_market_series(window)

    asset_ret  = log_returns(asset_df["close"])
    market_ret = log_returns(market_df["close"])

    aligned = pd.DataFrame({"asset": asset_ret, "market": market_ret}).dropna()
    metrics = beta_and_market_adjusted_return(aligned["asset"], aligned["market"])

    return json.dumps({
        "ticker":                 ticker.upper(),
        "window":                 window,
        "start":                  str(asset_df.attrs["start"]),
        "end":                    str(asset_df.attrs["end"]),
        "beta":                   round(metrics["beta"], 4),
        "market_adjusted_return": round(metrics["market_adjusted_return"], 6),
        "idiosyncratic_vol":      round(metrics["idiosyncratic_vol"], 6),
        "r_value":                round(metrics["r_value"], 4),
    })


@tool
def compute_all_metrics(ticker: str, window: str = "6M") -> str:
    """
    Convenience wrapper: compute ALL Signal Processing metrics for a ticker
    in a single call (log_return, volatility, beta, market_adjusted_return,
    idiosyncratic_vol).

    Use this when the lens is broad or you're not sure which subset is most
    relevant. Use the individual tools when you know you only need one or
    two metrics, to keep the findings packet focused.

    Args:
        ticker: stock ticker symbol
        window: one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    Returns JSON: all metrics plus ticker, window, start, end.
    """
    logger.debug("Running formulas for %s over a %s window", ticker, window)

    asset_df  = fetch_price_series(ticker, window)
    market_df = fetch_market_series(window)
    metrics   = compute_all(asset_df["close"], market_df["close"])

    return json.dumps({
        "ticker":  ticker.upper(),
        "window":  window,
        "start":   str(asset_df.attrs["start"]),
        "end":     str(asset_df.attrs["end"]),
        **{k: round(v, 6) for k, v in metrics.items()},
    })


@tool
def get_fred_series(series_id: str, window: str = "6M") -> str:
    """
    Fetch a FRED macroeconomic data series as context for your analysis.
    Only call this if the Narrative Analyst's lens makes macro context
    genuinely relevant (e.g. rate-sensitivity, inflation exposure).

    Common series IDs:
        "DGS10"       — 10-Year Treasury yield (rate-sensitive stocks)
        "CPIAUCSL"    — Consumer Price Index (inflation context)
        "BAMLH0A0HYM2"— ICE BofA High Yield spread (credit stress)
        "UNRATE"      — Unemployment rate (macro environment)
        "DPCREDIT"    — Fed discount rate

    Args:
        series_id: FRED series ID string
        window:    one of "1M", "3M", "4M", "6M", "1Y", "2Y"
    Returns JSON: {series_id, window, start, end, data: [{date, value}]}
    """
    series = fetch_fred_series(series_id, window)
    data = [
        {"date": str(idx.date()), "value": round(float(val), 4)}
        for idx, val in series.items()
    ]
    return json.dumps({
        "series_id": series_id,
        "window":    window,
        "start":     str(series.index.min().date()),
        "end":       str(series.index.max().date()),
        "data":      data,
    })


@tool
def generate_signal_opinion(findings_json: str, lens: str = "") -> str:
    """
    Given a JSON string of computed findings and the original lens from
    the Narrative Analyst, produce a 2-3 sentence qualitative opinion
    and a confidence score (0.0–1.0).

    Call this AFTER you have computed the quantitative metrics. Pass the
    full findings dict as a JSON string (e.g. the direct output of one of
    the compute_* tools, or a manually assembled dict with at minimum
    'ticker'/'asset', 'window' dict with start/end/label, and whatever
    metrics you computed).

    Args:
        findings_json: JSON string. Must include 'asset' (or 'ticker') and
                       'window' (dict with 'start', 'end', 'label').
                       Include any metric keys that are present.
        lens: the hypothesis/context string from the Narrative Analyst.
    Returns JSON: {"opinion": str, "confidence": float}
    """
    try:
        findings = json.loads(findings_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in findings_json: {e}"})

    # Normalise 'ticker' -> 'asset' so opinion.py can find it.
    if "asset" not in findings and "ticker" in findings:
        findings["asset"] = findings["ticker"]

    # Normalise flat window keys -> nested window dict.
    if "window" not in findings or not isinstance(findings["window"], dict):
        findings["window"] = {
            "label": findings.get("window", "unknown"),
            "start": findings.get("start", "unknown"),
            "end":   findings.get("end",   "unknown"),
        }
    else:
        findings["window"] = {
            "label": findings["window"].get("label") or findings.get("window_label", "unknown"),
            "start": findings["window"].get("start") or findings.get("start", "unknown"),
            "end":   findings["window"].get("end")   or findings.get("end",   "unknown"),
        }

    result = generate_opinion(findings, lens=lens or None)
    return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Agent wiring
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    fetch_prices,
    compute_log_return,
    compute_volatility,
    compute_beta_metrics,
    compute_all_metrics,
    get_fred_series,
    generate_signal_opinion,
]

class AgentWhiteboxLogger(BaseCallbackHandler):
    """Intercepts LLM lifecycle events to print structured decisions directly to the console."""
    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for g in generation:
                # 1. Check if the LLM decided to execute local analysis tools.
                if hasattr(g, 'message') and getattr(g.message, 'tool_calls', None):
                    print("\n" + "═"*50)
                    print("🤖 [AGENT DECISION] -> LLM requesting tool execution:")
                    for tool_call in g.message.tool_calls:
                        print(f"   🔧 Tool: {tool_call['name']}({tool_call['args']})")
                    print("═"*50 + "\n")

                # 2. The LLM generated a plain-text final response with NO tool call.
                #    The adapter posts this once after the graph completes.
                elif g.text:
                    print("\n" + "📝"*3 + " [AGENT FINAL TEXT — WILL POST ONCE] " + "📝"*3)
                    print(g.text)
                    print("═"*80 + "\n")


class ReliableDeliveryLangGraphAdapter(LangGraphAdapter):
    """
    Run the analysis with local tools only, then publish exactly one final
    response to Band after the graph completes.
    """

    async def on_message(
        self,
        msg: Any,
        tools: Any,
        history: Any,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        logger.info("[HANDLE] Message %s in room %s", msg.id, room_id)

        # Pass no Thenvoi platform tools to the graph. The factory still adds
        # this agent's local analysis tools, and this adapter handles delivery.
        graph = self.graph_factory([]) if self.graph_factory else self._static_graph
        if not graph:
            raise RuntimeError("No graph available")

        messages: list[Any] = []
        if is_session_bootstrap:
            if self.graph_factory and room_id not in self._bootstrapped_rooms:
                messages.append(("system", self._system_prompt))
                self._bootstrapped_rooms.add(room_id)
            if history:
                messages.extend(history)

        if participants_msg:
            messages.append(("user", f"[System]: {participants_msg}"))

        if contacts_msg:
            messages.append(("user", f"[System]: {contacts_msg}"))

        messages.append(("user", msg.format_for_llm()))
        graph_input = {"messages": messages}

        final_text: str | None = None

        try:
            async for event in graph.astream_events(
                graph_input,
                config={
                    "configurable": {"thread_id": room_id},
                    "recursion_limit": self.recursion_limit,
                },
                version="v2",
            ):
                event_type = event.get("event")
                if event_type == "on_chat_model_end":
                    candidate = self._extract_plain_model_text(event)
                    if candidate:
                        final_text = candidate

            if not final_text:
                raise RuntimeError("Signal Processing produced no final response text.")

            mentions = self._reply_mentions(msg, tools)
            logger.info("Posting one final Signal Processing response to Band.")
            await tools.send_message(final_text, mentions)

            logger.info("[DONE] Message %s processed successfully", msg.id)

        except Exception as e:
            logger.error("Error processing message %s: %s", msg.id, e, exc_info=SIGNAL_DEBUG)
            try:
                await tools.send_event(content=f"Error: {e}", message_type="error")
            except Exception:
                pass
            raise

    @staticmethod
    def _extract_plain_model_text(event: dict[str, Any]) -> str | None:
        output = event.get("data", {}).get("output")
        if not output:
            return None

        text = ReliableDeliveryLangGraphAdapter._message_text(output)
        if text:
            return text

        for generation in getattr(output, "generations", []) or []:
            candidates = generation if isinstance(generation, list) else [generation]
            for candidate in candidates:
                message = getattr(candidate, "message", candidate)
                text = ReliableDeliveryLangGraphAdapter._message_text(message)
                if text:
                    return text

        return None

    @staticmethod
    def _reply_mentions(msg: Any, tools: Any) -> list[str]:
        participants = getattr(tools, "participants", []) or []
        sender_id = getattr(msg, "sender_id", None)

        for participant in participants:
            if participant.get("id") == sender_id:
                handle = participant.get("handle") or participant.get("name")
                return [handle] if handle else []

        for participant in participants:
            handle = participant.get("handle") or participant.get("name")
            if handle and "signal-processing" not in handle and "signal_processing" not in handle:
                return [handle]

        return []

    @staticmethod
    def _message_text(message: Any) -> str | None:
        if getattr(message, "tool_calls", None):
            return None

        content = getattr(message, "content", None)
        if not content:
            content = getattr(message, "text", None)

        if isinstance(content, str):
            text = content.strip()
            return text or None

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "\n".join(parts).strip()
            return text or None

        return None


async def main():
    # Dynamically extract credentials via our shared config layer
    agent_id, api_key = load_agent_credentials("signal_processing")
    logger.info(f"Loaded agent: {agent_id}")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the Signal Processing agent.")

    model = os.getenv("SIGNAL_PROCESSING_MODEL") or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    llm = ChatOpenAI(
        model=model,
        api_key=groq_api_key,
        base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        temperature=0,
        callbacks=[AgentWhiteboxLogger()] if SIGNAL_DEBUG else None,
    )

    adapter = ReliableDeliveryLangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
        custom_section=SYSTEM_PROMPT,
        additional_tools=TOOLS,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )

    logger.info(f"Signal Processing agent is live on Groq model {model}. Press Ctrl+C to stop.")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
