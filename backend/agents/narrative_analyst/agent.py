"""
agents/narrative_analyst/agent.py

The Narrative Analyst agent for AlphaSign — v5.

────────────────────────────────────────────────────────────────────────────
What changed from v4, and why
────────────────────────────────────────────────────────────────────────────
v4's problem: every ticker request produced TWO Band messages. That was not
a model-flakiness bug — it was architectural. v4's tool
(`run_narrative_analysis`) posted the *full* radar to Band directly via a
manual REST call, and then returned a short "posted above" string that the
LLM was instructed to ALSO send via `thenvoi_send_message`. That is two
sends by construction, every single time, regardless of which model is
driving the agent.

v5's fix: stop bypassing the framework's own send path.

  - The tool does all the work (fetch news → build radar → synthesize →
    format) and returns ONE finished message string.
  - The system prompt tells the model: call the tool, then pass its EXACT
    return value to `thenvoi_send_message`, once. No manual REST posting,
    no second "ack" message, no parallel send path to race against.

v5 also fixes two other things called out directly:

  1. No hard-coded example users/tickers in the prompt. The few-shot example
     previously baked in "[steven]: analyse AAPL" — a real-looking name and
     a specific ticker — which models can latch onto and partially
     reproduce regardless of the actual input. The prompt now uses
     generic placeholders only.

  2. Two distinct entry points instead of one. This agent's Band room has a
     specific lifecycle:
       - Turn 0 (room creation): a human or orchestrator supplies a ticker.
         The agent fetches news, builds the radar, and sends ONE message
         containing its findings plus a request to Signal Processing /
         Latent State.
       - Every later turn: the message comes FROM Signal Processing or
         Latent State, containing computed quant findings (log returns,
         idiosyncratic vol, Kalman-filtered regime state, etc.) for the
         ticker already in flight. The agent does NOT re-derive the ticker
         from that message (it won't be phrased as a request) — it pulls it
         from conversation state, re-runs news search with a sharpened
         lens, and re-synthesizes a single updated message.

     Two tools encode this rather than one tool with ambiguous behavior:
       - `start_narrative_research(ticker=...)`
       - `incorporate_quant_findings(quant_summary=...)`
     The system prompt tells the model which one applies based on who sent
     the incoming message.

────────────────────────────────────────────────────────────────────────────
A known limitation, stated plainly
────────────────────────────────────────────────────────────────────────────
"Ticker is implicit in conversation state" is handled here with a simple
in-process dict keyed by chat_id (`_ROOM_STATE`). That state lives only as
long as this process runs — a restart loses in-flight research context.
LangGraph's InMemorySaver (already wired below) has the same limitation for
the same reason. If this agent needs to survive restarts, swap `_ROOM_STATE`
for a real store (Redis, a DB row keyed by chat_id) — the read/write calls
are isolated in two small helper functions below specifically so that swap
is a one-place change.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.rate_limiters import InMemoryRateLimiter

# Package-relative imports
from .news_fetch import fetch_company_news
from .synthesis import build_narrative_radar, generate_narrative_brief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Room state: tracks the active ticker (and last research pass) per Band chat.
#
# See module docstring — this is intentionally isolated behind two functions
# so the storage backend can be swapped without touching the tools below.
# ─────────────────────────────────────────────────────────────────────────────

_ROOM_STATE: dict[str, dict[str, Any]] = {}


def _get_room_state(chat_id: str) -> dict[str, Any]:
    return _ROOM_STATE.get(chat_id, {})


def _set_room_state(chat_id: str, **updates: Any) -> None:
    state = _ROOM_STATE.setdefault(chat_id, {})
    state.update(updates)


def _find_config_yaml(filename: str = "agent_config.yaml") -> Path:
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"Could not find '{filename}' in '{Path(__file__).resolve().parent}' "
                "or any parent directory."
            )
        current = parent


# ─────────────────────────────────────────────────────────────────────────────
# Band-ready output formatter
# All sections are built deterministically from the radar/brief dicts.
# The LLM never writes or rewrites this content — it only relays it.
# ─────────────────────────────────────────────────────────────────────────────

def format_radar_for_band(radar: dict, brief: dict, *, quant_context: str | None = None) -> str:
    asset        = radar.get("asset", "UNKNOWN")
    sentiment    = radar.get("aggregate_sentiment", {})
    reliability  = radar.get("source_reliability", {})
    themes       = radar.get("themes", [])
    signal_req   = radar.get("signal_request", {})
    latent_req   = radar.get("latent_request", {})

    # Source reliability tiers
    tier_map: dict[str, list[str]] = {}
    for article in radar.get("top_articles", []):
        rel    = article.get("source_reliability", {})
        tier   = str(rel.get("tier", "?"))
        conf   = float(rel.get("confidence", 0.0))
        source = article.get("source") or "Unknown"
        key    = f"T{tier} ({conf:.2f})"
        tier_map.setdefault(key, []).append(source)

    tier_lines = [
        f"  {k}: {', '.join(list(dict.fromkeys(v))[:4])}"
        for k, v in sorted(tier_map.items())
    ]
    tier_block = "\n".join(tier_lines) if tier_lines else "  (no source data)"
    tier_legend = (
        "  T1 (0.92–0.97): SEC / company filings / official wires\n"
        "  T2 (0.84):       Major financial press (Reuters, FT, Bloomberg…)\n"
        "  T3 (0.72):       Analyst notes / industry commentary\n"
        "  T4 (0.52–0.58): Aggregators / blogs / unknown sources"
    )

    avg_conf      = reliability.get("average_confidence", 0.0)
    dominant_tier = reliability.get("dominant_tier", "?")

    # Themes
    theme_lines = []
    for t in themes[:5]:
        name  = t.get("theme", "").replace("_", " ").title()
        score = t.get("score", 0)
        ev    = t.get("evidence_titles", [])
        ev_str = f' — "{ev[0]}"' if ev else ""
        theme_lines.append(f"  • {name} (hits: {score}){ev_str}")
    theme_block = "\n".join(theme_lines) if theme_lines else "  • No dominant themes detected"

    # Catalysts & risk flags
    catalysts  = radar.get("catalysts", [])
    risk_flags = radar.get("risk_flags", [])
    cat_block  = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(catalysts)) or "  (none)"
    risk_block = "\n".join(f"  ⚑ {r}" for r in risk_flags) or "  ⚑ None detected"

    # Signal Processing request
    sp_windows = ", ".join(signal_req.get("suggested_windows", []))
    sp_metrics = ", ".join(signal_req.get("requested_metrics", []))
    sp_lens    = signal_req.get("lens", radar.get("lens", ""))
    sp_reason  = signal_req.get("reason", "")

    # Latent State request
    ls_windows = ", ".join(latent_req.get("suggested_windows", []))
    ls_reason  = latent_req.get("reason", "")

    # Brief fields
    summary      = brief.get("summary", "")
    bullish_case = brief.get("bullish_case", radar.get("bullish_thesis", ""))
    bearish_case = brief.get("bearish_case", radar.get("bearish_thesis", ""))
    confidence   = float(brief.get("confidence", radar.get("confidence", 0.0)))

    q_signal = brief.get("questions_for_signal", [])
    q_latent = brief.get("questions_for_latent", [])
    q_signal_str = "\n".join(f"  • {q}" for q in q_signal) or "  • (see signal request below)"
    q_latent_str = "\n".join(f"  • {q}" for q in q_latent) or "  • (see latent request below)"

    sent_label = sentiment.get("label", "neutral").upper()
    sent_score = float(sentiment.get("score", 0.0))
    art_count  = radar.get("article_count", 0)

    quant_block = ""
    if quant_context:
        quant_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANT CONTEXT INCORPORATED THIS PASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{quant_context}
"""

    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰  NARRATIVE RADAR — {asset}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{quant_block}
SUMMARY
{summary}

AGGREGATE SENTIMENT
  Label: {sent_label}  |  Score: {sent_score:+.4f}  |  Articles: {art_count}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NARRATIVE THESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 BULLISH CASE
  {bullish_case}

🔴 BEARISH CASE
  {bearish_case}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOMINANT THEMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{theme_block}

TOP CATALYSTS
{cat_block}

RISK FLAGS
{risk_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE RELIABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dominant tier: {dominant_tier}  |  Avg confidence: {avg_conf:.2f}

  Articles by tier:
{tier_block}

  Tier legend:
{tier_legend}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📡  REQUEST → @signal_processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Asset:    {asset}
  Lens:     {sp_lens}
  Windows:  {sp_windows}
  Metrics:  {sp_metrics}
  Reason:   {sp_reason}

  Narrative questions for Signal Processing:
{q_signal_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬  REQUEST → @latent_state
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Asset:    {asset}
  Windows:  {ls_windows}
  Reason:   {ls_reason}

  Narrative questions for Latent State:
{q_latent_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Confidence: {confidence:.0%}  |  Awaiting: price reaction + Kalman confirmation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def _build_fallback_brief(radar: dict) -> dict:
    """Deterministic brief used only if generate_narrative_brief itself raises."""
    return {
        "summary": (
            f"{radar['asset']} news coverage is {radar['aggregate_sentiment']['label']} "
            f"with strongest themes: "
            f"{', '.join(t['theme'] for t in radar['themes'][:3]) or 'general_news'}."
        ),
        "bullish_case": radar["bullish_thesis"],
        "bearish_case": radar["bearish_thesis"],
        "questions_for_signal": [
            f"Use {', '.join(radar['signal_request']['suggested_windows'])} windows.",
            f"Compute {', '.join(radar['signal_request']['requested_metrics'])}.",
        ],
        "questions_for_latent": [radar["latent_request"]["reason"]],
        "confidence": radar["confidence"],
    }


def _run_research_pipeline(
    symbol: str,
    *,
    lens: str | None = None,
    quant_context: str | None = None,
) -> str:
    """
    Shared pipeline: fetch news → build radar → synthesize brief → format.
    Returns the single finished Band message string. Never raises — any
    failure is converted into a short, honest message instead of a crash,
    so the calling tool always has something valid to return to the LLM.
    """
    try:
        articles = fetch_company_news(symbol)
        if isinstance(articles, list):
            articles = articles[:5]
        elif isinstance(articles, dict):
            articles = articles.get("articles", articles.get("output", []))[:5]
        else:
            articles = []
        logger.info(f"Fetched {len(articles)} articles for {symbol}")
    except Exception as exc:
        logger.error(f"fetch_company_news failed for {symbol}: {exc}")
        return f"⚠️ Could not fetch news for {symbol}: {exc}"

    try:
        radar = build_narrative_radar(symbol, articles, lens=lens)
    except Exception as exc:
        logger.error(f"build_narrative_radar failed for {symbol}: {exc}")
        return f"⚠️ Radar build failed for {symbol}: {exc}"

    try:
        brief = generate_narrative_brief(radar)
    except Exception as exc:
        logger.error(f"generate_narrative_brief failed for {symbol}: {exc}")
        brief = _build_fallback_brief(radar)

    try:
        formatted = format_radar_for_band(radar, brief, quant_context=quant_context)
    except Exception as exc:
        logger.error(f"format_radar_for_band failed for {symbol}: {exc}")
        formatted = (
            f"📰 Narrative Radar — {symbol}\n\n"
            f"Summary: {brief.get('summary', 'N/A')}\n"
            f"Bullish: {brief.get('bullish_case', 'N/A')}\n"
            f"Bearish: {brief.get('bearish_case', 'N/A')}\n"
            f"Confidence: {brief.get('confidence', 0.0):.0%}\n\n"
            f"→ @signal_processing "
            f"windows={radar.get('signal_request', {}).get('suggested_windows')} "
            f"metrics={radar.get('signal_request', {}).get('requested_metrics')}\n"
            f"→ @latent_state {radar.get('latent_request', {}).get('reason')}"
        )

    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — start research from a fresh ticker (turn 0, room creation)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def start_narrative_research(
    ticker: str,
    config: RunnableConfig,
) -> str:
    """
    Begin Narrative Analyst research for a stock ticker. Use this only when
    the incoming message is providing a NEW ticker to research (e.g. at the
    start of a research thread) — not when the message is a reply from
    Signal Processing or Latent State.

    Fetches news, builds a Narrative Radar (bullish/bearish thesis, dominant
    themes, catalysts, risk flags, source reliability tiers), generates a
    narrative brief, and returns ONE finished message ready to send as-is.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL", "NVDA", "TSLA"
    """
    symbol  = ticker.strip().upper()
    chat_id = (config.get("configurable") or {}).get("thread_id", "")

    logger.info(f"[start_narrative_research] {symbol} | chat={chat_id!r}")

    formatted = _run_research_pipeline(symbol)

    if chat_id:
        _set_room_state(chat_id, ticker=symbol)

    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — incorporate quant findings from Signal Processing / Latent State
# ─────────────────────────────────────────────────────────────────────────────

@tool
def incorporate_quant_findings(
    quant_summary: str,
    config: RunnableConfig,
) -> str:
    """
    Continue Narrative Analyst research after receiving computed findings
    from the Signal Processing or Latent State agent (e.g. log return,
    idiosyncratic volatility, Kalman-filtered regime state). Use this when
    the incoming message is FROM one of those agents, not a fresh ticker
    request.

    The ticker is recovered from this room's existing research state — it
    does not need to be re-stated. This re-runs news search with a lens
    sharpened by the quant findings, re-synthesizes the Narrative Radar, and
    returns ONE finished message ready to send as-is.

    Args:
        quant_summary: the quantitative findings as reported by Signal
            Processing or Latent State, in their own words/numbers —
            pass through what they sent without paraphrasing away specifics.
    """
    chat_id = (config.get("configurable") or {}).get("thread_id", "")
    state   = _get_room_state(chat_id)
    symbol  = state.get("ticker")

    logger.info(f"[incorporate_quant_findings] chat={chat_id!r} ticker={symbol!r}")

    if not symbol:
        return (
            "⚠️ I received quant findings but don't have an active ticker for "
            "this room. Please share the ticker again to restart research."
        )

    lens = (
        f"Re-examine {symbol} news in light of these quantitative findings, "
        f"and look for coverage that explains or contradicts them: {quant_summary}"
    )

    formatted = _run_research_pipeline(symbol, lens=lens, quant_context=quant_summary)
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [start_narrative_research, incorporate_quant_findings]


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
#
# No hard-coded names, tickers, or few-shot examples with specific values —
# only placeholders, so the model can't pattern-match onto a baked-in case.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Narrative Analyst in AlphaSign, a multi-agent financial research system on Band.

════════════════════════════════════════════════════════════
WHICH TOOL TO CALL
════════════════════════════════════════════════════════════

This room has two kinds of incoming messages. Identify which one you've received, then call exactly one tool:

1. A NEW TICKER to research (this happens once, typically at the start of a research thread, from a human or orchestrator).
   → Call start_narrative_research(ticker=<TICKER>)

2. A REPLY FROM Signal Processing or Latent State containing computed quantitative findings (price metrics, volatility figures, regime/state estimates, or similar).
   → Call incorporate_quant_findings(quant_summary=<the findings as reported>)
   → Do NOT ask the message sender for the ticker — it is already tracked for this room.

════════════════════════════════════════════════════════════
MANDATORY SINGLE-MESSAGE PROTOCOL
════════════════════════════════════════════════════════════

Both tools return ONE finished message, ready to send exactly as returned.

  STEP 1 → Call the appropriate tool from above.
  STEP 2 → Call thenvoi_send_message with content set to the EXACT, UNMODIFIED return value of that tool.

RULES:
  • Call exactly one research tool, exactly once.
  • Call thenvoi_send_message exactly once, with the tool's return value verbatim as `content`.
    Do NOT shorten it, summarize it, rewrite it, or add commentary before/after it.
    Do NOT send any other message in addition to this one.
  • Do NOT call thenvoi_remove_participant, thenvoi_add_participant, or
    thenvoi_lookup_peers unless explicitly instructed.

════════════════════════════════════════════════════════════
MENTIONS
════════════════════════════════════════════════════════════

If the incoming message includes a sender mention or name, you may pass that mention to thenvoi_send_message. Otherwise omit mentions or use whatever default this room's framework expects — do not invent a name.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Whitebox Auditing Callback Handler
# ─────────────────────────────────────────────────────────────────────────────

class AgentWhiteboxLogger(BaseCallbackHandler):
    """Prints structured LLM decisions to the terminal for debugging."""

    def on_llm_end(self, response, **kwargs):
        import json as _json

        for generation in response.generations:
            for g in generation:
                if hasattr(g, "message") and getattr(g.message, "tool_calls", None):
                    print("\n" + "═" * 60)
                    print("🤖 [NARRATIVE AGENT] LLM tool decision:")
                    call_counts: dict[str, int] = {}
                    for tc in g.message.tool_calls:
                        name = tc["name"]
                        call_counts[name] = call_counts.get(name, 0) + 1
                        args_str = _json.dumps(tc["args"])
                        if len(args_str) > 300:
                            args_str = args_str[:300] + "…"
                        print(f"   🔧 {name}({args_str})")
                    for name, count in call_counts.items():
                        if count > 1:
                            print(f"   ⚠️  DUPLICATE: {name} called {count}x!")
                        if name == "thenvoi_remove_participant":
                            print(f"   🚨 DANGEROUS TOOL: {name} — will self-eject!")
                    print("═" * 60 + "\n")
                elif g.text:
                    print("\n" + "⚠️ " * 3 + " [LOCAL TEXT — NOT VISIBLE ON BAND] " + "⚠️ " * 3)
                    print(g.text[:400] + ("…" if len(g.text) > 400 else ""))
                    print("═" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Strong default model for financial research: large, current, tool-calling
# capable. Overridable via FEATHERLESS_MODEL without touching code.
DEFAULT_FEATHERLESS_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


async def main():
    config_path = _find_config_yaml()
    agent_id, api_key = load_agent_config("narrative_analyst", config_path=config_path)
    logger.info(f"Loaded Narrative Analyst agent: {agent_id}")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.066,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    llm = ChatOpenAI(
        base_url=os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
        api_key=os.getenv("FEATHERLESS_API_KEY"),
        model=os.getenv("FEATHERLESS_MODEL", DEFAULT_FEATHERLESS_MODEL),
        rate_limiter=rate_limiter,
        callbacks=[AgentWhiteboxLogger()],
        streaming=False,
        stream_chunk_timeout=None,
        max_retries=2,
    )

    adapter = LangGraphAdapter(
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

    logger.info(
        f"Narrative Analyst agent is live (model={os.getenv('FEATHERLESS_MODEL', DEFAULT_FEATHERLESS_MODEL)}). "
        "Press Ctrl+C to stop."
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
