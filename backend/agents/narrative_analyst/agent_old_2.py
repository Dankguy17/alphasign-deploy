"""
agents/narrative_analyst/agent.py

The Narrative Analyst agent for AlphaSign with Whitebox Auditing and Rate Limiting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import BaseCallbackHandler

# Package-relative imports for local agent submodules
from .article_extract import extract_article_text
from .news_fetch import fetch_company_news, fetch_yfinance_news, fetch_yahoo_rss_news
from .sentiment import score_text_sentiment
from .source_reliability import score_source_reliability
from .synthesis import build_narrative_radar, generate_narrative_brief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_config_yaml(filename: str = "agent_config.yaml") -> Path:
    """Walk UP from this file until agent_config.yaml is found."""
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"Could not find '{filename}' in '{Path(__file__).resolve().parent}' "
                "or any of its parent directories."
            )
        current = parent


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
#
# The core design principle: give Qwen 7B the minimum number of decisions
# to make. The entire news → radar → brief pipeline is collapsed into a
# single Python-side tool (run_narrative_analysis). The LLM only needs to
# make TWO tool calls: run_narrative_analysis, then thenvoi_send_message.
#
# Qwen 7B reliably fails at 3+ sequential tool calls with data-passing
# between them. One call + one send is well within its capability.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Narrative Analyst in AlphaSign, a multi-agent financial risk system on Band.

HOW TO SEND MESSAGES (CRITICAL)
────────────────────────────────
You MUST call thenvoi_send_message to post anything to the room.
Plain text output is invisible — only tool calls reach Band.

For the mentions field: extract the sender's name from the [Name]: prefix of the
incoming message and prefix it with @. Example: "[steven]: ..." → mentions=["@steven"].
Never use UUIDs or agent IDs as mention targets. Fallback: ["@all"].

RESEARCH REQUEST (ticker analysis)
────────────────────────────────────
STEP 1. Call run_narrative_analysis(ticker=<TICKER>). Wait for the result.
STEP 2. Call thenvoi_send_message(content=<OUTPUT FROM STEP 1>, mentions=["@<sender>"]).
That is all. Do not write plain text. Do not skip Step 2.

FOLLOW-UP QUESTIONS
────────────────────
Skip Step 1. Call thenvoi_send_message directly with your answer.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Whitebox Auditing Callback Handler
# ─────────────────────────────────────────────────────────────────────────────

class AgentWhiteboxLogger(BaseCallbackHandler):
    """Intercepts LLM lifecycle events to print structured choices to the terminal."""
    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for g in generation:
                if hasattr(g, 'message') and getattr(g.message, 'tool_calls', None):
                    print("\n" + "═"*60)
                    print("🤖 [NARRATIVE AGENT DECISION] -> LLM requested tool execution:")
                    sent_to_band = False
                    for tool_call in g.message.tool_calls:
                        print(f"   🔧 Tool: {tool_call['name']}")
                        print(f"      Args: {json.dumps(tool_call['args'])}")
                        if tool_call['name'] == 'thenvoi_send_message':
                            sent_to_band = True
                    if sent_to_band:
                        print("   ✅ thenvoi_send_message active -> posting to Band")
                    else:
                        print("   ⏳ Intermediate tool call — pipeline continuing")
                    print("═"*60 + "\n")
                elif g.text:
                    print("\n" + "📝"*3 + " [AGENT FINAL TEXT — LOCAL ONLY] " + "📝"*3)
                    print(g.text)
                    print("⚠️  Warning: No tool call — this text was NOT sent to Band.")
                    print("═"*80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def run_narrative_analysis(ticker: str) -> str:
    """
    Run the complete Narrative Analyst research pipeline for a stock ticker.
    Fetches news, builds a Narrative Radar, and returns a formatted brief.
    Call this for any new ticker research request.
    Pass the EXACT output of this tool as the content argument of thenvoi_send_message.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL", "NVDA", "TSLA"
    """
    symbol = ticker.upper()
    print(f"⚡ [LOCAL TOOL] run_narrative_analysis starting pipeline for {symbol}")

    # Step 1 — fetch articles
    try:
        articles = fetch_company_news(ticker)
        if isinstance(articles, list):
            articles = articles[:5]
        elif isinstance(articles, dict):
            articles = articles.get("articles", articles.get("output", []))[:5]
        logger.info(f"Fetched {len(articles)} articles for {symbol}")
    except Exception as e:
        logger.error(f"fetch_company_news failed for {symbol}: {e}")
        return json.dumps({"error": f"News fetch failed: {e}"})

    # Step 2 — build radar (needs a non-empty list; fall back to empty radar gracefully)
    try:
        radar = build_narrative_radar(symbol, articles)
    except Exception as e:
        logger.error(f"build_narrative_radar failed for {symbol}: {e}")
        return json.dumps({"error": f"Radar build failed: {e}"})

    # Step 3 — generate brief
    try:
        brief = generate_narrative_brief(radar)
    except Exception as e:
        logger.error(f"generate_narrative_brief failed for {symbol}: {e}")
        return json.dumps({"error": f"Brief generation failed: {e}"})

    print(f"⚡ [LOCAL TOOL] run_narrative_analysis complete for {symbol}")
    return brief if isinstance(brief, str) else json.dumps(brief, ensure_ascii=False)


@tool
def extract_text_from_article(url: str) -> str:
    """
    Extract raw body text from a specific news article URL.

    Args:
        url: the direct URL of the article
    """
    print(f"⚡ [LOCAL TOOL] extract_text_from_article scraping {url}")
    text = extract_article_text(url)
    return json.dumps({"url": url, "extracted_text": text})


@tool
def score_sentiment(text: str) -> str:
    """
    Score financial sentiment using deterministic lexical analysis.

    Args:
        text: text body to analyse
    """
    sentiment = score_text_sentiment(text)
    return json.dumps(sentiment)


@tool
def check_source_reliability(source: str) -> str:
    """
    Return the reliability tier of a media source or publisher name.

    Args:
        source: name of the news organisation or publisher
    """
    reliability = score_source_reliability(source)
    return json.dumps({"source": source, "reliability_tier": reliability})


# ─────────────────────────────────────────────────────────────────────────────
# Agent wiring
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    run_narrative_analysis,
    extract_text_from_article,
    score_sentiment,
    check_source_reliability,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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
        base_url=os.getenv("FEATHERLESS_BASE_URL"),
        api_key=os.getenv("FEATHERLESS_API_KEY"),
        model=os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        rate_limiter=rate_limiter,
        callbacks=[AgentWhiteboxLogger()],   # FIX: must be passed HERE to actually fire
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

    logger.info("Narrative Analyst agent is live. Press Ctrl+C to stop.")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
