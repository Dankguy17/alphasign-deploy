"""
agents/narrative_analyst/agent.py

Band-connected Narrative Analyst agent for AlphaSign.

Run from backend/:
    python -m agents.narrative_analyst.agent

For local, no-Band testing:
    python scripts/test_narrative_agent_local.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter

from shared.config import load_agent_credentials

from .article_extract import extract_article_text
from .news_fetch import fetch_company_news, fetch_yfinance_news, fetch_yahoo_rss_news
from .prompts import SYSTEM_PROMPT
from .sentiment import score_text_sentiment
from .source_reliability import score_source_reliability
from .synthesis import build_narrative_radar, generate_narrative_brief


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_band_credentials(agent_id: str, api_key: str) -> None:
    """
    Fail early with a useful message when agent_config.yaml still contains
    placeholders. Band otherwise returns a generic 401 later in startup.
    """
    try:
        parsed = UUID(str(agent_id))
    except ValueError as exc:
        raise RuntimeError(
            "Invalid narrative_analyst.agent_id in agent_config.yaml. "
            "Copy the UUID from your Band Remote Agent settings."
        ) from exc

    if str(parsed) == "00000000-0000-0000-0000-000000000002":
        raise RuntimeError(
            "agent_config.yaml still contains the placeholder narrative_analyst agent_id. "
            "Create a Band Remote Agent for narrative_analyst and paste its real agent_id."
        )

    lowered_key = str(api_key or "").strip().lower()
    if (
        not lowered_key
        or lowered_key.startswith("band_api_key")
        or lowered_key.startswith("your_")
        or lowered_key.startswith("optional_")
    ):
        raise RuntimeError(
            "agent_config.yaml still contains a placeholder narrative_analyst api_key. "
            "Paste the real API key shown when you create the Band Remote Agent."
        )


@tool
def search_company_news(ticker: str, company_name: str = "", lens: str = "", days_back: int = 14) -> str:
    """
    Fetch recent company/ticker news from free-first sources.

    Uses NewsAPI when NEWS_API_KEY exists, then free Yahoo RSS and yfinance
    fallback sources. Returns JSON list of article dicts.
    """
    articles = fetch_company_news(
        ticker=ticker,
        company_name=company_name or None,
        lens=lens or None,
        days_back=days_back,
        limit=int(os.getenv("NARRATIVE_MAX_ARTICLES", "25")),
    )
    return json.dumps({"ticker": ticker.upper(), "article_count": len(articles), "articles": articles})


@tool
def fetch_free_yahoo_news(ticker: str) -> str:
    """
    Fetch keyless Yahoo/yfinance news only. Useful if NewsAPI is unavailable.
    """
    articles = []
    articles.extend(fetch_yahoo_rss_news(ticker, limit=15))
    articles.extend(fetch_yfinance_news(ticker, limit=15))
    return json.dumps({"ticker": ticker.upper(), "article_count": len(articles), "articles": articles})


@tool
def extract_article_text_tool(url: str, max_chars: int = 4000) -> str:
    """
    Extract readable text from a news article URL for deeper analysis.
    """
    return json.dumps(extract_article_text(url, max_chars=max_chars))


@tool
def score_news_sentiment(text: str) -> str:
    """
    Score text with a free local financial sentiment heuristic.
    Returns label, numeric score, and driver terms.
    """
    return json.dumps(score_text_sentiment(text))


@tool
def score_source_reliability_tool(article_json: str) -> str:
    """
    Score one article/source with the Source Reliability Engine.

    Returns tier, confidence, source_type, and reason.
    """
    return json.dumps(score_source_reliability(json.loads(article_json)))


@tool
def build_narrative_radar_tool(ticker: str, articles_json: str, lens: str = "") -> str:
    """
    Convert fetched articles into a structured Narrative Radar.

    articles_json can be either a raw JSON list or the direct output from
    search_company_news / fetch_free_yahoo_news.
    """
    payload = json.loads(articles_json)
    if isinstance(payload, dict):
        articles = payload.get("articles", [])
    else:
        articles = payload
    radar = build_narrative_radar(ticker=ticker, articles=articles, lens=lens or None)
    return json.dumps(radar)


@tool
def generate_narrative_brief_tool(radar_json: str) -> str:
    """
    Generate a concise analyst brief from a Narrative Radar.

    Uses Featherless/AI-ML API if configured; otherwise returns a deterministic
    no-LLM brief.
    """
    radar = json.loads(radar_json)
    return json.dumps(generate_narrative_brief(radar))


TOOLS = [
    search_company_news,
    fetch_free_yahoo_news,
    extract_article_text_tool,
    score_news_sentiment,
    score_source_reliability_tool,
    build_narrative_radar_tool,
    generate_narrative_brief_tool,
]


class AgentWhiteboxLogger(BaseCallbackHandler):
    """Print tool decisions so the team can debug live deliberation."""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n[NARRATIVE AGENT] Message received by local agent; starting LLM reasoning...\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else "unknown_tool"
        print(f"\n[NARRATIVE AGENT TOOL START] {name}: {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)
        if len(preview) > 800:
            preview = preview[:800] + "... [truncated]"
        print(f"\n[NARRATIVE AGENT TOOL END] {preview}\n")

    def on_tool_error(self, error, **kwargs):
        print(f"\n[NARRATIVE AGENT TOOL ERROR] {error}\n")

    def on_llm_error(self, error, **kwargs):
        print(f"\n[NARRATIVE AGENT LLM ERROR] {error}\n")

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for item in generation:
                message = getattr(item, "message", None)
                tool_calls = getattr(message, "tool_calls", None) if message else None
                if tool_calls:
                    print("\n" + "=" * 72)
                    print("[NARRATIVE AGENT DECISION] LLM requested tool execution:")
                    for tool_call in tool_calls:
                        print(f"  Tool: {tool_call['name']}({tool_call['args']})")
                    print("=" * 72 + "\n")
                elif getattr(item, "text", None):
                    print("\n[NARRATIVE AGENT LOCAL TEXT - NOT SENT TO BAND]")
                    print(item.text)
                    print("If this was intended for the room, thenvoi_send_message must be called.\n")


def _build_llm() -> object:
    """
    Build the controller model. Defaults to Featherless/open-source models,
    with Gemini as a fallback option to match the Signal agent.
    """
    provider = os.getenv("NARRATIVE_LLM_PROVIDER", "featherless").lower()
    callbacks = [AgentWhiteboxLogger()]

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=float(os.getenv("NARRATIVE_REQUESTS_PER_SECOND", "0.066")),
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            rate_limiter=rate_limiter,
            callbacks=callbacks,
        )

    if provider == "aimlapi":
        return ChatOpenAI(
            model=os.getenv("AIML_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("AIML_API_KEY", ""),
            base_url=os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1"),
            rate_limiter=rate_limiter,
            callbacks=callbacks,
        )

    return ChatOpenAI(
        model=os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("FEATHERLESS_API_KEY", ""),
        base_url=os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
        rate_limiter=rate_limiter,
        callbacks=callbacks,
        timeout=float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("NARRATIVE_LLM_MAX_RETRIES", "1")),
    )


async def main():
    agent_id, api_key = load_agent_credentials("narrative_analyst")
    _validate_band_credentials(agent_id, api_key)
    logger.info("Loaded Narrative Analyst agent: %s", agent_id)
    logger.info("Band REST URL: %s", os.getenv("THENVOI_REST_URL") or os.getenv("BAND_REST_URL"))
    logger.info("Band WS URL: %s", os.getenv("THENVOI_WS_URL") or os.getenv("BAND_WS_URL"))
    logger.info("LLM provider: %s", os.getenv("NARRATIVE_LLM_PROVIDER", "featherless"))
    logger.info("Featherless model: %s", os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    adapter = LangGraphAdapter(
        llm=_build_llm(),
        checkpointer=InMemorySaver(),
        custom_section=SYSTEM_PROMPT,
        additional_tools=TOOLS,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL") or os.getenv("BAND_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL") or os.getenv("BAND_REST_URL"),
    )

    logger.info("Starting Band websocket runtime. Keep this terminal open.")
    logger.info("If a Band mention reaches this process, you will see '[NARRATIVE AGENT] Message received'.")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
