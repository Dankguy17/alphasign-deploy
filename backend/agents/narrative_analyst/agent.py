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

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

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


def _json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=True)


def _load_json_dict(value: str) -> dict:
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_articles_payload(value: str) -> tuple[list[dict], str | None]:
    """
    Parse article payloads without crashing the whole Band execution.

    LLMs sometimes try to pass the displayed output of search_company_news as a
    manually quoted string, which can produce invalid JSON. In that case return
    a tool-readable error instead of raising and marking the Band message failed.
    """
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        return [], (
            "Invalid articles_json. Do not manually quote or rewrite article JSON. "
            "Use build_full_narrative_report for normal ticker research, or pass "
            f"the exact raw output from search_company_news. JSON error: {exc}"
        )

    if isinstance(payload, dict):
        articles = payload.get("articles", [])
    elif isinstance(payload, list):
        articles = payload
    else:
        return [], "articles_json must decode to a dict with 'articles' or a list of article objects."

    if not isinstance(articles, list):
        return [], "The 'articles' field must be a list."

    return [article for article in articles if isinstance(article, dict)], None


def _format_band_report(radar: dict, brief: dict) -> str:
    top_articles = radar.get("top_articles", [])[:5]
    article_lines = []
    for idx, article in enumerate(top_articles, start=1):
        reliability = article.get("source_reliability", {})
        article_lines.append(
            f"{idx}. {article.get('title', 'Untitled')} "
            f"({article.get('source', 'Unknown source')}, "
            f"{reliability.get('tier_label', 'Unscored')}, "
            f"confidence {reliability.get('confidence', 'n/a')})"
        )

    reliability_summary = radar.get("source_reliability", {})
    signal_request = json.dumps(radar.get("signal_request", {}), indent=2)
    latent_request = json.dumps(radar.get("latent_request", {}), indent=2)

    return "\n".join(
        [
            f"## Narrative Radar: {radar.get('asset', 'UNKNOWN')}",
            "",
            f"**Summary:** {brief.get('summary', 'No summary generated.')}",
            "",
            "**Top Evidence:**",
            *(article_lines or ["No focused articles found."]),
            "",
            "**Source Reliability:**",
            f"- Average confidence: {reliability_summary.get('average_confidence', 'n/a')}",
            f"- Highest tier: {reliability_summary.get('highest_tier', 'n/a')}",
            f"- Tier counts: {reliability_summary.get('tier_counts', {})}",
            "",
            f"**Bullish case:** {brief.get('bullish_case', radar.get('bullish_thesis', 'n/a'))}",
            f"**Bearish case:** {brief.get('bearish_case', radar.get('bearish_thesis', 'n/a'))}",
            "",
            "**Risk flags:**",
            *[f"- {flag}" for flag in radar.get("risk_flags", [])],
            "",
            "**Signal Processing request:**",
            f"```json\n{signal_request}\n```",
            "",
            "**Latent State request:**",
            f"```json\n{latent_request}\n```",
        ]
    )


def _parse_tickers(tickers: str) -> list[str]:
    """Parse comma/space separated ticker input into unique uppercase symbols."""
    cleaned = tickers.replace(",", " ").replace(";", " ")
    stopwords = {"AND", "OR", "WITH", "VS", "VERSUS", "STOCK", "STOCKS", "TICKER", "TICKERS"}
    parsed: list[str] = []
    for token in cleaned.split():
        symbol = "".join(ch for ch in token.upper() if ch.isalnum() or ch in {".", "-"})
        if symbol and symbol not in stopwords and symbol not in parsed:
            parsed.append(symbol)
    return parsed


def _format_multi_band_report(results: list[dict]) -> str:
    lines = [
        "## Multi-Stock Narrative Radar",
        "",
        "I researched the requested tickers and selected the most relevant follow-up requests for Signal Processing and Latent State.",
        "",
    ]

    signal_requests = []
    latent_requests = []

    for result in results:
        radar = result["radar"]
        brief = result["brief"]
        reliability = radar.get("source_reliability", {})
        lines.extend(
            [
                f"### {radar.get('asset', 'UNKNOWN')}",
                f"**Summary:** {brief.get('summary', 'No summary generated.')}",
                f"**Source reliability:** average confidence {reliability.get('average_confidence', 'n/a')}, highest tier {reliability.get('highest_tier', 'n/a')}",
                f"**Bullish case:** {brief.get('bullish_case', radar.get('bullish_thesis', 'n/a'))}",
                f"**Bearish case:** {brief.get('bearish_case', radar.get('bearish_thesis', 'n/a'))}",
                "**Top evidence:**",
            ]
        )
        for article in radar.get("top_articles", [])[:3]:
            source_reliability = article.get("source_reliability", {})
            lines.append(
                f"- {article.get('title', 'Untitled')} "
                f"({article.get('source', 'Unknown source')}, "
                f"{source_reliability.get('tier_label', 'Unscored')})"
            )
        lines.append("")
        signal_requests.append(radar.get("signal_request", {}))
        latent_requests.append(radar.get("latent_request", {}))

    lines.extend(
        [
            "## Requests For Signal Processing",
            "```json",
            json.dumps(signal_requests, indent=2),
            "```",
            "",
            "## Requests For Latent State",
            "```json",
            json.dumps(latent_requests, indent=2),
            "```",
        ]
    )
    return "\n".join(lines)


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
    return _json_dumps({"ticker": ticker.upper(), "article_count": len(articles), "articles": articles})


@tool
def fetch_free_yahoo_news(ticker: str) -> str:
    """
    Fetch keyless Yahoo/yfinance news only. Useful if NewsAPI is unavailable.
    """
    articles = []
    articles.extend(fetch_yahoo_rss_news(ticker, limit=15))
    articles.extend(fetch_yfinance_news(ticker, limit=15))
    return _json_dumps({"ticker": ticker.upper(), "article_count": len(articles), "articles": articles})


@tool
def extract_article_text_tool(url: str, max_chars: int = 4000) -> str:
    """
    Extract readable text from a news article URL for deeper analysis.
    """
    return _json_dumps(extract_article_text(url, max_chars=max_chars))


@tool
def score_news_sentiment(text: str) -> str:
    """
    Score text with a free local financial sentiment heuristic.
    Returns label, numeric score, and driver terms.
    """
    return _json_dumps(score_text_sentiment(text))


@tool
def score_source_reliability_tool(article_json: str) -> str:
    """
    Score one article/source with the Source Reliability Engine.

    Returns tier, confidence, source_type, and reason.
    """
    article = _load_json_dict(article_json)
    if not article:
        return _json_dumps({
            "error": "Invalid article_json. Pass one article object as valid JSON.",
        })
    return _json_dumps(score_source_reliability(article))


@tool
def build_narrative_radar_tool(ticker: str, articles_json: str, lens: str = "") -> str:
    """
    Convert fetched articles into a structured Narrative Radar.

    articles_json can be either a raw JSON list or the direct output from
    search_company_news / fetch_free_yahoo_news.
    """
    articles, error = _parse_articles_payload(articles_json)
    if error:
        return _json_dumps({"error": error})
    radar = build_narrative_radar(ticker=ticker, articles=articles, lens=lens or None)
    return _json_dumps(radar)


@tool
def generate_narrative_brief_tool(radar_json: str) -> str:
    """
    Generate a concise analyst brief from a Narrative Radar.

    Uses Featherless/AI-ML API if configured; otherwise returns a deterministic
    no-LLM brief.
    """
    radar = _load_json_dict(radar_json)
    if not radar:
        return _json_dumps({"error": "Invalid radar_json. Pass a valid Narrative Radar JSON object."})
    return _json_dumps(generate_narrative_brief(radar))


@tool
def build_full_narrative_report(
    ticker: str,
    company_name: str = "",
    lens: str = "",
    days_back: int = 14,
) -> str:
    """
    Preferred tool for normal ticker research.

    Fetches focused news, scores source reliability, builds the Narrative Radar,
    creates the analyst brief, and returns a complete Band-ready message.

    After this tool returns, call thenvoi_send_message with the returned
    'band_message' value as the content. Do not rewrite article JSON manually.
    """
    articles = fetch_company_news(
        ticker=ticker,
        company_name=company_name or None,
        lens=lens or None,
        days_back=days_back,
        limit=int(os.getenv("NARRATIVE_MAX_ARTICLES", "25")),
    )
    radar = build_narrative_radar(
        ticker=ticker,
        articles=articles,
        lens=lens or None,
    )
    brief = generate_narrative_brief(radar)
    band_message = _format_band_report(radar, brief)
    return _json_dumps({
        "ticker": ticker.upper(),
        "band_message": band_message,
        "radar": radar,
        "brief": brief,
    })


@tool
def build_multi_ticker_narrative_report(
    tickers: str,
    lens: str = "",
    days_back: int = 14,
) -> str:
    """
    Preferred tool when the user asks about multiple stock tickers.

    Args:
        tickers: Comma or space separated ticker symbols, e.g. "AAPL, MSFT, NVDA".
        lens: Optional research lens that applies to the full basket.
        days_back: Recent-news lookback window.

    Returns a Band-ready multi-stock report plus a list of per-ticker Signal
    Processing and Latent State requests. After this tool returns, call
    thenvoi_send_message with the returned 'band_message' value as content.
    """
    symbols = _parse_tickers(tickers)
    if not symbols:
        return _json_dumps({
            "error": "No ticker symbols found. Ask the user for one or more stock tickers.",
        })

    max_tickers = int(os.getenv("NARRATIVE_MAX_TICKERS", "5"))
    symbols = symbols[:max_tickers]

    results = []
    for symbol in symbols:
        articles = fetch_company_news(
            ticker=symbol,
            lens=lens or None,
            days_back=days_back,
            limit=int(os.getenv("NARRATIVE_MAX_ARTICLES", "25")),
        )
        per_ticker_lens = lens or f"Assess whether recent {symbol} news is creating a tradable narrative shift."
        radar = build_narrative_radar(
            ticker=symbol,
            articles=articles,
            lens=per_ticker_lens,
        )
        brief = generate_narrative_brief(radar)
        results.append({
            "ticker": symbol,
            "radar": radar,
            "brief": brief,
        })

    band_message = _format_multi_band_report(results)
    return _json_dumps({
        "tickers": symbols,
        "band_message": band_message,
        "results": results,
        "signal_requests": [result["radar"].get("signal_request", {}) for result in results],
        "latent_requests": [result["radar"].get("latent_request", {}) for result in results],
    })


TOOLS = [
    search_company_news,
    fetch_free_yahoo_news,
    extract_article_text_tool,
    score_news_sentiment,
    score_source_reliability_tool,
    build_full_narrative_report,
    build_multi_ticker_narrative_report,
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
    # Match the known-working verify_setup_gui.py connection path as closely
    # as possible: load .env from backend/, then let thenvoi load the agent
    # credentials from agent_config.yaml.
    load_dotenv()
    agent_id, api_key = load_agent_config("narrative_analyst")
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
