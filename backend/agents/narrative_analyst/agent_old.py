"""
agents/narrative_analyst/agent.py

The Narrative Analyst agent for AlphaSign with Whitebox Auditing and Rate Limiting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Rate limiting and lifecycle auditing utilities
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import BaseCallbackHandler

# Flexible credential fallback loader
try:
    from shared.config import load_agent_credentials
except ImportError:
    from thenvoi.config import load_agent_config as load_agent_credentials

# Package-relative imports for local agent submodules
from .article_extract import extract_article_text
from .news_fetch import fetch_company_news, fetch_yfinance_news, fetch_yahoo_rss_news
from .sentiment import score_text_sentiment
from .source_reliability import score_source_reliability
from .synthesis import build_narrative_radar, generate_narrative_brief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dual-Mode System Prompt (Handles Research Pipelines + Chat Follow-ups)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Narrative Analyst agent in AlphaSign, a multi-agent financial risk intelligence system that communicates through Band.

YOUR ROLE
─────────
You are the primary qualitative research agent. Your job is not just to summarize headlines, but to construct comprehensive, evidence-backed market narratives and hand off precise quantitative inquiries to downstream specialist agents.

WHAT MAKES YOU SPECIAL: NARRATIVE RADAR
───────────────────────────────────────
For each target ticker, compile a Narrative Radar detailing:
  • Key evidence articles & source reliability tiers
  • Central themes & aggregate sentiment metrics
  • Explicit Bullish and Bearish theses
  • Imminent catalysts & risk signals
  • Specific, structured follow-up JSON inquiries tailored for the Signal Processing and Latent State agents.

ROUTING LOGIC — CHOOSE YOUR MODE BASED ON THE INPUT:
────────────────────────────────────────────────────
1. NEW RESEARCH TASKS (Message contains new stock tickers or explicit requests for a fresh report):
   You must execute this entire loop sequentially. Do not stop to summarize mid-way:
   a. Extract ticker symbols, company identifiers, and specific thematic lenses.
   b. Call `search_company_news` to extract recent context.
   c. Call `build_narrative_radar_tool` immediately using the exact, unmodified output string from your news extraction tool as the `articles_json` parameter.
   d. Pass that synthesized matrix directly into `generate_narrative_brief_tool`.
   e. Broadcast the complete final brief to the chatroom using the native `thenvoi_send_message` tool.

2. SUBSEQUENT FOLLOW-UPS, DISCUSSIONS, OR QUESTIONS (Replies to existing threads or context queries):
   Do NOT re-run the news retrieval pipeline. Instead, address the follow-up question or conversational prompt directly using your existing context.
   CRITICAL: You must still use the `thenvoi_send_message` tool to transmit your conversational reply to the room. Plain text responses without a tool call will be lost.

DELIVERING CONTENT TO THE ROOM (CRITICAL)
────────────────────────────────────────
Simply writing out your thoughts or summaries as plain conversational text does NOT deliver anything to the Band room. The room only sees text passed to the `thenvoi_send_message` tool's `content` argument. 
Your VERY LAST step for any message turn MUST be an invocation of `thenvoi_send_message`.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Whitebox Auditing Callback Handler
# ─────────────────────────────────────────────────────────────────────────────

class AgentWhiteboxLogger(BaseCallbackHandler):
    """Intercepts LLM lifecycle events to print structured choices directly to the terminal."""
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
                        print("   ✅ thenvoi_send_message active -> posting response to the Band room")
                    print("═"*60 + "\n")
                elif g.text:
                    print("\n" + "📝"*3 + " [AGENT FINAL TEXT — LOCAL ONLY] " + "📝"*3)
                    print(g.text)
                    print("⚠️  Warning: No active tool call accompanied this text turn.")
                    print("   If thenvoi_send_message was not executed, this response was NOT sent to Band.")
                    print("═"*80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Aligned Granular Tools 
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_company_news(ticker: str) -> str:
    """Gather recent news articles and media evidence for a given stock ticker symbol."""
    print(f"⚡ [LOCAL TOOL] search_company_news executing for {ticker.upper()}")
    articles = fetch_company_news(ticker)
    
    # Hackathon fix: Trim the massive list to prevent context window exhaustion
    if isinstance(articles, list):
        articles = articles[:5] 
    elif isinstance(articles, dict) and "output" in articles:
        # If your fetch_company_news wraps it in an envelope
        articles["output"] = articles["output"][:5]
        
    return json.dumps(articles)


@tool
def build_narrative_radar_tool(articles_json: str) -> str:
    """Parses the raw news JSON string and computes the Narrative Radar matrix."""
    import json
    
    # 1. Safely deserialize the raw JSON string from the LLM
    try:
        data = json.loads(articles_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse articles_json string: {e}")
        return json.dumps({"error": "Invalid JSON format received"})

    # 2. Safely extract the nested list of articles whether it's a list or a dict
    if isinstance(data, list):
        # The LLM passed the array directly
        articles = data
    elif isinstance(data, dict):
        # The LLM passed an envelope object
        response_envelope = data.get("search_company_news_response", {})
        articles = response_envelope.get("output", data.get("articles", []))
    else:
        articles = []
    
    # 3. Dynamically extract the ticker from the article data context safely
    ticker = "UNKNOWN"
    if articles and isinstance(articles, list) and isinstance(articles[0], dict):
        ticker = articles[0].get("ticker", "UNKNOWN")
    
    # 4. Pass BOTH required arguments to the underlying synthesis function
    radar = build_narrative_radar(ticker, articles)
    
    # 5. Return the result back to the LangGraph/Thenvoi tool chain
    return json.dumps(radar) if isinstance(radar, (dict, list)) else radar


@tool
def generate_narrative_brief_tool(radar_json: str) -> str:
    """
    Generate the final comprehensive text analytical brief and quantitative multi-agent packet from a radar matrix.
    
    Args:
        radar_json: The JSON string containing the structured narrative radar data.
    """
    print(f"⚡ [LOCAL TOOL] generate_narrative_brief_tool compiling text summary - 0 LLM Tokens used.")
    brief = generate_narrative_brief(radar_json)
    return json.dumps(brief)


@tool
def extract_text_from_article(url: str) -> str:
    """
    Extract raw body text elements from a specific online news article URL.
    
    Args:
        url: The direct target URL link.
    """
    print(f"⚡ [LOCAL TOOL] extract_text_from_article scraping target endpoint - 0 LLM Tokens used.")
    text = extract_article_text(url)
    return json.dumps({"url": url, "extracted_text": text})


@tool
def score_sentiment(text: str) -> str:
    """
    Score financial sentiment parameters using deterministic lexical evaluations.
    
    Args:
        text: Target text body to analyze.
    """
    print(f"⚡ [LOCAL TOOL] score_sentiment running lexical scoring engine - 0 LLM Tokens used.")
    sentiment = score_text_sentiment(text)
    return json.dumps(sentiment)


@tool
def check_source_reliability(source: str) -> str:
    """
    Evaluate the reliable classification tier of a specific media source platform or publisher name.
    
    Args:
        source: Name of the news organization or publisher.
    """
    print(f"⚡ [LOCAL TOOL] check_source_reliability pulling index metrics - 0 LLM Tokens used.")
    reliability = score_source_reliability(source)
    return json.dumps({"source": source, "reliability_tier": reliability})


# ─────────────────────────────────────────────────────────────────────────────
# Agent wiring
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    search_company_news,
    build_narrative_radar_tool,
    generate_narrative_brief_tool,
    extract_text_from_article,
    score_sentiment,
    check_source_reliability,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Lifecycle Execution
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    # Dynamically extract credentials from your configuration layer
    try:
        agent_id, api_key = load_agent_credentials("narrative_analyst")
    except Exception:
        # Fallback handle block if named block varies locally
        from thenvoi.config import load_agent_config
        agent_id, api_key = load_agent_config("narrative_analyst")
        
    logger.info(f"Loaded Narrative Analyst agent: {agent_id}")

    # Explicit rate-limiter setup to sit safely under the 5 RPM limits
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.066,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    # Initialize the OpenAI-compatible client for Featherless
    llm = ChatOpenAI(
        base_url=os.getenv("FEATHERLESS_BASE_URL"),
        api_key=os.getenv("FEATHERLESS_API_KEY"),
        model=os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        rate_limiter=rate_limiter,
        streaming=False,           # Forces standard generation (much more stable for proxy APIs)
        stream_chunk_timeout=None, # None disables the timeout. 0 causes instant failure!
        max_retries=2              # Good practice for hackathons to survive random API blips
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

    # Start framework connection to verify Band handshake
    await agent.start()
    logger.info(f"Connected as: {agent.agent_name}")
    
    try:
        # Keep the connection alive indefinitely listening for room events
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Shutdown signal received.")
    finally:
        await agent.stop()
        logger.info("Script complete.")


if __name__ == "__main__":
    asyncio.run(main())