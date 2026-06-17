"""
agents/narrative_analyst/news_fetch.py

Free-first news retrieval for the Narrative Analyst agent.

This module deliberately has no LLM or Band dependency. It returns plain
Python dictionaries so it can be tested locally and reused by the agent tools.

Sources, in priority order:
  1. NewsAPI, if NEWS_API_KEY is present.
  2. Yahoo Finance RSS, free and keyless.
  3. yfinance Ticker.news, free and keyless but response shape can vary.

Run a quick live smoke test from backend/:
    python -m agents.narrative_analyst.news_fetch MSFT
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

import httpx
from dateutil import parser as date_parser


NEWSAPI_URL = "https://newsapi.org/v2/everything"
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

PLACEHOLDER_KEY_PREFIXES = (
    "your_",
    "optional_",
    "replace_",
    "paste_",
)


COMPANY_HINTS: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "GOOG": "Alphabet Google",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "NFLX": "Netflix",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WMT": "Walmart",
    "DIS": "Disney",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(value: Any) -> str | None:
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        parsed = date_parser.parse(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _days_ago(published_at: str | None) -> int | None:
    if not published_at:
        return None
    try:
        parsed = date_parser.parse(published_at)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, (_utc_now() - parsed.astimezone(timezone.utc)).days)
    except (ValueError, TypeError):
        return None


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def _is_real_key(value: str | None) -> bool:
    """Return False for empty/default placeholder values from .env examples."""
    if not value:
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    if lowered in {"none", "null", "na", "n/a", "changeme", "todo"}:
        return False
    return not lowered.startswith(PLACEHOLDER_KEY_PREFIXES)


def _article(
    *,
    title: str,
    url: str,
    source: str,
    published_at: str | None,
    description: str = "",
    content: str = "",
    ticker: str = "",
    provider: str = "",
) -> dict[str, Any]:
    published = _parse_date(published_at)
    return {
        "ticker": ticker.upper(),
        "title": _clean_text(title),
        "url": str(url or ""),
        "source": _clean_text(source),
        "published_at": published,
        "days_ago": _days_ago(published),
        "description": _clean_text(description),
        "content": _clean_text(content),
        "provider": provider,
    }


def company_hint(ticker: str, company_name: str | None = None) -> str:
    """Return a search-friendly company name hint for a ticker."""
    explicit = _clean_text(company_name)
    if explicit:
        return explicit
    return COMPANY_HINTS.get(ticker.upper(), ticker.upper())


def build_news_query(ticker: str, company_name: str | None = None, lens: str | None = None) -> str:
    """
    Build a finance-specific query.

    NewsAPI supports simple boolean syntax. Keep this under its 500-character
    limit and bias toward investable context instead of generic brand mentions.
    """
    symbol = ticker.upper().strip()
    company = company_hint(symbol, company_name)
    base = f'("{symbol}" OR "{company}") AND (stock OR shares OR earnings OR revenue OR guidance OR analyst OR market)'
    lens_text = _clean_text(lens)
    if lens_text:
        lens_words = " ".join(lens_text.split()[:12])
        return f"{base} AND ({lens_words})"
    return base


def search_newsapi(
    ticker: str,
    company_name: str | None = None,
    lens: str | None = None,
    days_back: int = 14,
    page_size: int = 20,
    sort_by: str = "publishedAt",
) -> list[dict[str, Any]]:
    """
    Search NewsAPI's /everything endpoint.

    Returns an empty list when NEWS_API_KEY is missing so callers can fall back
    to free keyless sources without treating it as a fatal error.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not _is_real_key(api_key):
        return []

    since = (_utc_now() - timedelta(days=days_back)).date().isoformat()
    params = {
        "q": build_news_query(ticker, company_name, lens),
        "searchIn": "title,description,content",
        "from": since,
        "language": "en",
        "sortBy": sort_by,
        "pageSize": min(max(page_size, 1), 100),
        "apiKey": api_key,
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(NEWSAPI_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        # Keep live demos resilient: a bad/expired NewsAPI key should not stop
        # the free Yahoo/yfinance fallbacks from working.
        print(f"[Narrative Analyst] NewsAPI unavailable ({exc.response.status_code}); using free fallbacks.")
        return []
    except httpx.HTTPError as exc:
        print(f"[Narrative Analyst] NewsAPI request failed ({exc}); using free fallbacks.")
        return []

    articles = []
    for raw in payload.get("articles", []):
        source = raw.get("source") or {}
        articles.append(
            _article(
                ticker=ticker,
                title=raw.get("title", ""),
                url=raw.get("url", ""),
                source=source.get("name", "NewsAPI"),
                published_at=raw.get("publishedAt"),
                description=raw.get("description", ""),
                content=raw.get("content", ""),
                provider="newsapi",
            )
        )
    return articles


def fetch_yahoo_rss_news(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch ticker headlines from Yahoo Finance RSS. No key required."""
    try:
        import feedparser
    except ImportError:
        return []

    url = f"{YAHOO_RSS_URL}?s={quote_plus(ticker.upper())}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:limit]:
        articles.append(
            _article(
                ticker=ticker,
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source="Yahoo Finance RSS",
                published_at=entry.get("published", None),
                description=entry.get("summary", ""),
                provider="yahoo_rss",
            )
        )
    return articles


def fetch_yfinance_news(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Fetch recent ticker news through yfinance.

    yfinance has changed the structure of Ticker.news across versions, so this
    parser handles both older flat records and newer nested content records.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        raw_items = yf.Ticker(ticker.upper()).news or []
    except Exception:
        return []

    articles = []
    for raw in raw_items[:limit]:
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        provider = content.get("provider") or raw.get("publisher") or {}
        click = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}

        if isinstance(provider, dict):
            source = provider.get("displayName") or provider.get("name") or "Yahoo Finance"
        else:
            source = str(provider or "Yahoo Finance")

        url = ""
        if isinstance(click, dict):
            url = click.get("url", "")
        elif isinstance(click, str):
            url = click
        if not url:
            url = raw.get("link", "")

        published = (
            content.get("pubDate")
            or content.get("displayTime")
            or raw.get("providerPublishTime")
            or raw.get("pubDate")
        )

        articles.append(
            _article(
                ticker=ticker,
                title=content.get("title") or raw.get("title", ""),
                url=url,
                source=source,
                published_at=published,
                description=content.get("summary") or raw.get("summary", ""),
                provider="yfinance",
            )
        )
    return articles


def dedupe_articles(articles: list[dict[str, Any]], similarity_threshold: int = 88) -> list[dict[str, Any]]:
    """
    Remove repeated/syndicated articles.

    Uses rapidfuzz when available, with a simple normalized-title fallback.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

    kept: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []

    for article in articles:
        title = _clean_text(article.get("title", "")).lower()
        url = str(article.get("url", "")).split("?")[0]
        if not title:
            continue
        if url and url in seen_urls:
            continue
        if fuzz:
            if any(fuzz.token_set_ratio(title, existing) >= similarity_threshold for existing in seen_titles):
                continue
        elif title in seen_titles:
            continue

        kept.append(article)
        seen_titles.append(title)
        if url:
            seen_urls.add(url)

    return kept


def fetch_company_news(
    ticker: str,
    company_name: str | None = None,
    lens: str | None = None,
    days_back: int = 14,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Fetch, merge, dedupe, and recency-sort company news."""
    articles: list[dict[str, Any]] = []
    articles.extend(search_newsapi(ticker, company_name, lens, days_back, page_size=limit))
    articles.extend(fetch_yahoo_rss_news(ticker, limit=limit))
    articles.extend(fetch_yfinance_news(ticker, limit=limit))

    deduped = dedupe_articles(articles)
    deduped.sort(key=lambda item: item.get("published_at") or "", reverse=True)
    return deduped[:limit]


if __name__ == "__main__":
    import json
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    result = fetch_company_news(symbol, days_back=14, limit=10)
    print(json.dumps(result, indent=2))
