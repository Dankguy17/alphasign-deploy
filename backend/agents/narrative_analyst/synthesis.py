"""
agents/narrative_analyst/synthesis.py

Turns news articles into AlphaSign's unique "Narrative Radar":

  - what happened
  - bullish thesis
  - bearish thesis
  - catalysts
  - risk flags
  - missing evidence
  - targeted requests for Signal Processing and Latent State

The deterministic radar works without any paid model. If a free hackathon
OpenAI-compatible key is configured, generate_narrative_brief can polish the
radar into a concise analyst-style message.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any

from dotenv import find_dotenv, load_dotenv

from .sentiment import aggregate_sentiment, score_article
from .source_reliability import aggregate_reliability, attach_reliability


load_dotenv(find_dotenv())


THEME_KEYWORDS: dict[str, set[str]] = {
    "earnings_guidance": {"earnings", "revenue", "profit", "eps", "guidance", "margin", "quarter"},
    "analyst_action": {"analyst", "upgrade", "downgrade", "price target", "rating", "outperform", "underperform"},
    "ai_product": {"ai", "artificial intelligence", "chip", "gpu", "cloud", "data center", "software"},
    "macro_rates": {"fed", "rate", "inflation", "treasury", "yield", "macro", "recession"},
    "legal_regulatory": {"lawsuit", "regulator", "regulatory", "probe", "investigation", "antitrust", "sec"},
    "competition": {"competitor", "rival", "market share", "competition", "pricing"},
    "supply_chain": {"supply", "shipment", "inventory", "factory", "manufacturing", "shortage"},
    "m_and_a": {"acquisition", "merger", "deal", "stake", "buyout"},
}


def _article_text(article: dict[str, Any]) -> str:
    return " ".join(
        str(article.get(key, ""))
        for key in ("title", "description", "content", "text")
        if article.get(key)
    ).lower()


def detect_themes(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, int] = defaultdict(int)
    evidence: dict[str, list[str]] = defaultdict(list)

    for article in articles:
        text = _article_text(article)
        title = str(article.get("title", ""))
        for theme, keywords in THEME_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword in text]
            if hits:
                scores[theme] += len(hits)
                if title and len(evidence[theme]) < 3:
                    evidence[theme].append(title)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"theme": theme, "score": score, "evidence_titles": evidence[theme]}
        for theme, score in ranked
    ]


def rank_articles(articles: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    ranked = []
    for article in articles:
        item = attach_reliability(score_article(article))
        days_ago = item.get("days_ago")
        recency_score = max(0, 14 - int(days_ago or 14)) / 14
        sentiment_score = abs(float(item["sentiment"]["score"]))
        reliability_score = float(item["source_reliability"]["confidence"])
        has_description = 0.2 if item.get("description") else 0.0
        item["relevance_score"] = round(
            recency_score + sentiment_score + reliability_score + has_description,
            4,
        )
        ranked.append(item)
    ranked.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    return ranked[:limit]


def _choose_windows(themes: list[dict[str, Any]], articles: list[dict[str, Any]]) -> list[str]:
    theme_names = {theme["theme"] for theme in themes[:3]}
    max_days = max((article.get("days_ago") or 0 for article in articles), default=0)

    if "earnings_guidance" in theme_names or max_days <= 14:
        return ["1M", "3M"]
    if "macro_rates" in theme_names or "legal_regulatory" in theme_names:
        return ["6M", "1Y"]
    return ["3M", "6M"]


def _requested_metrics(themes: list[dict[str, Any]]) -> list[str]:
    theme_names = {theme["theme"] for theme in themes[:3]}
    metrics = ["log_return", "volatility"]
    if theme_names & {"macro_rates", "competition", "legal_regulatory"}:
        metrics.extend(["beta", "market_adjusted_return", "idiosyncratic_vol"])
    if "earnings_guidance" in theme_names:
        metrics.extend(["market_adjusted_return", "idiosyncratic_vol"])
    return list(dict.fromkeys(metrics))


def build_narrative_radar(
    ticker: str,
    articles: list[dict[str, Any]],
    lens: str | None = None,
    max_articles: int = 8,
) -> dict[str, Any]:
    """
    Build the core structured output for the Narrative Analyst.

    This can be sent directly to Band as JSON or fed into an LLM for a more
    polished analyst brief.
    """
    symbol = ticker.upper()
    ranked_articles = rank_articles(articles, limit=max_articles)
    themes = detect_themes(ranked_articles)
    sentiment = aggregate_sentiment(ranked_articles)
    reliability = aggregate_reliability(ranked_articles)
    windows = _choose_windows(themes, ranked_articles)
    metrics = _requested_metrics(themes)

    top_titles = [article["title"] for article in ranked_articles[:5] if article.get("title")]
    top_theme = themes[0]["theme"] if themes else "general_news"

    bullish = "Positive evidence is limited; wait for Signal Processing before making a bullish claim."
    bearish = "Negative evidence is limited; wait for Signal Processing before making a bearish claim."
    if sentiment["score"] > 0.15:
        bullish = f"Recent coverage leans positive around {top_theme.replace('_', ' ')}."
        bearish = "Main risk is that price action may already reflect the good news."
    elif sentiment["score"] < -0.15:
        bullish = "Contrarian upside may exist if quantitative data shows the selloff is market-driven."
        bearish = f"Recent coverage leans negative around {top_theme.replace('_', ' ')}."
    elif themes:
        bullish = f"There is enough activity around {top_theme.replace('_', ' ')} to test for constructive momentum."
        bearish = "Mixed coverage means the agent should separate company-specific moves from broad-market noise."

    lens_text = lens or f"Assess whether recent {symbol} news is creating a tradable narrative shift."
    signal_request = {
        "from": "narrative_analyst",
        "to": "signal_processing",
        "asset": symbol,
        "lens": lens_text,
        "suggested_windows": windows,
        "requested_metrics": metrics,
        "reason": (
            "Narrative coverage suggests these metrics can test whether the news is "
            "showing up as momentum, abnormal volatility, or company-specific movement."
        ),
    }
    latent_request = {
        "from": "narrative_analyst",
        "to": "latent_state",
        "asset": symbol,
        "lens": lens_text,
        "suggested_windows": windows[-1:],
        "reason": "Check whether the news-linked move looks like a trend/regime shift rather than short noise.",
    }

    confidence = min(
        0.95,
        0.25
        + 0.04 * len(ranked_articles)
        + min(0.2, abs(sentiment["score"]))
        + 0.25 * reliability["average_confidence"],
    )

    return {
        "packet_type": "narrative_radar",
        "agent": "narrative_analyst",
        "asset": symbol,
        "lens": lens_text,
        "article_count": len(ranked_articles),
        "top_articles": [
            {
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at"),
                "url": article.get("url", ""),
                "sentiment": article.get("sentiment", {}),
                "source_reliability": article.get("source_reliability", {}),
            }
            for article in ranked_articles
        ],
        "themes": themes,
        "aggregate_sentiment": sentiment,
        "source_reliability": reliability,
        "bullish_thesis": bullish,
        "bearish_thesis": bearish,
        "catalysts": top_titles[:3],
        "risk_flags": _risk_flags(themes, sentiment),
        "missing_evidence": [
            "Need Signal Processing to confirm price reaction and abnormal volatility.",
            "Need Latent State to check whether recent movement is a persistent trend.",
        ],
        "signal_request": signal_request,
        "latent_request": latent_request,
        "confidence": round(confidence, 2),
    }


def _risk_flags(themes: list[dict[str, Any]], sentiment: dict[str, Any]) -> list[str]:
    flags = []
    theme_names = {theme["theme"] for theme in themes[:4]}
    if "legal_regulatory" in theme_names:
        flags.append("Legal or regulatory coverage appears in the news cluster.")
    if "macro_rates" in theme_names:
        flags.append("Macro/rate sensitivity may be influencing the narrative.")
    if sentiment.get("label") == "mixed":
        flags.append("News tone is mixed, so single-source conclusions are risky.")
    if not flags:
        flags.append("No major narrative risk flag detected from headlines alone.")
    return flags


def _call_openai_compatible(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    provider = os.getenv("NARRATIVE_LLM_PROVIDER", "featherless").lower()
    if provider == "aimlapi":
        base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
        api_key = os.getenv("AIML_API_KEY", "")
        model = os.getenv("AIML_MODEL", "gpt-4o-mini")
    else:
        base_url = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
        api_key = os.getenv("FEATHERLESS_API_KEY", "")
        model = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    if not api_key or api_key.startswith("your_"):
        raise RuntimeError(f"Missing API key for NARRATIVE_LLM_PROVIDER={provider}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
        timeout=float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "60")),
    )
    return response.choices[0].message.content or ""


def generate_narrative_brief(radar: dict[str, Any], use_llm: bool | None = None) -> dict[str, Any]:
    """
    Return a concise analyst-style brief. Falls back to deterministic text when
    no free model key is configured.
    """
    if use_llm is None:
        use_llm = os.getenv("NARRATIVE_USE_LLM", "true").lower() == "true"

    if use_llm:
        system_prompt = (
            "You are the Narrative Analyst agent in AlphaSign. Write concise, "
            "evidence-backed financial narrative analysis. Do not invent facts. "
            "Return only valid JSON with keys: summary, bullish_case, bearish_case, "
            "questions_for_signal, questions_for_latent, confidence."
        )
        try:
            raw = _call_openai_compatible(system_prompt, json.dumps(radar, indent=2))
            return _parse_json_or_fallback(raw, radar)
        except Exception:
            pass

    return {
        "summary": (
            f"{radar['asset']} news coverage is {radar['aggregate_sentiment']['label']} "
            f"with strongest theme(s): {', '.join(t['theme'] for t in radar['themes'][:3]) or 'general_news'}."
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


def _parse_json_or_fallback(raw: str, radar: dict[str, Any]) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"summary": raw}

    data.setdefault("bullish_case", radar["bullish_thesis"])
    data.setdefault("bearish_case", radar["bearish_thesis"])
    data.setdefault("questions_for_signal", [radar["signal_request"]["reason"]])
    data.setdefault("questions_for_latent", [radar["latent_request"]["reason"]])
    data.setdefault("confidence", radar["confidence"])
    return data


if __name__ == "__main__":
    sample_articles = [
        {
            "ticker": "MSFT",
            "title": "Microsoft shares rise after cloud revenue beats estimates",
            "description": "Analysts point to AI demand and stronger margins.",
            "source": "Sample",
            "published_at": "2026-06-16T12:00:00Z",
            "url": "https://example.com/msft-cloud",
        },
        {
            "ticker": "MSFT",
            "title": "Regulators widen antitrust probe into cloud software contracts",
            "description": "Investors weigh legal risk against AI growth.",
            "source": "Sample",
            "published_at": "2026-06-15T12:00:00Z",
            "url": "https://example.com/msft-probe",
        },
    ]
    radar = build_narrative_radar("MSFT", sample_articles)
    print(json.dumps(radar, indent=2))
    print(json.dumps(generate_narrative_brief(radar, use_llm=False), indent=2))
