"""
agents/narrative_analyst/source_reliability.py

Evidence-tier scoring for the Narrative Analyst agent.

The goal is not to prove that every article is true. The goal is to make the
agent explain how much weight it should place on each source before sending a
claim to Signal Processing, Latent State, or Executive.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from urllib.parse import urlparse


TIER_1_DOMAINS = {
    "sec.gov",
    "investor.apple.com",
    "microsoft.com",
    "investor.microsoft.com",
    "nvidia.com",
    "investor.nvidia.com",
    "ir.tesla.com",
    "abc.xyz",
    "investor.fb.com",
    "aboutamazon.com",
    "ir.aboutamazon.com",
    "federalreserve.gov",
    "treasury.gov",
    "bea.gov",
    "bls.gov",
    "census.gov",
}

PRESS_RELEASE_DOMAINS = {
    "businesswire.com",
    "globenewswire.com",
    "prnewswire.com",
}

TIER_2_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "investors.com",
    "investing.com",
}

TIER_3_DOMAINS = {
    "morningstar.com",
    "zacks.com",
    "gurufocus.com",
    "fool.com",
    "seekingalpha.com",
    "stockstory.org",
    "stockstory.com",
    "trefis.com",
}

AGGREGATOR_DOMAINS = {
    "finance.yahoo.com",
    "news.google.com",
    "msn.com",
}

TIER_2_SOURCES = {
    "reuters",
    "associated press",
    "ap",
    "bloomberg",
    "the wall street journal",
    "wsj",
    "financial times",
    "cnbc",
    "marketwatch",
    "barron's",
    "barrons",
    "investing.com",
}

TIER_3_SOURCES = {
    "morningstar",
    "zacks",
    "gurufocus.com",
    "motley fool",
    "seeking alpha",
    "stockstory",
    "trefis",
}


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _matches_domain(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _source(article: dict[str, Any]) -> str:
    return str(article.get("source") or "").strip().lower()


def _text(article: dict[str, Any]) -> str:
    return " ".join(
        str(article.get(key, ""))
        for key in ("title", "description", "content", "source", "url")
        if article.get(key)
    ).lower()


def score_source_reliability(article: dict[str, Any]) -> dict[str, Any]:
    """
    Score one article's evidence reliability.

    Tiers:
      Tier 1: official filings, official IR/company sources, government data,
              official press-release wires.
      Tier 2: major publications and recognized financial media.
      Tier 3: analyst/research/commentary sources.
      Tier 4: aggregators, broad feeds, blogs, unknown sources.
    """
    host = _domain(str(article.get("url", "")))
    source = _source(article)
    text = _text(article)

    if _matches_domain(host, TIER_1_DOMAINS) or host.endswith(".gov"):
        return {
            "tier": 1,
            "tier_label": "Tier 1",
            "confidence": 0.97,
            "source_type": "official_filing_government_or_company",
            "reason": "Official company, SEC, or government source.",
        }

    if _matches_domain(host, PRESS_RELEASE_DOMAINS) or "press release" in text:
        return {
            "tier": 1,
            "tier_label": "Tier 1",
            "confidence": 0.92,
            "source_type": "official_press_release_wire",
            "reason": "Press-release wire, often used for official company announcements.",
        }

    if _matches_domain(host, TIER_2_DOMAINS) or source in TIER_2_SOURCES:
        return {
            "tier": 2,
            "tier_label": "Tier 2",
            "confidence": 0.84,
            "source_type": "major_publication",
            "reason": "Recognized financial or major news publication.",
        }

    if _matches_domain(host, TIER_3_DOMAINS) or source in TIER_3_SOURCES:
        return {
            "tier": 3,
            "tier_label": "Tier 3",
            "confidence": 0.72,
            "source_type": "analyst_or_industry_research",
            "reason": "Analyst, research, or market commentary source.",
        }

    if _matches_domain(host, AGGREGATOR_DOMAINS) or "yahoo finance" in source:
        return {
            "tier": 4,
            "tier_label": "Tier 4",
            "confidence": 0.58,
            "source_type": "aggregator_or_reposted_feed",
            "reason": "Aggregator feed; useful for discovery but should be verified upstream.",
        }

    return {
        "tier": 4,
        "tier_label": "Tier 4",
        "confidence": 0.52,
        "source_type": "unknown_or_unverified",
        "reason": "Source not recognized by the reliability map.",
    }


def attach_reliability(article: dict[str, Any]) -> dict[str, Any]:
    scored = dict(article)
    scored["source_reliability"] = score_source_reliability(article)
    return scored


def aggregate_reliability(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "average_confidence": 0.0,
            "highest_tier": None,
            "tier_counts": {},
            "source_type_counts": {},
        }

    reliabilities = [
        article.get("source_reliability") or score_source_reliability(article)
        for article in articles
    ]
    tier_counts = Counter(item["tier_label"] for item in reliabilities)
    source_type_counts = Counter(item["source_type"] for item in reliabilities)
    avg = sum(float(item["confidence"]) for item in reliabilities) / len(reliabilities)
    highest = min(int(item["tier"]) for item in reliabilities)

    return {
        "average_confidence": round(avg, 3),
        "highest_tier": f"Tier {highest}",
        "tier_counts": dict(tier_counts),
        "source_type_counts": dict(source_type_counts),
    }
