"""
agents/narrative_analyst/sentiment.py

Lightweight financial sentiment scoring.

This is intentionally free and dependency-light. It is not trying to replace
FinBERT; it gives the News agent a deterministic local signal even when no
model endpoint is available. The LLM synthesis step can still refine the final
language using the scored evidence.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


BULLISH_TERMS = {
    "beat": 1.2,
    "beats": 1.2,
    "raise": 1.0,
    "raises": 1.0,
    "raised": 1.0,
    "upgrade": 1.1,
    "upgraded": 1.1,
    "outperform": 1.1,
    "growth": 0.8,
    "profit": 0.7,
    "profits": 0.7,
    "record": 0.7,
    "surge": 0.9,
    "rally": 0.9,
    "strong": 0.7,
    "expands": 0.7,
    "partnership": 0.5,
    "buyback": 0.8,
    "dividend": 0.5,
}

BEARISH_TERMS = {
    "miss": 1.2,
    "misses": 1.2,
    "cut": 1.0,
    "cuts": 1.0,
    "downgrade": 1.1,
    "downgraded": 1.1,
    "underperform": 1.1,
    "lawsuit": 1.0,
    "probe": 0.9,
    "investigation": 0.9,
    "recall": 1.0,
    "delay": 0.8,
    "delays": 0.8,
    "weak": 0.7,
    "slump": 0.9,
    "falls": 0.8,
    "drop": 0.8,
    "decline": 0.8,
    "loss": 0.9,
    "losses": 0.9,
    "risk": 0.6,
    "warns": 0.8,
}


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def score_text_sentiment(text: str) -> dict[str, Any]:
    """
    Score one text block on a -1.0 to 1.0 scale.

    Returns drivers so the final narrative can explain why the label exists.
    """
    tokens = _tokenize(text)
    counts = Counter(tokens)

    positive = sum(BULLISH_TERMS[token] * count for token, count in counts.items() if token in BULLISH_TERMS)
    negative = sum(BEARISH_TERMS[token] * count for token, count in counts.items() if token in BEARISH_TERMS)
    total = positive + negative

    if total == 0:
        score = 0.0
    else:
        score = (positive - negative) / total

    if score >= 0.2:
        label = "positive"
    elif score <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    drivers = []
    for token, count in counts.most_common():
        if token in BULLISH_TERMS:
            drivers.append({"term": token, "direction": "positive", "count": count})
        elif token in BEARISH_TERMS:
            drivers.append({"term": token, "direction": "negative", "count": count})
        if len(drivers) >= 6:
            break

    return {
        "label": label,
        "score": round(score, 4),
        "positive_weight": round(positive, 4),
        "negative_weight": round(negative, 4),
        "drivers": drivers,
    }


def score_article(article: dict[str, Any] | str) -> dict[str, Any]:
    # 1. Normalize raw string items into a standard dict format
    if isinstance(article, str):
        article = {
            "title": article,
            "description": "",
            "content": article,
            "text": ""
        }

    # 2. Extract and combine the text fields safely
    text = " ".join(
        str(article.get(key, ""))
        for key in ("title", "description", "content", "text")
        if article.get(key)
    )
    
    # 3. Complete processing using the standardized dict structure
    scored = dict(article)
    scored["sentiment"] = score_text_sentiment(text)
    return scored


def aggregate_sentiment(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {"label": "neutral", "score": 0.0, "article_count": 0}

    scores = []
    label_counts: Counter[str] = Counter()
    for article in articles:
        sentiment = article.get("sentiment") or score_article(article)["sentiment"]
        scores.append(float(sentiment.get("score", 0.0)))
        label_counts[str(sentiment.get("label", "neutral"))] += 1

    avg = sum(scores) / len(scores)
    if avg >= 0.15:
        label = "positive"
    elif avg <= -0.15:
        label = "negative"
    else:
        label = "mixed" if label_counts["positive"] and label_counts["negative"] else "neutral"

    return {
        "label": label,
        "score": round(avg, 4),
        "article_count": len(articles),
        "label_counts": dict(label_counts),
    }


if __name__ == "__main__":
    import json
    import sys

    sample = " ".join(sys.argv[1:]) or "Company beats earnings but warns of demand risk."
    print(json.dumps(score_text_sentiment(sample), indent=2))
