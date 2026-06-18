"""
agents/signal_processing/opinion.py

Takes the computed findings (from calculations.py) plus optional Band
thread context, and produces a short natural-language "opinion" string
via an LLM call.

This is the swap point for model providers. The default provider is Groq.
Set SIGNAL_OPINION_PROVIDER=groq|gemini|featherless|aimlapi to switch
providers without changing the function signature.

Usage:
    python -m agents.signal_processing.opinion   (from backend/)
    python opinion.py                            (from this folder)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv, find_dotenv

# find_dotenv() walks UP the directory tree from this file until it
# finds a .env — so it works whether you run from backend/, from
# agents/signal_processing/, or anywhere else. No relative path hacks.
load_dotenv(find_dotenv())

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = """You are the Signal Processing agent in AlphaSign, a \
multi-agent financial risk intelligence system. You have just computed \
quantitative metrics for a stock over a specific time window using \
standard formulas (log return, volatility, beta, market-adjusted return, \
idiosyncratic volatility).

Your job is to write a substantive signal opinion interpreting these \
numbers in plain language. Rules:

- Only make claims that follow directly from the numbers provided.
- If "lens" or prior context from other agents is given, relate your \
numbers to it explicitly (e.g. "this supports/does not support the \
hypothesis that...").
- Explain what the most important numbers imply for momentum, market-relative \
performance, volatility, and company-specific movement when those metrics are \
present.
- If multiple metrics point in different directions, say that clearly instead \
of forcing a bullish or bearish conclusion.
- Do not invent additional data, news, or context not given to you.
- State your confidence (0.0-1.0) as a measure of how strong/clear the \
signal is, not how interesting it is.
- Output ONLY valid JSON: {"opinion": "...", "confidence": 0.0}
- Keep "opinion" to one analytical paragraph of 4-6 sentences.
"""


def _build_user_prompt(findings: dict, lens: str | None, prior_context: str | None) -> str:
    window = findings.get("window")
    if not isinstance(window, dict):
        window = {"label": window or "unknown", "start": "unknown", "end": "unknown"}

    window_label = window.get("label") or findings.get("window_label") or "unknown"
    window_start = window.get("start") or findings.get("start") or "unknown"
    window_end = window.get("end") or findings.get("end") or "unknown"

    lines = [
        f"Asset: {findings.get('asset') or findings.get('ticker') or 'unknown'}",
        f"Window: {window_label} ({window_start} to {window_end})",
    ]
    # Only include metrics that are actually present in this findings packet
    # (since the agent may have chosen to compute only a subset).
    metric_labels = {
        "log_return":             "Log return (most recent day)",
        "volatility":             "Volatility (window std dev of log returns)",
        "beta":                   "Beta vs. S&P 500",
        "market_adjusted_return": "Market-adjusted return (most recent day)",
        "idiosyncratic_vol":      "Idiosyncratic volatility",
    }
    for key, label in metric_labels.items():
        if key in findings:
            lines.append(f"{label}: {findings[key]:.4f}")

    if lens:
        lines.append(f"\nLens (why this was computed): {lens}")
    if prior_context:
        lines.append(f"\nPrior context from other agents:\n{prior_context}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────

def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content


def _call_openai_compatible(
    system_prompt: str, user_prompt: str, base_url: str, api_key: str, model: str
) -> str:
    if not api_key:
        raise RuntimeError(f"Missing API key for OpenAI-compatible model '{model}'.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    return _call_openai_compatible(
        system_prompt,
        user_prompt,
        base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        api_key=os.getenv("GROQ_API_KEY", ""),
        model=os.getenv("SIGNAL_OPINION_MODEL")
        or os.getenv("SIGNAL_PROCESSING_MODEL")
        or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
    )


def _call_featherless(system_prompt: str, user_prompt: str) -> str:
    return _call_openai_compatible(
        system_prompt,
        user_prompt,
        base_url=os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
        api_key=os.getenv("FEATHERLESS_API_KEY", ""),
        model=os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )


def _call_aimlapi(system_prompt: str, user_prompt: str) -> str:
    return _call_openai_compatible(
        system_prompt,
        user_prompt,
        base_url=os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1"),
        api_key=os.getenv("AIML_API_KEY", ""),
        model=os.getenv("AIML_MODEL", "gpt-4o-mini"),
    )


_PROVIDERS = {
    "groq":        _call_groq,
    "gemini":      _call_gemini,
    "featherless": _call_featherless,
    "aimlapi":     _call_aimlapi,
}


def generate_opinion(
    findings: dict,
    lens: str | None = None,
    prior_context: str | None = None,
) -> dict:
    """
    findings: dict with at minimum 'asset' and 'window' (dict with
              start/end/label). Any subset of the computed metrics is
              accepted — generate_opinion only describes what's present.

    Returns: {"opinion": str, "confidence": float}

    Provider is selected via SIGNAL_OPINION_PROVIDER env var
    ("groq" | "gemini" | "featherless" | "aimlapi"), default "groq".
    """
    provider_name = os.getenv("SIGNAL_OPINION_PROVIDER", "groq")
    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown SIGNAL_OPINION_PROVIDER '{provider_name}'. "
            f"Expected one of: {list(_PROVIDERS)}"
        )

    user_prompt = _build_user_prompt(findings, lens, prior_context)
    raw = _PROVIDERS[provider_name](SYSTEM_PROMPT, user_prompt)
    return _parse_response(raw)


def _parse_response(raw: str) -> dict:
    """Parse the LLM's JSON response, with a fallback if it adds stray text."""
    import json
    import re

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        return {
            "opinion":    str(data.get("opinion", "")).strip(),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"opinion": raw, "confidence": 0.5}


if __name__ == "__main__":
    sample_findings = {
        "asset":  "AAPL",
        "window": {"start": "2025-12-13", "end": "2026-06-13", "label": "6M"},
        "log_return":             0.012,
        "volatility":             0.018,
        "beta":                   1.21,
        "market_adjusted_return": -0.005,
        "idiosyncratic_vol":      0.045,
    }
    result = generate_opinion(
        sample_findings,
        lens="competitor hardware defect -- assess relative strength",
    )
    print(result)
