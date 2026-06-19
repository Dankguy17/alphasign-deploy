"""
agents/latent_state/opinion.py

LLM interpretation layer for Kalman predictions. This module uses Groq
via its OpenAI-compatible API because the Latent Space agent is intended to
reason over filtered state estimates rather than fetch data itself.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = """You are the Latent Space agent in AlphaSign, a multi-agent
financial risk intelligence system. You receive Kalman-filter outputs computed
from market or macro time-series data supplied by another agent.

Write a short 2-3 sentence interpretation of the latent state. Rules:
- Only use the Kalman metrics and lens provided.
- Explain whether the filtered trend supports continuation, stabilization, or
  a possible regime change.
- Treat structural_regime_shift as a statistical warning, not proof of news.
- Output ONLY valid JSON: {"summary": "...", "confidence": 0.0}
"""


def _build_user_prompt(kalman_result: dict, lens: str | None) -> str:
    lines = []
    if "predictions" in kalman_result:
        lines.append("Kalman predictions:")
        for prediction in kalman_result["predictions"]:
            lines.extend(_prediction_lines(prediction))
            lines.append("")
    else:
        lines.extend(_prediction_lines(kalman_result))

    if lens:
        lines.append(f"Lens: {lens}")
    return "\n".join(lines).strip()


def _prediction_lines(prediction: dict) -> list[str]:
    return [
        f"Series: {prediction.get('series_name', 'series')}",
        f"Window: {prediction.get('start')} to {prediction.get('end')}",
        f"Observations: {prediction.get('observations')}",
        f"Latest observation: {prediction.get('latest_observation')}",
        f"Filtered level: {prediction.get('filtered_level')}",
        f"Kalman trend slope: {prediction.get('kalman_trend_slope')}",
        f"Predicted next value: {prediction.get('predicted_next_value')}",
        f"Predicted next change: {prediction.get('predicted_next_change')}",
        f"Predicted next return: {prediction.get('predicted_next_return')}",
        f"Prediction variance: {prediction.get('prediction_variance')}",
        f"Noise variance: {prediction.get('noise_variance')}",
        f"Latest innovation z-score: {prediction.get('latest_innovation_z')}",
        f"Structural regime shift: {prediction.get('structural_regime_shift')}",
    ]


def generate_latent_summary(kalman_result: dict, lens: str | None = None) -> dict:
    """
    Generate a short Groq-written interpretation for a Kalman result.

    Requires GROQ_API_KEY. The model can be overridden with LATENT_STATE_MODEL
    or GROQ_MODEL, defaulting to llama-3.3-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY is required for latent summary generation")

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=api_key,
        model=(
            os.getenv("LATENT_STATE_MODEL")
            or os.getenv("GROQ_MODEL")
            or DEFAULT_GROQ_MODEL
        ),
        base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        temperature=0.2,
        max_tokens=300,
    )
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(kalman_result, lens)),
        ]
    )
    return _parse_response(response.content)


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        return {
            "summary": str(data.get("summary", "")).strip(),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"summary": raw, "confidence": 0.5}


if __name__ == "__main__":
    sample = {
        "series_name": "AAPL",
        "start": "2026-01-01",
        "end": "2026-06-15",
        "observations": 112,
        "latest_observation": 203.1,
        "filtered_level": 202.8,
        "kalman_trend_slope": 0.18,
        "predicted_next_value": 202.98,
        "predicted_next_change": -0.12,
        "predicted_next_return": -0.0006,
        "prediction_variance": 1.3,
        "noise_variance": 0.8,
        "latest_innovation_z": 0.7,
        "structural_regime_shift": False,
    }
    print(generate_latent_summary(sample, lens="assess continuation after earnings move"))
