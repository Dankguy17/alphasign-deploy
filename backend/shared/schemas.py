"""
shared/schemas.py

Pydantic models for the two packet types agents exchange via Band chat
messages: findings_packet and request_packet.

Lock this file on Day 1. Every agent imports from here so the JSON shape
is identical across the whole pipeline. Adding fields later is fine
(e.g. "opinion" for Layer 1, "lens" for Layer 2+) — just add them as
Optional with sensible defaults so earlier layers keep working.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TimeWindow(BaseModel):
    """The exact date range a computation was run over."""
    start: date
    end: date
    label: str  # e.g. "6M", "1Y", "4D" — human-readable, not load-bearing


class Headline(BaseModel):
    """A single news headline with its publication date."""
    text: str
    date: date
    days_ago: int


class FindingsPacket(BaseModel):
    """
    Published by a specialist agent (signal_processing, narrative_analyst,
    latent_state) after computing its numbers for a given asset/window.

    `opinion` and `request` are optional so Layer 0 (numbers only) and
    Layer 1 (+ opinion) work without breaking the schema for later layers.
    """
    agent: Literal["signal_processing", "narrative_analyst", "latent_state"]
    asset: str  # ticker, e.g. "AAPL"
    window: Optional[TimeWindow] = None
    lens: Optional[str] = None  # contextual reason this window/asset was examined

    # --- agent-specific numeric fields (populate whichever apply) ---
    log_return: Optional[float] = None
    idiosyncratic_vol: Optional[float] = None
    market_adjusted_return: Optional[float] = None
    beta: Optional[float] = None

    impact_z_score: Optional[float] = None
    threshold_crossed: Optional[bool] = None
    headlines: Optional[list[Headline]] = None

    kalman_trend_slope: Optional[float] = None
    noise_variance: Optional[float] = None
    structural_regime_shift: Optional[bool] = None

    # --- interpretive layer (Layer 1+) ---
    opinion: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RequestPacket(BaseModel):
    """
    Published by any specialist agent to ask another specialist to
    (re)compute findings for a given asset, optionally under a specific
    lens and/or specific time windows.

    `suggested_windows` is advisory — the receiving agent can choose
    different windows if the data suggests a more informative one.
    """
    from_agent: Literal["signal_processing", "narrative_analyst", "latent_state"] = Field(
        ..., alias="from"
    )
    to_agent: Literal["signal_processing", "narrative_analyst", "latent_state"] = Field(
        ..., alias="to"
    )
    asset: str
    lens: str
    suggested_windows: Optional[list[str]] = None  # e.g. ["1Y", "6M", "3M"]
    round: int = 1

    class Config:
        populate_by_name = True


class SessionConfig(BaseModel):
    """Initial user input that kicks off a session."""
    ticker: str
    sector: Optional[str] = None
    max_rounds: int = 2  # deliberation cap, per agent
