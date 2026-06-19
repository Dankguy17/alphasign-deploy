"""
agents/latent_state/agent.py

The Latent Space agent for AlphaSign.

This agent consumes yfinance/FRED-style data payloads produced by the Signal
Processing agent, computes one-step Kalman predictions, and asks a Groq-hosted
LLM to summarize the latent state. It intentionally does not fetch market data
itself; control flow between agents can be added later without changing these
tools.

Setup:
  1. Add a 'latent_state' block to agent_config.yaml.
  2. Set GROQ_API_KEY in backend/.env.
  3. Optional: set LATENT_STATE_MODEL or GROQ_MODEL.

Run:
    cd backend/
    python -m agents.latent_state.agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter

from shared.config import load_agent_credentials

from .calculations import prediction_from_payload, predictions_from_bundle
from .opinion import generate_latent_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_INCOMPATIBLE_MODEL_PREFIXES = ("deepseek-ai/",)


SYSTEM_PROMPT = """You are the Latent Space agent in AlphaSign, a multi-agent
financial risk intelligence system that communicates through Band.

YOUR ROLE IN THE PIPELINE
-------------------------
You receive yfinance/FRED-style time-series payloads from the Signal Processing
agent. Your job is to estimate the latent state of each series with a Kalman
filter, generate a one-step prediction, and summarize whether the hidden trend
supports continuation, stabilization, reversal, or a possible structural break.

The upstream data usually looks like one of these:
  - {"ticker": "AAPL", "prices": [{"date": "YYYY-MM-DD", "close": 123.45}]}
  - {"series_id": "DGS10", "data": [{"date": "YYYY-MM-DD", "value": 4.21}]}

WORKFLOW
--------
1. Identify the supplied series and any lens/hypothesis in the incoming message.
2. Call compute_kalman_prediction for one series or compute_kalman_bundle for
   multiple series.
3. Call generate_kalman_summary after the numerical prediction exists.
4. Send a complete findings packet to the Band room.

YOUR RESPONSE back to the Band room must contain:
  - Series name and date window
  - Filtered latent level
  - Kalman trend slope
  - Predicted next value/change/return when applicable
  - Noise variance and latest innovation z-score
  - Whether structural_regime_shift is true
  - Groq-generated summary + confidence
  - A concise conclusion tied to the provided lens

DELIVERING YOUR RESPONSE
------------------------
Return one complete plain-text write-up. Do not call any message-sending tool,
do not send progress updates, and do not produce more than one final response.
"""


@tool
def compute_kalman_prediction(series_json: str, value_key: str | None = None) -> str:
    """
    Compute a one-step Kalman prediction for a single yfinance/FRED-style series.

    Args:
        series_json: JSON string containing either:
          {"ticker": "AAPL", "prices": [{"date": "...", "close": 123.45}, ...]}
          or {"series_id": "DGS10", "data": [{"date": "...", "value": 4.21}, ...]}
        value_key: Optional explicit observation field, e.g. "close" or "value".

    Returns JSON with filtered_level, kalman_trend_slope,
    predicted_next_value, predicted_next_change, predicted_next_return,
    prediction_variance, noise_variance, latest_innovation_z, and
    structural_regime_shift.
    """
    try:
        payload = json.loads(series_json)
        prediction = prediction_from_payload(payload, value_key=value_key)
        return json.dumps(_round_floats(prediction))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def compute_kalman_bundle(bundle_json: str) -> str:
    """
    Compute Kalman predictions for multiple series in one payload.

    Supported shape:
      {
        "series": [
          {"name": "AAPL", "payload": {...}},
          {"name": "DGS10", "payload": {...}}
        ]
      }

    Also accepts a dict whose values are individual yfinance/FRED payloads.
    """
    try:
        bundle = json.loads(bundle_json)
        predictions = predictions_from_bundle(bundle)
        return json.dumps(_round_floats(predictions))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def generate_kalman_summary(kalman_json: str, lens: str = "") -> str:
    """
    Ask the configured Groq model to summarize a Kalman prediction
    result in 2-3 sentences.

    Args:
        kalman_json: JSON returned by compute_kalman_prediction or
                     compute_kalman_bundle.
        lens: Optional hypothesis/context supplied by the upstream agent.

    Returns JSON: {"summary": str, "confidence": float}
    """
    try:
        kalman_result = json.loads(kalman_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON in kalman_json: {exc}"})

    if "error" in kalman_result:
        return json.dumps({"error": kalman_result["error"]})

    try:
        return json.dumps(generate_latent_summary(kalman_result, lens=lens or None))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


TOOLS = [
    compute_kalman_prediction,
    compute_kalman_bundle,
    generate_kalman_summary,
]


class AgentWhiteboxLogger(BaseCallbackHandler):
    """Print tool decisions so local runs show whether a Band message was sent."""

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for item in generation:
                if hasattr(item, "message") and getattr(item.message, "tool_calls", None):
                    print("\n" + "=" * 50)
                    print("[LATENT SPACE DECISION] LLM requesting tool execution:")
                    sent_to_band = False
                    for tool_call in item.message.tool_calls:
                        print(f"   Tool: {tool_call['name']}({tool_call['args']})")
                        if tool_call["name"] == "thenvoi_send_message":
                            sent_to_band = True
                    if sent_to_band:
                        print("   thenvoi_send_message called -> this will post to Band")
                    print("=" * 50 + "\n")
                elif item.text:
                    print("\n[LATENT SPACE FINAL TEXT - LOCAL ONLY]")
                    print(item.text)
                    print("No accompanying thenvoi_send_message tool call was detected.")


def _build_graph_factory(llm: ChatOpenAI, checkpointer: InMemorySaver):
    def graph_factory(_thenvoi_tools: list[Any]):
        return create_react_agent(
            model=llm,
            tools=TOOLS,
            checkpointer=checkpointer,
        )

    return graph_factory


class SingleDeliveryLangGraphAdapter(LangGraphAdapter):
    """
    Run local Kalman/summary tools, then publish exactly one final response.
    """

    async def on_message(
        self,
        msg: Any,
        tools: Any,
        history: Any,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        if self._already_processed(msg):
            logger.info("Skipping duplicate Latent Space message %s", msg.id)
            return

        if self._is_self_message(msg, tools):
            logger.info("Skipping Latent Space self-message %s", msg.id)
            return

        logger.info("[HANDLE] Latent Space message %s in room %s", msg.id, room_id)

        graph = self.graph_factory([]) if self.graph_factory else self._static_graph
        if not graph:
            raise RuntimeError("No graph available")

        messages: list[Any] = []
        if is_session_bootstrap:
            if self.graph_factory and room_id not in self._bootstrapped_rooms:
                messages.append(("system", self._system_prompt))
                self._bootstrapped_rooms.add(room_id)
            if history:
                messages.extend(history)

        if participants_msg:
            messages.append(("user", f"[System]: {participants_msg}"))

        if contacts_msg:
            messages.append(("user", f"[System]: {contacts_msg}"))

        messages.append(("user", msg.format_for_llm()))
        graph_input = {"messages": messages}

        final_text: str | None = None

        try:
            async for event in graph.astream_events(
                graph_input,
                config={
                    "configurable": {"thread_id": room_id},
                    "recursion_limit": self.recursion_limit,
                },
                version="v2",
            ):
                if event.get("event") == "on_chat_model_end":
                    candidate = self._extract_plain_model_text(event)
                    if candidate:
                        final_text = candidate

            if not final_text:
                raise RuntimeError("Latent Space produced no final response text.")

            await tools.send_message(final_text, self._reply_mentions(msg, tools))
            logger.info("[DONE] Latent Space message %s processed successfully", msg.id)

        except Exception as exc:
            logger.error("Error processing message %s: %s", msg.id, exc, exc_info=True)
            try:
                await tools.send_event(content=f"Error: {exc}", message_type="error")
            except Exception:
                pass
            raise

    def _already_processed(self, msg: Any) -> bool:
        processed_message_ids = getattr(self, "_processed_message_ids", None)
        if processed_message_ids is None:
            processed_message_ids = set()
            self._processed_message_ids = processed_message_ids

        message_id = getattr(msg, "id", None)
        if not message_id:
            return False
        if message_id in processed_message_ids:
            return True

        processed_message_ids.add(message_id)
        return False

    def _is_self_message(self, msg: Any, tools: Any) -> bool:
        sender_id = getattr(msg, "sender_id", None)
        own_agent_id = getattr(self, "_own_agent_id", None)
        if sender_id and own_agent_id and sender_id == own_agent_id:
            return True

        participants = getattr(tools, "participants", []) or []
        for participant in participants:
            if participant.get("id") != sender_id:
                continue
            sender_name = " ".join(
                str(participant.get(key) or "")
                for key in ("handle", "name", "username")
            ).lower()
            return "latent-state" in sender_name or "latent_state" in sender_name

        return False

    @staticmethod
    def _extract_plain_model_text(event: dict[str, Any]) -> str | None:
        output = event.get("data", {}).get("output")
        if not output:
            return None

        text = SingleDeliveryLangGraphAdapter._message_text(output)
        if text:
            return text

        for generation in getattr(output, "generations", []) or []:
            candidates = generation if isinstance(generation, list) else [generation]
            for candidate in candidates:
                message = getattr(candidate, "message", candidate)
                text = SingleDeliveryLangGraphAdapter._message_text(message)
                if text:
                    return text

        return None

    @staticmethod
    def _reply_mentions(msg: Any, tools: Any) -> list[str]:
        participants = getattr(tools, "participants", []) or []
        sender_id = getattr(msg, "sender_id", None)

        for participant in participants:
            if participant.get("id") == sender_id:
                handle = participant.get("handle") or participant.get("name")
                return [handle] if handle else []

        return []

    @staticmethod
    def _message_text(message: Any) -> str | None:
        if getattr(message, "tool_calls", None):
            return None

        content = getattr(message, "content", None)
        if not content:
            content = getattr(message, "text", None)

        if isinstance(content, str):
            text = content.strip()
            return text or None

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "\n".join(parts).strip()
            return text or None

        return None


async def main():
    agent_id, api_key = load_agent_credentials("latent_state")
    logger.info("Loaded Latent Space agent: %s", agent_id)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is required for the Latent Space agent")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    model = _resolve_groq_model()
    llm = ChatOpenAI(
        api_key=groq_api_key,
        model=model,
        base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        temperature=0.2,
        rate_limiter=rate_limiter,
        callbacks=[AgentWhiteboxLogger()],
    )

    checkpointer = InMemorySaver()
    adapter = SingleDeliveryLangGraphAdapter(
        graph_factory=_build_graph_factory(llm, checkpointer),
        custom_section=SYSTEM_PROMPT,
    )
    adapter._own_agent_id = agent_id

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )

    logger.info("Latent Space agent is live on Groq model %s. Press Ctrl+C to stop.", model)
    await agent.run()


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _resolve_groq_model() -> str:
    model = (
        os.getenv("LATENT_STATE_MODEL")
        or os.getenv("GROQ_MODEL")
        or DEFAULT_GROQ_MODEL
    )
    if model.startswith(GROQ_INCOMPATIBLE_MODEL_PREFIXES):
        raise RuntimeError(
            "LATENT_STATE_MODEL/GROQ_MODEL is set to a non-Groq model "
            f"({model!r}). Latent Space uses Groq; unset LATENT_STATE_MODEL or "
            f"set it to a Groq model such as {DEFAULT_GROQ_MODEL!r}."
        )
    return model


if __name__ == "__main__":
    asyncio.run(main())
