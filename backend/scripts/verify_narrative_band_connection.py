"""
Verify the Narrative Analyst Band connection.

This checks the same credentials used by the live agent and starts the Band SDK
runtime briefly. It cannot prove that a specific chat room is routing messages,
but it separates "credentials/runtime can connect" from "room mention delivery".

Run from backend/:
    python scripts/verify_narrative_band_connection.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter

from agents.narrative_analyst.agent import _validate_band_credentials
from agents.narrative_analyst.prompts import SYSTEM_PROMPT
from shared.config import load_agent_credentials


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv(find_dotenv())

    agent_id, api_key = load_agent_credentials("narrative_analyst")
    _validate_band_credentials(agent_id, api_key)

    rest_url = os.getenv("THENVOI_REST_URL") or os.getenv("BAND_REST_URL")
    ws_url = os.getenv("THENVOI_WS_URL") or os.getenv("BAND_WS_URL")

    logger.info("Using narrative_analyst agent_id: %s", agent_id)
    logger.info("Using REST URL: %s", rest_url)
    logger.info("Using WS URL: %s", ws_url)

    adapter = LangGraphAdapter(
        llm=FakeListChatModel(responses=["Connection doctor response."]),
        checkpointer=InMemorySaver(),
        custom_section=SYSTEM_PROMPT,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting agent runtime for 20 seconds...")
    await agent.start()
    logger.info("Connected as: %s", getattr(agent, "agent_name", "<unknown>"))
    logger.info("Runtime started. If Band shows the agent online, websocket connection is working.")
    logger.info("Waiting 20 seconds so you can check the Band UI...")
    await asyncio.sleep(20)
    await agent.stop()
    logger.info("Connection doctor finished cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
