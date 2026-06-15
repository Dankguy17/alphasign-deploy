# This script loads a Gemini agent and connects it to the Band platform.
#
# Note: you must first create a .env file with your GEMINI_API_KEY and the Band URLs
#       as well as append the following to agent_config.yaml:
#
#         my_agent:
#           agent_id: "your-agent-id"
#           api_key: "your-api-key"
#
# Run `python verify_setup_gui.py` --> go to app.band.ai/dashboard
#     --> Add new chat room --> Click "+" on Participants --> Add your agent
#
# Now you'll be able to send it messages by saying "@<agent-name> <message>"

import asyncio
import logging
import os
from dotenv import load_dotenv
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_setup():
    # 1. This loads the GEMINI_API_KEY and Band URLs into memory
    load_dotenv()

    # 2. This searches agent_config.yaml for the 'my_agent' block
    agent_id, api_key = load_agent_config("my_agent")
    logger.info(f"Loaded agent: {agent_id}")

    # 3. This automatically pulls GEMINI_API_KEY from memory behind the scenes
    adapter = LangGraphAdapter(
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        checkpointer=InMemorySaver(),
    )

    # 4. Connects everything together
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )

    # ADD THIS INSTEAD:
    logger.info("Agent is live! Press Ctrl+C to stop.")
    await agent.run() # This keeps the script running and listening

asyncio.run(verify_setup())