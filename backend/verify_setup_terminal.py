# This does the same thing as verify_setup_gui.py, but runs entirely in the terminal

import asyncio
import logging
from dotenv import load_dotenv
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_setup():
    load_dotenv()

    # Load agent credentials
    agent_id, api_key = load_agent_config("my_agent")
    logger.info(f"Loaded agent: {agent_id}")

    # Initialize the LLM explicitly so we can test it directly
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Create adapter using our model instance
    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
    )

    # Create agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    # 1. Start the framework connection to verify Band handshake
    await agent.start()
    logger.info(f"Connected as: {agent.agent_name}")
    await asyncio.sleep(1)

    logger.info("Executing prompt directly through the LangChain LLM layer...")
    
    try:
        # 2. Programmatically query Gemini directly using the underlying LangChain object
        # This completely bypasses any hidden SDK adapter attributes
        messages = [HumanMessage(content="Hello! Tell me what 2 + 2 equals.")]
        
        # Invoke the model directly
        response = await llm.ainvoke(messages)
        
        # 3. Print out the response text
        print(f"\n🤖 [Gemini Direct Response]: {response.content}\n")
            
    except Exception as e:
        logger.error(f"Failed to execute Gemini call: {e}", exc_info=True)

    # 4. Clean shutdown
    await agent.stop()
    logger.info("Script complete.")

asyncio.run(verify_setup())

# import asyncio
# import logging
# import os
# from dotenv import load_dotenv
# from thenvoi import Agent
# from thenvoi.adapters import LangGraphAdapter
# from thenvoi.config import load_agent_config
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.checkpoint.memory import InMemorySaver

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# async def verify_setup():
#     # 1. This loads the GEMINI_API_KEY and Band URLs into memory
#     load_dotenv()

#     # 2. This searches agent_config.yaml for the 'my_agent' block
#     agent_id, api_key = load_agent_config("my_agent")
#     logger.info(f"Loaded agent: {agent_id}")

#     # 3. This automatically pulls GEMINI_API_KEY from memory behind the scenes
#     adapter = LangGraphAdapter(
#         llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
#         checkpointer=InMemorySaver(),
#     )

#     # 4. Connects everything together
#     agent = Agent.create(
#         adapter=adapter,
#         agent_id=agent_id,
#         api_key=api_key,
#         ws_url=os.getenv("THENVOI_WS_URL"),
#         rest_url=os.getenv("THENVOI_REST_URL"),
#     )

#     # ADD THIS INSTEAD:
#     logger.info("Agent is live! Press Ctrl+C to stop.")
#     await agent.run() # This keeps the script running and listening

# asyncio.run(verify_setup())