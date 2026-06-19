"""
agents/executive/agent.py

The Executive agent for AlphaSign.

This agent monitors the chat history across the Band framework. Once the 
Narrative Analyst, Signal Processing, and Latent State agents have shared 
their findings, this agent reads the history, asks a Groq-hosted LLM to 
synthesize a comprehensive executive summary, and generates a finalized PDF.

Setup:
  1. Add an 'executive' block to agent_config.yaml.
  2. Set GROQ_API_KEY in backend/.env.
  3. Ensure fpdf2 is installed (pip install fpdf2).

Run:
    cd backend/
    python -m agents.executive.agent
"""

from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter

# Assuming this exists in your shared codebase
from shared.config import load_agent_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Executive Agent in AlphaSign, a multi-agent
financial risk intelligence system communicating through Band.

YOUR ROLE IN THE PIPELINE
-------------------------
You are the final step in the analysis pipeline. Your job is to monitor the 
Band room and gather the insights produced by the other agents:
  1. Narrative Analyst
  2. Signal Processing
  3. Latent Space

When you see that all three agents have completed their respective outputs, 
or when you are explicitly prompted that the time limit has expired, you must:
  1. Read the chat history to synthesize a comprehensive "Executive Summary".
  2. Structure the summary cleanly with an overview, key findings, and a final conclusion.
  3. Call the `generate_executive_pdf` tool with your synthesized text to create the final PDF artifact.
  4. Send a final message back to the Band room confirming that the PDF has been 
     generated and provide a brief TL;DR of the summary.

DELIVERING YOUR RESPONSE TO THE ROOM
------------------------------------
You must use the `thenvoi_send_message` tool to communicate back to the room. 
Your final step must be a call to `thenvoi_send_message` letting the human and 
other agents know the PDF has been successfully generated. Do not end with 
only plain text.
"""

@tool
def generate_executive_pdf(summary_text: str, filename: str = "executive_summary.pdf") -> str:
    """
    Generates a PDF file containing the compiled executive summary text.

    Args:
        summary_text: The complete, formatted text of the executive summary.
        filename: The desired name of the output PDF file (default: executive_summary.pdf).

    Returns a status string indicating success or failure.
    """
    try:
        from fpdf import FPDF
        
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add Title
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, "AlphaSign Executive Summary", ln=True, align="C")
        pdf.ln(10)
        
        # Add Body Text
        pdf.set_font("Helvetica", size=11)
        
        # Encode string to latin-1 to avoid fpdf character map errors 
        # (fpdf2 handles unicode much better, but this ensures basic safety)
        clean_text = summary_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, clean_text)
        
        # Output to file
        output_path = os.path.join(os.getcwd(), filename)
        pdf.output(output_path)
        
        return f"SUCCESS: PDF generated and saved to {output_path}"
    
    except Exception as exc:
        logger.error(f"Failed to generate PDF: {exc}")
        return f"ERROR: Failed to generate PDF. Details: {str(exc)}"


TOOLS = [
    generate_executive_pdf,
]

class AgentWhiteboxLogger(BaseCallbackHandler):
    """Print tool decisions so local runs show whether a Band message was sent."""

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for item in generation:
                if hasattr(item, "message") and getattr(item.message, "tool_calls", None):
                    print("\n" + "=" * 50)
                    print("[EXECUTIVE DECISION] LLM requesting tool execution:")
                    sent_to_band = False
                    for tool_call in item.message.tool_calls:
                        print(f"   Tool: {tool_call['name']}({tool_call['args']})")
                        if tool_call["name"] == "thenvoi_send_message":
                            sent_to_band = True
                    if sent_to_band:
                        print("   thenvoi_send_message called -> this will post to Band")
                    print("=" * 50 + "\n")
                elif item.text:
                    print("\n[EXECUTIVE FINAL TEXT - LOCAL ONLY]")
                    print(item.text)
                    print("No accompanying thenvoi_send_message tool call was detected.")

async def main():
    agent_id, api_key = load_agent_credentials("executive")
    logger.info("Loaded Executive agent: %s", agent_id)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is required for the Executive agent")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    model = (
        os.getenv("GROQ_MODEL")
        or DEFAULT_GROQ_MODEL
    )
    
    llm = ChatOpenAI(
        api_key=groq_api_key,
        model=model,
        base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        temperature=0.1, # Lower temperature for a strictly factual executive summary
        rate_limiter=rate_limiter,
        callbacks=[AgentWhiteboxLogger()],
    )

    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
        custom_section=SYSTEM_PROMPT,
        additional_tools=TOOLS,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )

    logger.info("Executive agent is live on Groq model %s. Press Ctrl+C to stop.", model)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())