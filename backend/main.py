"""
backend/main.py

AlphaSign driver — starts all three Band agents in a single process.

Usage:
    cd backend/
    python main.py

Environment variables (all in backend/.env):
    MAX_TURNS_PER_SESSION   Max agent turns before a PDF report is generated.
                            Counts one turn every time any agent sends a message.
                            Hard-capped at 3. Default: 3.
    CONVERSATION_LOG_PATH   Where to write the running conversation log.
                            Default: alphasign_conversation.txt
    PDF_OUTPUT_PATH         Where the Executive summary PDF is written.
                            Default: alphasign_report.pdf
    GROQ_API_KEY            Required for Latent State and Executive summary.
    FEATHERLESS_API_KEY     Required for Narrative Analyst.
    THENVOI_WS_URL          Band WebSocket URL.
    THENVOI_REST_URL        Band REST URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Local imports
from adapter import AlphaSignAdapter
from start_agent import StartAgent
from agents.executive.agent_executive import generate_executive_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alphasign.main")

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_TURNS         = int(os.getenv("MAX_TURNS_PER_SESSION", "3"))
HARD_MAX_TURNS    = 3
CONV_LOG_PATH     = Path(os.getenv("CONVERSATION_LOG_PATH", "alphasign_conversation.txt"))
PDF_OUTPUT_PATH   = Path(os.getenv("PDF_OUTPUT_PATH", "alphasign_report.pdf"))


# ── Shared state ──────────────────────────────────────────────────────────────

class SessionState:
    """
    Single shared object passed into every agent's on_final_response callback.
    Thread-safe via asyncio (all callbacks fire on the same event loop).
    """

    def __init__(self, adapter: AlphaSignAdapter, max_turns: int):
        self.adapter    = adapter
        self.max_turns  = max(1, min(max_turns, HARD_MAX_TURNS))
        self.turn_count = 0
        self.messages: list[dict] = []   # {agent, room_id, text, ts}
        self._report_triggered = False
        self._agent_tasks: list[asyncio.Task] = []
        self._log_path = CONV_LOG_PATH
        self._log_path.write_text("")    # clear / create on startup

    def bind_agent_tasks(self, tasks: list[asyncio.Task]) -> None:
        """Register the live agent loops so the turn limit can stop them."""
        self._agent_tasks = tasks

    def _stop_agents(self) -> None:
        """Hard-stop every agent loop; the adapter and report task stay alive."""
        logger.info("Hard turn limit reached — cancelling all agent runtimes")
        for task in self._agent_tasks:
            if not task.done():
                task.cancel()

    # Called from each agent's on_final_response hook (same asyncio thread)
    def record(self, agent_name: str, room_id: str, text: str) -> None:
        # Agent WebSocket loops can continue receiving messages after the report
        # threshold. Do not let those callbacks extend the configured session.
        if self.turn_count >= self.max_turns:
            logger.info(
                "Ignoring %s response — hard turn limit %d reached",
                agent_name,
                self.max_turns,
            )
            return

        ts = datetime.now(timezone.utc).isoformat()
        entry = {"agent": agent_name, "room_id": room_id, "text": text, "ts": ts}
        self.messages.append(entry)
        self.turn_count += 1

        # Append to running log file
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{'─'*60}\n")
            fh.write(f"[{ts}] {agent_name.upper()} (room={room_id})\n")
            fh.write(text + "\n")

        # Forward to adapter's message queue so the frontend can poll it
        self.adapter.enqueue(entry)

        logger.info(
            "Turn %d/%d — %s (room=%s)",
            self.turn_count, self.max_turns, agent_name, room_id,
        )

        # Trigger report when limit is reached (only once)
        if self.turn_count >= self.max_turns and not self._report_triggered:
            self._report_triggered = True
            asyncio.get_event_loop().create_task(self._generate_report())
            self._stop_agents()

    async def set_max_turns(self, requested: int) -> int:
        """Update the live limit while enforcing the application-wide ceiling."""
        self.max_turns = max(1, min(requested, HARD_MAX_TURNS))
        logger.info("Session turn limit configured to %d", self.max_turns)
        if self.turn_count >= self.max_turns and not self._report_triggered:
            self._report_triggered = True
            asyncio.create_task(self._generate_report())
            self._stop_agents()
        return self.max_turns

    async def _generate_report(self) -> None:
        logger.info("Turn limit reached — generating Executive PDF report…")
        try:
            conversation_text = self._log_path.read_text(encoding="utf-8")
            pdf_bytes = await asyncio.to_thread(
                generate_executive_report, conversation_text, str(PDF_OUTPUT_PATH)
            )
            self.adapter.enqueue({
                "type": "report_ready",
                "path": str(PDF_OUTPUT_PATH),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("PDF report written to %s", PDF_OUTPUT_PATH)
        except Exception as exc:
            logger.error("Executive report generation failed: %s", exc, exc_info=True)


# ── Agent launchers ───────────────────────────────────────────────────────────

async def _run_narrative_analyst(session: SessionState) -> None:
    """Import and run the Narrative Analyst agent."""
    from agents.narrative_analyst.agent_narrative_analyst import main as na_main
    await na_main(on_final_response=session.record)


async def _run_signal_processing(session: SessionState) -> None:
    """Import and run the Signal Processing agent."""
    from agents.signal_processing.agent_signal_processing import main as sp_main
    await sp_main(on_final_response=session.record)


async def _run_latent_state(session: SessionState) -> None:
    """Import and run the Latent State agent."""
    from agents.latent_state.agent_latent_state import main as ls_main
    await ls_main(on_final_response=session.record)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run() -> None:
    adapter = AlphaSignAdapter()
    session = SessionState(adapter=adapter, max_turns=MAX_TURNS)
    adapter.configure_turn_limit(lambda: session.max_turns, session.set_max_turns)
    start_agent = StartAgent(Path(__file__).with_name("agent_config.yaml"))
    adapter.configure_start_agent(
        start_agent.send_ticker,
        start_agent.create_room,
        start_agent.close_room,
    )

    logger.info(
        "AlphaSign starting — %d agents, max %d turns, log=%s, pdf=%s",
        3, session.max_turns, CONV_LOG_PATH, PDF_OUTPUT_PATH,
    )

    # Start the HTTP adapter server as a background task
    adapter_task = asyncio.create_task(adapter.serve(), name="adapter")

    # Start all three agents concurrently — each runs its own Band WebSocket loop
    agent_tasks = [
        asyncio.create_task(_run_narrative_analyst(session), name="narrative_analyst"),
        asyncio.create_task(_run_signal_processing(session),  name="signal_processing"),
        asyncio.create_task(_run_latent_state(session),       name="latent_state"),
    ]
    session.bind_agent_tasks(agent_tasks)

    all_tasks = [adapter_task] + agent_tasks

    # Graceful shutdown on Ctrl-C or SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("Received %s — shutting down…", sig_name)
        for task in all_tasks:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig.name)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            pass

    try:
        done, pending = await asyncio.wait(
            all_tasks,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            exc = task.exception()
            if exc:
                logger.error("Task %s raised: %s", task.get_name(), exc, exc_info=exc)
    except asyncio.CancelledError:
        pass
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        logger.info("AlphaSign stopped.")


if __name__ == "__main__":
    asyncio.run(run())
