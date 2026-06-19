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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Local imports
from adapter import AlphaSignAdapter
from start_agent import StartAgent
from agents.executive.agent_executive import generate_executive_report
from live_protocol import GroqProtocolNormalizer

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
        self._normalization_tasks: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.Lock()
        self._close_room: Callable[[str | None], Awaitable[dict[str, object]]] | None = None
        self._log_path = CONV_LOG_PATH
        self._log_path.write_text("")    # clear / create on startup
        self._normalizer = GroqProtocolNormalizer()
        self._normalizer.clear()

    def bind_agent_tasks(self, tasks: list[asyncio.Task]) -> None:
        """Register the live agent loops so the turn limit can stop them."""
        self._agent_tasks = tasks
        self._loop = asyncio.get_running_loop()

    def bind_room_closer(
        self, closer: Callable[[str | None], Awaitable[dict[str, object]]]
    ) -> None:
        self._close_room = closer

    async def _deactivate_room(self, room_id: str) -> None:
        """Remove all runtime agents so Band cannot wake them again."""
        if not self._close_room:
            return
        try:
            await self._close_room(room_id)
            logger.info("Band room %s deactivated at the turn limit", room_id)
        except Exception as exc:
            logger.error("Failed to deactivate Band room %s: %s", room_id, exc)

    def _stop_agents(self, tasks: list[asyncio.Task] | None = None) -> None:
        """Hard-stop every agent loop; the adapter and report task stay alive."""
        logger.info("Hard turn limit reached — cancelling all agent runtimes")
        for task in tasks if tasks is not None else self._agent_tasks:
            if not task.done():
                task.cancel()

    def _schedule_normalization(self, entry: dict) -> None:
        """Create and retain a normalization task until it has completed."""
        task = asyncio.create_task(
            self._normalize_entry(entry),
            name=f"normalize-{entry['agent']}-{entry['ts']}",
        )
        self._normalization_tasks.add(task)
        task.add_done_callback(self._normalization_tasks.discard)

    async def _finish_cards_then_stop(self) -> None:
        """Let every accepted response become a card before stopping agents."""
        # Capture this generation. A new ticker may start replacement tasks
        # while the old session is still finishing normalization.
        tasks_to_stop = list(self._agent_tasks)
        # Preserve the brief grace period agents need to finish their Band send.
        await asyncio.sleep(0.5)
        pending = list(self._normalization_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._stop_agents(tasks_to_stop)

    async def start_runtime(self) -> None:
        """Reset session state and ensure all agent loops are live for a ticker."""
        old_tasks = list(self._agent_tasks)
        for task in old_tasks:
            if not task.done():
                task.cancel()
        if old_tasks:
            await asyncio.gather(*old_tasks, return_exceptions=True)

        with self._state_lock:
            self.turn_count = 0
            self._report_triggered = False
        self.messages.clear()
        self._log_path.write_text("")
        self._normalizer.clear()

        tasks = [
            asyncio.create_task(_run_narrative_analyst(self), name="narrative_analyst"),
            asyncio.create_task(_run_signal_processing(self), name="signal_processing"),
            asyncio.create_task(_run_latent_state(self), name="latent_state"),
        ]
        self.bind_agent_tasks(tasks)
        logger.info("Agent runtimes started for a new ticker session")

    # Called from each agent's on_final_response hook (same asyncio thread)
    def record(self, agent_name: str, room_id: str, text: str) -> bool:
        """Accept a response only while the session is live.

        The return value is a transmission permit: agents must not post to Band
        when it is false.
        """
        # Agent WebSocket loops can continue receiving messages after the report
        # threshold. Do not let those callbacks extend the configured session.
        with self._state_lock:
            if self.turn_count >= self.max_turns:
                logger.info(
                    "Blocking %s Band response — hard turn limit %d reached",
                    agent_name,
                    self.max_turns,
                )
                return False
            self.turn_count += 1
            reached_limit = self.turn_count >= self.max_turns

        ts = datetime.now(timezone.utc).isoformat()
        entry = {"agent": agent_name, "room_id": room_id, "text": text, "ts": ts}
        self.messages.append(entry)

        # Append to running log file
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{'─'*60}\n")
            fh.write(f"[{ts}] {agent_name.upper()} (room={room_id})\n")
            fh.write(text + "\n")

        # Forward to adapter's message queue so the frontend can poll it
        self.adapter.enqueue(entry)
        if self._loop:
            self._loop.call_soon_threadsafe(self._schedule_normalization, entry)

        logger.info(
            "Turn %d/%d — %s (room=%s)",
            self.turn_count, self.max_turns, agent_name, room_id,
        )

        # Trigger report when limit is reached (only once)
        if reached_limit and not self._report_triggered:
            self._report_triggered = True
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._generate_report(), self._loop)
                asyncio.run_coroutine_threadsafe(self._deactivate_room(room_id), self._loop)
                # Keep the process alive until every accepted response has been
                # normalized and delivered to the dashboard.
                asyncio.run_coroutine_threadsafe(
                    self._finish_cards_then_stop(), self._loop
                )
        return True

    async def _normalize_entry(self, entry: dict) -> None:
        """Turn one raw Band message into the stable, persisted UI protocol."""
        try:
            card = await self._normalizer.normalize(
                entry["agent"], entry["room_id"], entry["text"]
            )
            event = {
                "type": "protocol_card",
                "agent": entry["agent"],
                "room_id": entry["room_id"],
                "source_ts": entry["ts"],
                "ts": datetime.now(timezone.utc).isoformat(),
                "card": card.model_dump(mode="json"),
            }
            self._normalizer.persist(event)
            self.adapter.enqueue(event)
        except Exception as exc:
            logger.error("Groq protocol normalization failed for %s: %s", entry["agent"], exc)
            fallback = {
                "type": "protocol_card",
                "agent": entry["agent"],
                "room_id": entry["room_id"],
                "source_ts": entry["ts"],
                "ts": datetime.now(timezone.utc).isoformat(),
                "card": self._normalizer.fallback(
                    entry["agent"], entry["text"]
                ).model_dump(mode="json"),
            }
            self._normalizer.persist(fallback)
            self.adapter.enqueue(fallback)

    async def set_max_turns(self, requested: int) -> int:
        """Update the live limit while enforcing the application-wide ceiling."""
        self.max_turns = max(1, min(requested, HARD_MAX_TURNS))
        logger.info("Session turn limit configured to %d", self.max_turns)
        if self.turn_count >= self.max_turns and not self._report_triggered:
            self._report_triggered = True
            asyncio.create_task(self._generate_report())
            asyncio.create_task(self._finish_cards_then_stop())
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
    start_agent = StartAgent()
    session.bind_room_closer(start_agent.close_room)
    adapter.configure_start_agent(
        start_agent.send_ticker,
        start_agent.create_room,
        start_agent.close_room,
        session.start_runtime,
    )

    logger.info(
        "AlphaSign starting — %d agents, max %d turns, log=%s, pdf=%s",
        3, session.max_turns, CONV_LOG_PATH, PDF_OUTPUT_PATH,
    )

    # Start the HTTP adapter server as a background task
    adapter_task = asyncio.create_task(adapter.serve(), name="adapter")

    # Start all three agents concurrently — each runs its own Band WebSocket loop
    await session.start_runtime()
    agent_tasks = session._agent_tasks

    all_tasks = [adapter_task] + agent_tasks

    # Graceful shutdown on Ctrl-C or SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("Received %s — shutting down…", sig_name)
        for task in [adapter_task, *session._agent_tasks]:
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
        shutdown_tasks = [adapter_task, *session._agent_tasks]
        for task in shutdown_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        logger.info("AlphaSign stopped.")


if __name__ == "__main__":
    asyncio.run(run())
