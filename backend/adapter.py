"""
backend/adapter.py

AlphaSign adapter — bridges Band agent messages to the Next.js frontend.

Exposes two endpoints on http://localhost:8765 (configurable via ADAPTER_PORT):

    GET  /stream          — Server-Sent Events stream.
                            Each agent message and the final report_ready event
                            are pushed here as JSON-encoded SSE data frames.
                            Frontend connects once and receives all live updates.

    GET  /messages        — Returns the full message history as JSON.
                            Useful for initial page load / reconnect.

    GET  /report          — Streams the PDF file when it's ready.
                            Returns 404 until the report has been generated.

    POST /reset           — Clears the in-memory queue and history.
                            Call this between sessions.

CORS is open (*) so the Next.js dev server (usually :3000) can connect freely.
Change ADAPTER_ALLOWED_ORIGIN in .env for production.

The adapter is started as a background asyncio task by main.py.
It can also be run standalone for testing:
    python adapter.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

logger = logging.getLogger("alphasign.adapter")

ADAPTER_PORT          = int(os.getenv("ADAPTER_PORT", "8765"))
ADAPTER_ALLOWED_ORIGIN = os.getenv("ADAPTER_ALLOWED_ORIGIN", "*")
PDF_OUTPUT_PATH       = Path(os.getenv("PDF_OUTPUT_PATH", "alphasign_report.pdf"))

# Maximum messages kept in memory (ring buffer — oldest dropped when full)
MAX_HISTORY = int(os.getenv("ADAPTER_MAX_HISTORY", "500"))


class AlphaSignAdapter:
    """
    In-process message bus.

    main.py calls adapter.enqueue(entry) from the agent callbacks.
    The HTTP server pushes those entries to connected SSE clients.
    """

    def __init__(self) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._subscribers: list[asyncio.Queue[dict | None]] = []
        self._app = self._build_app()
        self._get_turn_limit: Callable[[], int] | None = None
        self._set_turn_limit: Callable[[int], Awaitable[int]] | None = None
        self._send_ticker: Callable[[str], Awaitable[str]] | None = None
        self._create_room: Callable[[], Awaitable[dict[str, object]]] | None = None
        self._close_room: Callable[[str | None], Awaitable[dict[str, object]]] | None = None

    def configure_turn_limit(
        self,
        getter: Callable[[], int],
        setter: Callable[[int], Awaitable[int]],
    ) -> None:
        """Expose the active session's turn limit to the HTTP GUI."""
        self._get_turn_limit = getter
        self._set_turn_limit = setter

    def configure_start_agent(
        self,
        sender: Callable[[str], Awaitable[str]],
        room_creator: Callable[[], Awaitable[dict[str, object]]],
        room_closer: Callable[[str | None], Awaitable[dict[str, object]]],
    ) -> None:
        """Register the Band-only start agent used by GUI session creation."""
        self._send_ticker = sender
        self._create_room = room_creator
        self._close_room = room_closer

    # ── Public API (called by main.py / SessionState) ─────────────────────

    def enqueue(self, entry: dict[str, Any]) -> None:
        """
        Push one entry into the history ring buffer and fan it out to all
        connected SSE subscribers. Safe to call from the asyncio event loop.
        """
        if "ts" not in entry:
            entry["ts"] = datetime.now(timezone.utc).isoformat()
        self._history.append(entry)

        for q in list(self._subscribers):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                logger.warning("SSE subscriber queue full — dropping message")

    async def serve(self) -> None:
        """Run the aiohttp server. Awaited by main.py as a background task."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", ADAPTER_PORT)
        await site.start()
        logger.info("AlphaSign adapter listening on http://0.0.0.0:%d", ADAPTER_PORT)
        # Run forever (cancelled by main.py on shutdown)
        try:
            await asyncio.Future()
        finally:
            await runner.cleanup()

    # ── HTTP application ──────────────────────────────────────────────────

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/stream",   self._handle_stream)
        app.router.add_get("/messages", self._handle_messages)
        app.router.add_get("/report",   self._handle_report)
        app.router.add_get("/config",   self._handle_get_config)
        app.router.add_post("/config",  self._handle_set_config)
        app.router.add_post("/reset",   self._handle_reset)
        app.router.add_post("/api/sessions", self._handle_start_session)
        app.router.add_post("/api/rooms", self._handle_create_room)
        app.router.add_post("/api/rooms/close", self._handle_close_room)
        app.on_response_prepare.append(self._add_cors)
        return app

    # ── CORS ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _add_cors(request: web.Request, response: web.StreamResponse) -> None:
        response.headers["Access-Control-Allow-Origin"]  = ADAPTER_ALLOWED_ORIGIN
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    # ── Handlers ─────────────────────────────────────────────────────────

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        """
        Server-Sent Events endpoint.
        Sends all historical messages immediately on connect, then streams
        live updates as they arrive.
        """
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type":  "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            },
        )
        await response.prepare(request)

        # Send history replay so the client is caught up immediately
        for entry in list(self._history):
            await self._sse_send(response, entry)

        # Subscribe to live updates
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=200)
        self._subscribers.append(queue)
        logger.debug("SSE client connected (%d total)", len(self._subscribers))

        try:
            while True:
                entry = await queue.get()
                if entry is None:
                    break   # None is the shutdown sentinel
                await self._sse_send(response, entry)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._subscribers.remove(queue)
            logger.debug("SSE client disconnected (%d remaining)", len(self._subscribers))

        return response

    async def _handle_messages(self, request: web.Request) -> web.Response:
        """Return full message history as JSON array."""
        return web.json_response(list(self._history))

    async def _handle_report(self, request: web.Request) -> web.Response:
        """Stream the PDF report if it exists."""
        if not PDF_OUTPUT_PATH.exists():
            return web.Response(
                status=404,
                text=json.dumps({"error": "Report not yet generated"}),
                content_type="application/json",
            )
        pdf_bytes = PDF_OUTPUT_PATH.read_bytes()
        return web.Response(
            body=pdf_bytes,
            content_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="alphasign_report.pdf"',
            },
        )

    async def _handle_get_config(self, request: web.Request) -> web.Response:
        turn_limit = self._get_turn_limit() if self._get_turn_limit else 3
        return web.json_response({"max_turns": turn_limit, "hard_max_turns": 3})

    async def _handle_set_config(self, request: web.Request) -> web.Response:
        if self._set_turn_limit is None:
            return web.json_response({"error": "Turn configuration unavailable"}, status=503)
        try:
            payload = await request.json()
            requested = int(payload["max_turns"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return web.json_response({"error": "max_turns must be an integer from 1 to 3"}, status=400)
        if not 1 <= requested <= 3:
            return web.json_response({"error": "max_turns must be from 1 to 3"}, status=400)
        turn_limit = await self._set_turn_limit(requested)
        return web.json_response({"max_turns": turn_limit, "hard_max_turns": 3})

    async def _handle_reset(self, request: web.Request) -> web.Response:
        """Clear history and notify subscribers."""
        self._history.clear()
        reset_event = {"type": "reset", "ts": datetime.now(timezone.utc).isoformat()}
        for q in list(self._subscribers):
            try:
                q.put_nowait(reset_event)
            except asyncio.QueueFull:
                pass
        return web.json_response({"ok": True})

    async def _handle_start_session(self, request: web.Request) -> web.Response:
        if self._send_ticker is None:
            return web.json_response({"error": "Band kickoff unavailable"}, status=503)
        try:
            payload = await request.json()
            ticker = str(payload["ticker"]).strip().upper()
        except (KeyError, TypeError, json.JSONDecodeError):
            return web.json_response({"error": "ticker is required"}, status=400)
        if not re.fullmatch(r"[A-Z^][A-Z0-9.^-]{0,14}", ticker):
            return web.json_response({"error": "invalid ticker"}, status=400)

        try:
            room_id = await self._send_ticker(ticker)
        except Exception:
            logger.exception("Could not send Band kickoff for %s", ticker)
            return web.json_response({"error": "Band kickoff failed"}, status=502)

        now = datetime.now(timezone.utc).isoformat()
        event = {
            "type": "session_started",
            "session_id": room_id,
            "ticker": ticker,
            "ts": now,
        }
        self.enqueue(event)
        return web.json_response({
            "session_id": room_id,
            "ticker": ticker,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "agents": [],
            "latest_event_id": str(uuid.uuid4()),
            "report_ready": False,
        }, status=201)

    async def _handle_create_room(self, request: web.Request) -> web.Response:
        if self._create_room is None:
            return web.json_response({"error": "Start agent unavailable"}, status=503)
        try:
            room = await self._create_room()
        except Exception as exc:
            logger.exception("Could not create and populate Band room")
            return web.json_response(
                {"error": f"Band room creation failed: {exc}"}, status=502
            )
        self.enqueue({"type": "room_created", **room})
        return web.json_response(room, status=201)

    async def _handle_close_room(self, request: web.Request) -> web.Response:
        if self._close_room is None:
            return web.json_response({"error": "Room closing unavailable"}, status=503)
        try:
            payload = await request.json()
            room_id = str(payload.get("room_id", "")).strip() or None
            result = await self._close_room(room_id)
        except (TypeError, json.JSONDecodeError):
            return web.json_response({"error": "Invalid room close request"}, status=400)
        except Exception as exc:
            logger.exception("Could not close Band room")
            return web.json_response({"error": f"Band room close failed: {exc}"}, status=502)
        self.enqueue({"type": "room_closed", **result})
        return web.json_response(result)

    # ── SSE helpers ───────────────────────────────────────────────────────

    @staticmethod
    async def _sse_send(response: web.StreamResponse, data: dict) -> None:
        payload = f"data: {json.dumps(data)}\n\n"
        await response.write(payload.encode())

    # ── Graceful shutdown ─────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Signal all SSE clients to disconnect."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _standalone():
        adapter = AlphaSignAdapter()
        # Push a test message so the stream endpoint has something to show
        adapter.enqueue({
            "agent": "test",
            "text": "AlphaSign adapter running in standalone mode.",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        await adapter.serve()

    asyncio.run(_standalone())
