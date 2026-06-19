"""Kick off the AlphaSign agent chain in its existing Band room."""

from __future__ import annotations

import os
from thenvoi import AgentTools, ThenvoiLink

from shared.config import load_agent_credentials


class StartAgent:
    """Uses the Band-only start agent to mention Narrative Analyst in a shared room."""

    def __init__(self) -> None:
        start_agent_id, start_agent_key = load_agent_credentials("start_agent")
        self._narrative_id, _ = load_agent_credentials("narrative_analyst")
        self._signal_id, _ = load_agent_credentials("signal_processing")
        self._latent_id, _ = load_agent_credentials("latent_state")
        # Executive report generation is local. A Band participant is optional
        # and only added when an ID is explicitly configured.
        self._executive_id = os.getenv("EXECUTIVE_AGENT_ID", "").strip()
        self._active_room_id: str | None = None
        self._link = ThenvoiLink(
            agent_id=start_agent_id,
            api_key=start_agent_key,
            ws_url=os.getenv("THENVOI_WS_URL", "wss://app.band.ai/api/v1/socket/websocket"),
            rest_url=os.getenv("THENVOI_REST_URL", "https://app.band.ai"),
        )

    async def send_ticker(self, ticker: str) -> str:
        """Send one kickoff message to the agents' existing shared room."""
        room_id, participants = await self._resolve_room()
        tools = AgentTools(room_id, self._link.rest, participants=participants)

        # AgentTools.send_message is the installed Band SDK equivalent of
        # band_send_message. The mention must be passed separately so Band
        # notifies (and wakes) the Narrative Analyst agent.
        mention = next(
            (
                p.get("handle")
                for p in participants
                if p.get("id") == self._narrative_id and p.get("handle")
            ),
            self._narrative_id,
        )
        await tools.send_message(ticker, mentions=[mention])
        return room_id

    async def create_room(self) -> dict[str, object]:
        """Create a Band room and populate it with every runtime agent."""
        tools = AgentTools("pending", self._link.rest)
        room_id = await tools.create_chatroom()
        tools = AgentTools(room_id, self._link.rest)
        for agent_id in self._room_agent_ids:
            await tools.add_participant(agent_id)

        participants = await self._participants(room_id)
        expected = set(self._room_agent_ids)
        present = {str(participant["id"]) for participant in participants}
        missing = expected - present
        if missing:
            raise RuntimeError(f"Band room is missing required agents: {sorted(missing)}")

        self._active_room_id = room_id
        return {"room_id": room_id, "participants": participants}

    async def close_room(self, room_id: str | None = None) -> dict[str, object]:
        """Close a room for this run by removing all runtime agents from it."""
        target_room_id = room_id or self._active_room_id
        if not target_room_id:
            raise RuntimeError("No active Band room is available to close")

        participants = await self._participants(target_room_id)
        present = {str(participant["id"]) for participant in participants}
        tools = AgentTools(target_room_id, self._link.rest, participants=participants)
        removed: list[dict[str, object]] = []
        for agent_id in self._room_agent_ids:
            if agent_id in present:
                removed.append(await tools.remove_participant(agent_id))

        if self._active_room_id == target_room_id:
            self._active_room_id = None
        return {"room_id": target_room_id, "closed": True, "removed": removed}

    async def _resolve_room(self) -> tuple[str, list[dict[str, str | None]]]:
        if self._active_room_id:
            return self._active_room_id, await self._participants(self._active_room_id)

        configured_room = os.getenv("BAND_ROOM_ID", "").strip()
        if configured_room:
            participants = await self._participants(configured_room)
            if not any(p["id"] == self._narrative_id for p in participants):
                raise RuntimeError(
                    "BAND_ROOM_ID does not contain the Narrative Analyst agent"
                )
            return configured_room, participants

        response = await self._link.rest.agent_api_chats.list_agent_chats(
            page=1, page_size=50
        )
        rooms = sorted(response.data, key=lambda room: room.updated_at, reverse=True)
        for room in rooms:
            participants = await self._participants(room.id)
            if any(p["id"] == self._narrative_id for p in participants):
                self._active_room_id = room.id
                return room.id, participants

        # A fresh deployment has no shared rooms yet. Create and populate one
        # instead of requiring a manual trip through the Band dashboard.
        room = await self.create_room()
        return str(room["room_id"]), room["participants"]  # type: ignore[return-value]

    @property
    def _room_agent_ids(self) -> tuple[str, ...]:
        agent_ids = (
            self._signal_id,
            self._narrative_id,
            self._latent_id,
        )
        return agent_ids + ((self._executive_id,) if self._executive_id else ())

    async def _participants(self, room_id: str) -> list[dict[str, str | None]]:
        response = await self._link.rest.agent_api_participants.list_agent_chat_participants(
            room_id
        )
        return [
            {
                "id": participant.id,
                "name": participant.name,
                "handle": participant.handle,
                "type": str(participant.type),
            }
            for participant in response.data
        ]
