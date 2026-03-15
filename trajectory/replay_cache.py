from __future__ import annotations

import hashlib
import json
from typing import Any

from .datatypes import InteractionRecord, ModelResponse


def _messages_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ReplayCache:
    def __init__(
        self,
        entries: dict[tuple[str, int], ModelResponse],
        message_hashes: dict[tuple[str, int], str] | None = None,
    ) -> None:
        self._entries = entries
        self._message_hashes = message_hashes or {}

    @classmethod
    def from_buffer(
        cls,
        buffer: list[InteractionRecord],
        branch_at_global_position: int | None = None,
    ) -> "ReplayCache":
        sorted_buffer = sorted(buffer, key=lambda record: record.timestamp)
        if branch_at_global_position is not None:
            sorted_buffer = sorted_buffer[:branch_at_global_position]

        entries: dict[tuple[str, int], ModelResponse] = {}
        message_hashes: dict[tuple[str, int], str] = {}
        for record in sorted_buffer:
            key = (record.agent_role, record.turn_index)
            entries[key] = ModelResponse(
                content=record.response_text,
                token_ids=record.token_ids,
                logprobs=record.logprobs,
                finish_reason=record.finish_reason,
            )
            message_hashes[key] = _messages_hash(record.messages)

        return cls(entries, message_hashes=message_hashes)

    def lookup(
        self,
        agent_role: str,
        turn_index: int,
        messages: list[dict[str, Any]] | None = None,
    ) -> ModelResponse | None:
        key = (agent_role, turn_index)
        entry = self._entries.get(key)
        if entry is None:
            return None

        expected_hash = self._message_hashes.get(key)
        if expected_hash is not None and messages is not None and _messages_hash(messages) != expected_hash:
            return None

        return entry

    def __len__(self) -> int:
        return len(self._entries)
