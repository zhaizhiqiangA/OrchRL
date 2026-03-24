from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from ._support.diagnostics import build_drift_artifact
from ._support.renderer import ChatRenderer
from ._support.replay_cache import ReplayCache
from ._support.validator import validate_runtime_request, validate_runtime_response
from .backend import BACKEND_URL_OVERRIDE_KEY, InferenceBackend
from .datatypes import InteractionRecord, ModelMappingEntry, ModelRequest


class ModelMonitor:
    def __init__(
        self,
        backend: InferenceBackend,
        model_mapping: dict[str, ModelMappingEntry],
        episode_id: str | None = None,
        replay_cache: ReplayCache | None = None,
        renderer: ChatRenderer | Mapping[str, ChatRenderer] | None = None,
    ) -> None:
        self._backend = backend
        self._model_mapping = model_mapping
        self._episode_id = episode_id or uuid.uuid4().hex
        self._replay_cache = replay_cache
        if isinstance(renderer, Mapping):
            self._renderer = dict(renderer)
        else:
            self._renderer = renderer

        self._buffer: list[InteractionRecord] = []
        self._turn_counters: dict[str, int] = {}
        self._buffer_generation = 0
        self._state_lock = threading.Lock()

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._port: int | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        if self._runner is not None:
            if self._port is None:
                raise RuntimeError("ModelMonitor is in inconsistent started state")
            return self._port

        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, host=host, port=port)
        await self._site.start()

        server = self._site._server
        if server is None or not server.sockets:
            await self.stop()
            raise RuntimeError("failed to bind ModelMonitor server socket")

        self._port = int(server.sockets[0].getsockname()[1])
        return self._port

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

        self._app = None
        self._runner = None
        self._site = None
        self._port = None

    def get_buffer(self) -> list[InteractionRecord]:
        with self._state_lock:
            return list(self._buffer)

    def clear_buffer(self) -> None:
        with self._state_lock:
            self._buffer.clear()
            self._turn_counters.clear()
            self._buffer_generation += 1

    async def _handle_chat_completions(self, http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": "request body must be a JSON object"}, status=400)

        agent_role = body.get("model")
        if not isinstance(agent_role, str) or agent_role not in self._model_mapping:
            return web.json_response({"error": f"unknown agent role: {agent_role}"}, status=400)
        mapping_entry = self._model_mapping[agent_role]

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return web.json_response({"error": "messages must be a list"}, status=400)

        generation_params = {k: v for k, v in body.items() if k not in {"model", "messages"}}
        if mapping_entry.actual_model is not None:
            generation_params["model"] = mapping_entry.actual_model
        if mapping_entry.backend_url is not None:
            generation_params[BACKEND_URL_OVERRIDE_KEY] = mapping_entry.backend_url

        with self._state_lock:
            turn_index = self._turn_counters.get(agent_role, 0)
            self._turn_counters[agent_role] = turn_index + 1
            generation_snapshot = self._buffer_generation

        request_id = uuid.uuid4().hex
        model_request = ModelRequest(
            request_id=request_id,
            agent_role=agent_role,
            messages=messages,
            generation_params=generation_params,
            sampling_fingerprint=dict(generation_params),
        )
        renderer = self._resolve_renderer(agent_role)
        if renderer is not None and model_request.prompt_ids is None:
            prompt_ids, render_fingerprint = renderer.render(
                messages,
                add_generation_prompt=True,
            )
            model_request.prompt_ids = prompt_ids
            model_request.render_fingerprint = render_fingerprint

        response = None
        replayed = False
        if self._replay_cache is not None:
            response = self._replay_cache.lookup(agent_role, turn_index, messages)
            replayed = response is not None

        if response is None:
            try:
                if model_request.prompt_ids is not None:
                    validate_runtime_request(model_request)
                response = await self._backend.generate(model_request)
                if model_request.prompt_ids is not None:
                    validate_runtime_response(response)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=502)
        elif model_request.prompt_ids is not None:
            try:
                validate_runtime_response(response)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=502)

        metadata = dict(getattr(response, "runtime_metadata", {}))
        if response.routed_experts is not None:
            metadata["routed_experts"] = response.routed_experts
        if model_request.prompt_ids is not None:
            metadata.setdefault(
                "drift_artifact",
                build_drift_artifact(
                    messages=messages,
                    runtime_prompt_ids=response.prompt_ids or model_request.prompt_ids,
                    rerendered_prompt_ids=model_request.prompt_ids,
                    response_ids=response.token_ids,
                    response_logprobs=response.logprobs,
                    render_fingerprint=model_request.render_fingerprint,
                    sampling_fingerprint=model_request.sampling_fingerprint,
                ),
            )
        if replayed:
            metadata["replayed"] = True

        record = InteractionRecord(
            agent_role=agent_role,
            turn_index=turn_index,
            timestamp=time.time(),
            messages=messages,
            generation_params=generation_params,
            response_text=response.content,
            token_ids=response.token_ids,
            logprobs=response.logprobs,
            finish_reason=response.finish_reason,
            episode_id=self._episode_id,
            prompt_ids=response.prompt_ids or model_request.prompt_ids,
            metadata=metadata,
        )
        with self._state_lock:
            if generation_snapshot == self._buffer_generation:
                self._buffer.append(record)

        payload: dict[str, Any] = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent_role,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        return web.json_response(payload)

    def _resolve_renderer(self, agent_role: str) -> ChatRenderer | None:
        if self._renderer is None:
            return None
        if isinstance(self._renderer, ChatRenderer):
            return self._renderer
        return self._renderer.get(agent_role)
