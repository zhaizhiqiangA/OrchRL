import asyncio

import httpx
import pytest
import pytest_asyncio

from trajectory import ChatRenderer
from trajectory.backend import InferenceBackend
from trajectory.datatypes import ModelMappingEntry, ModelRequest, ModelResponse
from trajectory.monitor import ModelMonitor

pytestmark = pytest.mark.asyncio


class RecordingBackend(InferenceBackend):
    def __init__(self, *, raise_error: bool = False):
        self.raise_error = raise_error
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.raise_error:
            raise RuntimeError("backend boom")
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[11, 22, 33],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


class OrderedDelayBackend(InferenceBackend):
    def __init__(self) -> None:
        self.slow_started = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = request.messages[0]["content"]
        if content == "slow":
            self.slow_started.set()
            await asyncio.sleep(0.05)
        return ModelResponse(
            content=f"done:{content}",
            token_ids=None,
            logprobs=None,
            finish_reason="stop",
        )


class GateBackend(InferenceBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        await self.release.wait()
        return ModelResponse(
            content="done",
            token_ids=None,
            logprobs=None,
            finish_reason="stop",
        )


@pytest_asyncio.fixture
async def monitor_server():
    mapping = {
        "verifier": ModelMappingEntry(actual_model="m1"),
        "searcher": ModelMappingEntry(actual_model="m2"),
    }
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    yield monitor, backend, port
    await monitor.stop()


async def test_routes_by_model_field(monitor_server):
    _, backend, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "verifier",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.6,
            },
        )

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].agent_role == "verifier"
    assert response.json()["choices"][0]["message"]["content"] == "echo:verifier"


async def test_collects_interaction_record_to_buffer(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "searcher",
                "messages": [{"role": "user", "content": "find x"}],
                "top_p": 0.9,
            },
        )

    buf = monitor.get_buffer()
    assert len(buf) == 1
    record = buf[0]
    assert record.response_text == "echo:searcher"
    assert record.token_ids == [11, 22, 33]
    assert record.logprobs == [-0.1, -0.2, -0.3]
    assert record.turn_index == 0


async def test_turn_index_increments_for_same_agent(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

    turns = [r.turn_index for r in monitor.get_buffer()]
    assert turns == [0, 1, 2]


async def test_unknown_agent_returns_400(monitor_server):
    _, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "ghost", "messages": []},
        )

    assert response.status_code == 400


async def test_clear_buffer_works(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
        )

    assert len(monitor.get_buffer()) == 1
    monitor.clear_buffer()
    assert monitor.get_buffer() == []


async def test_invalid_json_returns_400():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                content="{bad-json",
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 400
    finally:
        await monitor.stop()


async def test_backend_exception_returns_502():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(raise_error=True), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )
        assert response.status_code == 502
    finally:
        await monitor.stop()


async def test_turn_index_assigned_by_arrival_order_under_concurrency():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    backend = OrderedDelayBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            slow_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "slow"}]},
                )
            )
            await backend.slow_started.wait()
            fast_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "fast"}]},
                )
            )
            slow_resp, fast_resp = await asyncio.gather(slow_task, fast_task)

        assert slow_resp.status_code == 200
        assert fast_resp.status_code == 200
        indices = {record.messages[0]["content"]: record.turn_index for record in monitor.get_buffer()}
        assert indices["slow"] == 0
        assert indices["fast"] == 1
    finally:
        await monitor.stop()


async def test_injects_actual_model_from_mapping_into_backend_request():
    mapping = {"verifier": ModelMappingEntry(actual_model="mapped-real-model")}
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                    "temperature": 0.2,
                },
            )

        assert response.status_code == 200
        assert backend.requests[0].generation_params["model"] == "mapped-real-model"
    finally:
        await monitor.stop()


async def test_injects_backend_url_override_from_mapping_into_backend_request():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1", backend_url="http://role-backend/")}
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

        assert response.status_code == 200
        assert backend.requests[0].generation_params["_backend_url"] == "http://role-backend/"
    finally:
        await monitor.stop()


async def test_clear_buffer_drops_inflight_response_after_clear():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    backend = GateBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            request_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
                )
            )
            await backend.started.wait()
            monitor.clear_buffer()
            backend.release.set()
            response = await request_task

        assert response.status_code == 200
        assert monitor.get_buffer() == []
    finally:
        await monitor.stop()


async def test_uses_role_specific_renderer_mapping():
    class RecordingRenderer(ChatRenderer):
        def __init__(self, prompt_ids):
            self._prompt_ids = list(prompt_ids)

        def render(self, messages, *, add_generation_prompt):
            return list(self._prompt_ids), {
                "add_generation_prompt": add_generation_prompt,
                "tokenizer_class": "RecordingRenderer",
            }

    mapping = {
        "verifier": ModelMappingEntry(actual_model="m1"),
        "searcher": ModelMappingEntry(actual_model="m2"),
    }
    backend = RecordingBackend()
    monitor = ModelMonitor(
        backend=backend,
        model_mapping=mapping,
        renderer={
            "verifier": RecordingRenderer([11, 22]),
            "searcher": RecordingRenderer([33, 44, 55]),
        },
    )
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            verifier_response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q1"}]},
            )
            searcher_response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "searcher", "messages": [{"role": "user", "content": "q2"}]},
            )

        assert verifier_response.status_code == 200
        assert searcher_response.status_code == 200
        assert backend.requests[0].prompt_ids == [11, 22]
        assert backend.requests[1].prompt_ids == [33, 44, 55]
    finally:
        await monitor.stop()
