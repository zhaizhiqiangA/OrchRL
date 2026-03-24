from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import orchrl


class _DummyRemoteCaller:
    def remote(self, fn):
        return object()


class _DummyWorker:
    __ray_call__ = _DummyRemoteCaller()


async def _raise_gather_sentinel(*args, **kwargs):
    raise RuntimeError("gather sentinel")


def test_launch_servers_reaches_gather_for_multi_node_version_check(monkeypatch) -> None:
    dummy_replica = SimpleNamespace(
        workers=[_DummyWorker()],
        world_size=1,
        config=SimpleNamespace(data_parallel_size=1),
        nnodes=2,
        gpus_per_replica_node=1,
        replica_rank=0,
        is_reward_model=False,
        model_config={},
        server_class=None,
        rollout_mode=None,
        servers=[],
    )

    monkeypatch.setattr(orchrl.asyncio, "gather", _raise_gather_sentinel)

    with pytest.raises(RuntimeError, match="gather sentinel"):
        asyncio.run(orchrl.launch_servers(dummy_replica))
