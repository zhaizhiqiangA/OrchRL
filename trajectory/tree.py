from __future__ import annotations

import asyncio
import logging

from .backend import InferenceBackend
from .datatypes import BranchResult, InteractionRecord, TreeEpisodeResult
from .pipe import AgentPipe, AgentPipeConfig
from .replay_cache import ReplayCache
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def tree_rollout(
    prompt: str,
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    k_branches: int = 3,
    max_concurrent_branches: int | None = None,
) -> TreeEpisodeResult:
    if k_branches < 1:
        raise ValueError("k_branches must be >= 1")
    if max_concurrent_branches is not None and max_concurrent_branches < 1:
        raise ValueError("max_concurrent_branches must be >= 1 when provided")

    pilot_pipe = AgentPipe(config=config, backend=backend)
    pilot_result = await pilot_pipe.run(prompt=prompt, reward_provider=reward_provider)
    pilot_buffer = _sorted_buffer(pilot_pipe.last_buffer())
    pilot_total_turns = len(pilot_buffer)

    if pilot_result.status != "success" or not pilot_buffer:
        return TreeEpisodeResult(
            pilot_result=pilot_result,
            branch_results=[],
            prompt=prompt,
            tree_metadata={
                "n_branch_points": 0,
                "k_branches": k_branches,
                "total_branches_collected": 0,
                "pilot_total_turns": pilot_total_turns,
            },
        )

    semaphore = (
        asyncio.Semaphore(max_concurrent_branches)
        if max_concurrent_branches is not None
        else None
    )

    async def run_branch(record: InteractionRecord, global_position: int) -> BranchResult | None:
        cache = ReplayCache.from_buffer(
            pilot_buffer,
            branch_at_global_position=global_position,
        )
        branch_pipe = AgentPipe(config=config, backend=backend, replay_cache=cache)

        async def execute() -> BranchResult | None:
            try:
                branch_result = await branch_pipe.run(
                    prompt=prompt,
                    reward_provider=reward_provider,
                    allow_partial=True,
                )
            except Exception as exc:
                _LOGGER.warning(
                    "tree_rollout dropped failed branch at position %s for %s[%s]: %s",
                    global_position,
                    record.agent_role,
                    record.turn_index,
                    exc,
                )
                return None

            if branch_result.status != "success":
                return None

            return BranchResult(
                episode_result=branch_result,
                branch_turn=global_position,
                branch_agent_role=record.agent_role,
                parent_episode_id=pilot_result.trajectory.episode_id,
            )

        if semaphore is None:
            return await execute()

        async with semaphore:
            return await execute()

    tasks = [
        run_branch(record, global_position)
        for global_position, record in enumerate(pilot_buffer)
        for _ in range(k_branches)
    ]
    branch_results = [result for result in await asyncio.gather(*tasks) if result is not None]

    return TreeEpisodeResult(
        pilot_result=pilot_result,
        branch_results=branch_results,
        prompt=prompt,
        tree_metadata={
            "n_branch_points": pilot_total_turns,
            "k_branches": k_branches,
            "total_branches_collected": len(branch_results),
            "pilot_total_turns": pilot_total_turns,
        },
    )


def _sorted_buffer(buffer: list[InteractionRecord]) -> list[InteractionRecord]:
    return sorted(buffer, key=lambda record: record.timestamp)
