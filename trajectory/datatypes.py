from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelMappingEntry:
    actual_model: str | None = None
    backend_url: str | None = None


@dataclass
class ModelRequest:
    request_id: str
    agent_role: str
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]


@dataclass
class ModelResponse:
    content: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str


@dataclass
class InteractionRecord:
    agent_role: str
    turn_index: int
    timestamp: float
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    episode_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict[str, Any]]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    episode_id: str
    agent_trajectories: dict[str, list[TurnData]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    trajectory: EpisodeTrajectory
    rewards: dict[str, float | list[float]]
    final_reward: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    failure_info: dict[str, Any] | None = None


@dataclass
class BranchResult:
    episode_result: EpisodeResult
    branch_turn: int
    branch_agent_role: str
    parent_episode_id: str


@dataclass
class TreeEpisodeResult:
    pilot_result: EpisodeResult
    branch_results: list[BranchResult]
    prompt: str
    tree_metadata: dict[str, Any] = field(default_factory=dict)
