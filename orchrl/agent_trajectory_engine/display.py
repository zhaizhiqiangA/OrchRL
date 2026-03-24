"""Structured display formatter for trajectory data.

Formats EpisodeResult and TreeEpisodeResult into human-readable
multi-layer terminal output, designed for live demo presentations.
"""

from __future__ import annotations

import re
from typing import Any

from .datatypes import BranchResult, EpisodeResult, EpisodeTrajectory, TreeEpisodeResult, TurnData

# ── Box drawing ──────────────────────────────────────────────────────

_HEAVY_LINE = "═" * 62
_LIGHT_LINE = "─" * 62
_THIN_LINE = "╌" * 62


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _truncate(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _preview_ids(ids: list[int] | None, show: int = 5) -> str:
    if ids is None:
        return "None"
    if len(ids) <= show * 2:
        return f"{ids}  (len={len(ids)})"
    head = ", ".join(str(x) for x in ids[:show])
    tail = ", ".join(str(x) for x in ids[-show:])
    return f"[{head}, ..., {tail}]  (len={len(ids)})"


def _preview_logprobs(logprobs: list[float] | None, show: int = 5) -> str:
    if logprobs is None:
        return "None"
    if len(logprobs) <= show * 2:
        return f"[{', '.join(f'{v:.4f}' for v in logprobs)}]  (len={len(logprobs)})"
    head = ", ".join(f"{v:.4f}" for v in logprobs[:show])
    tail = ", ".join(f"{v:.4f}" for v in logprobs[-show:])
    return f"[{head}, ..., {tail}]  (len={len(logprobs)})"


def _sorted_turns(trajectory: EpisodeTrajectory) -> list[TurnData]:
    all_turns = [
        turn
        for turns in trajectory.agent_trajectories.values()
        for turn in turns
    ]
    return sorted(all_turns, key=lambda t: t.timestamp)


def _semantic_hint(turn: TurnData) -> str:
    text = turn.response_text
    verify = _extract_tag(text, "verify")
    search = _extract_tag(text, "search")
    answer = _extract_tag(text, "answer")
    if verify:
        return f"verify={verify}"
    if search:
        return f"query={_truncate(search, 40)}"
    if answer:
        return f"answer={_truncate(answer, 40)}"
    return ""


# ── Public API ───────────────────────────────────────────────────────


def format_episode_overview(result: EpisodeResult) -> str:
    """Layer 1: episode-level summary."""
    traj = result.trajectory
    turns = _sorted_turns(traj)

    agent_summary = "  ".join(
        f"{role}: {len(turns_list)} turn{'s' if len(turns_list) != 1 else ''}"
        for role, turns_list in traj.agent_trajectories.items()
    )

    prompt_hint = ""
    for turn in turns:
        for msg in turn.messages:
            if msg.get("role") == "user":
                prompt_hint = _truncate(msg.get("content", ""), 60)
                break
        if prompt_hint:
            break

    lines = [
        _HEAVY_LINE,
        "  Episode Overview",
        _HEAVY_LINE,
        f"  Episode ID   : {traj.episode_id}",
        f"  Status       : {result.status}",
        f"  Final Reward : {result.final_reward}",
    ]
    if prompt_hint:
        lines.append(f"  Prompt       : \"{prompt_hint}\"")
    lines += [
        "",
        f"  Agents       : {agent_summary}",
        f"  Total Turns  : {len(turns)} (chronological)",
        _HEAVY_LINE,
    ]
    return "\n".join(lines)


def format_turn_detail(turn: TurnData, index: int, total: int) -> str:
    """Layer 2: per-turn training-critical fields."""
    hint = _semantic_hint(turn)
    hint_str = f"  [{hint}]" if hint else ""

    response_preview = _truncate(turn.response_text, 80)
    response_chars = len(turn.response_text)

    lines = [
        _LIGHT_LINE,
        f"  Turn {index + 1}/{total}  │  agent: {turn.agent_role}  │  turn_index: {turn.turn_index}{hint_str}",
        _LIGHT_LINE,
    ]

    n_msgs = len(turn.messages) if turn.messages else 0
    msg_roles = [m.get("role", "?") for m in (turn.messages or [])]
    lines.append(f"  Messages     : {n_msgs} messages  [{' → '.join(msg_roles)}]")
    lines.append(f"  Response     : \"{response_preview}\"  ({response_chars} chars)")

    lines += [
        "",
        "  ── Training-Critical Fields ──",
        f"  prompt_ids   : {_preview_ids(turn.prompt_ids)}",
        f"  token_ids    : {_preview_ids(turn.token_ids)}",
        f"  logprobs     : {_preview_logprobs(turn.logprobs)}",
        f"  finish       : {turn.finish_reason}",
    ]

    if turn.replayed or turn.branch_phase:
        lines += [
            "",
            "  ── Branch Semantics ──",
            f"  replayed     : {turn.replayed}",
            f"  branch_phase : {turn.branch_phase}",
        ]

    routed = turn.routed_experts
    if routed is not None:
        lines += [
            "",
            f"  routed_experts : present (len={len(routed)})",
        ]

    return "\n".join(lines)


def format_training_mapping() -> str:
    """Layer 3: static table showing MATE → VERL field mapping."""
    lines = [
        _HEAVY_LINE,
        "  MATE → VERL Training Field Mapping",
        _HEAVY_LINE,
        "  MATE Field       │  VERL Consumer                   │  Purpose",
        "  ─────────────────┼──────────────────────────────────┼──────────────────────────",
        "  prompt_ids        │  DataProto.prompts (no re-tok)   │  PPO/GRPO prompt input",
        "  token_ids         │  DataProto.responses             │  response token sequence",
        "  logprobs          │  DataProto.old_log_probs         │  importance sampling ratio",
        "  agent_role        │  per-policy batch splitting      │  multi-agent routing",
        "  turn_index        │  per-turn record alignment       │  multi-turn credit assign",
        "  finish_reason     │  truncation / completion flag    │  sample quality control",
        "  replayed          │  tree branch skip predicate      │  avoid training on prefix",
        "  branch_phase      │  branch_point / post_branch      │  structured CA grouping",
        _HEAVY_LINE,
    ]
    return "\n".join(lines)


def format_episode(result: EpisodeResult, *, show_mapping: bool = False) -> str:
    """Full 2-layer display: overview + all turns."""
    traj = result.trajectory
    turns = _sorted_turns(traj)

    parts = [format_episode_overview(result)]
    for idx, turn in enumerate(turns):
        parts.append(format_turn_detail(turn, idx, len(turns)))

    if show_mapping:
        parts.append("")
        parts.append(format_training_mapping())

    return "\n".join(parts)


def format_tree_overview(result: TreeEpisodeResult) -> str:
    """Summary for a tree rollout result."""
    meta = result.tree_metadata
    n_branch_points = meta.get("n_branch_points", 0)
    k = meta.get("k_branches", 0)
    collected = meta.get("total_branches_collected", 0)

    lines = [
        _HEAVY_LINE,
        "  Tree Rollout Overview",
        _HEAVY_LINE,
        f"  Prompt           : \"{_truncate(result.prompt, 60)}\"",
        f"  Pilot Status     : {result.pilot_result.status}",
        f"  Pilot Reward     : {result.pilot_result.final_reward}",
        f"  Branch Points    : {n_branch_points}",
        f"  K per Point      : {k}",
        f"  Branches Collected: {collected}",
    ]

    if result.branch_results:
        lines.append("")
        lines.append("  Branches:")
        for i, br in enumerate(result.branch_results):
            phase_info = ""
            br_turns = _sorted_turns(br.episode_result.trajectory)
            for t in br_turns:
                if t.branch_phase == "branch_point":
                    phase_info = f"branch_point={t.agent_role}[{t.turn_index}]"
                    break
            lines.append(
                f"    [{i}] turn={br.branch_turn}  agent={br.branch_agent_role}  "
                f"reward={br.episode_result.final_reward}  {phase_info}"
            )

    lines.append(_HEAVY_LINE)
    return "\n".join(lines)


def format_tree(result: TreeEpisodeResult, *, expand_pilot: bool = True, show_mapping: bool = False) -> str:
    """Full display for tree rollout: overview + optionally expand pilot."""
    parts = [format_tree_overview(result)]
    if expand_pilot:
        parts.append("")
        parts.append("  ── Pilot Episode Detail ──")
        parts.append(format_episode(result.pilot_result))
    if show_mapping:
        parts.append("")
        parts.append(format_training_mapping())
    return "\n".join(parts)
