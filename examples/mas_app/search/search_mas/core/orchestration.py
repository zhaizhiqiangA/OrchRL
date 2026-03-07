from __future__ import annotations


def append_team_context(team_context: str, agent_id: str, response: str) -> str:
    return team_context + f'\nThe output of "{agent_id}": {response}\n'


def append_tool_observation(team_context: str, observation: str) -> str:
    return team_context + f"\n{observation}\n"
