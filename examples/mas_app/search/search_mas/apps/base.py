from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceStep:
    loop_index: int
    agent_id: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MASRunResult:
    question: str
    final_response: str
    final_answer: str | None
    approved: bool
    trace: list[TraceStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trace"] = [asdict(step) for step in self.trace]
        return payload


class BaseMASApplication:
    def solve(self, question: str) -> MASRunResult:
        raise NotImplementedError
