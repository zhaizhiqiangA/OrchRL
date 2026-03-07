from __future__ import annotations

import re
from dataclasses import dataclass

from ...core.agent import BaseChatAgent

TAG_FLAGS = re.IGNORECASE | re.DOTALL


def extract_tag_content(text: str, tag_name: str) -> str | None:
    pattern = re.compile(rf"<{tag_name}>(.*?)</{tag_name}>", TAG_FLAGS)
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


class SearchAgent(BaseChatAgent):
    @staticmethod
    def extract_search_query(text: str) -> str | None:
        return extract_tag_content(text, "search")


class AnswerAgent(BaseChatAgent):
    @staticmethod
    def extract_answer(text: str) -> str | None:
        return extract_tag_content(text, "answer")


@dataclass
class VerifierDecision:
    decision: str
    approved: bool


class VerifierAgent(BaseChatAgent):
    @staticmethod
    def parse_decision(text: str) -> VerifierDecision:
        lower = text.lower()
        if "<verify>yes</verify>" in lower:
            return VerifierDecision(decision="yes", approved=True)
        if "<verify>no</verify>" in lower:
            return VerifierDecision(decision="no", approved=False)
        # Keep parity with DrMAS: fallback to "approved".
        return VerifierDecision(decision="yes_by_default", approved=True)
