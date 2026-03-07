from __future__ import annotations

from typing import Any

from ..base import BaseMASApplication, MASRunResult
from ...core.llm import (
    build_generation_config,
    build_llm_backend,
    resolve_agent_llm_config,
)
from .agents import AnswerAgent, SearchAgent, VerifierAgent
from .orchestrator import SearchRouterOrchestrator
from .prompts import ANSWER_PROMPT, SEARCH_PROMPT, VERIFIER_PROMPT
from .search_client import build_search_client


class SearchMASApplication(BaseMASApplication):
    """Inference-only Search MAS app.

    Flow:
    Verifier(router) -> Search (if no) / Answer (if yes).
    """

    def __init__(self, orchestrator: SearchRouterOrchestrator) -> None:
        self.orchestrator = orchestrator

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SearchMASApplication":
        llm_cfg = config.get("llm", {})
        if not isinstance(llm_cfg, dict) or not llm_cfg:
            raise ValueError("Config missing `llm` section")

        agents_cfg = config.get("agents", {})
        if not isinstance(agents_cfg, dict):
            raise ValueError("Config `agents` must be a dict")
        verifier_cfg = agents_cfg.get("verifier", {})
        search_cfg = agents_cfg.get("searcher", {})
        answer_cfg = agents_cfg.get("answerer", {})
        if not isinstance(verifier_cfg, dict):
            raise ValueError("Config `agents.verifier` must be a dict")
        if not isinstance(search_cfg, dict):
            raise ValueError("Config `agents.searcher` must be a dict")
        if not isinstance(answer_cfg, dict):
            raise ValueError("Config `agents.answerer` must be a dict")

        verifier_llm_cfg = resolve_agent_llm_config(llm_cfg, verifier_cfg, "verifier")
        search_llm_cfg = resolve_agent_llm_config(llm_cfg, search_cfg, "searcher")
        answer_llm_cfg = resolve_agent_llm_config(llm_cfg, answer_cfg, "answerer")

        verifier = VerifierAgent(
            name="Verifier Agent",
            prompt_template=VERIFIER_PROMPT,
            llm=build_llm_backend(verifier_llm_cfg),
            generation_config=build_generation_config(verifier_llm_cfg, verifier_cfg),
        )
        searcher = SearchAgent(
            name="Search Agent",
            prompt_template=SEARCH_PROMPT,
            llm=build_llm_backend(search_llm_cfg),
            generation_config=build_generation_config(search_llm_cfg, search_cfg),
        )
        answerer = AnswerAgent(
            name="Answer Agent",
            prompt_template=ANSWER_PROMPT,
            llm=build_llm_backend(answer_llm_cfg),
            generation_config=build_generation_config(answer_llm_cfg, answer_cfg),
        )

        app_cfg = config.get("application", {})
        orchestrator = SearchRouterOrchestrator(
            verifier=verifier,
            searcher=searcher,
            answerer=answerer,
            search_client=build_search_client(config.get("search", {})),
            max_turns=int(app_cfg.get("max_turns", 4)),
            force_final_answer_on_max_turn=bool(app_cfg.get("force_final_answer_on_max_turn", True)),
        )
        return cls(orchestrator=orchestrator)

    def solve(self, question: str) -> MASRunResult:
        out = self.orchestrator.run(question)
        return MASRunResult(
            question=question,
            final_response=out.final_response,
            final_answer=out.final_answer,
            approved=out.router_approved,
            trace=out.trace,
            metadata={"loop_count": out.loop_count},
        )
