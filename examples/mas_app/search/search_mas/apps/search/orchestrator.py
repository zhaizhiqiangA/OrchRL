from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base import TraceStep
from ...core.orchestration import append_team_context, append_tool_observation
from .agents import AnswerAgent, SearchAgent, VerifierAgent
from .search_client import BaseSearchClient, SearchClientResult


@dataclass
class SearchOrchestrationOutput:
    final_response: str
    final_answer: str | None
    router_approved: bool
    trace: list[TraceStep]
    loop_count: int


class SearchRouterOrchestrator:
    def __init__(
        self,
        verifier: VerifierAgent,
        searcher: SearchAgent,
        answerer: AnswerAgent,
        search_client: BaseSearchClient,
        max_turns: int = 4,
        force_final_answer_on_max_turn: bool = True,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.verifier = verifier
        self.searcher = searcher
        self.answerer = answerer
        self.search_client = search_client
        self.max_turns = max_turns
        self.force_final_answer_on_max_turn = force_final_answer_on_max_turn

    def run(self, question: str) -> SearchOrchestrationOutput:
        team_context = ""
        trace: list[TraceStep] = []
        final_response = ""
        final_answer: str | None = None
        router_approved = False
        loop_count = 0

        for loop_i in range(self.max_turns):
            loop_count = loop_i + 1
            verifier_output = self.verifier.run(question, team_context, step=loop_i)
            decision = self.verifier.parse_decision(verifier_output)
            team_context = append_team_context(team_context, self.verifier.name, verifier_output)
            trace.append(
                TraceStep(
                    loop_index=loop_i,
                    agent_id=self.verifier.name,
                    output=verifier_output,
                    metadata={"decision": decision.decision, "approved": decision.approved},
                )
            )

            if decision.approved:
                router_approved = True
                answer_output = self.answerer.run(question, team_context, step=loop_i)
                final_response = answer_output
                final_answer = self.answerer.extract_answer(answer_output)
                team_context = append_team_context(team_context, self.answerer.name, answer_output)
                trace.append(
                    TraceStep(
                        loop_index=loop_i,
                        agent_id=self.answerer.name,
                        output=answer_output,
                        metadata={"route": "answer", "final_answer": final_answer},
                    )
                )
                break

            search_output = self.searcher.run(question, team_context, step=loop_i)
            query = self.searcher.extract_search_query(search_output)
            team_context = append_team_context(team_context, self.searcher.name, search_output)
            trace.append(
                TraceStep(
                    loop_index=loop_i,
                    agent_id=self.searcher.name,
                    output=search_output,
                    metadata={"route": "search", "query": query},
                )
            )

            search_result = self._run_search(query=query)
            observation = f"<information>{search_result.result_text}</information>"
            team_context = append_tool_observation(team_context, observation)
            trace.append(
                TraceStep(
                    loop_index=loop_i,
                    agent_id="SearchTool",
                    output=observation,
                    metadata=asdict(search_result),
                )
            )

        if not final_response and self.force_final_answer_on_max_turn:
            answer_output = self.answerer.run(question, team_context, step=loop_count)
            final_response = answer_output
            final_answer = self.answerer.extract_answer(answer_output)
            trace.append(
                TraceStep(
                    loop_index=loop_count,
                    agent_id=self.answerer.name,
                    output=answer_output,
                    metadata={"route": "forced_answer", "final_answer": final_answer},
                )
            )

        return SearchOrchestrationOutput(
            final_response=final_response,
            final_answer=final_answer,
            router_approved=router_approved,
            trace=trace,
            loop_count=loop_count,
        )

    def _run_search(self, query: str | None) -> SearchClientResult:
        if query is None or not query.strip():
            return SearchClientResult(
                query=query or "",
                result_text='{"result": "Search query is empty."}',
                status="invalid_query",
                error="empty query",
                raw_response=None,
                metadata={},
            )
        return self.search_client.search(query.strip())
