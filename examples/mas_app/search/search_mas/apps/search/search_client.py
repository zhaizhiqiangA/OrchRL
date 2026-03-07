from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class SearchClientResult:
    query: str
    result_text: str
    status: str
    error: str | None = None
    raw_response: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSearchClient:
    def search(self, query: str) -> SearchClientResult:
        raise NotImplementedError


class DisabledSearchClient(BaseSearchClient):
    def search(self, query: str) -> SearchClientResult:
        result_text = json.dumps(
            {"result": "Search backend is disabled. Configure search.retrieval_service_url to enable retrieval."},
            ensure_ascii=False,
        )
        return SearchClientResult(
            query=query,
            result_text=result_text,
            status="disabled",
            metadata={"provider": "disabled"},
        )


class HttpSearchClient(BaseSearchClient):
    def __init__(
        self,
        retrieval_service_url: str,
        topk: int = 3,
        return_scores: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.retrieval_service_url = retrieval_service_url
        self.topk = topk
        self.return_scores = return_scores
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.session = requests.Session()

    def search(self, query: str) -> SearchClientResult:
        payload = {
            "query": query,
            "topk": self.topk,
            "return_scores": self.return_scores,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        last_err: Exception | None = None
        for i in range(self.max_retries):
            try:
                response = self.session.post(
                    self.retrieval_service_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                result_text = _format_api_response(data)
                return SearchClientResult(
                    query=query,
                    result_text=result_text,
                    status="success",
                    raw_response=data,
                    metadata={"http_status": response.status_code},
                )
            except Exception as err:  # pragma: no cover
                last_err = err
                if i < self.max_retries - 1:
                    time.sleep(self.retry_backoff_sec * (2**i))
                    continue
                break

        return SearchClientResult(
            query=query,
            result_text=json.dumps(
                {"result": f"Search request failed after retries: {last_err}"},
                ensure_ascii=False,
            ),
            status="error",
            error=str(last_err),
            raw_response=None,
            metadata={"provider": "http"},
        )


def build_search_client(search_cfg: dict[str, Any]) -> BaseSearchClient:
    provider = str(search_cfg.get("provider", "http")).lower()
    if provider in {"disabled", "none"}:
        return DisabledSearchClient()

    url = search_cfg.get("retrieval_service_url")
    if not url:
        return DisabledSearchClient()

    return HttpSearchClient(
        retrieval_service_url=url,
        topk=int(search_cfg.get("topk", 3)),
        return_scores=bool(search_cfg.get("return_scores", True)),
        timeout=float(search_cfg.get("timeout", 30.0)),
        max_retries=int(search_cfg.get("max_retries", 3)),
        retry_backoff_sec=float(search_cfg.get("retry_backoff_sec", 1.0)),
    )


def _format_api_response(api_response: Any) -> str:
    if not isinstance(api_response, dict):
        return json.dumps({"result": str(api_response)}, ensure_ascii=False)

    results = api_response.get("result")
    if results is None:
        return json.dumps({"result": "No search results found."}, ensure_ascii=False)

    if isinstance(results, str):
        return json.dumps({"result": results}, ensure_ascii=False)

    if not isinstance(results, list):
        return json.dumps({"result": str(results)}, ensure_ascii=False)

    per_query_results: list[str] = []
    for block in results:
        if isinstance(block, list):
            per_query_results.append(_passages_to_string(block))
        elif isinstance(block, dict):
            per_query_results.append(_single_passage_to_string(block))
        else:
            per_query_results.append(str(block))

    merged = "\n---\n".join([item for item in per_query_results if item.strip()])
    if not merged:
        merged = "No search results found."
    return json.dumps({"result": merged}, ensure_ascii=False)


def _passages_to_string(passages: list[Any]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(passages):
        line = _single_passage_to_string(item)
        if not line:
            continue
        lines.append(f"Doc {idx + 1}: {line}")
    return "\n".join(lines)


def _single_passage_to_string(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    document = item.get("document")
    if isinstance(document, dict):
        contents = document.get("contents") or document.get("content") or document.get("text")
        if contents is not None:
            return str(contents).strip()
    for key in ("contents", "content", "text"):
        if key in item:
            return str(item[key]).strip()
    return json.dumps(item, ensure_ascii=False)
