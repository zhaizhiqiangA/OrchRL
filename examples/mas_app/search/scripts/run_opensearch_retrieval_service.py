from __future__ import annotations

import argparse
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query text.")
    topk: int | None = Field(default=None, ge=1, le=100, description="Number of docs to retrieve.")
    return_scores: bool = Field(default=True, description="Whether to include retrieval scores.")


class RetrievalService:
    def __init__(
        self,
        client: OpenSearch,
        index: str,
        fields: list[str],
        default_topk: int,
        request_timeout: float,
    ) -> None:
        self.client = client
        self.index = index
        self.fields = fields
        self.default_topk = default_topk
        self.request_timeout = request_timeout

    def retrieve(self, query: str, topk: int, return_scores: bool) -> dict[str, Any]:
        search_body: dict[str, Any] = {
            "size": topk,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.fields,
                    "type": "best_fields",
                }
            },
        }

        try:
            response = self.client.search(
                index=self.index,
                body=search_body,
                request_timeout=self.request_timeout,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenSearch search failed: {exc}") from exc

        hits = response.get("hits", {}).get("hits", [])
        passages: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source") or {}
            text = _extract_text(source, self.fields)
            if not text:
                continue

            entry: dict[str, Any] = {
                "id": str(hit.get("_id", "")),
                "document": {"contents": text},
            }
            if return_scores:
                entry["score"] = float(hit.get("_score") or 0.0)

            metadata = _extract_metadata(source)
            if metadata:
                entry["metadata"] = metadata

            passages.append(entry)

        return {
            "result": [passages],
            "metadata": {
                "provider": "opensearch",
                "index": self.index,
                "count": len(passages),
            },
        }

    def health(self) -> dict[str, Any]:
        try:
            ping_ok = bool(self.client.ping())
            index_exists = bool(self.client.indices.exists(index=self.index))
            return {
                "status": "ok" if ping_ok else "degraded",
                "opensearch_ping": ping_ok,
                "index": self.index,
                "index_exists": index_exists,
            }
        except Exception as exc:
            return {
                "status": "error",
                "opensearch_ping": False,
                "index": self.index,
                "index_exists": False,
                "error": str(exc),
            }


def _extract_text(source: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = source.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_metadata(source: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("title", "source", "url"):
        value = source.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a FastAPI retrieval service backed by OpenSearch for Search MAS."
    )
    parser.add_argument("--host", default="0.0.0.0", help="FastAPI host.")
    parser.add_argument("--port", type=int, default=18080, help="FastAPI port.")
    parser.add_argument("--opensearch-url", default="http://127.0.0.1:9200", help="OpenSearch endpoint.")
    parser.add_argument("--index", default="drmas_search_docs", help="OpenSearch index name.")
    parser.add_argument(
        "--fields",
        default="contents,content,text",
        help="Comma-separated text fields used for retrieval.",
    )
    parser.add_argument("--default-topk", type=int, default=3, help="Default topk when request topk is not set.")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="OpenSearch request timeout in seconds.")
    parser.add_argument("--username", default=None, help="OpenSearch username.")
    parser.add_argument("--password", default=None, help="OpenSearch password.")
    parser.add_argument(
        "--verify-certs",
        action="store_true",
        help="Enable SSL certificate verification when using https OpenSearch endpoint.",
    )
    parser.add_argument(
        "--auto-create-index",
        action="store_true",
        help="Create index with text mapping if it does not exist.",
    )
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    return parser


def _build_client(args: argparse.Namespace) -> OpenSearch:
    kwargs: dict[str, Any] = {
        "hosts": [args.opensearch_url],
        "verify_certs": bool(args.verify_certs),
    }
    if args.username:
        kwargs["http_auth"] = (args.username, args.password or "")
    return OpenSearch(**kwargs)


def _ensure_index(client: OpenSearch, index: str, fields: list[str], auto_create: bool) -> None:
    try:
        exists = bool(client.indices.exists(index=index))
    except Exception as exc:
        raise RuntimeError(f"Failed to check index `{index}`: {exc}") from exc

    if exists:
        return
    if not auto_create:
        logging.warning(
            "Index `%s` does not exist. Retrieval may fail until documents are indexed. "
            "Use --auto-create-index to create an empty index.",
            index,
        )
        return

    mappings = {"mappings": {"properties": {field: {"type": "text"} for field in fields}}}
    try:
        client.indices.create(index=index, body=mappings)
    except Exception as exc:
        raise RuntimeError(f"Failed to create index `{index}`: {exc}") from exc


def create_app(service: RetrievalService) -> FastAPI:
    app = FastAPI(title="Search MAS OpenSearch Retrieval Service")

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="`query` cannot be empty.")
        topk = req.topk or service.default_topk
        return service.retrieve(query=query, topk=topk, return_scores=req.return_scores)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    return app


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.default_topk < 1:
        raise ValueError("--default-topk must be >= 1")

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not fields:
        raise ValueError("--fields must contain at least one field")

    client = _build_client(args)
    _ensure_index(client=client, index=args.index, fields=fields, auto_create=bool(args.auto_create_index))

    service = RetrievalService(
        client=client,
        index=args.index,
        fields=fields,
        default_topk=int(args.default_topk),
        request_timeout=float(args.request_timeout),
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
