from __future__ import annotations

from ..datatypes import ModelRequest, ModelResponse


def validate_runtime_request(request: ModelRequest) -> None:
    if request.prompt_ids is None:
        raise ValueError("runtime prompt_ids are required on canonical token paths")


def validate_runtime_response(response: ModelResponse) -> None:
    if response.token_ids is None or not response.token_ids:
        raise ValueError("response token_ids must not be empty")
    if response.logprobs is not None and len(response.token_ids) != len(response.logprobs):
        raise ValueError("response token_ids/logprobs length mismatch")
