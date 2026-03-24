from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from ._support.renderer import ChatRenderer
from .datatypes import ModelRequest, ModelResponse

BACKEND_URL_OVERRIDE_KEY = "_backend_url"
MODEL_OVERRIDE_KEY = "model"

_SUPPORTED_SAMPLING_PARAM_KEYS = {
    "best_of",
    "detokenize",
    "early_stopping",
    "frequency_penalty",
    "ignore_eos",
    "include_stop_str_in_output",
    "length_penalty",
    "max_new_tokens",
    "max_tokens",
    "min_p",
    "min_tokens",
    "n",
    "presence_penalty",
    "prompt_logprobs",
    "repetition_penalty",
    "seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop",
    "stop_token_ids",
    "temperature",
    "top_k",
    "top_p",
    "truncate_prompt_tokens",
}

_CONTROL_PARAM_KEYS = {
    BACKEND_URL_OVERRIDE_KEY,
    MODEL_OVERRIDE_KEY,
    "logprobs",
    "return_token_ids",
    "stream",
}

_LOGGER = logging.getLogger(__name__)


class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from an inference backend."""


class VLLMBackend(InferenceBackend):
    """Inference backend that forwards OpenAI-style requests to vLLM."""

    def __init__(
        self,
        backend_url: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
        tokenizer: Any | None = None,
        renderer: ChatRenderer | None = None,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.actual_model = actual_model
        self.timeout = timeout
        self._tokenizer = tokenizer
        if renderer is not None:
            self._renderer = renderer
        elif tokenizer is not None:
            self._renderer = ChatRenderer.from_tokenizer(tokenizer, model_name=actual_model)
        else:
            self._renderer = None

    @classmethod
    def with_tokenizer(
        cls,
        backend_url: str,
        model_path: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
    ) -> VLLMBackend:
        """Create a VLLMBackend with a local tokenizer for token_ids extraction."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            backend_url=backend_url,
            actual_model=actual_model or model_path,
            timeout=timeout,
            tokenizer=tokenizer,
            renderer=ChatRenderer.from_tokenizer(
                tokenizer,
                model_name=actual_model or model_path,
            ),
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        generation_params = dict(request.generation_params)
        backend_url_override = generation_params.pop(BACKEND_URL_OVERRIDE_KEY, None)
        target_backend_url = self.backend_url
        if isinstance(backend_url_override, str) and backend_url_override:
            target_backend_url = backend_url_override.rstrip("/")

        payload: dict[str, Any] = {
            "messages": request.messages,
            **generation_params,
        }
        payload["logprobs"] = True
        payload["return_token_ids"] = True
        if self.actual_model:
            payload["model"] = self.actual_model
        elif "model" not in payload:
            payload["model"] = request.agent_role

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{target_backend_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("malformed response: missing or invalid choices")

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        routed_experts = choice.get("routed_experts")

        token_ids: list[int] | None = None
        prompt_ids = request.prompt_ids
        render_fingerprint = dict(request.render_fingerprint)
        logprobs: list[float] | None = None
        logprobs_data = choice.get("logprobs")
        if isinstance(logprobs_data, dict) and isinstance(logprobs_data.get("content"), list):
            values: list[float] = []
            for token_info in logprobs_data["content"]:
                if not isinstance(token_info, dict):
                    continue
                value = token_info.get("logprob")
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
            logprobs = values

        raw_token_ids = choice.get("token_ids")
        if isinstance(raw_token_ids, list):
            token_ids = raw_token_ids
        elif self._tokenizer is not None and logprobs_data is not None:
            token_ids = self._extract_token_ids_from_logprobs(logprobs_data)
            if token_ids is None and content:
                token_ids = self._tokenizer.encode(content, add_special_tokens=False)

        if prompt_ids is None and self._renderer is not None:
            prompt_ids, render_fingerprint = self._renderer.render(
                request.messages,
                add_generation_prompt=True,
            )

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            prompt_ids=prompt_ids,
            routed_experts=routed_experts,
            runtime_metadata={"render_fingerprint": render_fingerprint},
        )

    def _extract_token_ids_from_logprobs(self, logprobs_data: dict[str, Any]) -> list[int] | None:
        """Extract token IDs from logprobs content using tokenizer vocabulary."""
        if self._tokenizer is None:
            return None
        content_list = logprobs_data.get("content")
        if not isinstance(content_list, list) or not content_list:
            return None
        ids: list[int] = []
        for entry in content_list:
            if not isinstance(entry, dict):
                continue
            token_str = entry.get("token")
            if not isinstance(token_str, str):
                continue
            tid = self._tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int):
                ids.append(tid)
            else:
                ids.append(self._tokenizer.unk_token_id or 0)
        return ids if ids else None


class VerlBackend(InferenceBackend):
    """Inference backend that sends canonical prompt IDs to direct VERL actors."""

    def __init__(
        self,
        server_manager: Any | None = None,
        *,
        tokenizer: Any | None = None,
        decoder: Callable[[list[int]], str] | None = None,
        policy_to_manager: Mapping[str, Any] | None = None,
        policy_to_tokenizer: Mapping[str, Any] | None = None,
        policy_to_decoder: Mapping[str, Callable[[list[int]], str]] | None = None,
        policy_to_actual_model: Mapping[str, str] | None = None,
    ) -> None:
        self._server_manager = server_manager
        self._tokenizer = tokenizer
        self._decoder = decoder
        self._policy_to_manager = dict(policy_to_manager or {})
        self._policy_to_tokenizer = dict(policy_to_tokenizer or {})
        self._policy_to_decoder = dict(policy_to_decoder or {})
        self._actual_model_to_policy: dict[str, str] = {}

        if policy_to_actual_model is not None:
            for policy_name, actual_model in policy_to_actual_model.items():
                if isinstance(actual_model, str) and actual_model:
                    self._actual_model_to_policy[actual_model] = policy_name

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.prompt_ids is None:
            raise ValueError("VerlBackend requires canonical prompt_ids")

        manager, tokenizer, decoder = self._resolve_runtime_handles(request)
        output = await manager.generate(
            request_id=request.request_id,
            prompt_ids=list(request.prompt_ids),
            sampling_params=self._build_sampling_params(request.generation_params),
        )
        token_ids = self._normalize_token_sequence(getattr(output, "token_ids", None))
        logprobs = self._normalize_logprobs(getattr(output, "log_probs", None))
        routed_experts = getattr(output, "routed_experts", None)
        raw_stop_reason = getattr(output, "stop_reason", None)
        content = getattr(output, "text", None)
        if not isinstance(content, str) or not content:
            content = self._decode_response_text(token_ids, tokenizer=tokenizer, decoder=decoder)

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=self._normalize_finish_reason(raw_stop_reason),
            prompt_ids=list(request.prompt_ids),
            routed_experts=routed_experts,
            runtime_metadata={
                "raw_stop_reason": raw_stop_reason,
                "render_fingerprint": dict(request.render_fingerprint),
                "sampling_fingerprint": dict(request.sampling_fingerprint),
            },
        )

    def _resolve_runtime_handles(
        self,
        request: ModelRequest,
    ) -> tuple[Any, Any | None, Callable[[list[int]], str] | None]:
        if not self._policy_to_manager:
            if self._server_manager is None:
                raise ValueError("VerlBackend is missing a server manager")
            return self._server_manager, self._tokenizer, self._decoder

        policy_name = self._resolve_policy_name(request)
        manager = self._policy_to_manager.get(policy_name)
        if manager is None:
            raise ValueError(f"No direct VERL manager configured for policy '{policy_name}'")

        tokenizer = self._policy_to_tokenizer.get(policy_name, self._tokenizer)
        decoder = self._policy_to_decoder.get(policy_name, self._decoder)
        return manager, tokenizer, decoder

    def _resolve_policy_name(self, request: ModelRequest) -> str:
        candidates: list[str] = []
        if isinstance(request.agent_role, str) and request.agent_role:
            candidates.append(request.agent_role)

        model_name = request.generation_params.get(MODEL_OVERRIDE_KEY)
        if isinstance(model_name, str) and model_name:
            candidates.append(model_name)

        for candidate in candidates:
            if candidate in self._policy_to_manager:
                return candidate

            policy_name = self._actual_model_to_policy.get(candidate)
            if policy_name is not None and policy_name in self._policy_to_manager:
                return policy_name

        if len(self._policy_to_manager) == 1:
            return next(iter(self._policy_to_manager))

        raise ValueError(f"Unable to resolve policy for VerlBackend request from candidates={candidates}")

    def _decode_response_text(
        self,
        token_ids: list[int] | None,
        *,
        tokenizer: Any | None,
        decoder: Callable[[list[int]], str] | None,
    ) -> str:
        if not token_ids:
            return ""
        if decoder is not None:
            return decoder(token_ids)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        raise ValueError("VerlBackend requires a tokenizer or decoder to recover text from token_ids")

    @staticmethod
    def _normalize_token_sequence(raw_token_ids: Any) -> list[int] | None:
        if raw_token_ids is None:
            return None
        if hasattr(raw_token_ids, "tolist"):
            raw_token_ids = raw_token_ids.tolist()
        return [int(token_id) for token_id in raw_token_ids]

    @staticmethod
    def _normalize_logprobs(raw_log_probs: Any) -> list[float] | None:
        if raw_log_probs is None:
            return None
        if hasattr(raw_log_probs, "tolist"):
            raw_log_probs = raw_log_probs.tolist()
        return [float(value) for value in raw_log_probs]

    @staticmethod
    def _build_sampling_params(generation_params: Mapping[str, Any]) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {"logprobs": True}

        for key, value in generation_params.items():
            if key in _CONTROL_PARAM_KEYS:
                if key == "stream" and value:
                    raise ValueError("VerlBackend does not support stream=True")
                continue

            if key not in _SUPPORTED_SAMPLING_PARAM_KEYS:
                raise ValueError(f"Unsupported generation parameter for VerlBackend: {key}")

            if key == "n" and value != 1:
                raise ValueError("VerlBackend currently supports only n=1")

            sampling_params[key] = value

        return sampling_params

    @staticmethod
    def _normalize_finish_reason(raw_stop_reason: Any) -> str:
        if raw_stop_reason in {"stop", "length", "content_filter", "tool_calls", "function_call"}:
            return str(raw_stop_reason)
        if raw_stop_reason in {"completed", "aborted", None}:
            return "stop"
        return "stop"
