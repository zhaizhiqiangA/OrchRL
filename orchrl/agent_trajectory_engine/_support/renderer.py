from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ChatRenderer:
    def __init__(self, tokenizer: Any, *, model_name: str | None = None) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, model_name: str | None = None) -> "ChatRenderer":
        return cls(tokenizer, model_name=model_name)

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[list[int] | None, dict[str, Any]]:
        try:
            prompt_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
            )
        except TypeError:
            prompt_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )

        return self._normalize_ids(prompt_ids), {
            "model_name": self._model_name,
            "add_generation_prompt": add_generation_prompt,
            "tokenizer_class": type(self._tokenizer).__name__,
        }

    @staticmethod
    def _normalize_ids(token_ids: Any) -> list[int] | None:
        if token_ids is None:
            return None
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            return None

        ids: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool):
                return None
            if not isinstance(token_id, int):
                return None
            ids.append(int(token_id))
        return ids if ids else None
