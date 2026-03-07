from __future__ import annotations

from typing import Any

from .llm import ChatGenerationConfig, OpenAICompatibleLLM


class BaseChatAgent:
    def __init__(
        self,
        name: str,
        prompt_template: str,
        llm: OpenAICompatibleLLM,
        generation_config: ChatGenerationConfig,
    ) -> None:
        self.name = name
        self.prompt_template = prompt_template
        self.llm = llm
        self.generation_config = generation_config

    def run(self, env_prompt: str, team_context: str, step: int) -> str:
        prompt = self.prompt_template.format(
            env_prompt=env_prompt,
            team_context=team_context,
            step=step,
        )
        messages = self.build_messages(prompt)
        response = self.llm.chat(messages, self.generation_config)
        print(f"{self.name}: {response.replace(chr(10), '\\n')}")
        return response

    def build_messages(self, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]
