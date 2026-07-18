import json
import os
from pathlib import Path
from typing import Optional

from maxed_agent.trace import trace

STUB_DATA_PATH = Path(__file__).parent / "fixtures" / "data" / "stub_responses.json"


class ModelClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        stub_responses: Optional[dict[str, list[str]]] = None,
    ):
        self.provider = provider or os.getenv("MODEL_PROVIDER", "stub")
        self._stub_data: dict[str, list[str]] = {}
        self._stub_counters: dict[str, int] = {}
        self._client = None
        self.model = "stub"

        if self.provider == "stub":
            if stub_responses is not None:
                self._stub_data = stub_responses
            elif STUB_DATA_PATH.exists():
                self._stub_data = json.loads(STUB_DATA_PATH.read_text())
        elif self.provider == "ollama":
            from openai import OpenAI

            self.model = os.getenv("OLLAMA_MODEL", "gemma2:9b")
            self._client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
            )
        elif self.provider == "gemini":
            from openai import OpenAI

            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self._client = OpenAI(
                base_url=os.getenv("GEMINI_BASE_URL"),
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        else:
            raise ValueError(f"unknown MODEL_PROVIDER: {self.provider}")

    def complete(self, task: str, system: str, prompt: str) -> str:
        if self.provider == "stub":
            responses = self._stub_data.get(task, [""])
            idx = self._stub_counters.get(task, 0)
            output = responses[min(idx, len(responses) - 1)]
            self._stub_counters[task] = idx + 1
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            output = response.choices[0].message.content or ""

        trace(
            task=task,
            prompt=prompt,
            output=output,
            meta={"provider": self.provider, "model": self.model},
        )
        return output
