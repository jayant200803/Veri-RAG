"""Provider-agnostic LLM interface.

Every provider implements `complete()`. Swapping backends is a one-line
env change (LLM_PROVIDER), which is what keeps this project at zero cost
and gives us an offline path for the live demo.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Base class for all LLM backends."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Return the model's text completion."""

    # ------------------------------------------------------------------
    def complete_json(self, prompt: str, *, system: str | None = None,
                      temperature: float = 0.0) -> dict[str, Any]:
        """Complete and parse JSON, tolerating markdown fences and prose.

        LLMs routinely wrap JSON in ```json fences or add a preamble.
        Rather than fail the whole agent run, we extract the first JSON
        object we can find. If that fails we return {} and the caller
        applies a safe fallback.
        """
        raw = self.complete(prompt, system=system, temperature=temperature)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        text = raw.strip()

        # Strip markdown code fences if present
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fall back to the first balanced {...} block
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}
