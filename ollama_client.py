"""
Minimal wrapper around the Ollama Python library.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional


try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None


@dataclass
class OllamaClient:
    """
    A tiny client wrapper.

    This exists mainly to centralize Ollama error handling and to keep the rest
    of the codebase clean and testable.
    """

    model: str

    def generate(
        self, prompt: str, temperature: float = 0.3, num_predict: Optional[int] = None
    ) -> str:
        """
        Generate text with `ollama.chat()` using the provided prompt.

        Args:
            prompt: Full prompt text to send as a user message.
            temperature: Sampling temperature.

        Returns:
            Model response content as a string.
        """

        if ollama is None:  # pragma: no cover
            raise RuntimeError(
                "Ollama Python library is not available. Install it with: pip install ollama"
            )

        try:
            options = {"temperature": temperature}
            # `num_predict` limits how many tokens Ollama generates (speed/length control).
            if num_predict is not None:
                options["num_predict"] = int(num_predict)

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat() call failed: {e}") from e

        # The Ollama client returns a dict with message content in response["message"]["content"].
        try:
            return response["message"]["content"].strip()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Unexpected Ollama response format: {e}; response={response}") from e

