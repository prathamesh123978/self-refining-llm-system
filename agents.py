"""
Agent implementations for the self-refining reliability pipeline.

Each agent has a single `run(input_text)` method as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ollama_client import OllamaClient
from prompts import (
    CRITIC_PROMPT_TEMPLATE,
    EVALUATOR_PROMPT_TEMPLATE,
    GENERATOR_PROMPT_TEMPLATE,
    REFINER_PROMPT_TEMPLATE,
)
from config import (
    MAX_TOKENS_CRITIC,
    MAX_TOKENS_EVALUATOR,
    MAX_TOKENS_GENERATOR,
    MAX_TOKENS_REFINER,
)


def _extract_block(text: str, start_marker: str) -> str:
    """
    Extract a block that begins with `start_marker` (line-based).

    The block ends at the next line that starts with an uppercase marker
    pattern like `SOMETHING:` or reaches the end of the string.
    """

    # Normalize line endings.
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Find the start position.
    idx = t.find(start_marker)
    if idx < 0:
        return ""

    # Start right after the marker line.
    remainder = t[idx + len(start_marker) :]
    remainder = remainder.lstrip("\n")

    # Terminate at the next marker line (e.g., QUESTION:, ANSWER:, CRITIQUE:).
    match = re.search(r"\n[A-Z][A-Z0-9_ ]+:\s*", "\n" + remainder)
    if match:
        return remainder[: match.start()].strip()

    return remainder.strip()


@dataclass
class GeneratorAgent:
    client: OllamaClient
    temperature: float = 0.3
    num_predict: int = MAX_TOKENS_GENERATOR

    def run(self, input_text: str) -> str:
        prompt = GENERATOR_PROMPT_TEMPLATE.format(question=input_text.strip())
        return self.client.generate(
            prompt=prompt, temperature=self.temperature, num_predict=self.num_predict
        )


@dataclass
class CriticAgent:
    client: OllamaClient
    temperature: float = 0.3
    num_predict: int = MAX_TOKENS_CRITIC

    def run(self, input_text: str) -> str:
        question = _extract_block(input_text, "QUESTION:")
        answer = _extract_block(input_text, "ANSWER:")

        prompt = CRITIC_PROMPT_TEMPLATE.format(question=question, answer=answer)
        return self.client.generate(
            prompt=prompt, temperature=self.temperature, num_predict=self.num_predict
        )


@dataclass
class RefinerAgent:
    client: OllamaClient
    temperature: float = 0.3
    num_predict: int = MAX_TOKENS_REFINER

    def run(self, input_text: str) -> str:
        question = _extract_block(input_text, "QUESTION:")
        answer = _extract_block(input_text, "ANSWER:")
        critique = _extract_block(input_text, "CRITIQUE:")

        prompt = REFINER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            critique=critique,
        )
        return self.client.generate(
            prompt=prompt, temperature=self.temperature, num_predict=self.num_predict
        )


@dataclass
class EvaluatorAgent:
    client: OllamaClient
    temperature: float = 0.3
    num_predict: int = MAX_TOKENS_EVALUATOR

    def run(self, input_text: str) -> str:
        question = _extract_block(input_text, "QUESTION:")
        answer = _extract_block(input_text, "ANSWER:")

        prompt = EVALUATOR_PROMPT_TEMPLATE.format(question=question, answer=answer)
        return self.client.generate(
            prompt=prompt, temperature=self.temperature, num_predict=self.num_predict
        )

