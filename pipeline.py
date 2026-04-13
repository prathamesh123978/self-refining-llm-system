"""
Self-refining LLM reliability pipeline.

Reliability engineering perspective:
- Generator proposes an answer.
- Critic reviews and reports risks.
- Refiner improves answer using critique (hallucination mitigation).
- Evaluator estimates correctness probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

from agents import CriticAgent, EvaluatorAgent, GeneratorAgent, RefinerAgent
from config import MODEL, REFLECTION_LOOPS, TEMPERATURE
from ollama_client import OllamaClient


def _parse_confidence(evaluator_output: str) -> Dict[str, Any]:
    """
    Extract confidence/reliability fields from evaluator structured output.
    """

    confidence_raw = None
    reliability_raw = None

    m = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})", evaluator_output, flags=re.IGNORECASE)
    if m:
        confidence_raw = int(m.group(1))
        confidence_raw = max(0, min(100, confidence_raw))

    m = re.search(
        r"RELIABILITY\s*:\s*(HIGH|MEDIUM|LOW)", evaluator_output, flags=re.IGNORECASE
    )
    if m:
        reliability_raw = m.group(1).upper()

    return {
        "confidence": confidence_raw,
        "reliability": reliability_raw,
        "raw": evaluator_output.strip(),
    }


@dataclass
class Pipeline:
    """
    Orchestrates all agents sequentially.
    """

    reflection_loops: int = REFLECTION_LOOPS
    temperature: float = TEMPERATURE
    model: str = MODEL

    def __post_init__(self) -> None:
        self.client = OllamaClient(model=self.model)
        self.generator = GeneratorAgent(client=self.client, temperature=self.temperature)
        self.critic = CriticAgent(client=self.client, temperature=self.temperature)
        self.refiner = RefinerAgent(client=self.client, temperature=self.temperature)
        self.evaluator = EvaluatorAgent(client=self.client, temperature=self.temperature)

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the pipeline and return intermediate artifacts.
        """

        query = query.strip()
        if not query:
            return {"error": "Empty question provided."}

        try:
            # Stage 1: generation.
            initial_answer = self.generator.run(query)
        except Exception as e:
            return {"error": f"Generator stage failed: {e}"}

        critique: str = ""
        refined_answer: str = initial_answer

        # Reflection loops:
        # Repeat critique+refine to reduce errors/hallucinations and improve clarity.
        for _pass_idx in range(max(0, int(self.reflection_loops))):
            try:
                critique_input = f"QUESTION:\n{query}\n\nANSWER:\n{refined_answer}"
                critique = self.critic.run(critique_input)
            except Exception as e:
                return {"error": f"Critic stage failed: {e}"}

            try:
                refine_input = (
                    f"QUESTION:\n{query}\n\nANSWER:\n{refined_answer}\n\nCRITIQUE:\n{critique}"
                )
                refined_answer = self.refiner.run(refine_input)
            except Exception as e:
                return {"error": f"Refiner stage failed: {e}"}

        try:
            evaluator_input = f"QUESTION:\n{query}\n\nANSWER:\n{refined_answer}"
            evaluator_output = self.evaluator.run(evaluator_input)
        except Exception as e:
            return {"error": f"Evaluator stage failed: {e}"}

        confidence = _parse_confidence(evaluator_output)

        return {
            "initial_answer": initial_answer,
            "critique": critique,
            "refined_answer": refined_answer,
            "confidence": confidence,
        }

