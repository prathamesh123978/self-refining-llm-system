"""
Entry point for the self-refining LLM reliability pipeline.

Run:
    python main.py
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from config import MODEL, REFLECTION_LOOPS, TEMPERATURE
from pipeline import Pipeline


def _maybe_rich_print(text: str) -> None:
    """
    Print with optional rich support. Falls back to built-in print().
    """

    try:
        from rich import print as rich_print  # type: ignore

        rich_print(text)
    except Exception:
        print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-refining reliability pipeline (Ollama + Mistral).")
    parser.add_argument(
        "--reflection-loops",
        type=int,
        default=REFLECTION_LOOPS,
        help="Number of critique+refine passes (advanced reflection feature).",
    )
    args = parser.parse_args()

    try:
        question = input("Enter question:\n").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return 0

    pipeline = Pipeline(
        reflection_loops=args.reflection_loops,
        temperature=TEMPERATURE,
        model=MODEL,
    )

    try:
        result: Dict[str, Any] = pipeline.run(question)
    except Exception as e:
        print(f"Ollama/pipeline failure: {e}")
        return 1

    if "error" in result:
        print(f"Pipeline error: {result['error']}")
        return 1

    initial_answer = result.get("initial_answer", "")
    critique = result.get("critique", "")
    refined_answer = result.get("refined_answer", "")
    confidence_obj = result.get("confidence", {}) or {}
    confidence_raw = confidence_obj.get("raw", "")

    _maybe_rich_print("========================")
    _maybe_rich_print("INITIAL ANSWER")
    _maybe_rich_print("========================")
    _maybe_rich_print(initial_answer)

    _maybe_rich_print("\n========================")
    _maybe_rich_print("CRITIQUE REPORT")
    _maybe_rich_print("========================")
    _maybe_rich_print(critique)

    _maybe_rich_print("\n========================")
    _maybe_rich_print("REFINED ANSWER")
    _maybe_rich_print("========================")
    _maybe_rich_print(refined_answer)

    _maybe_rich_print("\n========================")
    _maybe_rich_print("CONFIDENCE ANALYSIS")
    _maybe_rich_print("========================")
    _maybe_rich_print(confidence_raw)

    return 0


if __name__ == "__main__":
    sys.exit(main())

