"""
Project configuration for the Self-Refining LLM reliability pipeline.

Change defaults here, or override at runtime (see main.py).
"""

# Local Ollama model name.
MODEL = "mistral"

# Default sampling temperature for generation.
TEMPERATURE = 0.3

# Number of refinement passes (Generator -> Critic -> Refiner, repeated).
# Example: REFLECTION_LOOPS=1 means one critique+refine pass after generation.
REFLECTION_LOOPS = 1

# Token limits (Ollama `num_predict`) to make the system faster.
# Smaller values => faster responses, potentially less detailed reasoning.
MAX_TOKENS_GENERATOR = 220
MAX_TOKENS_CRITIC = 150
MAX_TOKENS_REFINER = 220
MAX_TOKENS_EVALUATOR = 120

