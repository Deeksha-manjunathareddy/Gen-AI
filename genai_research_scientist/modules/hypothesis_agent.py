from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

from .generator import generate_hypotheses_from_insights
from .models import Hypothesis  # Use Pydantic model instead of dataclass


class HypothesisAgent:
    """
    Agent that generates multiple hypotheses and selects the most promising one.

    Uses async LiteLLM with Pydantic structured outputs.
    """

    async def run(
        self,
        question: str,
        insights: List[Dict[str, Any]],
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Create 3–5 candidate hypotheses grounded in the extracted insights and
        return the full list plus the primary (best) hypothesis.

        Uses async LLM with Pydantic structured outputs.
        Uses PLANNING_MODEL from .env if llm_model is not provided.
        Currently, a simple heuristic selects the first hypothesis as primary.
        """
        hypotheses = await generate_hypotheses_from_insights(
            question, insights, llm_model=llm_model, temperature=temperature
        )
        best = hypotheses[0] if hypotheses else None
        return hypotheses, best


