from __future__ import annotations

from typing import List, Dict, Any, Optional

from .generator import design_experiment_from_hypothesis


class ExperimentAgent:
    """
    Agent that transforms a chosen hypothesis into a concrete experiment plan.

    Uses async LiteLLM with Pydantic structured outputs.
    """

    async def run(
        self,
        best_hypothesis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate an experiment specification including objectives, setup,
        metrics, and risks.

        Uses async LLM with Pydantic structured outputs.
        Uses PLANNING_MODEL from .env if llm_model is not provided.
        """
        if not best_hypothesis:
            return {}
        return await design_experiment_from_hypothesis(
            best_hypothesis, insights, llm_model=llm_model, temperature=temperature
        )


