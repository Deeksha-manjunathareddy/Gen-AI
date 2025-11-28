from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

from .generator import create_structured_draft


class DraftAgent:
    """
    Agent that assembles a structured research-style draft from all upstream
    reasoning steps (insights, hypotheses, experiment plan).

    Uses async LiteLLM with Pydantic structured outputs.
    """

    async def run(
        self,
        question: str,
        insights: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        best_hypothesis: Dict[str, Any],
        experiment_plan: Dict[str, Any],
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Tuple[str, List[str]]:
        """
        Return a Markdown-formatted draft and a list of flattened citation
        strings suitable for rendering in the UI.

        Uses async LLM with Pydantic structured outputs.
        Uses PLANNING_MODEL from .env if llm_model is not provided.
        """
        return await create_structured_draft(
            question,
            insights,
            hypotheses,
            best_hypothesis,
            experiment_plan,
            llm_model=llm_model,
            temperature=temperature,
        )


