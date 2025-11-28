from __future__ import annotations

from typing import List, Dict, Any, Optional

from .generator import summarize_documents_to_insights


class ExtractionAgent:
    """
    Agent responsible for extracting key insights, methods, and results
    from retrieved documents.

    Uses async LiteLLM with Pydantic structured outputs.
    """

    async def run(
        self,
        docs: List[Dict[str, Any]],
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw documents into a structured list of insights.

        Each insight contains summary text plus lightweight method and result
        fields. Uses async LLM with Pydantic structured outputs.
        Uses PLANNING_MODEL from .env if llm_model is not provided.
        """
        return await summarize_documents_to_insights(
            docs, llm_model=llm_model, temperature=temperature
        )


