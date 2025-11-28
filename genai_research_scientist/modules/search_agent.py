from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from .llm_client import call_llm_structured
from .rag_pipeline import LocalRAGPipeline

load_dotenv()

# Get PLANNING_MODEL from environment
PLANNING_MODEL_ENV = os.getenv("PLANNING_MODEL")


class LiteratureDocument(BaseModel):
    """Structured representation of a literature document."""
    title: str
    abstract: str
    authors: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float = 1.0


class LiteratureSearchResult(BaseModel):
    """Structured result from LLM literature search."""
    documents: List[LiteratureDocument]


class SearchAgent:
    """
    Agent responsible for literature retrieval using either the local TF-IDF corpus
    or an external LLM synthesis step.

    The user selects the retrieval mode in the UI; this class simply routes the
    request to the appropriate backend.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        rag_pipeline: Optional[LocalRAGPipeline] = None,
    ):
        self.llm_model = llm_model or PLANNING_MODEL_ENV
        self.rag_pipeline = rag_pipeline

    def attach_rag_pipeline(self, rag_pipeline: Optional[LocalRAGPipeline]):
        """Allow callers to lazily inject a TF-IDF pipeline once available."""
        self.rag_pipeline = rag_pipeline

    async def _search_with_llm(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Use LLM to search for relevant literature based on the query.
        
        The LLM generates relevant literature documents that would be useful
        for answering the research question.
        """
        if not self.llm_model:
            # Fallback: return empty list if no model
            return []

        system_prompt = (
            "You are an expert research librarian. Given a research question, "
            "generate a list of relevant academic papers, articles, or research documents "
            "that would be useful for answering the question. "
            "For each document, provide: title, abstract, authors (if known), year, venue, and URL (if available). "
            "Generate realistic and relevant literature that would actually exist for this topic."
        )
        
        user_prompt = (
            f"Research question: {query}\n\n"
            f"Generate {top_k} relevant academic papers or research documents that would help answer this question. "
            "Include papers that cover different aspects of the topic. "
            "For each paper, provide a realistic title, abstract (2-3 sentences), and other metadata."
        )

        try:
            result = await call_llm_structured(
                model=self.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=LiteratureSearchResult,
                temperature=0.3,  # Slightly higher for more diverse results
                max_tokens=2000,
            )

            if result and result.documents:
                # Convert to expected format
                search_results = []
                for idx, doc in enumerate(result.documents):
                    # Combine abstract and title for text field
                    text = f"{doc.abstract}"
                    if doc.authors:
                        citation = f"{doc.authors} – {doc.title}"
                    else:
                        citation = doc.title
                    if doc.year:
                        citation += f" ({doc.year})"
                    if doc.venue:
                        citation += f" – {doc.venue}"

                    search_results.append({
                        "id": f"doc_{idx+1}",
                        "title": doc.title,
                        "text": text,
                        "source": doc.url or "LLM Generated Literature",
                        "summary": doc.abstract[:600] + ("..." if len(doc.abstract) > 600 else ""),
                        "score": doc.relevance_score,
                        "citation": citation,
                    })
                return search_results
        except Exception:
            # If LLM search fails, return empty list
            pass

        return []

    def _search_local(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Use the bundled TF-IDF pipeline for retrieval."""
        if not self.rag_pipeline:
            return []
        return self.rag_pipeline.search(query, top_k=top_k)

    async def run(
        self,
        query: str,
        top_k: int = 8,
        llm_model: Optional[str] = None,
        retrieval_mode: str = "llm",
    ) -> List[Dict[str, Any]]:
        """
        Run a literature search using the configured retrieval strategy.

        Args:
            query: Research question or search query
            top_k: Number of documents to retrieve
            llm_model: Optional model override (uses PLANNING_MODEL if not provided)
            retrieval_mode: "llm" for LLM-generated literature, "local" for TF-IDF

        Returns:
            List of documents with fields: id, title, text, source, summary, score, citation
        """
        if llm_model:
            self.llm_model = llm_model

        mode = (retrieval_mode or "llm").lower()
        if mode == "local":
            local_results = self._search_local(query, top_k=top_k)
            if local_results:
                return local_results
            # Fallback to LLM if corpus is empty
        return await self._search_with_llm(query, top_k=top_k)
