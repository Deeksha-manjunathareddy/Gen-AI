"""
Pydantic models for structured LLM outputs across all agents.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Insight(BaseModel):
    """Structured representation of an insight extracted from a document."""
    
    id: str
    title: str
    summary: str
    methods: str
    results: str
    raw_text: str
    citation: str


class Hypothesis(BaseModel):
    """Structured representation of a research hypothesis."""
    
    id: str
    statement: str
    rationale: str
    testability: str
    risks: str


class HypothesisList(BaseModel):
    """List of hypotheses returned from LLM."""
    
    hypotheses: List[Hypothesis] = Field(..., description="List of 3-5 testable hypotheses")


class ExperimentPlan(BaseModel):
    """Structured representation of an experiment plan."""
    
    overview: str
    objectives_md: str = Field(..., description="Markdown formatted objectives")
    setup_md: str = Field(..., description="Markdown formatted experimental setup")
    metrics_md: str = Field(..., description="Markdown formatted metrics")
    risks_md: str = Field(..., description="Markdown formatted risks and mitigations")


class DraftSections(BaseModel):
    """Structured sections of a research draft."""
    
    abstract: str
    introduction: str
    research_gap_table_md: str
    related_work: str
    hypotheses: str
    novelty: str
    methodology: str
    architecture_pipeline: str
    expected_outcomes: str

