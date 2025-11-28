from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from .llm_client import call_llm_structured
from .models import HypothesisList, ExperimentPlan, DraftSections

load_dotenv()

# Get PLANNING_MODEL from environment
PLANNING_MODEL_ENV = os.getenv("PLANNING_MODEL")
TARGET_WORD_COUNT = 1800


# Pydantic model for insight extraction
class InsightExtraction(BaseModel):
    """Temporary model for extracting insights from documents."""
    summary: str
    methods: str
    results: str


def _truncate(text: str, max_len: int = 400) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _build_gap_rows(
    question: str,
    insights: List[Dict[str, Any]],
    best_hypothesis: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    key_insights = insights[:3] if insights else []
    if key_insights:
        for ins in key_insights:
            gap = ins.get("title", "Unspecified gap")
            evidence = _truncate(ins.get("summary", ""), 180)
            opportunity = (
                best_hypothesis.get("statement", "")
                if best_hypothesis
                else f"Investigate unmet needs around {question}"
            )
            rows.append(
                {
                    "gap": gap,
                    "evidence": evidence or "Limited evidence captured in corpus.",
                    "opportunity": opportunity or "Design focused experiments to explore this space.",
                }
            )
    while len(rows) < 3:
        rows.append(
            {
                "gap": f"Unverified aspect #{len(rows)+1}",
                "evidence": f"No strong literature signal was found for this angle of '{question}'.",
                "opportunity": "Collect new data or broaden the corpus to validate this gap.",
            }
        )
    return rows[:5]


def _gap_table_md(question: str, insights: List[Dict[str, Any]], best_hypothesis: Optional[Dict[str, Any]]) -> str:
    rows = _build_gap_rows(question, insights, best_hypothesis)
    lines = ["| Research Gap | Supporting Evidence | Opportunity |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['gap']} | {row['evidence']} | {row['opportunity']} |")
    return "\n".join(lines)


def _novelty_text(question: str, best_hypothesis: Optional[Dict[str, Any]]) -> str:
    if best_hypothesis:
        return (
            f"The proposed direction centers on **{best_hypothesis.get('statement', '').strip()}**. "
            "This is novel because it aggregates the strongest insights from the literature review into "
            "a singular, testable proposition, emphasizing controlled evaluation and explicit research gap closure."
        )
    return (
        "The novelty stems from synthesizing sparse literature signals about "
        f"'{question}' into a unified hypothesis-experiment pair that has not yet been systematically studied."
    )


def _architecture_text(question: str, experiment_plan: Dict[str, Any]) -> str:
    steps = [
        "1. **Question parsing & retrieval** – normalize the research question and source evidence via local TF-IDF or LLM search.",
        "2. **Insight extraction** – convert raw documents into structured insights (summary, methods, results).",
        "3. **Hypothesis ranking** – generate 3–5 candidates, select the most testable and impactful statement.",
        "4. **Experiment design** – expand the primary hypothesis into objectives, setup, metrics, and risk controls.",
        "5. **Draft assembly** – stitch insights, hypotheses, and experiment guidance into a publication-style manuscript.",
    ]
    extra = experiment_plan.get("overview")
    if extra:
        steps.append(f"6. **Execution alignment** – ground implementation details in: {extra}")
    steps.append(
        f"7. **Human review loop** – domain experts vet the auto-generated plan before running any real-world study on '{question}'."
    )
    return "\n".join(steps)


def _supplemental_discussion(question: str, insights: List[Dict[str, Any]]) -> str:
    paragraphs = [
        "The corpus signals multiple secondary directions that merit documentation even if they were not prioritized "
        "in the main experiment plan.",
    ]
    for ins in insights[:4]:
        paragraphs.append(
            f"- **{ins.get('title', 'Additional Insight')}**: {ins.get('summary', 'Summary unavailable.')}"
        )
    paragraphs.append(
        "Future work should revisit these leads, collect more diverse datasets, and stress-test the hypotheses "
        f"across domains adjacent to '{question}'."
    )
    return "\n".join(paragraphs)


async def summarize_documents_to_insights(
    docs: List[Dict[str, Any]],
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert retrieved documents into structured 'insights' using async LLM with Pydantic.

    Uses PLANNING_MODEL from .env if llm_model is not provided.
    """
    insights: List[Dict[str, Any]] = []
    model = llm_model or PLANNING_MODEL_ENV

    # If no model is available, use fallback heuristics
    if not model:
        for idx, d in enumerate(docs):
            text = d.get("text", "")
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            summary = ". ".join(sentences[:2]) + ("." if sentences else "")
            
            insight = {
                "id": f"insight_{idx+1}",
                "title": d.get("title", f"Document {idx+1}"),
                "summary": _truncate(summary, 500) or _truncate(text, 500),
                "methods": "Heuristic extraction from text; replace with LLM-based method parsing in production.",
                "results": "High-level outcomes inferred from sample text; for demo purposes only.",
                "raw_text": text,
                "citation": f"{d.get('title', 'Unknown')} ({d.get('source', 'Sample')})",
            }
            insights.append(insight)
        return insights

    # Use LLM with structured output for each document
    for idx, d in enumerate(docs):
        text = d.get("text", "")
        
        system_prompt = (
            "You are an expert research assistant. Given a paper's text, "
            "extract a concise summary, key methods, and main results. "
            "Be precise and factual."
        )
        user_prompt = f"Paper title: {d.get('title', '')}\n\nFull text:\n{text[:6000]}"

        extracted = await call_llm_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=InsightExtraction,
            temperature=temperature,
            max_tokens=800,
        )

        if extracted:
            summary = extracted.summary.strip()
            methods = extracted.methods.strip()
            results = extracted.results.strip()
        else:
            # Fallback to heuristics
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            summary = ". ".join(sentences[:2]) + ("." if sentences else "")
            methods = "Heuristic extraction from text; replace with LLM-based method parsing in production."
            results = "High-level outcomes inferred from sample text; for demo purposes only."

        if not summary:
            summary = _truncate(text, 500)

        insight = {
            "id": f"insight_{idx+1}",
            "title": d.get("title", f"Document {idx+1}"),
            "summary": _truncate(summary, 500),
            "methods": methods,
            "results": results,
            "raw_text": text,
            "citation": f"{d.get('title', 'Unknown')} ({d.get('source', 'Sample')})",
        }
        insights.append(insight)

    return insights


async def generate_hypotheses_from_insights(
    question: str,
    insights: List[Dict[str, Any]],
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Generate 3–5 testable hypotheses using async LLM with Pydantic structured output.

    Uses PLANNING_MODEL from .env if llm_model is not provided.
    """
    model = llm_model or PLANNING_MODEL_ENV

    # Fallback template if no model
    if not model:
        base = (
            "Based on current literature around '{q}', "
            "we hypothesize that {variation}."
        )
        variations = [
            "introducing a carefully aligned large language model will improve task performance compared to existing baselines",
            "combining supervised finetuning with human preference optimization yields better alignment than either alone",
            "scaling data diversity has a larger impact on generalization than scaling model size alone",
            "incorporating explicit human feedback loops reduces harmful or low-quality outputs in real-world settings",
            "hybrid symbolic–neural approaches offer more controllable behavior than purely neural models",
        ]
        selected_vars = variations[: max(3, min(5, len(variations)))]

        hypotheses: List[Dict[str, Any]] = []
        for i, v in enumerate(selected_vars):
            statement = base.format(q=question, variation=v)
            hypotheses.append(
                {
                    "id": f"H{i+1}",
                    "statement": statement,
                    "rationale": (
                        "Synthesized from patterns in the retrieved literature and "
                        "general alignment research trends."
                    ),
                    "testability": (
                        "Can be evaluated empirically using controlled benchmarks "
                        "and A/B tests against competitive baselines."
                    ),
                    "risks": (
                        "External validity may be limited; results could depend on "
                        "datasets, implementation details, and evaluation metrics."
                    ),
                }
            )
        return hypotheses

    # Use LLM with structured output
    context = "\n".join(
        f"- {ins.get('title', '')}: {ins.get('summary', '')}"
        for ins in insights[:8]
    )

    system_prompt = (
        "You are a senior research scientist. Given a research question and "
        "some literature insights, propose 3-5 numbered, testable hypotheses. "
        "Each hypothesis should include: a clear statement, rationale, testability description, and main risks."
    )
    user_prompt = (
        f"Research question:\n{question}\n\n"
        f"Literature insights:\n{context}\n"
    )

    result = await call_llm_structured(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=HypothesisList,
        temperature=temperature,
        max_tokens=1200,
    )

    if result and result.hypotheses:
        return [h.model_dump() for h in result.hypotheses]

    # Fallback to template
    base = (
        "Based on current literature around '{q}', "
        "we hypothesize that {variation}."
    )
    variations = [
        "introducing a carefully aligned large language model will improve task performance compared to existing baselines",
        "combining supervised finetuning with human preference optimization yields better alignment than either alone",
        "scaling data diversity has a larger impact on generalization than scaling model size alone",
    ]
    hypotheses: List[Dict[str, Any]] = []
    for i, v in enumerate(variations[:3]):
        statement = base.format(q=question, variation=v)
        hypotheses.append(
            {
                "id": f"H{i+1}",
                "statement": statement,
                "rationale": "Synthesized from patterns in the retrieved literature.",
                "testability": "Can be evaluated empirically using controlled benchmarks.",
                "risks": "External validity may be limited.",
            }
        )
    return hypotheses


async def design_experiment_from_hypothesis(
    hypothesis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Turn the best hypothesis into a compact experiment design using async LLM with Pydantic.

    Uses PLANNING_MODEL from .env if llm_model is not provided.
    """
    model = llm_model or PLANNING_MODEL_ENV
    h_text = hypothesis.get("statement", "N/A")

    # Fallback template if no model
    if not model:
        objectives_md = (
            "- **Primary**: Quantify the effect of the proposed intervention on key metrics.\n"
            "- **Secondary**: Analyze failure modes and sensitivity to dataset/domain shifts.\n"
        )
        setup_md = (
            "- **Datasets**: Select 2–3 publicly available benchmarks relevant to the question.\n"
            "- **Models**: Compare a strong baseline vs. a variant implementing the hypothesis.\n"
            "- **Protocol**: Train or evaluate under matched conditions; control for confounders.\n"
        )
        metrics_md = (
            "- **Core task metrics** (e.g., accuracy, F1, BLEU, reward).\n"
            "- **Robustness** metrics across domains or perturbations.\n"
            "- **Human evaluation** for qualitative aspects where applicable.\n"
        )
        risks_md = (
            "- **Data leakage** or overfitting to benchmarks.\n"
            "- **Mis-specified metrics** that do not reflect true alignment or quality.\n"
            "- **Compute constraints**, limiting exploration of ablations.\n"
        )
        overview = (
            f"This experiment tests the following hypothesis:\n\n> {h_text}\n\n"
            "We compare a baseline system to an intervention reflecting this hypothesis, "
            "under controlled experimental conditions."
        )
        return {
            "overview": overview,
            "objectives_md": objectives_md,
            "setup_md": setup_md,
            "metrics_md": metrics_md,
            "risks_md": risks_md,
        }

    # Use LLM with structured output
    context = "\n".join(
        f"- {ins.get('title', '')}: {ins.get('summary', '')}"
        for ins in insights[:8]
    )

    system_prompt = (
        "You are an expert experimentalist. Given a primary hypothesis and "
        "some literature context, design a concise experiment plan with "
        "sections: overview, objectives_md, setup_md, metrics_md, risks_md. "
        "The *_md fields should be Markdown formatted with bullet points."
    )
    user_prompt = (
        f"Primary hypothesis:\n{h_text}\n\n"
        f"Relevant insights:\n{context}\n"
    )

    result = await call_llm_structured(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=ExperimentPlan,
        temperature=temperature,
        max_tokens=1200,
    )

    if result:
        return result.model_dump()

    # Fallback to template
    objectives_md = (
        "- **Primary**: Quantify the effect of the proposed intervention on key metrics.\n"
        "- **Secondary**: Analyze failure modes and sensitivity to dataset/domain shifts.\n"
    )
    setup_md = (
        "- **Datasets**: Select 2–3 publicly available benchmarks relevant to the question.\n"
        "- **Models**: Compare a strong baseline vs. a variant implementing the hypothesis.\n"
    )
    metrics_md = (
        "- **Core task metrics** (e.g., accuracy, F1, BLEU, reward).\n"
        "- **Robustness** metrics across domains or perturbations.\n"
    )
    risks_md = (
        "- **Data leakage** or overfitting to benchmarks.\n"
        "- **Mis-specified metrics** that do not reflect true alignment or quality.\n"
    )
    overview = (
        f"This experiment tests the following hypothesis:\n\n> {h_text}\n\n"
        "We compare a baseline system to an intervention reflecting this hypothesis, "
        "under controlled experimental conditions."
    )
    return {
        "overview": overview,
        "objectives_md": objectives_md,
        "setup_md": setup_md,
        "metrics_md": metrics_md,
        "risks_md": risks_md,
    }


def assemble_research_draft_sections(
    question: str,
    insights: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    best_hypothesis: Dict[str, Any],
    experiment_plan: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build each section of the research-style draft as Markdown strings (template-based fallback).
    """
    # Abstract
    abstract = (
        f"**Research Question**: {question}\n\n"
        "This work explores this question using a combination of automated "
        "literature review, hypothesis generation, and experiment design. "
        "We summarize prior work, propose several testable hypotheses, and outline "
        "an experiment to evaluate the most promising direction."
    )

    # Introduction
    intro = (
        "Modern machine learning and AI systems increasingly rely on large-scale "
        "models and data. However, systematically exploring a new research question "
        "still requires significant manual effort. This document was generated by a "
        "GenAI Personal Research Scientist that automates early-stage reasoning.\n\n"
        "We focus on the following research question:\n\n"
        f"> {question}\n\n"
        "The goal is not to replace human expertise but to accelerate ideation and "
        "experiment planning."
    )

    gap_table_md = _gap_table_md(question, insights, best_hypothesis)

    # Related Work
    if insights:
        bullets = []
        for ins in insights[:5]:
            bullets.append(
                f"- **{ins.get('title', 'Untitled')}** – "
                f"{_truncate(ins.get('summary', ''), 220)}"
            )
        related = (
            "The automated literature review surfaced the following indicative works:\n\n"
            + "\n".join(bullets)
        )
    else:
        related = (
            "The current corpus did not yield strong matches. In a real deployment, "
            "this section would summarize key prior work from large external indexes."
        )

    # Hypotheses
    if hypotheses:
        lines = []
        for h in hypotheses:
            prefix = "⭐ **Primary Hypothesis**" if best_hypothesis and h["id"] == best_hypothesis["id"] else f"**{h['id']}**"
            lines.append(f"- {prefix}: {h['statement']}")
        hyp_section = "We consider the following candidate hypotheses:\n\n" + "\n".join(
            lines
        )
    else:
        hyp_section = "No hypotheses were generated for this query."

    novelty = _novelty_text(question, best_hypothesis)

    # Methodology
    methodology = (
        "The proposed experiment is designed to evaluate the primary hypothesis "
        "in a controlled setting.\n\n"
        "#### Overview\n\n"
        f"{experiment_plan.get('overview', 'N/A')}\n\n"
        "#### Objectives\n\n"
        f"{experiment_plan.get('objectives_md', 'N/A')}\n\n"
        "#### Experimental Setup\n\n"
        f"{experiment_plan.get('setup_md', 'N/A')}\n\n"
        "#### Metrics\n\n"
        f"{experiment_plan.get('metrics_md', 'N/A')}\n\n"
        "#### Risks & Mitigations\n\n"
        f"{experiment_plan.get('risks_md', 'N/A')}\n"
    )

    architecture = _architecture_text(question, experiment_plan)

    # Expected Outcomes
    expected = (
        "We expect that the intervention representing the primary hypothesis "
        "will outperform the baseline on core task metrics, while also influencing "
        "secondary properties such as robustness or human preference alignment. "
        "However, results may vary by dataset and evaluation protocol, and negative "
        "or mixed findings would also be informative."
    )

    return {
        "abstract": abstract,
        "introduction": intro,
        "research_gap_table_md": gap_table_md,
        "related_work": related,
        "hypotheses": hyp_section,
        "novelty": novelty,
        "methodology": methodology,
        "architecture_pipeline": architecture,
        "expected_outcomes": expected,
    }


async def create_structured_draft(
    question: str,
    insights: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    best_hypothesis: Dict[str, Any],
    experiment_plan: Dict[str, Any],
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
) -> Tuple[str, List[str]]:
    """
    Assemble the full draft as Markdown plus a flat list of citation strings.
    Uses async LLM with Pydantic structured output if model is available.

    Uses PLANNING_MODEL from .env if llm_model is not provided.
    """
    model = llm_model or PLANNING_MODEL_ENV

    disclaimer_text = (
        "> **Disclaimer:** This research note is auto-generated for brainstorming research gaps. "
        "Review original sources and involve domain experts before executing any experiments."
    )

    # Try LLM with structured output first
    if model:
        context_insights = "\n".join(
            f"- {ins.get('title', '')}: {ins.get('summary', '')}"
            for ins in insights[:8]
        )
        context_hyps = "\n".join(
            f"- {h.get('id', '')}: {h.get('statement', '')}" for h in hypotheses
        )

        system_prompt = (
            "You are an expert academic writer. Given a research question, literature insights, "
            "hypotheses, and an experiment plan, write a ~5 page (1800-2200 words) research-style draft. "
            "Return structured sections: abstract, introduction, research_gap_table_md (Markdown table "
            "with columns 'Research Gap | Supporting Evidence | Opportunity' and at least 3 rows), "
            "related_work, hypotheses, novelty (describe unique contribution), methodology, "
            "architecture_pipeline (step-by-step system pipeline), expected_outcomes. "
            "Each field must be detailed, factual, and grounded in the supplied inputs."
        )
        user_prompt = (
            f"Research question:\n{question}\n\n"
            f"Insights:\n{context_insights}\n\n"
            f"Hypotheses:\n{context_hyps}\n\n"
            f"Experiment plan overview:\n{experiment_plan.get('overview', '')}\n"
        )

        result = await call_llm_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=DraftSections,
            temperature=temperature,
            max_tokens=4000,
        )

        if result:
            sections = result.model_dump()
            draft_body = "\n\n".join([
                disclaimer_text,
                "# Abstract",
                sections["abstract"],
                "\n# Introduction",
                sections["introduction"],
                "\n# Research Gap Analysis",
                sections["research_gap_table_md"],
                "\n# Related Work",
                sections["related_work"],
                "\n# Hypotheses",
                sections["hypotheses"],
                "\n# Proposed Novelty",
                sections["novelty"],
                "\n# Methodology",
                sections["methodology"],
                "\n# Architecture & Pipeline",
                sections["architecture_pipeline"],
                "\n# Expected Outcomes",
                sections["expected_outcomes"],
            ])
        else:
            # Fallback to template
            sections = assemble_research_draft_sections(
                question, insights, hypotheses, best_hypothesis, experiment_plan
            )
            draft_body = "\n\n".join([
                disclaimer_text,
                "# Abstract",
                sections["abstract"],
                "\n# Introduction",
                sections["introduction"],
                "\n# Research Gap Analysis",
                sections["research_gap_table_md"],
                "\n# Related Work",
                sections["related_work"],
                "\n# Hypotheses",
                sections["hypotheses"],
                "\n# Proposed Novelty",
                sections["novelty"],
                "\n# Methodology",
                sections["methodology"],
                "\n# Architecture & Pipeline",
                sections["architecture_pipeline"],
                "\n# Expected Outcomes",
                sections["expected_outcomes"],
            ])
    else:
        # Template-based fallback
        sections = assemble_research_draft_sections(
            question, insights, hypotheses, best_hypothesis, experiment_plan
        )
        draft_body = "\n\n".join([
            disclaimer_text,
            "# Abstract",
            sections["abstract"],
            "\n# Introduction",
            sections["introduction"],
            "\n# Research Gap Analysis",
            sections["research_gap_table_md"],
            "\n# Related Work",
            sections["related_work"],
            "\n# Hypotheses",
            sections["hypotheses"],
            "\n# Proposed Novelty",
            sections["novelty"],
            "\n# Methodology",
            sections["methodology"],
            "\n# Architecture & Pipeline",
            sections["architecture_pipeline"],
            "\n# Expected Outcomes",
            sections["expected_outcomes"],
        ])

    if len(draft_body.split()) < TARGET_WORD_COUNT:
        draft_body = "\n\n".join([
            draft_body,
            "# Additional Discussion",
            _supplemental_discussion(question, insights),
        ])

    md_parts = [draft_body, "\n# References"]

    citations = []
    for ins in insights:
        cit = ins.get("citation")
        if cit and cit not in citations:
            citations.append(cit)

    # Append references section
    if citations:
        md_parts.append("\n".join(f"- {c}" for c in citations))
    else:
        md_parts.append("References could not be extracted from the local corpus.")

    full_md = "\n\n".join(md_parts)
    return full_md, citations
