# Agents Analysis: LiteLLM + Pydantic Structured Outputs Pattern

## Summary

All production agents in `genai_research_scientist` now follow the same contract as the Hardware Analyzer reference:

- Async calls through `litellm.acompletion`
- `response_format=` wired to Pydantic BaseModels
- Model selection driven by `PLANNING_MODEL` (or the sidebar override)
- Temperature defaults to `0.0` for deterministic structured JSON

This document captures the current capabilities, highlights the recent upgrades, and outlines what we can build next.

---

## Agents Using the Pattern Today

| Agent | Purpose | Structured Model | Notes |
|-------|---------|------------------|-------|
| `SearchAgent` | Generates literature candidates via LLM | `LiteratureDocument`, `LiteratureSearchResult` | Shares the same model as downstream steps; no Semantic Scholar dependency |
| `ExtractionAgent` | Summarises docs into insights | `InsightExtraction` → `Insight` | Async, Pydantic output, schema validated |
| `HypothesisAgent` | Produces 3–5 hypotheses + primary selection | `HypothesisList` / `Hypothesis` | Replaces dataclass with BaseModel |
| `ExperimentAgent` | Designs experiment plan | `ExperimentPlan` | Structured Markdown fields |
| `DraftAgent` | Creates final research draft | `DraftSections` | Emits Markdown + citations |

Every agent inherits the model selected in the sidebar (or `PLANNING_MODEL` fallback) so the entire pipeline operates on a single provider per run.

---

## Architecture Snapshot

```mermaid
graph TD
    A[Sidebar Model Selector] --> B[SearchAgent (LLM literature)]
    B --> C[ExtractionAgent]
    C --> D[HypothesisAgent]
    D --> E[ExperimentAgent]
    E --> F[DraftAgent]
    F --> G[Downloads / plan.txt / hardware_analysis.json]
```

- All nodes use async LiteLLM calls with schema enforcement.
- SearchAgent now seeds the chain with LLM-generated literature, keeping the workflow self-consistent even without a local corpus.

---

## Recent Updates

- **LLM-first retrieval:** SearchAgent no longer depends on TF-IDF or Semantic Scholar. It uses the same model chosen for the rest of the workflow and emits structured metadata.
- **Centralised model selector:** Users can switch between OpenAI, Gemini, and Claude directly in the sidebar. Missing API keys trigger warnings to keep the UX safe.
- **Session-aware downloads:** Once a draft is generated, Markdown/PDF export buttons stay active until a new run resets the state.
- **Documentation refresh:** Added `MODEL_CONFIGURATION.md` and `CONVERSION_SUMMARY.md` to track setup details and migration history.

---

## Future Scope

1. **Hybrid Retrieval:** Augment the LLM search with vector stores or web connectors, falling back to the LLM when no external hits are found.
2. **Agent Benchmarks:** Introduce automated evals (BLEU, ROUGE, schema validators) to compare providers and spot regressions when models change.
3. **Fine-Tuned Personas:** Allow each agent to run on a different specialised model (e.g., Gemini for search, Claude for drafting) while retaining structured contracts.
4. **User Corpora Uploads:** Add an ingestion API that chunks and indexes user PDFs/notes, then feeds them into `SearchAgent` before the LLM augmentation step.
5. **Guardrails & Auditing:** Log structured outputs with trace IDs so we can audit LLM reasoning or replay steps when bugs appear.

---

Need another view or metric? Add it here once new agents or flows ship. For now, the entire pipeline is aligned around the Hardware Analyzer pattern. 🎉