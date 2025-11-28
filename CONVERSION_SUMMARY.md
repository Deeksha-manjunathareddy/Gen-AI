# Conversion Summary: All Agents to Hardware Analyzer Pattern

## Overview

All agents in the GenAI Research Scientist project have been converted to use the same pattern as the Hardware Analyzer:
- ✅ **Async LiteLLM** (`litellm.acompletion`)
- ✅ **Pydantic structured outputs** (`response_format=PydanticModel`)
- ✅ **PLANNING_MODEL from .env** (no hard-coded models)
- ✅ **Structured API requests** with validated outputs

---

## Files Modified

### 1. `genai_research_scientist/modules/llm_client.py`
**Changes:**
- Added `call_llm_structured()` async function for Pydantic structured outputs
- Added `get_api_key_for_model()` helper function
- Added `PLANNING_MODEL` support from environment
- Kept `call_llm()` for backward compatibility (synchronous)

**Key Features:**
```python
async def call_llm_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: Type[T],  # Pydantic BaseModel
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Optional[T]:
```

### 2. `genai_research_scientist/modules/models.py` (NEW)
**Created:**
- `Insight` - Structured insight from documents
- `Hypothesis` - Research hypothesis (replaces dataclass)
- `HypothesisList` - List of hypotheses for LLM output
- `ExperimentPlan` - Experiment design structure
- `DraftSections` - Research draft sections

### 3. `genai_research_scientist/modules/generator.py`
**Changes:**
- All functions converted to `async`
- All LLM calls use `call_llm_structured()` with Pydantic models
- Functions use `PLANNING_MODEL` from `.env` if `llm_model` not provided
- Added `InsightExtraction` Pydantic model for document extraction

**Functions Updated:**
- `summarize_documents_to_insights()` → `async`
- `generate_hypotheses_from_insights()` → `async` with `HypothesisList`
- `design_experiment_from_hypothesis()` → `async` with `ExperimentPlan`
- `create_structured_draft()` → `async` with `DraftSections`

### 4. `genai_research_scientist/modules/extraction_agent.py`
**Changes:**
- `run()` method converted to `async`
- Temperature default changed to `0.0` (better for structured outputs)
- Uses `PLANNING_MODEL` from `.env` if model not provided

### 5. `genai_research_scientist/modules/hypothesis_agent.py`
**Changes:**
- Removed `@dataclass Hypothesis`, now uses Pydantic `Hypothesis` from `models.py`
- `run()` method converted to `async`
- Temperature default changed to `0.0`

### 6. `genai_research_scientist/modules/experiment_agent.py`
**Changes:**
- `run()` method converted to `async`
- Temperature default changed to `0.0`

### 7. `genai_research_scientist/modules/draft_agent.py`
**Changes:**
- `run()` method converted to `async`
- Temperature default changed to `0.0`

### 8. `genai_research_scientist/app.py`
**Changes:**
- Added `asyncio` import
- Updated `run_workflow()` to use `asyncio.run()` for async agent calls
- Introduced a sidebar model selector with API-key validation/warnings
- Removed the retrieval dropdown and locked the experience to the selected LLM provider
- Download buttons now react to session state so exports stay enabled after an analysis
- All agent calls wrapped with `asyncio.run()`

### 9. `genai_research_scientist/modules/search_agent.py`
**Changes:**
- Replaced Semantic Scholar/TF-IDF logic with an **LLM-powered** literature search that emits `LiteratureDocument` and `LiteratureSearchResult` objects.
- SearchAgent now inherits the active model from the sidebar (`PLANNING_MODEL` fallback) so the same provider handles search, analysis, and drafting.
- Results include consistent metadata (title, abstract, citation) suitable for downstream agents.

---

## Pattern Consistency

All agents now follow the **Hardware Analyzer pattern**:

```python
# 1. Define Pydantic model
class MyOutput(BaseModel):
    field1: str
    field2: List[str]

# 2. Async function with structured output
async def my_agent_function(input_data: str) -> MyOutput:
    model = llm_model or PLANNING_MODEL_ENV
    
    result = await call_llm_structured(
        model=model,
        system_prompt="...",
        user_prompt=f"...{input_data}...",
        response_format=MyOutput,
        temperature=0.0,
    )
    
    return result  # Validated Pydantic model
```

---

## Environment Variables Required

Add to your `.env` file:

```env
# Required: Model for structured outputs
PLANNING_MODEL=anthropic/claude-3-5-sonnet-20240620

# API Keys (based on PLANNING_MODEL provider)
ANTHROPIC_API_KEY=sk-...
# OR
OPENAI_API_KEY=sk-...
# OR
GOOGLE_API_KEY=...
```

---

## Benefits

1. **Type Safety**: Pydantic models provide runtime validation
2. **Consistency**: All agents use the same pattern
3. **Reliability**: Structured outputs reduce parsing errors
4. **Flexibility**: Easy to change models via `.env`
5. **Performance**: Async operations for better concurrency

---

## Migration Notes

- **SearchAgent** now uses LLM responses instead of Semantic Scholar or TF-IDF, keeping retrieval consistent with the chosen provider.
- **Temperature**: Changed from `0.3` to `0.0` for structured outputs (more deterministic)
- **Backward Compatibility**: `call_llm()` still available for sync calls
- **Streamlit**: Uses `asyncio.run()` to wrap async calls (Streamlit doesn't natively support async)

---

## Testing

To test the conversion:

1. Set `PLANNING_MODEL` in `.env`
2. Set appropriate API key in `.env`
3. Run: `streamlit run app.py`
4. Enter a research question and run the pipeline
5. All agents should use structured LLM outputs

---

## Next Steps

- All agents now work with LLM API requests and structured outputs
- No more manual JSON parsing
- Unified LLM search removes the Semantic Scholar dependency
- Consistent pattern across all agents

---

## Future Scope

- **Vector-powered retrieval:** integrate FAISS/Pinecone/LanceDB so SearchAgent can combine dense retrieval with LLM re-ranking.
- **Parallel orchestration:** evaluate LangGraph/CrewAI to run extraction, hypothesis, and drafting steps concurrently for lower latency.
- **Automated evals:** add regression suites that check schema compliance and factual consistency per agent/provider combination.
- **User document ingestion:** allow uploading PDFs/URLs, automatically chunk them, and feed them to the LLM search phase.

