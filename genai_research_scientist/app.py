import os
import asyncio
from datetime import datetime
from typing import Dict, Any

import streamlit as st

from modules.agents import (
    SearchAgent,
    ExtractionAgent,
    HypothesisAgent,
    ExperimentAgent,
    DraftAgent,
)
from modules.generator import TARGET_WORD_COUNT
from modules.utils import (
    load_sample_corpus,
    generate_markdown_download,
    generate_pdf_download,
)
from modules.rag_pipeline import LocalRAGPipeline


def init_state():
    """Initialize Streamlit session state keys."""
    defaults = {
        "question": "",
        "search_results": [],
        "extracted_insights": [],
        "hypotheses": [],
        "best_hypothesis": None,
        "experiment_plan": None,
        "research_draft_md": "",
        "citations": [],
        "pipeline_ran": False,
        "rag_pipeline": None,
        "draft_word_count": 0,
        "draft_page_estimate": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_local_rag_pipeline() -> LocalRAGPipeline:
    """Load (or reuse) the TF-IDF corpus pipeline."""
    pipeline = st.session_state.get("rag_pipeline")
    if pipeline is None:
        corpus = load_sample_corpus()
        pipeline = LocalRAGPipeline(corpus)
        st.session_state["rag_pipeline"] = pipeline
    return pipeline


def configure_page():
    st.set_page_config(
        page_title="GenAI Personal Research Scientist",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def check_api_key_available(model: str) -> tuple[bool, str]:
    """Check if API key is available for the given model."""
    model_lower = model.lower()
    
    if "anthropic" in model_lower or "claude" in model_lower:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return False, "ANTHROPIC_API_KEY not found in environment variables"
        return True, "✅ API key available"
    elif "openai" in model_lower or "gpt" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment variables"
        return True, "✅ API key available"
    elif "gemini" in model_lower or "google" in model_lower:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False, "GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables"
        return True, "✅ API key available"
    elif "groq" in model_lower:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False, "GROQ_API_KEY not found in environment variables"
        return True, "✅ API key available"
    else:
        return True, "⚠️ API key check not implemented for this provider"


# Model options (defined at module level for reuse)
MODEL_OPTIONS = {
    "OpenAI – gpt-4o": "gpt-4o",
    "OpenAI – gpt-4o-mini": "gpt-4o-mini",
    "Gemini – gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "Gemini – gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "Gemini – gemini-1.5-flash": "gemini/gemini-1.5-flash",
    "Claude – claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20240620",
    "Claude – claude-3-opus": "anthropic/claude-3-opus-20240229",
    "Claude – claude-3-haiku": "anthropic/claude-3-haiku-20240307",
}


def sidebar():
    st.sidebar.title("🧪 GenAI Research Scientist")
    st.sidebar.markdown(
        "Your **personal AI research assistant** for literature review, "
        "hypothesis generation, and experiment design."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")

    # Model selection with API key checking
    model_options = MODEL_OPTIONS
    
    # Check which models have API keys available
    available_models = []
    model_status = {}
    
    for label, model_name in model_options.items():
        has_key, status = check_api_key_available(model_name)
        model_status[label] = (has_key, status, model_name)
        if has_key:
            available_models.append(label)
        else:
            available_models.append(f"{label} ⚠️ (API key missing)")
    
    # Get PLANNING_MODEL from environment as default
    planning_model = os.getenv("PLANNING_MODEL", "")
    default_index = 0
    
    if planning_model:
        # Find matching model in options
        for idx, (label, model_name) in enumerate(model_options.items()):
            if model_name == planning_model:
                if label in available_models:
                    default_index = available_models.index(label)
                else:
                    # Find the version with warning
                    for avail_label in available_models:
                        if label in avail_label:
                            default_index = available_models.index(avail_label)
                            break
                break
    
    selected_label = st.sidebar.selectbox(
        "LLM Model",
        available_models,
        index=default_index,
        key="llm_model_selectbox",
        help="Select a model. Models with ⚠️ don't have API keys configured.",
    )
    
    # Extract model name (remove warning if present)
    clean_label = selected_label.replace(" ⚠️ (API key missing)", "")
    llm_model = model_options.get(clean_label, "")
    
    # Show API key status
    has_key, status_msg, _ = model_status.get(clean_label, (False, "Unknown model", ""))
    if has_key:
        st.sidebar.success(status_msg)
    else:
        st.sidebar.error(f"❌ {status_msg}")
        st.sidebar.warning("⚠️ Currently can't use this model. Please add the API key to your `.env` file.")
        llm_model = ""  # Disable model if no API key

    temperature = st.sidebar.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="temperature_slider",
        help="Lower temperature (0.0) for structured outputs, higher for creative tasks.",
    )

    st.sidebar.caption(
        "Tip: Use **0.0** for structured outputs (recommended), higher for "
        "**brainstorming**."
    )

    retrieval_label = st.sidebar.radio(
        "Literature retrieval",
        (
            "LLM literature synthesis",
            "Local TF-IDF corpus",
        ),
        key="retrieval_mode_radio",
        help="Switch between hallucination-resistant local search or broader LLM-generated literature.",
    )
    retrieval_mode = "local" if "Local" in retrieval_label else "llm"
    if retrieval_mode == "local":
        st.sidebar.info("Drop your .txt/.md files into `assets/sample_docs` to expand the local corpus.")
    else:
        st.sidebar.caption("Requires an API key for the selected LLM provider.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Download")
    
    # Check if draft exists in session state
    draft_md = st.session_state.get("research_draft_md", "")
    draft_word_count = st.session_state.get("draft_word_count", 0)
    draft_page_estimate = st.session_state.get("draft_page_estimate", 0.0)

    if draft_md and draft_md.strip():
        md_bytes = generate_markdown_download(draft_md)
        st.sidebar.download_button(
            label="⬇️ Download Draft (Markdown)",
            data=md_bytes,
            file_name=f"research_draft_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
        if draft_word_count:
            st.sidebar.caption(
                f"Approx length: {draft_word_count} words (~{draft_page_estimate:.1f} pages)."
            )
            if draft_word_count < TARGET_WORD_COUNT:
                st.sidebar.warning(
                    "Draft is shorter than the 5-page target. Re-run with a richer question if needed."
                )

        pdf_bytes = generate_pdf_download(draft_md)
        if pdf_bytes:
            st.sidebar.download_button(
                label="⬇️ Download Draft (PDF)",
                data=pdf_bytes,
                file_name=f"research_draft_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        else:
            st.sidebar.caption(
                "Install `reportlab` to enable **PDF** export (see README)."
            )
    else:
        st.sidebar.caption("Run an analysis to enable draft downloads.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with ❤️ using Streamlit + LiteLLM. Optimized for light/dark mode."
    )

    config = {
        "llm_model": llm_model,
        "temperature": temperature,
        "retrieval_mode": retrieval_mode,
    }
    
    # Store in session state for access in other functions
    st.session_state["sidebar_config"] = config
    
    return config


def run_workflow(question: str, config: Dict[str, Any]):
    """Main orchestrator that runs the multi-agent workflow (wraps async)."""
    # Progress bar
    progress = st.progress(0)

    retrieval_mode = config.get("retrieval_mode", "llm")
    rag_pipeline = get_local_rag_pipeline() if retrieval_mode == "local" else None

    # 1. Search Agent (async)
    search_msg = (
        "📚 Searching local TF-IDF corpus..."
        if retrieval_mode == "local"
        else "🔍 Searching relevant literature with LLM..."
    )
    with st.spinner(search_msg):
        search_agent = SearchAgent(
            llm_model=config.get("llm_model"),
            rag_pipeline=rag_pipeline,
        )
        search_results = asyncio.run(
            search_agent.run(
                question,
                top_k=8,
                llm_model=config.get("llm_model"),
                retrieval_mode=retrieval_mode,
            )
        )
    progress.progress(20)

    # Initialize other agents
    extraction_agent = ExtractionAgent()
    hypothesis_agent = HypothesisAgent()
    experiment_agent = ExperimentAgent()
    draft_agent = DraftAgent()

    # 2. Extraction Agent (async)
    with st.spinner("📑 Extracting key insights and methods..."):
        extracted_insights = asyncio.run(extraction_agent.run(
            search_results,
            llm_model=config.get("llm_model") or None,
            temperature=config.get("temperature", 0.0),
        ))
    progress.progress(45)

    # 3. Hypothesis Agent (async)
    with st.spinner("💡 Generating candidate hypotheses..."):
        hypotheses, best_hypothesis = asyncio.run(hypothesis_agent.run(
            question,
            extracted_insights,
            llm_model=config.get("llm_model") or None,
            temperature=config.get("temperature", 0.0),
        ))
    progress.progress(65)

    # 4. Experiment Agent (async)
    with st.spinner("🧪 Designing experiment for best hypothesis..."):
        experiment_plan = asyncio.run(experiment_agent.run(
            best_hypothesis,
            extracted_insights,
            llm_model=config.get("llm_model") or None,
            temperature=config.get("temperature", 0.0),
        ))
    progress.progress(80)

    # 5. Draft Agent (async)
    with st.spinner("📝 Writing structured research draft..."):
        research_draft_md, citations = asyncio.run(draft_agent.run(
            question,
            extracted_insights,
            hypotheses,
            best_hypothesis,
            experiment_plan,
            llm_model=config.get("llm_model") or None,
            temperature=config.get("temperature", 0.0),
        ))
    progress.progress(100)

    st.success("✅ End-to-end research pipeline completed.")

    word_count = len(research_draft_md.split())
    page_estimate = round(max(word_count / 400, 0.25), 1) if word_count else 0.0

    # Persist to session state
    st.session_state["question"] = question
    st.session_state["search_results"] = search_results
    st.session_state["extracted_insights"] = extracted_insights
    st.session_state["hypotheses"] = hypotheses
    st.session_state["best_hypothesis"] = best_hypothesis
    st.session_state["experiment_plan"] = experiment_plan
    st.session_state["research_draft_md"] = research_draft_md
    st.session_state["citations"] = citations
    st.session_state["pipeline_ran"] = True
    st.session_state["draft_word_count"] = word_count
    st.session_state["draft_page_estimate"] = page_estimate


def home_tab():
    st.header("🧪 GenAI Personal Research Scientist")
    st.warning(
        "This workspace is for brainstorming only. It analyzes potential research gaps "
        "and drafts a working paper, but you must verify every insight before acting on it."
    )
    st.markdown(
        """
        This tool acts as your **AI research collaborator**.  

        **What it can do:**
        - 🔍 Retrieve and summarize relevant literature using LLM
        - 💡 Generate 3–5 **testable hypotheses**
        - 🧪 Propose an **experiment design** for the best hypothesis
        - 📝 Draft a **structured research write-up** with citations

        Retrieval can run purely on your **local TF-IDF corpus** or via **LLM literature synthesis**—
        pick the option that fits your workflow in the sidebar.

        All agents use LLM API requests (or deterministic fallbacks) with structured outputs for reliable results.
        """
    )

    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.markdown(
            """
            1. Enter a **clear research question** (e.g., *“How can large language models help with code review?”*).  
            2. Choose your **retrieval method** in the sidebar: LLM synthesis (requires API key) or the local TF-IDF corpus.  
            3. Click **Run Research Pipeline**.  
            4. Explore the results in each tab: **Literature**, **Hypotheses**, **Experiment**, and **Draft**.  
            5. Download the final ~5-page draft from the **sidebar**.
            """
        )

    st.markdown("---")

    # Use a single text area bound to session_state for stable behaviour.
    # We only set the default BEFORE the widget is instantiated to avoid
    # Streamlit's "cannot be modified after instantiation" error.
    if "research_question" not in st.session_state:
        st.session_state["research_question"] = ""

    def _set_example_question():
        st.session_state["research_question"] = (
            "What are effective techniques for aligning large language models "
            "with human preferences?"
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        run_button = st.button(
            "🚀 Run Research Pipeline",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.button(
            "✨ Use Example Question",
            use_container_width=True,
            on_click=_set_example_question,
        )

    question = st.text_area(
        "Enter your research question",
        key="research_question",
        placeholder="Example: How can reinforcement learning improve sample efficiency in robotics?",
        height=120,
    )

    if run_button:
        if not question.strip():
            st.error("Please enter a research question before running the pipeline.")
            return
        try:
            # Get config from session state (sidebar was already called in main())
            # Widget values are automatically stored in session state via their keys
            config = st.session_state.get("sidebar_config", {
                "llm_model": st.session_state.get("llm_model_selectbox", ""),
                "temperature": st.session_state.get("temperature_slider", 0.0),
                "retrieval_mode": "llm",
            })
            # Update config with current widget values from session state
            if "llm_model_selectbox" in st.session_state:
                selected_label = st.session_state["llm_model_selectbox"]
                # Extract model from selected label
                clean_label = selected_label.replace(" ⚠️ (API key missing)", "")
                config["llm_model"] = MODEL_OPTIONS.get(clean_label, "")
            if "temperature_slider" in st.session_state:
                config["temperature"] = st.session_state["temperature_slider"]
            if "retrieval_mode_radio" in st.session_state:
                label = st.session_state["retrieval_mode_radio"]
                config["retrieval_mode"] = "local" if "Local" in label else "llm"
            
            run_workflow(question.strip(), config)
            st.experimental_rerun()
        except Exception as e:
            st.error(
                "An unexpected error occurred while running the pipeline. "
                "Please see details below and try again."
            )
            st.exception(e)


def literature_tab():
    st.header("📚 Literature Overview")
    retrieval_mode = st.session_state.get("sidebar_config", {}).get("retrieval_mode", "llm")
    if retrieval_mode == "local":
        st.caption("Mode: Local TF-IDF corpus. Add files under `assets/sample_docs` to customize results.")
    else:
        st.caption("Mode: LLM literature synthesis (API key required).")
    if not st.session_state.get("pipeline_ran"):
        st.info("Run the research pipeline from the **Home** tab to see results here.")
        return

    results = st.session_state.get("search_results", [])
    if not results:
        st.warning("No literature results were found for this query.")
        return

    if retrieval_mode == "local":
        st.markdown("Top retrieved documents from the **local TF-IDF corpus**:")
    else:
        st.markdown("Top retrieved documents from the **LLM literature synthesis**:")

    for idx, doc in enumerate(results, start=1):
        with st.expander(f"📄 Document {idx}: {doc['title']}", expanded=(idx == 1)):
            st.markdown(f"**Title:** {doc['title']}")
            st.markdown(f"**Source:** {doc.get('source', 'LLM Generated Literature')}")
            st.markdown(f"**Relevance Score:** `{doc.get('score', 0):.3f}`")
            st.markdown("---")
            st.markdown("**Summary Snippet:**")
            st.markdown(doc.get("summary", "No summary available."))


def hypotheses_tab():
    st.header("💡 Hypotheses")
    if not st.session_state.get("pipeline_ran"):
        st.info("Run the research pipeline from the **Home** tab to see results here.")
        return

    hypotheses = st.session_state.get("hypotheses", [])
    best_hypothesis = st.session_state.get("best_hypothesis")

    if not hypotheses:
        st.warning("No hypotheses were generated.")
        return

    st.markdown("The system generated the following **testable hypotheses**:")

    for i, h in enumerate(hypotheses, start=1):
        is_best = best_hypothesis and (h["id"] == best_hypothesis["id"])
        label = f"H{i}: {h['statement']}"
        if is_best:
            label = f"⭐ **Primary Hypothesis** – {label}"
        with st.expander(label, expanded=is_best):
            st.markdown(f"**Rationale:** {h.get('rationale', 'N/A')}")
            st.markdown(f"**Testability:** {h.get('testability', 'N/A')}")
            st.markdown(f"**Risk/Limitations:** {h.get('risks', 'N/A')}")


def experiment_tab():
    st.header("🧪 Experiment Design")
    if not st.session_state.get("pipeline_ran"):
        st.info("Run the research pipeline from the **Home** tab to see results here.")
        return

    plan = st.session_state.get("experiment_plan")
    if not plan:
        st.warning("No experiment plan is available.")
        return

    st.subheader("Primary Hypothesis")
    st.markdown(f"> {st.session_state['best_hypothesis']['statement']}")
    st.markdown("---")

    st.subheader("Experiment Overview")
    st.markdown(plan.get("overview", "N/A"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Objectives")
        st.markdown(plan.get("objectives_md", "N/A"))
        st.markdown("### 📊 Metrics")
        st.markdown(plan.get("metrics_md", "N/A"))
    with col2:
        st.markdown("### 🧫 Experimental Setup")
        st.markdown(plan.get("setup_md", "N/A"))
        st.markdown("### ⚠️ Risks & Mitigations")
        st.markdown(plan.get("risks_md", "N/A"))


def draft_tab():
    st.header("📝 Research Draft")
    st.info(
        "⚠️ This draft is for brainstorming research ideas and highlighting gaps. "
        "Always review the cited literature and validate the plan with domain experts."
    )
    if not st.session_state.get("pipeline_ran"):
        st.info("Run the research pipeline from the **Home** tab to see results here.")
        return

    draft_md = st.session_state.get("research_draft_md", "")
    citations = st.session_state.get("citations", [])

    if not draft_md:
        st.warning("No research draft is available yet.")
        return

    st.markdown("Below is your **auto-generated research draft**:")
    st.markdown("---")
    st.markdown(draft_md)
    word_count = st.session_state.get("draft_word_count")
    if word_count:
        pages = st.session_state.get("draft_page_estimate", 0.0)
        st.caption(f"Approximate length: {word_count} words (~{pages:.1f} pages).")
        if word_count < TARGET_WORD_COUNT:
            st.warning("Draft is shorter than the requested ~5 pages. Consider refining the question or rerunning.")

    st.markdown("---")
    st.subheader("📚 References")
    if citations:
        for i, c in enumerate(citations, start=1):
            st.markdown(f"{i}. {c}")
    else:
        st.caption("No explicit citations found; likely due to limited sample corpus.")


def main():
    configure_page()
    init_state()

    # Call sidebar once - it updates session state internally
    config = sidebar()

    tabs = st.tabs(["🏠 Home", "📚 Literature", "💡 Hypotheses", "🧪 Experiment", "📝 Draft"])
    with tabs[0]:
        home_tab()
    with tabs[1]:
        literature_tab()
    with tabs[2]:
        hypotheses_tab()
    with tabs[3]:
        experiment_tab()
    with tabs[4]:
        draft_tab()


if __name__ == "__main__":
    main()


