## GenAI Personal Research Scientist

An interactive **Streamlit** app that acts as your **personal AI research scientist**.  
It takes a research question and automatically performs:

- **Literature retrieval** (LLM-powered search aligned with your selected provider)
- **Insight extraction**
- **Hypothesis generation**
- **Experiment design**
- **Structured research draft generation**
- **Citation-backed responses**
- **One-click downloads** (Markdown + optional PDF)
- **Model-aware validation** (sidebar selector with API-key checks)

All results are presented in a clean, modern multi-tab UI with download support for the final draft.

---

### 🧱 Folder Structure

```text
genai_research_scientist/
│── app.py
│── requirements.txt
│── README.md
│── USER_MANUAL.txt
│
├── modules/
│   ├── rag_pipeline.py
│   ├── generator.py
│   ├── utils.py
│   ├── agents.py              # Facade that re-exports all agents
│   ├── search_agent.py
│   ├── extraction_agent.py
│   ├── hypothesis_agent.py
│   ├── experiment_agent.py
│   ├── draft_agent.py
│
├── assets/
│   ├── logo.png               # (optional placeholder – add your own)
│   ├── sample_docs/
│       ├── sample_alignment.txt
```

---

### ⚙️ Installation (Local)

1. **Clone or copy the project folder**

   Place the `genai_research_scientist` directory in your workspace.

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   Your browser will open automatically (`http://localhost:8501` by default).

---

### 🧠 Multi-Agent Workflow

The system uses multiple modular agents, each in its own file:

- **`SearchAgent`** (`modules/search_agent.py`)  
  Calls the active LiteLLM provider to generate realistic literature candidates (`LiteratureDocument` models). The selected model in the sidebar (or `PLANNING_MODEL`) is reused for every downstream step so the experience remains provider-consistent. (Legacy TF‑IDF via `LocalRAGPipeline` is still available for experiments, but is no longer the default.)

- **`ExtractionAgent`** (`modules/extraction_agent.py`)  
  Converts raw documents into structured insights (summary, methods, results).

- **`HypothesisAgent`** (`modules/hypothesis_agent.py`)  
  Generates 3–5 candidate hypotheses and selects a primary one.

- **`ExperimentAgent`** (`modules/experiment_agent.py`)  
  Designs an experiment plan (objectives, setup, metrics, risks) for the primary hypothesis.

- **`DraftAgent`** (`modules/draft_agent.py`)  
  Assembles a full research-style draft (Abstract, Introduction, Related Work, Hypotheses,
  Methodology, Expected Outcomes, References).

All of these are re-exported through `modules/agents.py` for a simple import path:

```python
from modules.agents import (
    SearchAgent,
    ExtractionAgent,
    HypothesisAgent,
    ExperimentAgent,
    DraftAgent,
)
```

---

### 🎨 UI / UX Overview

The Streamlit app (`app.py`) provides:

- **Sidebar**
  - Project description and settings
  - **Model selector with API-key checks** (OpenAI, Gemini, Anthropic, etc.)
  - Creativity (temperature) slider
  - Draft download buttons (Markdown + optional PDF) that remain enabled after each run

- **Tabs**
  - **Home** – ask a question, run the pipeline, see quick instructions
  - **Literature** – retrieved documents, relevance scores, expanders with summaries
  - **Hypotheses** – 3–5 hypotheses with rationale, testability, and risks
  - **Experiment** – structured experiment design for the primary hypothesis
  - **Draft** – complete research draft in Markdown, plus references list

The color palette is chosen to work in both light and dark modes, and uses emojis and
expanders for a friendly, readable layout.

---

### 📚 Using Your Own Documents

The current default flow relies on LLM-backed literature suggestions so you can get end-to-end results without curating a corpus. If you want to plug in your own documents instead:

1. Add one or more `.txt` files into `assets/sample_docs`.
2. Update `SearchAgent` (or re-enable the legacy `LocalRAGPipeline`) to read from those documents instead of the LLM search.
3. Restart the app; whichever retrieval strategy you enable will feed data into the rest of the pipeline.

> **Tip:** You can hybridise both approaches—use local TF‑IDF for high-precision corporate docs and fall back to the LLM for additional context.

---

### 🤖 Integrating Real LLM APIs (OpenAI / Groq / HF)

The reference code deliberately avoids direct network calls so that it is:

- Easy to copy, run, and deploy.
- Safe to inspect without API keys.

To hook in your own LLMs:

1. Identify key generation points (currently template-based):
   - `summarize_documents_to_insights` in `modules/generator.py`
   - `generate_hypotheses_from_insights` in `modules/generator.py`
   - `design_experiment_from_hypothesis` in `modules/generator.py`
   - `create_structured_draft` in `modules/generator.py`

2. Replace the template logic with LLM calls, for example:

   ```python
   import openai

   def generate_hypotheses_from_insights(question, insights):
       prompt = build_prompt(question, insights)  # your custom function
       response = openai.ChatCompletion.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": prompt}],
           temperature=0.4,
       )
       # Parse response into a list of hypothesis dicts
       ...
   ```

3. Pass API keys through environment variables or Streamlit secrets.

---

### ☁️ Deployment on Streamlit Cloud

1. **Create a GitHub repository**
   - Log in to GitHub and click **New repository**.
   - Name it (e.g.) `genai-personal-research-scientist`.
   - Add a short description and keep it public (easiest for Streamlit Cloud).
   - Commit/push the entire `genai_research_scientist` project contents.

2. **Prepare the repo structure**
   - At the root of the repo, you should see:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `modules/` directory
     - `assets/` directory

3. **Create a Streamlit Cloud app**
   - Go to `https://share.streamlit.io` (or the Streamlit Community Cloud URL).
   - Sign in with your GitHub account.
   - Click **New app**.
   - Select your repository and branch.
   - Set the **Main file path** to `app.py`.

4. **Deploy**
   - Click **Deploy**.
   - Streamlit Cloud will install dependencies from `requirements.txt` and start the app.
   - Once running, you get a shareable URL for your GenAI Research Scientist.

5. **Secrets and API keys (if you integrate real models)**
   - In the app settings on Streamlit Cloud, add your secrets under **Secrets**.
   - Access them in code via `st.secrets["YOUR_KEY_NAME"]`.

---

### 🛠️ Common Issues & Fixes

- **App fails to start (ModuleNotFoundError)**  
  Make sure you deployed the entire folder structure and that `requirements.txt` contains
  `streamlit` and `scikit-learn`. Reboot the Streamlit app after updating.

- **No PDF download button**  
  Ensure `reportlab` is installed (locally or via `requirements.txt` on Streamlit Cloud).

- **No documents / empty results**  
  Add `.txt` files to `assets/sample_docs`, or adjust your question to be more general.

---

### 📄 Additional Documentation

For a more step-by-step, user-focused guide, see `USER_MANUAL.txt` in the project root.  
It explains how to launch the app, run the pipeline, interpret each tab, and customize the corpus.

---

### 🚀 Recent Updates

- **Unified LLM workflow:** SearchAgent now uses the same provider selected in the sidebar, so every step (search → draft) stays in sync.
- **Model selector with guardrails:** The sidebar surfaces OpenAI, Gemini, and Claude options, warning the user if an API key is missing before a run starts.
- **Persistent downloads:** Once a draft is generated, the Markdown/PDF export buttons remain active until the session resets.
- **Structured logging:** All agents emit validated Pydantic models, making debugging and analytics easier.

---

### 🔭 Future Scope

- **Bring-your-own vector store:** Add plug-and-play adapters for FAISS, Pinecone, or LanceDB for teams that prefer classic RAG over the built-in LLM retrieval.
- **Provider benchmarking:** Capture latency/cost metrics per model to recommend the best option for a given use case.
- **Per-user secrets:** Let authenticated users inject their own API keys at runtime (e.g., via Streamlit secrets or a secure vault UI).
- **Document ingestion UI:** Add an uploader for PDFs/notes with automatic chunking and indexing, then feed that corpus into the pipeline.
- **Quality evals:** Integrate automated evaluators (BLEU/ROUGE/JSON schema checks) to guard against regressions when swapping models.


