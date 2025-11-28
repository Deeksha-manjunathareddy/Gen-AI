# Model Configuration Guide

## Supported Models

You can use **any model supported by LiteLLM**, including:

- **Anthropic (Claude)**: `anthropic/claude-3-5-sonnet-20240620`, `anthropic/claude-3-opus`, etc.
- **Google (Gemini)**: `gemini/gemini-1.5-pro`, `gemini/gemini-1.5-flash`, etc.
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.
- **Groq**: `groq/llama3-70b-8192`, `groq/mixtral-8x7b-32768`, etc.
- **And many more** - See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for full list

---

## Configuration Examples

### Option 1: Using Gemini

**`.env` file:**
```env
PLANNING_MODEL=gemini/gemini-1.5-pro
GOOGLE_API_KEY=your-google-api-key-here
```

**Or:**
```env
PLANNING_MODEL=gemini/gemini-1.5-flash
GEMINI_API_KEY=your-gemini-api-key-here
```

### Option 2: Using OpenAI

**`.env` file:**
```env
PLANNING_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Option 3: Using Claude (Anthropic)

**`.env` file:**
```env
PLANNING_MODEL=anthropic/claude-3-5-sonnet-20240620
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

### Option 4: Using Groq

**`.env` file:**
```env
PLANNING_MODEL=groq/llama3-70b-8192
GROQ_API_KEY=your-groq-api-key-here
```

---

## Model String Format

LiteLLM uses the format: `provider/model-name`

Examples:
- `gemini/gemini-1.5-pro` ✅
- `gpt-4o` ✅ (OpenAI doesn't need prefix)
- `anthropic/claude-3-5-sonnet-20240620` ✅
- `groq/llama3-70b-8192` ✅

---

## API Key Environment Variables

The system automatically detects which API key to use based on the model:

| Model Provider | Environment Variable | Example |
|---------------|---------------------|---------|
| **Anthropic** | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| **OpenAI** | `OPENAI_API_KEY` | `sk-...` |
| **Google/Gemini** | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `...` |
| **Groq** | `GROQ_API_KEY` | `...` |
| **Others** | Provider-specific | See LiteLLM docs |

---

## Quick Start: Switch to Gemini

1. **Get your Google API key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Update your `.env` file:**
   ```env
   PLANNING_MODEL=gemini/gemini-1.5-pro
   GOOGLE_API_KEY=your-api-key-here
   ```

3. **Restart your application:**
   ```bash
   streamlit run app.py
   ```

That's it! All agents will now use Gemini with structured outputs.

---

## Model Recommendations

### For Structured Outputs (Pydantic):
- ✅ **Claude 3.5 Sonnet** - Excellent structured output support
- ✅ **GPT-4o** - Very reliable structured outputs
- ✅ **Gemini 1.5 Pro** - Good structured output support
- ⚠️ **Gemini 1.5 Flash** - Faster but may have occasional formatting issues

### For Cost-Effective Options:
- **Gemini 1.5 Flash** - Fast and affordable
- **GPT-4o-mini** - Good balance of cost and quality
- **Claude 3 Haiku** - Fast and cost-effective

---

## Testing Different Models

You can easily switch between models by changing `PLANNING_MODEL` in your `.env` file:

```env
# Try Gemini
PLANNING_MODEL=gemini/gemini-1.5-pro
GOOGLE_API_KEY=your-key

# Switch to OpenAI
PLANNING_MODEL=gpt-4o
OPENAI_API_KEY=your-key

# Switch to Claude
PLANNING_MODEL=anthropic/claude-3-5-sonnet-20240620
ANTHROPIC_API_KEY=your-key
```

No code changes needed! Just update `.env` and restart.

---

## Troubleshooting

### "API key not set" error:
- Make sure the correct environment variable is set for your model provider
- Check that the variable name matches (e.g., `GOOGLE_API_KEY` for Gemini)

### Model not found:
- Check the exact model name in [LiteLLM documentation](https://docs.litellm.ai/docs/providers)
- Some models require specific prefixes (e.g., `gemini/` for Gemini models)

### Structured output issues:
- Some models handle structured outputs better than others
- If you get parsing errors, try a different model or check the model's structured output support

---

## Notes

- **All models use the same Pydantic structured output pattern** - no code changes needed
- **Temperature is set to 0.0** by default for more deterministic structured outputs
- **The system automatically detects** which API key to use based on the model name
- **You can have multiple API keys** in your `.env` file - only the relevant one will be used

---

## Recent Updates

- Added an in-app **model selector** with automatic API-key checks so end users can switch between OpenAI, Gemini, and Claude directly from the Streamlit sidebar.
- The sidebar now highlights missing keys with a warning (“Currently can't use this model”) and disables model invocation until the proper environment variable is provided.
- Download buttons are automatically enabled once a draft is generated, making it easier to export Markdown/PDF outputs after switching models.
- SearchAgent has been refactored to use the selected LLM for literature retrieval, so all steps (search, extraction, hypotheses, experiment, draft) consistently use the same provider.

---

## Future Scope

- **Per-user model overrides:** allow authenticated users to supply their own API keys at runtime (e.g., via Streamlit secrets or a secure key vault) without touching `.env`.
- **Advanced provider metadata:** expose latency/cost estimates beside each model option to guide users toward the best trade-off for their workflow.
- **Dynamic capability detection:** automatically grey out models that do not support JSON/Pydantic structured outputs or function calling.
- **Model presets:** ship curated profiles (e.g., “Low-cost fast draft”, “High-accuracy research”) that adjust both the model and temperature/LLM parameters together.

