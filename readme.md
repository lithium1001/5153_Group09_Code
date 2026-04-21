# BT5153 Group 09 — Mobile Review Analysis System

An AI-powered system for analyzing Amazon mobile phone reviews using a multi-agent LangGraph pipeline with hybrid retrieval, aspect-level sentiment analysis, episodic memory, and an evaluator-optimizer quality loop.

---

## Running on Google Colab

### Step 1 — Upload the ZIP

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook (or open `BT5153_MAS.ipynb` directly).
2. In the left sidebar, click the **Files** icon, then **Upload to session storage**.
3. Upload `Amazon_Unlocked_Mobile.zip` — the notebook extracts it automatically on first run. Do **not** unzip it manually.

> Alternatively, mount Google Drive and place the ZIP there, then update the `DATA_PATH` variable in Cell B to point to its Drive location.

### Step 2 — Set the OpenAI API Key via Google Secret

1. In the left sidebar, click the **Key** icon (Secrets).
2. Click **Add new secret**.
3. Set the name to `OPENAI_API_KEY` and paste your OpenAI API key as the value.
4. Enable **Notebook access** for this secret.

Cell B reads the key automatically with:

```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

### Step 3 — Run All Cells in Order

Click **Runtime → Run all** (or `Ctrl+F9`). The cells must run top-to-bottom:

| Cell | What happens |
|------|-------------|
| A | Installs all dependencies (`langgraph`, `langchain`, `gradio`, `sentence-transformers`, etc.) |
| B | Loads config and API key |
| C | Imports libraries |
| D | Loads and preprocesses 413,840 reviews → filters to ~105,160 clean reviews; runs LLM brand normalization (cached after first run to `brand_mapping.pkl`) |
| E–F | Defines prompt templates and Pydantic output schemas |
| G | Sets up episodic memory and context compression |
| H | Builds the hybrid search index (BM25 + bi-encoder embeddings — this may take a few minutes) |
| I–O | Defines agents and assembles the LangGraph workflow |
| P | Runs unit tests |
| Q | Launches the **Gradio web interface** — a public URL will appear in the output |

> **First-run time**: Cell D (brand normalization) can take 5–15 minutes. Subsequent runs are fast because results are cached in `brand_mapping.pkl`. Cell H (embedding precomputation over 105K reviews) takes a few minutes on Colab's free GPU/CPU.

---

## Using the Interface

Once Cell Q runs, a Gradio URL appears (e.g. `https://xxxxxx.gradio.live`). Open it and type a query:

- **Compare brands**: `"Compare Samsung and Apple battery life"`
- **Analyze one brand**: `"What do people say about the camera on OnePlus?"`
- **Search by feature**: `"Which phones have the best value for money?"`
- **Follow-up**: `"What about their display quality?"` (uses session memory from the previous turn)

Results include a comparison table, per-aspect sentiment breakdown, 4 interactive Plotly charts, a quality score from the reflection agent, and the full conversation history.

Click **Clear Session** to reset the in-memory cache and start a fresh conversation.

---

## Architecture Overview

The pipeline is a LangGraph state machine with 8 agents:

```
Coordinator → Cache Inject → [Search] → Analysis → Reflection → [retry loop] → Visualization
```

- **Coordinator**: Parses query intent (`compare / analyze / search / off_topic`), extracts brand names and aspects, injects episodic memory and conversation history.
- **Search (ReAct)**: Hybrid 3-stage retrieval — BM25 + dense bi-encoder embeddings (`all-MiniLM-L6-v2`) fused with RRF, then cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`). Returns top-30 reviews per brand.
- **Analysis**: Async per-brand LLM calls using chain-of-thought prompting for aspect-level sentiment. Outputs polarity, confidence, and per-aspect summaries.
- **Reflection (Evaluator-Optimizer)**: Scores result quality (coverage, diversity, confidence). If score < 7/10, feeds per-brand critiques back into Analysis for up to 2 retry passes at higher temperature.
- **Visualization**: Generates 4 Plotly charts (sentiment distribution, aspect heatmap, diverging bar, brand scatter).

### Session Caching

- **In-RAM**: Fetched reviews and sentiment outputs are cached for the session, so follow-up queries on the same brands skip re-fetching and re-analysis.
- **On-disk**: `episodic_memory.json` persists the last 20 query resolutions across sessions for few-shot context. `brand_mapping.pkl` caches LLM brand normalizations.

---

## Configuration

Key parameters in Cell B (`CONFIG` dict):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gpt-4o-mini` | OpenAI model used for all LLM calls |
| `max_reviews` | `30` | Reviews per brand fed to the analysis agent |
| `rerank_candidates` | `50` | Candidates passed to the cross-encoder |
| `final_top_k` | `30` | Reviews returned after re-ranking |
| `reflection_threshold` | `7` | Quality score below this triggers a retry |
| `reflection_max_iter` | `2` | Maximum retry passes |
| `temperature` | `0.0` | Base temperature (raised to `0.4` on retry) |

---

## Requirements

All dependencies are installed automatically in Cell A. No `requirements.txt` is needed. The only external requirement is a valid **OpenAI API key** with access to `gpt-4o-mini`.

Hugging Face models (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) are downloaded automatically from the Hub on first run.

---

## Dataset

`Amazon_Unlocked_Mobile.zip` contains `Amazon_Unlocked_Mobile.csv` — 413,840 unlocked mobile phone reviews from Amazon. The preprocessing pipeline (Cell D) filters this to ~105,160 high-quality reviews across 277 normalized brands.
