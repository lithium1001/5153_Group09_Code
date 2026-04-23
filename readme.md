# BT5153 Group 09

---

## Running on Google Colab

### Step 1 — Upload the ZIP

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook (or open `BT5153_MAS.ipynb` directly).
2. Before running, go to Runtime → Change runtime type and set Hardware accelerator to **GPU**. Cell H precomputes embeddings for 105K reviews — on CPU this takes 20–30 minutes; on GPU it finishes in 2–3 minutes.
3. In the left sidebar, click the **Files** icon, then **Upload to session storage**.
4. Upload `data09.zip` — the notebook extracts it automatically on first run. Do **not** unzip it manually.


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

Click **Runtime → Run all**. The cells must run top-to-bottom:

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

> **First-run time**: Cell D (brand normalization) can take 5–15 minutes. Subsequent runs are fast because results are cached in `brand_mapping.pkl`. Cell H (embedding precomputation over 105K reviews) takes a few minutes on Colab's GPU.

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

The only external requirement is a valid **OpenAI API key** with access to `gpt-4o-mini`.


