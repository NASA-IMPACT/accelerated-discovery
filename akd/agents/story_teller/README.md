# Story Teller Agent

Turn **scientific literature** (a publication DOI/URL) plus **Earth-observation data**
(a STAC collection) into a publishable, VEDA-style **MDX** web story.

The pipeline scrapes the literature, fetches the STAC collection, filters the data down to
what's relevant to the paper, drafts a narrative, injects the data, and emits validated MDX.
See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full flow diagram and stage-by-stage detail.

```
DOI ──scrape──┐
              ├─▶ DataScout ─▶ ScriptWriter ─▶ DataInjection ─▶ MDXBuilder ─▶ story.mdx
STAC ─fetch──┘   (filter)      (blueprint+draft)   (add data)    (+validate)
```

---

## 1. Environment setup (one-time)

```bash
# Clone and enter the repo
git clone git@github.com:NASA-IMPACT/veda-accelerated-discovery.git
cd veda-accelerated-discovery

# Check out the story-teller branch
git checkout story-teller-showcase

# Create + activate the virtualenv (Python 3.12)
uv venv --python 3.12
source .venv/bin/activate

# Install project dependencies
uv sync

# Install marimo (not yet in project deps — required for the marimo runner)
uv pip install marimo

# Create your .env and add your OpenAI key
cp .env.example .env
#   edit .env →  OPENAI_API_KEY=sk-... in all the places.
```

> **Note: You need to have an OpenAI key**

**Verify the key works** before running (a bad/expired key is the #1 failure):

```bash
curl -s https://api.openai.com/v1/models \
  -H "Authorization: Bearer $(grep OPENAI_API_KEY .env | cut -d= -f2)" | head -c 120
```

A JSON list of models = good. `401 invalid_api_key` = fix the key.

---

## 2. Run it — marimo notebook (recommended)

[`story_teller_marimo.py`](./story_teller_marimo.py) is an interactive app that wraps the agent
with safeguards: a **live collection dropdown** (only valid STAC IDs) and a **source checker**
(catches bot-walled / paywalled DOIs before spending LLM calls).

```bash
# from the repo root, with the venv active:
marimo run akd/agents/story_teller/story_teller_marimo.py
```

Opens `http://localhost:2718`.

**In the browser:**

1. Check the **API-key** card — green "connected" pill means you're ready.
2. **Sidebar ① STAC source** — keep the STAC URL as `https://openveda.cloud/api/stac` (VEDA),
   then pick a **Collection id** from the searchable dropdown.
3. **Sidebar ② Publication** — paste a **DOI** on the *same topic* as the collection.
4. **Sidebar ③ Actions → 🔍 Check source** — confirm it scraped real content (not a bot-wall).
5. **🚀 Generate story** — runs the pipeline (~1 min, several LLM calls).
6. **Result** — download the `.mdx`, or expand the rendered / raw previews.

---

## 3. Gotchas & known issues

- **Collection ID must exist on the chosen STAC.** The `main.ipynb` example uses
  `vulcan-ffco2-yeargrid-v4`, which lives on the **GHG Center** STAC (`earth.gov/ghgcenter`) and
  is **not** on openveda. If you switch the URL to openveda, pick an openveda collection or the
  fetch fails. (The marimo runner's live dropdown prevents this.)
- **Bot-walled / paywalled DOIs poison the story.** Gated publishers (AGU/Wiley/Elsevier) often
  return a "verify you are human" page with HTTP 200; the scraper treats it as content and the
  story comes out about *web security* instead of your topic. **Prefer open-access** DOIs
  (Copernicus/ACP, arXiv, PLOS). The marimo **Check source** step flags this.
- **Keep collection and DOI topically matched.** Don't pair a fish-population paper with a
  deer-population dataset — the relevance filter will discard the data and the story suffers.
- **Invalid/expired API key** surfaces as a `litellm.AuthenticationError` at the first LLM stage
  (DataScout). Verify the key as shown in section 1.
