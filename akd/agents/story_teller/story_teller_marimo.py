import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    _CSS = """
    <style>
    :root{
      --st-accent:#7c3aed; --st-accent2:#06b6d4;
      --st-ink:#0f172a; --st-muted:#64748b;
      --st-card:#ffffff; --st-border:#e6e8ef; --st-soft:#f6f7fb;
    }
    @media (prefers-color-scheme: dark){
      :root{ --st-ink:#e7ebf3; --st-muted:#94a3b8;
             --st-card:#161a23; --st-border:#262c3a; --st-soft:#11151d; }
    }
    .st-hero{
      background:linear-gradient(120deg,#7c3aed,#06b6d4);
      border-radius:18px; padding:26px 30px; color:#fff;
      box-shadow:0 18px 40px -18px rgba(124,58,237,.65);
    }
    .st-hero .kick{ text-transform:uppercase; letter-spacing:.14em;
      font-size:.7rem; font-weight:700; opacity:.85; }
    .st-hero h1{ margin:6px 0 8px; font-size:1.65rem; font-weight:750;
      letter-spacing:-.02em; }
    .st-hero p{ margin:0; opacity:.93; font-size:.92rem; max-width:62ch;
      line-height:1.5; }
    .st-card{
      background:var(--st-card); border:1px solid var(--st-border);
      border-radius:14px; padding:16px 18px; margin:2px 0;
    }
    .st-card h3{ margin:0 0 6px; font-size:.98rem; color:var(--st-ink);
      display:flex; align-items:center; gap:9px; font-weight:680; }
    .st-card p{ margin:0; color:var(--st-muted); font-size:.85rem; line-height:1.5; }
    .st-num{ display:inline-flex; width:22px; height:22px; border-radius:50%;
      background:linear-gradient(120deg,#7c3aed,#06b6d4); color:#fff;
      font-size:.74rem; font-weight:700; align-items:center; justify-content:center; }
    .st-side-title{ font-weight:750; font-size:1.12rem;
      background:linear-gradient(120deg,#7c3aed,#06b6d4);
      -webkit-background-clip:text; background-clip:text; color:transparent; }
    .st-side-sec{ text-transform:uppercase; letter-spacing:.11em; font-size:.66rem;
      font-weight:700; color:var(--st-muted); margin:16px 0 4px; }
    .st-hint{ font-size:.74rem; color:var(--st-muted); line-height:1.45; margin:0; }
    .st-pill{ display:inline-block; padding:2px 11px; border-radius:999px;
      font-size:.72rem; font-weight:650; }
    .st-pill.ok{ background:rgba(16,185,129,.15); color:#10b981; }
    .st-pill.bad{ background:rgba(239,68,68,.15); color:#ef4444; }
    .st-pill.muted{ background:var(--st-soft); color:var(--st-muted);
      border:1px solid var(--st-border); }
    .st-kv{ width:100%; border-collapse:collapse; font-size:.85rem; }
    .st-kv td{ padding:6px 10px; border-bottom:1px solid var(--st-border); }
    .st-kv td:first-child{ color:var(--st-muted); width:34%; }
    .st-kv code{ font-size:.82rem; }
    </style>
    """

    _HERO = """
    <div class="st-hero">
      <div class="kick">Accelerated Discovery · VEDA</div>
      <h1>📖 Story Teller</h1>
      <p>Turn a <b>STAC collection</b> and a <b>publication DOI</b> into a publishable,
      VEDA-style <b>MDX</b> story. Configure the run in the sidebar, check your source,
      then generate &amp; download.</p>
    </div>
    """

    mo.vstack([mo.Html(_CSS), mo.Html(_HERO)])
    return (mo,)


@app.cell
def _(mo):
    # --- Make the flat-import story_teller package importable + fix CWD for data/ ---
    import os
    import sys
    from pathlib import Path

    try:
        nb_dir = Path(mo.notebook_dir())
    except Exception:
        nb_dir = Path.cwd()

    # story_teller_agent.py uses flat imports (e.g. `from data_scout_agent import ...`),
    # so its own directory must be on sys.path. akd/* resolves via the installed package.
    project_root = nb_dir.parent.parent.parent
    for _p in (str(nb_dir), str(project_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # get_collection_items() writes to a relative `data/` dir → run from the notebook dir.
    if (nb_dir / "data").exists() or nb_dir.name == "story_teller":
        os.chdir(nb_dir)
    return (nb_dir,)


@app.cell
def _(mo, nb_dir):
    from dotenv import load_dotenv
    import os as _os

    _env = nb_dir.parent.parent.parent / ".env"
    load_dotenv(_env)
    api_key = _os.environ.get("OPENAI_API_KEY")

    if api_key:
        _badge = f'<span class="st-pill ok">connected · {api_key[:7]}…</span>'
        _sub = f"OpenAI key loaded from <code>{_env.name}</code> · {len(api_key)} chars"
    else:
        _badge = '<span class="st-pill bad">missing</span>'
        _sub = f"No <code>OPENAI_API_KEY</code> in <code>{_env}</code> — add one (keys: Nish, np0069@uah.edu)"

    mo.Html(
        f"""
        <div class="st-card">
          <h3><span class="st-num">✓</span> API key &nbsp;{_badge}</h3>
          <p>{_sub}</p>
        </div>
        """
    )
    return (api_key,)


@app.cell
def _(api_key, mo):
    # Import the agent once the key is present.
    mo.stop(not api_key)

    from story_teller_agent import (
        StoryTellerAgent,
        StoryTellerAgentConfig,
        StoryTellerAgentInputSchema,
    )
    from helper import scrape_text_from_urls
    return (
        StoryTellerAgent,
        StoryTellerAgentConfig,
        StoryTellerAgentInputSchema,
        scrape_text_from_urls,
    )


@app.cell
def _(mo):
    # --- Controls (created here, displayed only in the sidebar) ---
    stac_url = mo.ui.text(
        value="https://openveda.cloud/api/stac",
        label="STAC URL",
        full_width=True,
    )
    return (stac_url,)


@app.cell
def _(mo, stac_url):
    # Fetch collection ids live from the chosen STAC so only valid ids are offered.
    import json
    import urllib.request

    _url = stac_url.value.rstrip("/") + "/collections?limit=500"
    collection_ids = []
    fetch_error = None
    try:
        _req = urllib.request.Request(_url, headers={"User-Agent": "story-teller-marimo"})
        with urllib.request.urlopen(_req, timeout=30) as _r:
            _data = json.load(_r)
        collection_ids = sorted(c["id"] for c in _data.get("collections", []))
    except Exception as e:  # noqa: BLE001
        fetch_error = str(e)
    return collection_ids, fetch_error


@app.cell
def _(collection_ids, fetch_error, mo):
    if fetch_error:
        coll_status = mo.Html(f'<p class="st-hint">⚠️ fetch failed: {fetch_error[:80]}</p>')
    else:
        coll_status = mo.Html(
            f'<p class="st-hint"><span class="st-pill muted">{len(collection_ids)} collections</span></p>'
        )
    return (coll_status,)


@app.cell
def _(collection_ids, mo):
    _default = "landfill-emit" if "landfill-emit" in collection_ids else (
        collection_ids[0] if collection_ids else None
    )
    collection = mo.ui.dropdown(
        options=collection_ids or ["—"],
        value=_default,
        label="Collection id",
        searchable=True,
        full_width=True,
    )
    return (collection,)


@app.cell
def _(mo):
    doi = mo.ui.text(
        value="https://doi.org/10.5194/acp-22-9617-2022",
        label="Publication DOI / URL",
        full_width=True,
    )
    return (doi,)


@app.cell
def _(mo):
    check_button = mo.ui.run_button(label="🔍 Check source", full_width=True)
    return (check_button,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(
        label="🚀 Generate story", kind="success", full_width=True
    )
    return (run_button,)


@app.cell
def _(check_button, coll_status, collection, doi, mo, run_button, stac_url):
    # --- The sidebar: all controls in one tidy panel ---
    mo.sidebar(
        [
            mo.Html('<div class="st-side-title">📖 Story Teller</div>'),
            mo.Html('<div class="st-side-sec">① STAC source</div>'),
            stac_url,
            coll_status,
            collection,
            mo.Html('<div class="st-side-sec">② Publication</div>'),
            doi,
            mo.Html(
                '<p class="st-hint">Keep the DOI on the <b>same topic</b> as the '
                "collection. Prefer open-access (ACP / arXiv / PLOS) — gated "
                "publishers often serve a bot-wall.</p>"
            ),
            mo.Html('<div class="st-side-sec">③ Actions</div>'),
            check_button,
            run_button,
        ]
    )
    return


@app.cell
def _(mo):
    mo.Html(
        '<div class="st-card"><h3><span class="st-num">1</span> Check the source</h3>'
        "<p>Scrape the DOI and screen it for bot-walls / paywalls before spending LLM calls.</p></div>"
    )
    return


@app.cell
async def _(check_button, doi, mo, scrape_text_from_urls):
    mo.stop(
        not check_button.value,
        mo.Html('<p class="st-hint">Click <b>🔍 Check source</b> in the sidebar.</p>'),
    )

    scraped_preview = await scrape_text_from_urls([doi.value])

    _lower = scraped_preview.lower()
    _flags = []
    if len(scraped_preview) < 1500:
        _flags.append(f"very short ({len(scraped_preview)} chars)")
    for _kw in (
        "verify you are human",
        "just a moment",
        "cloudflare",
        "captcha",
        "enable javascript",
        "unusual traffic",
        "access denied",
        "are you a robot",
    ):
        if _kw in _lower:
            _flags.append(f"“{_kw}”")

    if _flags:
        _verdict = mo.callout(
            mo.md(
                "**Looks like a bot-wall / paywall — not the paper.** Detected: "
                + ", ".join(_flags)
                + ". The story would be garbage; try an open-access DOI."
            ),
            kind="warn",
        )
    else:
        _verdict = mo.callout(
            mo.md(f"**Good source** — scraped **{len(scraped_preview):,} chars** of real content."),
            kind="success",
        )

    mo.vstack(
        [
            _verdict,
            mo.accordion(
                {"Preview — first 2,000 chars": mo.md(f"```\n{scraped_preview[:2000]}\n```")}
            ),
        ]
    )
    return


@app.cell
def _(collection, doi, mo, stac_url):
    mo.Html(
        f"""
        <div class="st-card"><h3><span class="st-num">2</span> Generate</h3>
        <table class="st-kv">
          <tr><td>STAC URL</td><td><code>{stac_url.value}</code></td></tr>
          <tr><td>Collection</td><td><code>{collection.value}</code></td></tr>
          <tr><td>DOI</td><td><code>{doi.value}</code></td></tr>
        </table>
        <p style="margin-top:8px">Click <b>🚀 Generate story</b> in the sidebar — several
        LLM calls, ~1 min.</p></div>
        """
    )
    return


@app.cell
async def _(
    StoryTellerAgent,
    StoryTellerAgentConfig,
    StoryTellerAgentInputSchema,
    api_key,
    collection,
    doi,
    mo,
    run_button,
    stac_url,
):
    mo.stop(
        not run_button.value,
        mo.Html('<p class="st-hint">Waiting — click <b>🚀 Generate story</b>.</p>'),
    )

    _config = StoryTellerAgentConfig(api_key=api_key, stac_url=stac_url.value)
    _agent = StoryTellerAgent(config=_config)
    _input = StoryTellerAgentInputSchema(urls=[doi.value], collection_ids=[collection.value])
    with mo.status.spinner(title="Running the Story Teller pipeline…"):
        _output = await _agent.arun(_input)

    story_mdx = _output.story_mdx
    return (story_mdx,)


@app.cell
def _(collection, mo, story_mdx):
    # --- Result ---
    _fname = f"story_{collection.value}.mdx"
    mo.vstack(
        [
            mo.Html(
                f'<div class="st-card"><h3><span class="st-num">3</span> Story '
                f'<span class="st-pill ok">{len(story_mdx):,} chars</span></h3></div>'
            ),
            mo.download(
                data=story_mdx.encode("utf-8"),
                filename=_fname,
                label=f"⬇️ Download {_fname}",
                mimetype="text/markdown",
            ),
            mo.accordion(
                {
                    "📄 Rendered preview": mo.md(story_mdx),
                    "🔤 Raw MDX source": mo.md(f"```mdx\n{story_mdx}\n```"),
                },
                multiple=True,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Note — pick a collection and DOI that share a common context.**

            The map visualization is only generated when the STAC collection and the
            publication describe the **same subject, region, and time period**. When they
            overlap, the relevance filter keeps the data and the story embeds an interactive
            map block for it. If the two are unrelated, the data gets filtered out and the
            story is produced **without any map**.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
