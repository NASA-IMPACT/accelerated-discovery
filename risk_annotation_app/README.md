# Risk Annotation UI

A lightweight FastAPI + Jinja2 web app for human annotation of model risk evaluation criteria.

Annotators can:
- Review a conversation (User <-> Model).
- Valdate automatically generated criteria.
- Judge whether LLM decisions were correct.
- Add new (missing) criteria interactively.
- Save updated annotations as JSON.

---

## 🚀 Setup (using [uv](https://github.com/astral-sh/uv))

This project is tested with **Python 3.12**.

1. **Create and activate a virtual environment with uv**

```bash
uv venv --python=3.12
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows (PowerShell)
```

2. **Install dependencies**

```bash
uv pip install -e .
```

---

## ▶️ Running the App

Start the FastAPI server with uvicorn:

```bash
uvicorn risk_annotation_ui.app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Deployment (Docker & Docker Compose)

The app is Docker-ready and automatically persists `annotations_<annotator>.json`

1. **Build the image**

```bash
docker-compose build
```

2. **Run the container**

```bash
docker-compose up -d
```

- The UI will be available at `http://<server>:8000`
- The included `docker-compose.yml` mounts your local repo folder into the container at `/app`. This means changes to `annotations_<annotator>.json` are autosaved to the host.
