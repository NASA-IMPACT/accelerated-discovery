import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

MASTER_FILE = BASE_DIR / "annotations.json"
if not MASTER_FILE.exists():
    raise FileNotFoundError(f"{MASTER_FILE} not found")

with open(MASTER_FILE, "r", encoding="utf-8") as f:
    master_annotations = json.load(f)

# Splitting master JSON into per-SME files
SME_TO_FILE = {}
for ann in master_annotations:
    sme = ann.get("SME", "unknown")
    # sme_file = Path(f"annotations_{sme}.json")
    sme_file = BASE_DIR / f"annotations_{sme}.json"
    SME_TO_FILE.setdefault(sme, sme_file)
    if not sme_file.exists():
        # Create the file if it doens't exists, writing only relevant entries
        with open(sme_file, "w", encoding="utf-8") as f:
            json.dump(
                [a for a in master_annotations if a.get("SME") == sme],
                f,
                indent=2,
                ensure_ascii=False,
            )


@app.get("/", response_class=HTMLResponse)
async def choose_annotator(request: Request):
    """Landing page to let users pick their SME name."""
    sme_names = sorted(SME_TO_FILE.keys())
    return templates.TemplateResponse(
        "choose_sme.html",
        {"request": request, "sme_names": sme_names},
    )


def _load_annotations(sme: str):
    file = SME_TO_FILE.get(sme)
    if not file or not file.exists():
        return []
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_annotations(sme: str, data):
    file = SME_TO_FILE[sme]
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.get("/annotate/{sme}", response_class=HTMLResponse)
async def index(request: Request, sme: str, idx: int = 0):
    """Show a single example for a specific SME."""
    annotations = _load_annotations(sme)
    if not annotations:
        return HTMLResponse(f"<h2>No data for SME '{sme}'</h2>", status_code=404)
    if idx < 0 or idx >= len(annotations):
        return HTMLResponse(f"<h2>Index {idx} out of range</h2>", status_code=404)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "example": annotations[idx],
            "idx": idx,
            "total": len(annotations),
            "sme": sme,
        },
    )


@app.post("/update/criterion/{sme}/{idx}/{risk_id}/{c_idx}")
async def update_criterion(
    sme: str,
    idx: int,
    risk_id: str,
    c_idx: int,
    is_valid: str = Form(...),
    comment: str = Form(""),
):
    annotations = _load_annotations(sme)
    annotations[idx]["risks"][risk_id]["criteria"][c_idx]["is_criterion_valid"] = (
        is_valid
    )
    annotations[idx]["risks"][risk_id]["criteria"][c_idx]["comment"] = comment
    _save_annotations(sme, annotations)
    return {"status": "ok"}


@app.post("/update/judgement/{sme}/{idx}/{risk_id}/{j_idx}")
async def update_judgement(
    sme: str,
    idx: int,
    risk_id: str,
    j_idx: int,
    correctness: str = Form(...),
    comment: str = Form(""),
):
    annotations = _load_annotations(sme)
    annotations[idx]["risks"][risk_id]["judgements"][j_idx][
        "is_llm_judgement_correct"
    ] = correctness
    annotations[idx]["risks"][risk_id]["judgements"][j_idx]["comment"] = comment
    _save_annotations(sme, annotations)
    return {"status": "ok"}


@app.post("/update/missing/{sme}/{idx}/{risk_id}", response_class=HTMLResponse)
async def update_missing(
    request: Request,
    sme: str,
    idx: int,
    risk_id: str,
    new_criterion: str = Form(""),
):
    annotations = _load_annotations(sme)
    risk = annotations[idx]["risks"][risk_id]
    if risk.get("missing_criteria") is None:
        risk["missing_criteria"] = []
    if new_criterion.strip():
        risk["missing_criteria"].append(new_criterion.strip())
    _save_annotations(sme, annotations)
    return templates.TemplateResponse(
        "missing_container.html",
        {"request": request, "risk_id": risk_id, "risk": risk, "idx": idx, "sme": sme},
    )


@app.post("/delete/missing/{sme}/{idx}/{risk_id}/{mc_idx}", response_class=HTMLResponse)
async def delete_missing(
    request: Request,
    sme: str,
    idx: int,
    risk_id: str,
    mc_idx: int,
):
    annotations = _load_annotations(sme)
    risk = annotations[idx]["risks"][risk_id]
    if risk.get("missing_criteria") and 0 <= mc_idx < len(risk["missing_criteria"]):
        risk["missing_criteria"].pop(mc_idx)
    _save_annotations(sme, annotations)
    return templates.TemplateResponse(
        "missing_container.html",
        {"request": request, "risk_id": risk_id, "risk": risk, "idx": idx, "sme": sme},
    )


@app.get("/download/{sme}")
async def download_json(sme: str):
    """Download the SME-specific JSON."""
    file = SME_TO_FILE.get(sme)
    if not file:
        return HTMLResponse(f"No file for SME '{sme}'", status_code=404)
    return FileResponse(file, media_type="application/json", filename=file.name)
