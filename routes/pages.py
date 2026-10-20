"""Server-rendered pages."""
import json

from flask import Blueprint, abort, current_app, render_template, request

from config import BASE_DIR

bp = Blueprint("pages", __name__)
_MODEL_INFO = None


def _kb():
    return current_app.extensions["caulicare"]["kb"]


def model_info() -> dict:
    """Evaluation figures shown on the site, read from the committed model metadata."""
    global _MODEL_INFO
    if _MODEL_INFO is None:
        meta = json.loads((BASE_DIR / "models" / "model_metadata.json").read_text(encoding="utf-8"))
        ev = meta["test_evaluation"]
        _MODEL_INFO = {
            "accuracy": meta["test_accuracy"],
            "train_accuracy": meta["train_accuracy"],
            "val_accuracy": meta["val_accuracy"],
            "macro_f1": meta["test_f1"],
            "macro_precision": meta["test_precision"],
            "macro_recall": meta["test_recall"],
            "test_images": ev["images"],
            "dataset_images": sum(meta["split"][s]["total"] for s in ("train", "val", "test")),
            "split": meta["split"],
            "params": meta["parameters"],
            "input_size": meta["input_size"],
            "per_class": ev["per_class"],
            "confusion": ev["confusion_matrix"],
        }
    return _MODEL_INFO


def _wake_similarity():
    # Free-tier services sleep when idle: start waking the similarity service while
    # the visitor is still on the page, so it is ready by the time they upload.
    current_app.extensions["caulicare"]["similarity"].warm_up()


@bp.get("/")
def home():
    _wake_similarity()
    return render_template("home.html", diseases=_kb().diseases, m=model_info(), active="home")


@bp.get("/analyze")
def analyze():
    _wake_similarity()
    return render_template("analyze.html", active="analyze")


@bp.get("/diseases")
def diseases():
    return render_template("diseases.html", diseases=_kb().diseases, m=model_info(), active="diseases")


@bp.get("/diseases/<disease_id>")
def disease_detail(disease_id):
    kb = _kb()
    record = kb.get(disease_id)
    if not record:
        abort(404)
    others = [d for d in kb.diseases if d["id"] != record["id"]]
    return render_template("disease_detail.html", d=kb.public_record(record), others=others,
                           m=model_info(), active="diseases")


@bp.get("/photo-guide")
def photo_guide():
    return render_template("photo_guide.html", active="guide")


@bp.get("/history")
def history():
    return render_template("history.html", active="history")


@bp.get("/report")
def report():
    return render_template("report.html", active="history")


@bp.get("/developers")
def developers():
    from services.predictor import HIGH_SCORE_THRESHOLD, LOW_SCORE_THRESHOLD

    return render_template("developers.html", base_url=request.host_url.rstrip("/"), low=LOW_SCORE_THRESHOLD,
                           high=HIGH_SCORE_THRESHOLD, batch_max=current_app.config["BATCH_MAX_IMAGES"],
                           active="developers")


@bp.get("/about")
def about():
    return render_template("about.html", diseases=_kb().diseases, m=model_info(), active="about")
