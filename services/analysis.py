"""Turns validation, model output and knowledge records into one analysis result.

Statements in `reasons` are derived only from what the system measured (image
checks, class probabilities, whether Grad-CAM ran). They never claim biological
reasoning the network did not perform. Disease information comes verbatim from
the knowledge base and is labelled separately from the model output.
"""
from __future__ import annotations

from collections import Counter

from services.image_validation import QualityReport
from services.knowledge_base import KnowledgeBase
from services.predictor import CONFUSABLE, LOW_SCORE_THRESHOLD

DISCLAIMER = (
    "CauliCare is an image-based screening aid, not a definitive agricultural diagnosis. "
    "Confirm important decisions with a local agricultural extension officer or plant-health laboratory."
)

NOT_CAULIFLOWER = {
    "title": "Please upload a cauliflower leaf image",
    "message": "The uploaded image does not appear to contain a cauliflower leaf, "
               "so CauliCare cannot perform disease detection on it.",
}
LEAF_CHECK_UNAVAILABLE = (
    "We couldn't check whether this photo shows a cauliflower leaf right now, so it was not analysed. "
    "Please try again in a minute."
)
LOW_QUALITY = {
    "title": "Image quality is too low for reliable analysis",
    "message": "Please upload another photo.",
}


DEFAULT_TOP_K = 3
# A second opinion needs at least this many similar reference photos, and this share of them agreeing.
SECOND_OPINION_MIN_NEIGHBOURS = 3
SECOND_OPINION_MIN_SHARE = 0.8


def decide(prediction: dict) -> tuple[str, str]:
    """The machine-readable outcome of a classified image.

    accepted   the top class cleared the reliability threshold (confidence >= 0.80)
    uncertain  a prediction is returned, but its confidence is below the threshold, so a
               caller should not act on it automatically (retake the photo or ask an expert)
    (rejected is used for images that are refused before classification)"""
    score = prediction["confidence"]
    if prediction["confidence_level"] == "uncertain":
        return "uncertain", (f"The top class has {_pct(score)} probability, below the {LOW_SCORE_THRESHOLD:.0%} "
                             "reliability threshold. Treat this result as unreliable.")
    if prediction["confidence_level"] == "moderate":
        return "accepted", (f"The top class has {_pct(score)} probability (moderate confidence). "
                            "Consider confirming it with a second photo.")
    return "accepted", f"The top class has {_pct(score)} probability (high confidence)."


def second_opinion(prediction: dict, similar: dict | None) -> dict:
    """Compare the CNN with the labels of the most similar reference photos (found by DINOv2, a
    different model that was not trained on these diseases).

    `agrees` is True / False only when the neighbours are near-unanimous; otherwise None, because
    a split vote says nothing. The CNN's label is never changed, only judged."""
    results = (similar or {}).get("results") if (similar or {}).get("available") else None
    if not results or len(results) < SECOND_OPINION_MIN_NEIGHBOURS:
        return {"available": False}
    label, votes = Counter(r["label"] for r in results).most_common(1)[0]
    share = votes / len(results)
    unanimous = share >= SECOND_OPINION_MIN_SHARE
    return {"available": True, "neighbours": len(results), "votes": votes, "consensus_label": label if unanimous else None,
            "agrees": (label == prediction["disease"]) if unanimous else None}


def apply_second_opinion(result: dict, prediction: dict, opinion: dict) -> dict:
    """Attach the second opinion; a strong disagreement turns an `accepted` decision into `uncertain`."""
    result["second_opinion"] = opinion
    if opinion.get("agrees") is False and result["decision"] == "accepted":
        result["decision"] = "uncertain"
        result["decision_reason"] = (
            f"The disease model says {prediction['disease']} ({_pct(prediction['confidence'])}), but "
            f"{opinion['votes']} of the {opinion['neighbours']} most similar reference photos are {opinion['consensus_label']}. "
            "The two disagree, so treat this result as unreliable and try a sharp close-up of a single leaf.")
        result["result"]["reliable"] = False
    return result


def top_k(prediction: dict, k: int = DEFAULT_TOP_K) -> list[dict]:
    items = list(prediction["all_predictions"].items())[: max(1, k)]  # already sorted by probability
    return [{"label": name, "confidence": p, "percent": round(p * 100, 1)} for name, p in items]


def compact_result(prediction: dict, k: int = DEFAULT_TOP_K) -> dict:
    """The small, stable contract for programmatic clients."""
    decision, reason = decide(prediction)
    return {
        "decision": decision,
        "decision_reason": reason,
        "result": {"label": prediction["disease"], "confidence": prediction["confidence"],
                   "confidence_percent": prediction["confidence_percent"],
                   "confidence_level": prediction["confidence_level"], "reliable": decision == "accepted",
                   "top_k": top_k(prediction, k)},
    }


def _pct(v: float) -> str:
    # Softmax scores are never shown as "100%" or "0.0%": both would read as certainty.
    if v >= 0.999:
        return ">99.9%"
    return "<0.1%" if v < 0.001 else f"{v * 100:.1f}%"


def explain(prediction: dict, quality: QualityReport) -> list[str]:
    top, runner = prediction["disease"], prediction["runner_up"]
    reasons = [f"The model gave the highest probability to {top} ({_pct(prediction['confidence'])})."]
    if quality.warnings:
        reasons.append("The image passed the quality checks, with the warnings listed below.")
    else:
        reasons.append("The image passed the brightness, exposure, sharpness and resolution checks.")
    reasons.append(f"The strongest competing class was {runner['disease']} ({_pct(runner['probability'])}).")
    if prediction["confusable_close_call"]:
        reasons.append("These two classes account for most of the model's errors on the held-out test set.")
    if prediction.get("explanation"):
        reasons.append("The visual explanation highlights the image regions that most influenced this prediction.")
    return reasons


def suitability_outcome(quality: QualityReport) -> dict | None:
    """Return a refusal outcome when the image should not be classified, else None."""
    if not quality.acceptable:
        return {"status": "low_quality", **LOW_QUALITY, "reasons": [p.message for p in quality.problems]}
    return None


def not_cauliflower_outcome(leaf_check: dict) -> dict:
    """Refusal for an image the cauliflower-leaf validator rejected."""
    return {
        "status": "not_cauliflower_leaf",
        **NOT_CAULIFLOWER,
        "leaf_check": leaf_check,
        "reasons": [
            "CauliCare only analyses photos of cauliflower leaves. This image was checked before any "
            "disease analysis and did not match the cauliflower leaf photos the check was trained on.",
            "No disease prediction was made, because the disease model would still pick one of its "
            "four leaf conditions for any picture.",
        ],
    }


def build_result(prediction: dict, quality: QualityReport, kb: KnowledgeBase, k: int = DEFAULT_TOP_K) -> dict:
    record = kb.get(prediction["disease"])
    top, runner = prediction["disease"], prediction["runner_up"]
    similar = None
    if prediction["confusable_close_call"]:
        similar = kb.compare(top, runner["disease"])
        note = "These two conditions can look similar in the image. Consider comparing the symptoms below."
    elif prediction["confidence_level"] != "high" or runner["probability"] >= 0.10:
        similar = kb.compare(top, runner["disease"])
        note = f"Closest alternative: {runner['disease']}."
    elif top in CONFUSABLE:
        # Even high-scoring predictions of these two classes are sometimes the other one.
        other = next(c for c in CONFUSABLE if c != top)
        similar = kb.compare(top, other)
        note = f"Commonly confused with {other}: most of the model's test-set errors are between these two conditions."
    if similar is not None:
        similar["note"] = note
        similar["alternative_probability"] = prediction["all_predictions"].get(similar["conditions"][1]["name"])
    return {
        "status": "ok",
        **compact_result(prediction, k),
        "prediction": prediction,
        "reasons": explain(prediction, quality),
        "disease": kb.public_record(record) if record else None,
        "similar": similar,
        "disclaimer": DISCLAIMER,
    }
