"""Small FastAPI service that scores one URL with the saved model.

The model is loaded once, when this file starts. A prediction does not train it again.
Feature extraction reads the URL text only and does not visit the address.
"""

import json
import os
import traceback
from urllib.parse import urlparse

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from features import API_MAX_URL_LENGTH, FEATURE_NAMES, features_from_normalized, normalize_url


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
COMPARISON_CHART_PATH = os.path.join(BASE_DIR, "results", "model_comparison.png")
MAX_BODY_BYTES = 32 * 1024

DISCLAIMER = (
    "URLGuard uses structural URL features and machine learning. "
    "Predictions may be incorrect. Do not rely on this result as a security guarantee."
)

app = FastAPI(title="URLGuard API")
# Local lab pages may use port 3000, or another port if 3000 is already taken.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None
metrics = None
load_error = None


def load_artifacts():
    """Load the model and metrics one time."""
    global bundle, metrics, load_error
    bundle = None
    metrics = None
    load_error = None

    if not os.path.exists(MODEL_PATH):
        load_error = "The trained model file was not found. Run python train.py before starting predictions."
        print(load_error)
    else:
        try:
            loaded = joblib.load(MODEL_PATH)
            if loaded.get("feature_names") != list(FEATURE_NAMES):
                load_error = "The saved model uses a different feature list. Run python train.py again."
                print(load_error)
            else:
                bundle = loaded
                print("Loaded model:", bundle.get("model_name"))
        except Exception as error:
            load_error = "The model file could not be loaded. Run python train.py again."
            print(load_error)
            print(error)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        print("metrics.json was not found. Model Insights will be unavailable until you train.")

    print("The model stays in memory. Restart the API if you train again.")


def class_probability(model, vector, class_value):
    """Return predict_proba for one class, using the model's own class order."""
    matrix = np.asarray([vector], dtype=float)
    probabilities = model.predict_proba(matrix)[0]
    classes = [int(label) for label in model.classes_]
    if class_value not in classes:
        raise RuntimeError("The model does not contain class " + str(class_value) + ".")
    predicted = int(model.predict(matrix)[0])
    return float(probabilities[classes.index(class_value)]), predicted


def build_reasons(features):
    """Describe URL traits that were measured. These do not prove why the model decided."""
    reasons = []
    if features["is_ip_address"] == 1:
        reasons.append("The hostname is an IP address.")
    else:
        reasons.append("No IP address was found in the hostname.")

    if features["uses_https"] == 1:
        reasons.append("HTTPS is used. This encrypts the connection, but it does not prove the site is legitimate.")
    else:
        reasons.append("The URL does not use HTTPS.")

    if features["at_count"] > 0:
        reasons.append("The URL contains an @ symbol.")
    if features["subdomain_count"] >= 2:
        reasons.append("The hostname contains multiple extra subdomain labels.")
    elif features["subdomain_count"] == 1:
        reasons.append("The hostname contains an extra subdomain label.")
    if features["suspicious_keyword_count"] > 0:
        reasons.append("The URL contains a login-related or account-related keyword.")
    if features["url_length"] >= 75:
        reasons.append("The URL is unusually long.")
    if features["hyphen_count"] >= 4:
        reasons.append("The URL contains several hyphens.")
    if features["digit_count"] >= 8:
        reasons.append("The URL contains many digits.")
    if features["question_mark_count"] > 0 or features["equals_count"] > 0:
        reasons.append("The URL contains query parameters.")
    if features["path_length"] >= 40:
        reasons.append("The URL path is long.")
    return reasons


def build_details(features):
    return [
        {"label": "URL length", "value": str(features["url_length"]) + " characters"},
        {"label": "HTTPS", "value": "Yes" if features["uses_https"] == 1 else "No"},
        {"label": "IP address", "value": "Yes" if features["is_ip_address"] == 1 else "No"},
        {"label": "Subdomains", "value": str(features["subdomain_count"])},
        {"label": "Suspicious keywords", "value": str(features["suspicious_keyword_count"])},
    ]


def scheme_was_added(original, normalized):
    text = original.strip().lower()
    return not text.startswith("http://") and not text.startswith("https://") and normalized.startswith("https://")


def www_was_removed(original, normalized):
    original_host = (urlparse(original.strip()).hostname or "").lower()
    normalized_host = (urlparse(normalized).hostname or "").lower()
    return original_host.startswith("www.") and not normalized_host.startswith("www.")


def error(message, status_code):
    return JSONResponse({"error": message}, status_code=status_code)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return error("The request is too large.", 413)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return error("Send a JSON body with a url field.", 400)

    if not isinstance(payload, dict) or "url" not in payload:
        return error('Send JSON like {"url": "https://example.com"}.', 400)

    url = payload.get("url")
    if not isinstance(url, str):
        return error("The url field must be a string.", 400)

    try:
        normalized = normalize_url(url, max_length=API_MAX_URL_LENGTH)
        features = features_from_normalized(normalized)
    except ValueError as exc:
        return error(str(exc), 400)

    if bundle is None or bundle.get("model") is None:
        return error(load_error or "The trained model is not available.", 503)

    try:
        vector = [features[name] for name in FEATURE_NAMES]
        phishing_probability, predicted = class_probability(bundle["model"], vector, 1)
    except Exception:
        traceback.print_exc()
        return error("The model could not score this URL.", 500)

    if predicted not in (0, 1):
        return error("The model returned an unexpected label.", 500)

    notes = []
    if scheme_was_added(url, normalized):
        notes.append("No scheme was entered, so the URL was analyzed as HTTPS.")
    if www_was_removed(url, normalized):
        notes.append(
            "A leading www. was removed before scoring. Every legitimate URL in the training file starts with www, so that prefix is not treated as proof of safety."
        )
    note = " ".join(notes) if notes else None

    return {
        "prediction": "PHISHING" if predicted == 1 else "SAFE",
        "phishing_probability": round(phishing_probability, 6),
        "model": bundle.get("model_name", "Trained model"),
        "reasons": build_reasons(features),
        "details": build_details(features),
        "analyzed_url": normalized,
        "scheme_note": note,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/model-info")
def model_info():
    if metrics is None:
        return error("Model results were not found. Run python train.py first.", 503)
    payload = dict(metrics)
    payload["confusion_matrix_image"] = "/api/confusion-matrix"
    payload["comparison_chart_image"] = "/api/comparison-chart"
    return payload


def png_response(path, missing_message):
    if not os.path.exists(path):
        return error(missing_message, 503)
    with open(path, "rb") as handle:
        payload = handle.read()
    return Response(content=payload, media_type="image/png")


@app.get("/api/confusion-matrix")
def confusion_matrix_image():
    return png_response(CONFUSION_MATRIX_PATH, "The confusion matrix image was not found. Run python train.py.")


@app.get("/api/comparison-chart")
def comparison_chart_image():
    return png_response(COMPARISON_CHART_PATH, "The comparison chart was not found. Run python train.py.")


@app.get("/")
def index():
    return {
        "name": "URLGuard API",
        "health": "/api/health",
        "predict": "/api/predict",
        "model_info": "/api/model-info",
    }


load_artifacts()


if __name__ == "__main__":
    # Port 5000 matches the lab instructions. Set PORT if that port is already taken.
    port = int(os.environ.get("PORT", "5000"))
    print("URLGuard API: http://127.0.0.1:" + str(port))
    uvicorn.run(app, host="127.0.0.1", port=port)
