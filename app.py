"""
app.py
======
Flask web application for browser-based emotion recognition.

The app loads saved model metadata so it can work with both the baseline
48x48 grayscale CNN and the transfer-learning MobileNetV2 model.
"""

from __future__ import annotations

import base64
import os
import time
from typing import List, cast

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
import tensorflow as tf
from keras import Model

from data_loader import mobilenet_v2_preprocess
from model import EMOTION_LABELS, infer_runtime_config, load_model_metadata

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
INFERENCE_FPS = 24
# Safari/iOS only allows camera access from secure contexts. `localhost` is
# treated as secure, but a phone using `http://<computer-lan-ip>:5000` is not.
# Serve HTTPS by default so browser camera mode works from Safari on the LAN.
# Set HER_DL_HTTPS=0 to force plain HTTP for local-only testing.
ENABLE_HTTPS = os.environ.get("HER_DL_HTTPS", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

_model: Model | None = None
_face_cascade = None
_model_trained = False
_runtime_config = {
    "input_size": [48, 48],
    "channels": 1,
    "preprocessing": "grayscale_0_1",
    "model_type": "baseline_cnn",
}

EMOTION_COLORS_HEX = {
    "Angry": "#FF4444",
    "Disgust": "#8B4513",
    "Fear": "#9B59B6",
    "Happy": "#F1C40F",
    "Sad": "#3498DB",
    "Surprise": "#E67E22",
    "Neutral": "#95A5A6",
}


# Model registry: keys map to model file paths and loaded bundles.
MODEL_SPECS = {
    "baseline": {"path": "models/best_model.keras", "display": "Baseline CNN"},
    "transfer": {
        "path": "models/transfer_best_model.keras",
        "display": "Transfer (MobileNetV2)",
    },
}

_model_registry: dict = {}


def load_model_bundle(key: str):
    """Load a model bundle for the given key into the registry and return it.

    Bundle format:
    {
        "model": Model | None,
        "metadata": dict,
        "runtime_config": dict,
        "trained": bool,
        "path": str,
        "params": int,
    }
    """
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise KeyError(f"Unknown model key: {key}")

    model_path = spec["path"]
    bundle = {
        "model": None,
        "metadata": {},
        "runtime_config": {},
        "trained": False,
        "path": model_path,
        "params": 0,
    }

    if os.path.exists(model_path):
        try:
            model = cast(Model, keras.models.load_model(model_path))
            metadata = load_model_metadata(model_path)
            runtime_config = infer_runtime_config(model, metadata)
            bundle.update(
                {
                    "model": model,
                    "metadata": metadata,
                    "runtime_config": runtime_config,
                    "trained": True,
                    "params": int(model.count_params()),
                }
            )
            print(f"[App] Loaded model for key '{key}': {model_path}")
        except Exception as exc:
            print(f"[App] Error loading model '{key}' from {model_path}: {exc}")
    else:
        print(f"[App] Model file not found for key '{key}': {model_path}")

    _model_registry[key] = bundle
    return bundle


def load_resources():
    """Load model registry, and face detector at startup."""
    global _face_cascade

    cascade_data = getattr(cv2, "data", None)
    cascade_root = (
        getattr(cascade_data, "haarcascades", "") if cascade_data is not None else ""
    )
    cascade_path = (
        cascade_root + "haarcascade_frontalface_default.xml"
        if cascade_root
        else "haarcascade_frontalface_default.xml"
    )
    _face_cascade = cv2.CascadeClassifier(cascade_path)
    print("[App] Haar Cascade loaded")

    # Load configured models
    for key in MODEL_SPECS.keys():
        load_model_bundle(key)

    # Set a default runtime config for the UI if baseline missing
    if (
        _model_registry.get("baseline")
        and _model_registry["baseline"]["runtime_config"]
    ):
        # Use baseline runtime config for UI defaults
        pass


def preprocess_face(face_gray, runtime_config=None):
    """Preprocess a face ROI using the selected model's runtime config."""
    config = runtime_config or _runtime_config
    input_shape = config.get("input_size", _runtime_config["input_size"])
    input_height = int(input_shape[0])
    input_width = int(input_shape[1]) if len(input_shape) > 1 else input_height
    channels = int(config.get("channels", _runtime_config["channels"]))
    preprocessing = config.get("preprocessing", _runtime_config["preprocessing"])

    resized = cv2.resize(
        face_gray, (input_width, input_height), interpolation=cv2.INTER_AREA
    )
    equalized = cv2.equalizeHist(resized)

    if channels == 3:
        rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        face_input = rgb.astype(np.float32)[np.newaxis, ...]
        if preprocessing == "mobilenet_v2":
            face_input = mobilenet_v2_preprocess(face_input)
        return face_input

    norm = equalized.astype(np.float32) / 255.0
    return norm[np.newaxis, ..., np.newaxis]


def decode_image(file_bytes):
    """Decode uploaded image bytes to a BGR numpy array."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _predict_from_bgr(img_bgr, model_key: str = "baseline", annotate: bool = True):
    """Run face detection + prediction using the specified model bundle.

    Returns a tuple: (results_list, annotated_bgr_or_none, runtime_config, inference_ms)
    """
    if model_key not in _model_registry:
        return [], None, {}, 0.0

    bundle = _model_registry[model_key]
    model = bundle.get("model")
    runtime_config = bundle.get("runtime_config") or _runtime_config

    if model is None:
        return [], None, runtime_config, 0.0

    t0 = time.perf_counter()

    h_orig, w_orig = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    assert _face_cascade is not None
    faces = _face_cascade.detectMultiScale(
        gray_enh, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results: List[dict] = []
    annotated_img = img_bgr.copy() if annotate else None

    if len(faces) > 0:
        face_inputs = []
        face_boxes = []
        for x, y, w, h in faces:
            roi = gray[y : y + h, x : x + w]
            face_inputs.append(preprocess_face(roi, runtime_config)[0])
            face_boxes.append((x, y, w, h))

        batch = np.stack(face_inputs, axis=0)
        probs_batch = model(batch, training=False).numpy()

        for i, (x, y, w, h) in enumerate(face_boxes):
            probs = probs_batch[i]
            pred_idx = int(np.argmax(probs))
            emotion = EMOTION_LABELS[pred_idx]
            confidence = float(probs[pred_idx])

            if emotion == "Neutral" and confidence < 0.6:
                ranked = np.argsort(probs)[::-1]
                for alt_idx in ranked:
                    alt_emotion = EMOTION_LABELS[int(alt_idx)]
                    if alt_emotion != "Neutral":
                        pred_idx = int(alt_idx)
                        emotion = alt_emotion
                        confidence = float(probs[pred_idx])
                        break

            results.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "confidence": confidence,
                    "color": EMOTION_COLORS_HEX.get(emotion, "#FFFFFF"),
                    "all_probs": {
                        label: float(p) for label, p in zip(EMOTION_LABELS, probs)
                    },
                }
            )

        if annotate and annotated_img is not None:
            # reuse annotate_image logic but adapted to work with our results
            annotated_img = annotated_img.copy()
            for face_info in results:
                x, y, w, h = face_info["bbox"]
                emotion = face_info["emotion"]
                confidence = face_info["confidence"]

                hex_color = EMOTION_COLORS_HEX.get(emotion, "#FFFFFF")
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                bgr = (b, g, r)

                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), bgr, 3)
                label = f"{emotion}: {confidence * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thick = 2
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                label_y = max(y - 10, th + 10)
                cv2.rectangle(
                    annotated_img,
                    (x, label_y - th - 8),
                    (x + tw + 10, label_y + 4),
                    bgr,
                    -1,
                )
                cv2.putText(
                    annotated_img,
                    label,
                    (x + 5, label_y - 2),
                    font,
                    scale,
                    (255, 255, 255),
                    thick,
                    cv2.LINE_AA,
                )

    inference_ms = (time.perf_counter() - t0) * 1000
    return results, annotated_img, runtime_config, round(inference_ms, 2)


def annotate_image(img, faces_data):
    """Draw emotion annotations on image, return as JPEG base64."""
    for face_info in faces_data:
        x, y, w, h = face_info["bbox"]
        emotion = face_info["emotion"]
        confidence = face_info["confidence"]

        hex_color = EMOTION_COLORS_HEX.get(emotion, "#FFFFFF")
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        bgr = (b, g, r)

        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 3)

        label = f"{emotion}: {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        label_y = max(y - 10, th + 10)
        cv2.rectangle(img, (x, label_y - th - 8), (x + tw + 10, label_y + 4), bgr, -1)
        cv2.putText(
            img,
            label,
            (x + 5, label_y - 2),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

        cl = min(w, h) // 5
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in corners:
            dx = cl if cx == x else -cl
            dy = cl if cy == y else -cl
            cv2.line(img, (cx, cy), (cx + dx, cy), bgr, 4)
            cv2.line(img, (cx, cy), (cx, cy + dy), bgr, 4)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.route("/")
def index():
    """Serve the main web interface."""
    any_trained = any(v.get("trained", False) for v in _model_registry.values())
    baseline_runtime = _model_registry.get("baseline", {}).get(
        "runtime_config", _runtime_config
    )
    return render_template(
        "index.html",
        model_trained=any_trained,
        emotion_labels=EMOTION_LABELS,
        emotion_colors=EMOTION_COLORS_HEX,
        runtime_config=baseline_runtime,
        inference_fps=INFERENCE_FPS,
        tf_version=tf.__version__,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict emotions from an uploaded image. Accepts optional `model_type` form field."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    model_type = request.form.get("model_type", "baseline")

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    img_bytes = file.read()
    img = decode_image(img_bytes)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        results, annotated_img, runtime_config, inference_ms = _predict_from_bgr(
            img, model_key=model_type, annotate=True
        )

        annotated_b64 = None
        if annotated_img is not None:
            _, buffer = cv2.imencode(
                ".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        return jsonify(
            {
                "faces_detected": len(results),
                "faces": results,
                "annotated_image": annotated_b64,
                "image_size": {"width": img.shape[1], "height": img.shape[0]},
                "inference_ms": inference_ms,
                "model_type": model_type,
                "runtime_config": runtime_config,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/camera/frame", methods=["POST"])
def api_camera_frame():
    """Accept a camera frame (base64 or file) and return predictions.

    Expects form fields:
    - `model_type` (optional): `baseline` or `transfer`
    - `image` (file upload) OR `image_b64` (base64 string)
    """
    model_type = request.form.get("model_type", "baseline")

    img = None
    if "image" in request.files:
        f = request.files["image"]
        img = decode_image(f.read())
    else:
        img_b64 = request.form.get("image_b64")
        if img_b64:
            header, _, b64data = img_b64.partition(",")
            try:
                img = decode_image(base64.b64decode(b64data))
            except Exception:
                return jsonify({"error": "Invalid base64 image"}), 400

    if img is None:
        return jsonify({"error": "No image provided"}), 400

    results, annotated_img, runtime_config, inference_ms = _predict_from_bgr(
        img, model_key=model_type, annotate=True
    )

    annotated_b64 = None
    if annotated_img is not None:
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return jsonify(
        {
            "faces_detected": len(results),
            "faces": results,
            "annotated_image": annotated_b64,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "inference_ms": inference_ms,
            "model_type": model_type,
            "runtime_config": runtime_config,
        }
    )


@app.route("/api/models")
def api_models():
    """Return available model slots and their status."""
    models_info = {}
    for k, v in _model_registry.items():
        models_info[k] = {
            "display": MODEL_SPECS.get(k, {}).get("display", k),
            "path": v.get("path"),
            "trained": bool(v.get("trained", False)),
            "params": int(v.get("params", 0)),
        }
    return jsonify(models_info)


if __name__ == "__main__":
    print("=" * 60)
    print("  EmotionNet — Web Application")
    print("=" * 60)
    load_resources()
    scheme = "https" if ENABLE_HTTPS else "http"
    ssl_context = "adhoc" if ENABLE_HTTPS else None
    print(f"\n[App] Starting Flask server at {scheme}://0.0.0.0:5000")
    if ENABLE_HTTPS:
        print(
            "[App] HTTPS enabled for Safari/iOS camera support. "
            "When opening from your phone, use https://<computer-lan-ip>:5000 "
            "and accept the local development certificate warning."
        )
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )
