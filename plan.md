# Plan: Move Real-Time Inference into the Flask App and Clean Up Training Modes

## Objective
Update the project so real-time camera inference happens through `app.py` in the browser/web app instead of through the terminal webcam script. Also add model selection in the web app and remove synthetic dataset generation/training options.

## Scope
- Update `app.py` to support browser-based real-time camera inference.
- Update the web UI/API so the user can choose between:
  - baseline dataset-trained model (`models/best_model.keras`), and
  - transfer-learning model (`models/transfer_best_model.keras`).
- Remove terminal-based webcam inference entirely by deleting or deprecating `realtime_inference.py`.
- Remove synthetic dataset generation and synthetic training/evaluation options from training and evaluation flows.
- Update docs and validation steps to match the new workflow.

## Current Issues
1. Real-time camera inference currently lives in `realtime_inference.py`, which opens a webcam from the terminal with OpenCV windows.
2. `app.py` only supports uploaded-image prediction and a synthetic `/api/demo` endpoint.
3. `app.py` loads only one default model path (`models/best_model.keras`) at startup.
4. The project still exposes synthetic dataset generation and synthetic training flags, but the desired workflow should use real FER2013 data only.

## Desired End State
- The Flask app is the only user-facing real-time inference interface.
- Browser camera frames are sent to the Flask backend for inference.
- The user can switch between baseline and transfer-learning models from the UI/API.
- Terminal webcam inference is removed.
- Synthetic data generation and synthetic training paths are removed from the CLI and docs.

---

## Implementation Plan

### 1. Refactor model loading in `app.py`
- Replace the single global `_model` with a model registry keyed by model type, for example:
  - `baseline`: `models/best_model.keras`
  - `transfer`: `models/transfer_best_model.keras`
- Store per-model metadata:
  - loaded model object,
  - runtime config,
  - trained/available status,
  - model path,
  - parameter count.
- Add a helper such as `load_model_bundle(model_key)` that loads one model and returns a normalized runtime bundle.
- Add a helper such as `get_selected_model(model_key)` that validates the requested model key and returns the active model bundle.
- Avoid falling back to an untrained model for production web inference; return a clear unavailable-model response instead.

### 2. Make prediction code reusable in `app.py`
- Extract shared image prediction logic from `/api/predict` into a reusable function, for example:
  - decode/accept BGR image,
  - convert to grayscale,
  - enhance contrast,
  - detect faces,
  - preprocess each face according to the selected model runtime config,
  - run inference,
  - return face predictions and optional annotated image.
- Pass the selected model bundle into this helper so the same logic works for baseline and transfer models.
- Keep uploaded-image inference working, but add `model_type` selection support via form field or query parameter.

### 3. Add real-time camera API support to `app.py`
- Add a web-camera frame endpoint, for example `POST /api/camera/frame`.
- Accept frames as one of:
  - base64 image data from the browser canvas, or
  - multipart image upload.
- Parameters:
  - `model_type`: `baseline` or `transfer`, default `baseline`.
  - optional face detection settings (`scale_factor`, `min_neighbors`, `min_face_size`) if useful.
- Return JSON with:
  - faces detected,
  - per-face bounding boxes,
  - emotion predictions,
  - probabilities,
  - selected model metadata,
  - inference time,
  - optionally an annotated JPEG frame as base64.

### 4. Update the web UI for browser camera inference
- Update `templates/index.html` to add:
  - model selector (`Baseline CNN` vs `Transfer Learning`),
  - camera start/stop controls,
  - live video preview from `navigator.mediaDevices.getUserMedia`,
  - canvas capture loop,
  - prediction overlay or annotated frame display,
  - status panel showing active model and inference FPS/latency.
- Use JavaScript to periodically capture frames from the camera and call `/api/camera/frame`.
- Throttle requests to a reasonable rate to avoid overwhelming CPU-only systems.
- Keep image upload inference available as a separate mode.

### 5. Add model selection endpoints
- Update `/api/status` to report all configured model slots:
  - model key,
  - model path,
  - loaded/available status,
  - runtime config,
  - parameter count.
- Add `GET /api/models` if useful for populating the UI model selector.
- Support model selection in:
  - `/api/predict`, and
  - `/api/camera/frame`.

### 6. Remove terminal webcam inference
- Remove `realtime_inference.py` entirely if no code imports it.
- If deletion is too disruptive, replace it with a short message pointing users to `uv run app.py` and the browser camera interface, then delete it in a follow-up cleanup.
- Remove README references to `realtime_inference.py` commands.
- Remove screenshot/video recording docs tied to terminal OpenCV windows unless reimplemented in the web UI.

### 7. Remove synthetic dataset generation and synthetic training options
- Remove or deprecate `generate_synthetic_fer2013` from `data_loader.py`.
- Remove CLI flags from `train_model.py`:
  - `--synthetic`,
  - `--synthetic_n`.
- Update `train_model.py` so missing real dataset paths produce a clear error instead of generating synthetic data.
- Remove synthetic options from `train_transfer.py`:
  - `--synthetic`,
  - `--synthetic_n`.
- Remove synthetic evaluation/demo paths from `test_model.py` if they depend on synthetic data.
- Remove `/api/demo` from `app.py`, because it generates a synthetic face image.
- Update docs so all training/evaluation examples use the real FER2013 dataset.

### 8. Update documentation
- Update `README.md` to describe the web-first workflow:
  - `uv run app.py`,
  - open the browser UI,
  - select baseline or transfer model,
  - start camera inference.
- Remove terminal webcam command examples.
- Remove synthetic training/evaluation examples.
- Document expected model file paths:
  - `models/best_model.keras`,
  - `models/transfer_best_model.keras`.
- Document how to train each model from real FER2013 data only.

### 9. Validation
- Run diagnostics after edits.
- Start the Flask app with `uv run app.py`.
- Verify `/api/status` reports both model slots correctly.
- Verify uploaded-image prediction works with `model_type=baseline`.
- Verify uploaded-image prediction works with `model_type=transfer` if the transfer model file exists.
- Verify browser camera mode starts and sends frames successfully.
- Verify no references remain to terminal webcam inference in docs.
- Verify no synthetic training flags remain in help output for training scripts.

---

## Success Criteria
- `app.py` supports real-time browser camera inference.
- The web UI allows selecting baseline or transfer-learning model.
- Upload prediction and camera prediction both honor selected model type.
- `realtime_inference.py` is removed or replaced with a web-app migration notice.
- Synthetic dataset generation/training options are removed from user-facing workflows.
- Documentation matches the new UV + Flask web workflow.
- Project diagnostics are clean after implementation.
