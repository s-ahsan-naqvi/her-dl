"""
test_model.py
=============
Evaluation and comparison suite for the FER-2013 emotion models.

This script can evaluate the baseline grayscale CNN, the new transfer-learning
MobileNetV2 model, or both side-by-side.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, cast

import keras
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import Model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import (
    EMOTION_COLORS,
    EMOTION_LABELS,
    IMG_SIZE,
    TRANSFER_IMG_SIZE,
    load_fer2013_from_dirs,
    mobilenet_v2_preprocess,
    preprocess_images,
    preprocess_transfer_images,
)
from model import (
    build_emotion_model,
    build_transfer_emotion_model,
    compile_model,
    infer_runtime_config,
    load_model_metadata,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Emotion Recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.keras",
        help="Path to a trained model (.keras)",
    )
    parser.add_argument(
        "--compare_model",
        type=str,
        default=None,
        help="Optional second model to compare against --model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="fer2013",
        help="Path to the FER2013 directory tree",
    )

    parser.add_argument(
        "--arch_only",
        action="store_true",
        help="Only test architecture (no model file needed)",
    )
    parser.add_argument(
        "--transfer_arch_only",
        action="store_true",
        help="Test the MobileNetV2 transfer architecture without a model file",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image for prediction"
    )
    return parser.parse_args()


def load_model_and_runtime_config(model_path: str):
    model = cast(Model, keras.models.load_model(model_path))
    metadata = load_model_metadata(model_path)
    runtime_config = infer_runtime_config(model, metadata)
    return model, metadata, runtime_config


def load_dataset_for_model(args, runtime_config):
    input_size = int(runtime_config["input_size"][0])
    channels = int(runtime_config["channels"])
    preprocessing = runtime_config["preprocessing"]

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")

    color_mode = "rgb" if channels == 3 else "grayscale"
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_from_dirs(
        args.data, input_size=input_size, color_mode=color_mode
    )

    if channels == 3 or preprocessing == "mobilenet_v2":
        return preprocess_transfer_images(X_tr, y_tr, X_v, y_v, X_te, y_te)
    return preprocess_images(X_tr, y_tr, X_v, y_v, X_te, y_te)


def plot_confusion_matrix(y_true, y_pred, save_path="logs/confusion_matrix.png"):
    """Plot both normalized and absolute confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=np.float32)
    np.divide(cm.astype(np.float32), row_sums, out=cm_norm, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Confusion Matrix — Emotion Recognition", fontsize=14, fontweight="bold"
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Absolute Counts", fontsize=11)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Normalized (Recall per Class)", fontsize=11)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Confusion matrix saved: {save_path}")
    return cm


def plot_per_class_metrics(report_dict, save_path="logs/per_class_metrics.png"):
    """Horizontal bar chart of precision, recall, F1 per emotion class."""
    classes = EMOTION_LABELS
    precision = [report_dict[c]["precision"] for c in classes]
    recall = [report_dict[c]["recall"] for c in classes]
    f1 = [report_dict[c]["f1-score"] for c in classes]
    support = [report_dict[c]["support"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Per-Class Performance Metrics", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", color="#FF5722", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colors = [EMOTION_COLORS[c] for c in classes]
    ax2.barh(classes, support, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Test Samples")
    ax2.set_title("Test Set Class Support")
    ax2.grid(axis="x", alpha=0.3)
    for i, (s, c) in enumerate(zip(support, classes)):
        ax2.text(s + 5, i, str(s), va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Per-class metrics saved: {save_path}")


def benchmark_inference_speed(model, input_shape, n_runs=200, batch_sizes=(1, 8, 32)):
    """Benchmark model inference speed for real-time feasibility."""
    print("\n[Benchmark] Inference Speed Test")
    print("-" * 60)
    print(f"{'Batch Size':>12} | {'Mean (ms)':>10} | {'Std (ms)':>8} | {'FPS':>8}")
    print("-" * 60)

    results = {}
    for bs in batch_sizes:
        dummy = np.random.uniform(-1.0, 1.0, (bs,) + tuple(input_shape)).astype(
            np.float32
        )
        for _ in range(10):
            _ = model(dummy, training=False)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy, training=False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        fps = float(bs / (mean_ms / 1000.0))

        print(f"{bs:>12} | {mean_ms:>10.2f} | {std_ms:>8.2f} | {fps:>8.1f}")
        results[bs] = {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}

    print("-" * 60)
    single_fps = results[1]["fps"]
    print(
        f"\n  Real-time feasibility (≥25 FPS at batch=1): {'✓ YES' if single_fps >= 25.0 else '✗ NO'}"
    )
    return results


def preprocess_single_image(image_path, runtime_config):
    import cv2

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    size = int(runtime_config["input_size"][0])
    channels = int(runtime_config["channels"])
    preprocessing = runtime_config["preprocessing"]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    face_equalized = cv2.equalizeHist(face_resized)

    if channels == 3:
        face_rgb = cv2.cvtColor(face_equalized, cv2.COLOR_GRAY2RGB)
        face_input = face_rgb.astype(np.float32)[np.newaxis, ...]
        if preprocessing == "mobilenet_v2":
            face_input = mobilenet_v2_preprocess(face_input)
    else:
        face_input = face_equalized.astype(np.float32) / 255.0
        face_input = face_input[np.newaxis, ..., np.newaxis]

    return face_input


def predict_single_image(model, image_path, runtime_config):
    img_input = preprocess_single_image(image_path, runtime_config)
    probs = model(img_input, training=False).numpy()[0]
    pred_class = int(np.argmax(probs))

    print(f"\n[Prediction] Image: {image_path}")
    print(
        f"  Predicted emotion: {EMOTION_LABELS[pred_class]} ({probs[pred_class] * 100:.1f}%)"
    )
    print("\n  All class probabilities:")
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        bar = "█" * int(prob * 30)
        marker = " ←" if i == pred_class else ""
        print(f"    {label:>10}: {prob * 100:5.1f}%  {bar}{marker}")


def build_inference_dataset(X, runtime_config, batch_size: int = 64):
    """Build an inference dataset with the preprocessing expected by the model."""
    preprocessing = runtime_config.get("preprocessing")

    ds = tf.data.Dataset.from_tensor_slices(np.asarray(X, dtype=np.float32)).batch(
        batch_size
    )
    if preprocessing == "mobilenet_v2":
        ds = ds.map(mobilenet_v2_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def run_full_evaluation(
    model,
    X_test,
    y_test_raw,
    model_label: str,
    save_prefix: str,
    runtime_config,
):
    """Run a complete evaluation suite and save plots/reports."""
    print(f"\n[Evaluation] Running comprehensive evaluation for {model_label}...")

    test_ds = build_inference_dataset(X_test, runtime_config, batch_size=64)
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = y_test_raw

    acc = float(accuracy_score(y_true, y_pred))
    top2_acc = float(top_k_accuracy_score(y_true, y_probs, k=2))
    top3_acc = float(top_k_accuracy_score(y_true, y_probs, k=3))

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS — {model_label}")
    print(f"{'=' * 60}")
    print(f"  Top-1 Accuracy:  {acc * 100:.2f}%")
    print(f"  Top-2 Accuracy:  {top2_acc * 100:.2f}%")
    print(f"  Top-3 Accuracy:  {top3_acc * 100:.2f}%")
    print(
        f"  Input shape:     {runtime_config['input_size'][0]}x{runtime_config['input_size'][1]}x{runtime_config['channels']}"
    )
    print(f"{'=' * 60}")

    report_text = cast(
        str,
        classification_report(
            y_true,
            y_pred,
            target_names=EMOTION_LABELS,
            digits=4,
            zero_division=cast(Any, 0),
        ),
    )
    print("\n[Evaluation] Classification Report:")
    print(report_text)

    report_dict = cast(
        Dict[str, object],
        classification_report(
            y_true,
            y_pred,
            target_names=EMOTION_LABELS,
            output_dict=True,
            zero_division=cast(Any, 0),
        ),
    )

    report_path = f"{save_prefix}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Emotion Recognition Evaluation — {model_label}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Top-1 Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Top-2 Accuracy: {top2_acc * 100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_acc * 100:.2f}%\n\n")
        f.write(str(report_text))
    print(f"[Evaluation] Report saved: {report_path}")

    cm = plot_confusion_matrix(
        y_true, y_pred, save_path=f"{save_prefix}_confusion_matrix.png"
    )
    plot_per_class_metrics(
        report_dict, save_path=f"{save_prefix}_per_class_metrics.png"
    )

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_indices = np.argsort(cm_no_diag, axis=None)[::-1]
    print("\n[Evaluation] Top confused class pairs:")
    shown = 0
    for flat_idx in flat_indices:
        true_cls, pred_cls = np.unravel_index(flat_idx, cm_no_diag.shape)
        count = int(cm_no_diag[true_cls, pred_cls])
        if count <= 0:
            continue
        print(
            f"  True={EMOTION_LABELS[true_cls]:>10} → Pred={EMOTION_LABELS[pred_cls]:>10}: {count} cases"
        )
        shown += 1
        if shown >= 5:
            break

    benchmark = benchmark_inference_speed(
        model,
        input_shape=(
            runtime_config["input_size"][0],
            runtime_config["input_size"][1],
            runtime_config["channels"],
        ),
    )

    return {
        "label": model_label,
        "accuracy": acc,
        "top2_accuracy": top2_acc,
        "top3_accuracy": top3_acc,
        "report": report_dict,
        "benchmark": benchmark,
        "runtime_config": runtime_config,
        "report_path": report_path,
        "confusion_matrix_path": f"{save_prefix}_confusion_matrix.png",
        "per_class_metrics_path": f"{save_prefix}_per_class_metrics.png",
    }


def test_architecture_only(transfer: bool = False):
    """Test that the chosen architecture builds, compiles, and runs a forward pass."""
    if transfer:
        print("[ArchTest] Building transfer-learning model...")
        model = build_transfer_emotion_model(
            input_shape=(TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE, 3)
        )
        dummy_shape = (TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE, 3)
    else:
        print("[ArchTest] Building baseline CNN model...")
        model = build_emotion_model()
        dummy_shape = (IMG_SIZE, IMG_SIZE, 1)

    model = compile_model(model)
    model.summary()
    print(f"\n[ArchTest] Parameters: {model.count_params():,}")

    for bs in [1, 4, 16]:
        dummy = np.random.randn(bs, *dummy_shape).astype(np.float32)
        out = model(dummy, training=False)
        probs_sum = tf.reduce_sum(out, axis=-1)
        assert out.shape == (bs, len(EMOTION_LABELS)), (
            f"Wrong output shape: {out.shape}"
        )
        assert np.allclose(probs_sum.numpy(), 1.0, atol=1e-5), "Probs don't sum to 1"
        print(
            f"  Batch={bs}: output={out.shape} ✓ (probs sum={probs_sum.numpy().mean():.6f})"
        )

    benchmark_inference_speed(model, input_shape=dummy_shape)
    print("\n✓ Architecture test PASSED")
    return model


def evaluate_single_model(args, model_path: str, label: str):
    print(f"[Evaluation] Loading model: {model_path}")
    model, metadata, runtime_config = load_model_and_runtime_config(model_path)
    print(f"[Evaluation] Model loaded. Parameters: {model.count_params():,}")
    print(f"[Evaluation] Runtime config: {runtime_config}")

    _, _, _, _, _, _, X_test, _, y_test_raw = load_dataset_for_model(
        args, runtime_config
    )

    prefix = os.path.join("logs", label.replace(" ", "_").lower())
    results = run_full_evaluation(
        model,
        X_test,
        y_test_raw,
        model_label=label,
        save_prefix=prefix,
        runtime_config=runtime_config,
    )

    if args.image:
        predict_single_image(model, args.image, runtime_config)

    results["model_path"] = model_path
    results["metadata"] = metadata
    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("  Emotion Recognition Model Evaluation")
    print("=" * 60)
    print()

    if args.arch_only:
        test_architecture_only(transfer=args.transfer_arch_only)
        return

    if not os.path.exists(args.model):
        print(f"[Warning] Model not found at '{args.model}'.")
        print("[Info] Running architecture test instead (train first to get a model).")
        test_architecture_only(transfer=args.transfer_arch_only)
        return

    primary_results = evaluate_single_model(args, args.model, "Primary Model")

    comparison_results = None
    if args.compare_model:
        if not os.path.exists(args.compare_model):
            print(f"[Warning] Compare model not found at '{args.compare_model}'.")
        else:
            comparison_results = evaluate_single_model(
                args, args.compare_model, "Comparison Model"
            )

    if comparison_results is not None:
        print("\n" + "=" * 72)
        print("  MODEL COMPARISON")
        print("=" * 72)
        print(f"  Primary model accuracy:     {primary_results['accuracy'] * 100:.2f}%")
        print(
            f"  Comparison model accuracy:  {comparison_results['accuracy'] * 100:.2f}%"
        )
        print(
            f"  Primary model top-2:        {primary_results['top2_accuracy'] * 100:.2f}%"
        )
        print(
            f"  Comparison model top-2:     {comparison_results['top2_accuracy'] * 100:.2f}%"
        )
        print("=" * 72)

        comparison_path = "logs/model_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "primary": {
                        "model_path": args.model,
                        "accuracy": primary_results["accuracy"],
                        "top2_accuracy": primary_results["top2_accuracy"],
                        "runtime_config": primary_results["runtime_config"],
                    },
                    "comparison": {
                        "model_path": args.compare_model,
                        "accuracy": comparison_results["accuracy"],
                        "top2_accuracy": comparison_results["top2_accuracy"],
                        "runtime_config": comparison_results["runtime_config"],
                    },
                },
                f,
                indent=2,
            )
        print(f"[Evaluation] Comparison summary saved: {comparison_path}")

    print("\n✓ Evaluation complete!")
    print(f"  Final test accuracy: {primary_results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
