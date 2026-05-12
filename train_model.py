from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import (
    compute_class_weights,
    create_tf_datasets,
    load_fer2013_from_dirs,
    preprocess_images,
    visualize_samples,
)
from model import (
    EMOTION_LABELS,
    build_emotion_model,
    compile_model,
    save_model_metadata,
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------------------------------------------------------------------
# CLI / data helpers
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Emotion Recognition CNN on FER2013",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="fer2013", help="Path to fer2013 directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to .keras model or .weights.h5 checkpoint to resume training from",
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--include_brightness_augment",
        action="store_true",
        help="Enable normalized RandomBrightness augmentation for baseline training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.keras",
        help="Where to save the final native Keras model",
    )
    parser.add_argument(
        "--no_savedmodel",
        action="store_true",
        help="Skip optional TensorFlow SavedModel export",
    )
    parser.add_argument(
        "--savedmodel_path",
        type=str,
        default="models/emotion_model_savedmodel",
        help="Where to export an optional TensorFlow SavedModel",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard callback even if tensorboard package is installed",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable balanced class weights during baseline training",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Categorical crossentropy label smoothing for baseline training",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="Metric monitored for checkpointing, LR reduction, and early stopping",
    )
    parser.add_argument(
        "--overfit_samples",
        type=int,
        default=0,
        help="Debug mode: train/validate/evaluate on the first N training samples",
    )
    return parser.parse_args()


def load_data(args):
    """Load and preprocess FER2013 data from directory."""
    if not args.data or not os.path.exists(args.data):
        print(f"[Error] Dataset directory not found: {args.data}")
        print("Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        sys.exit(1)

    X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_from_dirs(args.data)
    return preprocess_images(X_tr, y_tr, X_v, y_v, X_te, y_te)


# ---------------------------------------------------------------------------
# Plotting / metadata
# ---------------------------------------------------------------------------


def plot_training_history(history, save_path="logs/training_curves.png"):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — EmotionNet", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(history.history["accuracy"], label="Train Accuracy", color="#2196F3", lw=2)
    ax.plot(
        history.history["val_accuracy"],
        label="Val Accuracy",
        color="#FF5722",
        lw=2,
        ls="--",
    )
    if "top_2_accuracy" in history.history:
        ax.plot(
            history.history["top_2_accuracy"],
            label="Train Top-2 Acc",
            color="#4CAF50",
            lw=1.5,
            alpha=0.7,
        )
    if "val_top_2_accuracy" in history.history:
        ax.plot(
            history.history["val_top_2_accuracy"],
            label="Val Top-2 Acc",
            color="#FF9800",
            lw=1.5,
            ls="--",
            alpha=0.7,
        )
    ax.axhline(y=0.655, color="gray", ls=":", lw=1.5, label="Human Accuracy (65.5%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(history.history["loss"], label="Train Loss", color="#2196F3", lw=2)
    ax.plot(
        history.history["val_loss"], label="Val Loss", color="#FF5722", lw=2, ls="--"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Training] Curves saved to: {save_path}")
    plt.close()


def _best_history_value(history, key, mode: str = "max"):
    values = history.history.get(key, [])
    if isinstance(values, list) and len(values) > 0:
        return float(min(values) if mode == "min" else max(values))
    return 0.0


def _metric_value(
    test_results: Sequence[float], metric_names: Sequence[str], name: str
):
    if name in metric_names:
        return float(test_results[metric_names.index(name)])
    return None


def save_training_metadata(
    args,
    history,
    test_results: Sequence[float],
    metric_names: Sequence[str],
    save_path="logs/training_metadata.json",
) -> Dict[str, Any]:
    """Save all training metadata for reproducibility."""
    test_loss = _metric_value(test_results, metric_names, "loss")
    test_accuracy = _metric_value(test_results, metric_names, "accuracy")
    test_top_2_accuracy = _metric_value(test_results, metric_names, "top_2_accuracy")

    metadata: Dict[str, Any] = {
        "model_type": "baseline_cnn",
        "model": "EmotionNet_MobileInspired",
        "dataset": "FER2013",
        "epochs_run": len(history.history.get("accuracy", [])),
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "class_weights_enabled": not args.no_class_weights,
        "augmentation_enabled": not args.no_augment,
        "brightness_augmentation_enabled": args.include_brightness_augment,
        "monitor": args.monitor,
        "overfit_samples": args.overfit_samples,
        "best_monitor_value": _best_history_value(
            history,
            args.monitor,
            "min" if "loss" in args.monitor.lower() else "max",
        ),
        "best_val_accuracy": _best_history_value(history, "val_accuracy"),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_top_2_accuracy": test_top_2_accuracy,
        "emotion_classes": EMOTION_LABELS,
        "input_shape": [48, 48, 1],
        "input_size": [48, 48],
        "channels": 1,
        "preprocessing": "grayscale_0_1",
        "architecture": {
            "input_size": "48x48x1",
            "depthwise_separable_convolutions": True,
            "global_average_pooling": True,
            "regularization": "L2 + Dropout",
            "label_smoothing": args.label_smoothing,
        },
        "artifacts": {
            "keras_model": args.model_path,
            "best_weights": _weights_checkpoint_path(args.model_path),
            "training_history_csv": "logs/training_history.csv",
            "training_curves": "logs/training_curves.png",
        },
        "training_time_note": "See logs/training_history.csv for per-epoch data",
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Training] Metadata saved to: {save_path}")
    return metadata


# ---------------------------------------------------------------------------
# Robust model persistence
# ---------------------------------------------------------------------------


def _weights_checkpoint_path(model_path: str) -> str:
    """Return a Keras-compatible weights checkpoint path for a model path."""
    path = Path(model_path)
    return str(path.with_suffix(".weights.h5"))


def _safe_unlink(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        print(f"[Warning] Could not remove existing file '{path}': {exc}")


def build_training_callbacks(
    model_save_path: str,
    log_dir: str = "logs/",
    monitor: str = "val_loss",
    patience: int = 15,
    use_tensorboard: bool = True,
) -> Tuple[List[tf.keras.callbacks.Callback], str]:
    """Create callbacks that checkpoint weights instead of full model objects.

    Whole-model checkpointing can fail on some TensorFlow/Keras combinations
    because optimizer/config internals contain tracked dictionaries. We only
    checkpoint weights during training, then rebuild and save a clean model at
    the end.
    """
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    weights_path = _weights_checkpoint_path(model_save_path)
    csv_log_path = os.path.join(log_dir, "training_history.csv")
    monitor_mode = "min" if "loss" in monitor.lower() else "max"

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode=monitor_mode,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode=monitor_mode,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode=monitor_mode,
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path, append=False),
    ]

    if use_tensorboard:
        try:
            callbacks.insert(
                -1, tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            )
        except Exception as exc:
            print(f"[Warning] TensorBoard callback unavailable: {exc}")

    return callbacks, weights_path


def build_clean_model_from_weights(
    weights_path: str, learning_rate: float, label_smoothing: float = 0.0
) -> tf.keras.Model:
    """Rebuild the baseline model and load weights without optimizer state."""
    clean_model = cast(tf.keras.Model, build_emotion_model())
    clean_model = cast(
        tf.keras.Model,
        compile_model(
            clean_model,
            learning_rate=learning_rate,
            label_smoothing=label_smoothing,
        ),
    )
    clean_model.load_weights(weights_path)
    return clean_model


def load_resume_model(
    resume_path: str, learning_rate: float, label_smoothing: float = 0.0
) -> tf.keras.Model:
    """Load a resume checkpoint from either native Keras or weights-only format."""
    model = cast(tf.keras.Model, build_emotion_model())
    model = cast(
        tf.keras.Model,
        compile_model(
            model,
            learning_rate=learning_rate,
            label_smoothing=label_smoothing,
        ),
    )

    if resume_path.endswith(".weights.h5"):
        model.load_weights(resume_path)
        print(f"[Training] Loaded resume weights from: {resume_path}")
        return model

    loaded = cast(
        tf.keras.Model, tf.keras.models.load_model(resume_path, compile=False)
    )
    model.set_weights(loaded.get_weights())
    print(f"[Training] Loaded resume model weights from: {resume_path}")
    return model


def save_native_keras_model(model: tf.keras.Model, model_path: str) -> bool:
    """Save a model in native `.keras` format with no optimizer state."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    _safe_unlink(model_path)

    try:
        cast(Any, model).save(model_path, include_optimizer=False)
        print(f"[Saving] Native Keras model saved to: {model_path}")
        return True
    except Exception as exc:
        print(f"[Warning] Native Keras save failed: {exc}")
        return False


def export_inference_saved_model(model: tf.keras.Model, export_path: str) -> bool:
    """Export a lightweight inference SavedModel using a plain tf.Module wrapper.

    Keras 3's `model.export()` is preferred when available, but this fallback
    avoids serializing the Keras object itself. It only exposes a TensorFlow
    serving function that calls the already-built model.
    """
    if os.path.exists(export_path):
        import shutil

        shutil.rmtree(export_path)

    sample_shape = tuple(int(dim) for dim in model.input_shape[1:])
    signature = [
        tf.TensorSpec(shape=(None, *sample_shape), dtype=tf.float32, name="image_input")
    ]

    if hasattr(model, "export"):
        try:
            model.export(export_path)
            print(f"[Saving] SavedModel exported to: {export_path}")
            return True
        except Exception as exc:
            print(
                f"[Warning] Keras SavedModel export failed; trying tf.Module fallback: {exc}"
            )

    class InferenceModule(tf.Module):
        def __init__(self, keras_model: tf.keras.Model):
            super().__init__()
            self.keras_model = keras_model

        @tf.function(input_signature=signature)
        def serving_default(self, image_input):
            return {"emotion_output": self.keras_model(image_input, training=False)}

    try:
        module = InferenceModule(model)
        tf.saved_model.save(
            module,
            export_path,
            signatures={"serving_default": module.serving_default},
        )
        print(f"[Saving] SavedModel exported to: {export_path}")
        return True
    except Exception as exc:
        print(f"[Warning] tf.Module SavedModel export failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print("=" * 60)
    print("  Real-Time Human Emotion Recognition — Training")
    print("  Group: Abdur Rahman Goraya (2022035), Ahsan Naqvi (2022073)")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version:      {tf.keras.__version__}")
    print(f"GPU available:     {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()

    print("[Step 1/5] Loading and preprocessing data...")
    (
        X_train,
        y_train,
        y_train_raw,
        X_val,
        y_val,
        _y_val_raw,
        X_test,
        y_test,
        _y_test_raw,
    ) = load_data(args)

    if args.overfit_samples > 0:
        n = min(int(args.overfit_samples), len(X_train))
        print(
            f"[Debug] Tiny-overfit mode enabled: using first {n} training samples for train/val/test."
        )
        X_train = X_train[:n]
        y_train = y_train[:n]
        y_train_raw = y_train_raw[:n]
        X_val = X_train
        y_val = y_train
        X_test = X_train
        y_test = y_train

    try:
        visualize_samples(X_train, y_train_raw, save_path="logs/sample_grid.png")
    except Exception as exc:
        print(f"[Warning] Could not save sample grid: {exc}")

    print("\n[Step 2/5] Building model...")
    if args.resume and os.path.exists(args.resume):
        print(f"[Training] Resuming from: {args.resume}")
        model = load_resume_model(
            args.resume,
            learning_rate=args.lr,
            label_smoothing=args.label_smoothing,
        )
    else:
        model = cast(tf.keras.Model, build_emotion_model())
        model = cast(
            tf.keras.Model,
            compile_model(
                model,
                learning_rate=args.lr,
                label_smoothing=args.label_smoothing,
            ),
        )

    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")

    print("\n[Step 3/5] Computing class weights...")
    class_weights = (
        None if args.no_class_weights else compute_class_weights(y_train_raw)
    )

    print("\n[Step 4/5] Building data pipelines...")
    train_ds, val_ds = create_tf_datasets(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        augment=(not args.no_augment),
        include_brightness=args.include_brightness_augment,
    )

    print(f"\n[Step 5/5] Training for up to {args.epochs} epochs...")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Label smoothing:  {args.label_smoothing}")
    print(f"  Class weights:    {'disabled' if args.no_class_weights else 'balanced'}")
    print(f"  Augmentation:     {'disabled' if args.no_augment else 'enabled'}")
    print(
        f"  Brightness aug:   {'enabled' if args.include_brightness_augment else 'disabled'}"
    )
    print(f"  Monitor:          {args.monitor}")
    if args.overfit_samples > 0:
        print(f"  Overfit samples:  {args.overfit_samples}")
    print(f"  Best weights:     {_weights_checkpoint_path(args.model_path)}")
    print(f"  Final Keras model:{args.model_path}")
    print()

    callbacks, weights_path = build_training_callbacks(
        model_save_path=args.model_path,
        monitor=args.monitor,
        use_tensorboard=(not args.no_tensorboard),
    )

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose="auto",
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed / 60:.1f} minutes")

    if not os.path.exists(weights_path):
        print(
            "[Warning] Best-weights checkpoint was not created; saving current weights."
        )
        model.save_weights(weights_path)

    print("\n[Evaluation] Rebuilding clean model from best weights...")
    best_model = build_clean_model_from_weights(
        weights_path,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
    )

    print("[Evaluation] Evaluating on held-out test set...")
    X_test_ds = np.asarray(X_test, dtype=np.float32)
    y_test_ds = np.asarray(y_test, dtype=np.float32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_ds, y_test_ds)).batch(64)
    raw_test_results = best_model.evaluate(test_ds, verbose="auto")
    test_results = (
        [float(x) for x in raw_test_results]
        if isinstance(raw_test_results, (list, tuple))
        else [float(raw_test_results)]
    )

    metric_names = list(best_model.metrics_names)
    print("\n" + "=" * 40)
    print("  FINAL TEST RESULTS")
    print("=" * 40)
    for name, val in zip(metric_names, test_results):
        print(f"  {name:>20}: {val:.4f}")
    print("\n  Human accuracy on FER2013: ~65.5%")
    print("=" * 40)

    plot_training_history(history)
    metadata = save_training_metadata(args, history, test_results, metric_names)

    save_native_keras_model(best_model, args.model_path)
    model_metadata_path = save_model_metadata(args.model_path, metadata)
    print(f"[Saving] Sidecar metadata saved to: {model_metadata_path}")

    if args.no_savedmodel:
        print("[Saving] SavedModel export skipped (--no_savedmodel).")
    else:
        export_inference_saved_model(best_model, args.savedmodel_path)

    print("\n✓ Training complete!")
    print(f"  Best weights:     {weights_path}")
    print(f"  Best model:       {args.model_path}")
    print("  Training curves:  logs/training_curves.png")
    print("  Metadata:         logs/training_metadata.json")


if __name__ == "__main__":
    main()
