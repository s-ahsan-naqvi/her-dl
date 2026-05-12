"""
train_transfer.py
=================
Two-stage transfer-learning training pipeline for FER-2013.

Stage 1: freeze the backbone and train the new classifier head.
Stage 2: unfreeze the top layers of the backbone and fine-tune with a very
low learning rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import cast

import keras
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras import Model

from data_loader import (
    EMOTION_LABELS,
    TRANSFER_IMG_SIZE,
    compute_class_weights,
    create_transfer_tf_datasets,
    load_fer2013_from_dirs,
    mobilenet_v2_preprocess,
    preprocess_transfer_images,
    visualize_samples,
)
from model import (
    build_transfer_emotion_model,
    compile_model,
    export_saved_model,
    get_backbone_layer,
    get_callbacks,
    save_model_metadata,
    set_backbone_trainable,
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 transfer-learning model on FER2013",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="fer2013", help="Path to the FER2013 directory"
    )

    parser.add_argument(
        "--backbone", type=str, default="MobileNetV2", help="Backbone to use"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=TRANSFER_IMG_SIZE,
        help="Transfer-learning input size",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--head_epochs",
        type=int,
        default=10,
        help="Stage 1: classifier head training epochs",
    )
    parser.add_argument(
        "--fine_tune_epochs", type=int, default=10, help="Stage 2: fine-tuning epochs"
    )
    parser.add_argument(
        "--fine_tune_at",
        type=int,
        default=-30,
        help="Index in backbone.layers where fine-tuning starts; negative values count from the end",
    )
    parser.add_argument(
        "--head_lr", type=float, default=1e-3, help="Learning rate for stage 1"
    )
    parser.add_argument(
        "--fine_tune_lr", type=float, default=1e-5, help="Learning rate for stage 2"
    )
    parser.add_argument(
        "--weights", type=str, default="imagenet", help="Backbone weights"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/transfer_best_model.keras",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/transfer", help="Directory for logs"
    )
    parser.add_argument(
        "--no_class_weights", action="store_true", help="Disable class weights"
    )
    parser.add_argument(
        "--freeze_bn",
        action="store_true",
        default=True,
        help="Freeze BatchNorm layers during fine-tuning",
    )
    parser.add_argument(
        "--no_freeze_bn",
        action="store_false",
        dest="freeze_bn",
        help="Allow BatchNorm layers to update during fine-tuning",
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Early stopping patience"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from an existing .keras model"
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard callback even if tensorboard package is installed",
    )
    return parser.parse_args()


def load_data(args):
    if not os.path.exists(args.data):
        print(f"[Error] Directory not found: {args.data}")
        print("Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        sys.exit(1)
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_from_dirs(
        args.data, input_size=args.input_size, color_mode="rgb"
    )

    return preprocess_transfer_images(X_tr, y_tr, X_v, y_v, X_te, y_te)


def plot_training_history(stage1_history, stage2_history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transfer Learning Training History", fontsize=14, fontweight="bold")

    # Accuracy plot
    ax = axes[0]
    ax.plot(
        stage1_history.history.get("accuracy", []),
        label="Stage 1 Train Acc",
        color="#2196F3",
    )
    ax.plot(
        stage1_history.history.get("val_accuracy", []),
        label="Stage 1 Val Acc",
        color="#2196F3",
        ls="--",
    )
    if (
        stage2_history is not None
        and len(stage2_history.history.get("accuracy", [])) > 0
    ):
        ax.plot(
            stage2_history.history.get("accuracy", []),
            label="Stage 2 Train Acc",
            color="#FF5722",
        )
        ax.plot(
            stage2_history.history.get("val_accuracy", []),
            label="Stage 2 Val Acc",
            color="#FF5722",
            ls="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Loss plot
    ax = axes[1]
    ax.plot(
        stage1_history.history.get("loss", []),
        label="Stage 1 Train Loss",
        color="#2196F3",
    )
    ax.plot(
        stage1_history.history.get("val_loss", []),
        label="Stage 1 Val Loss",
        color="#2196F3",
        ls="--",
    )
    if stage2_history is not None and len(stage2_history.history.get("loss", [])) > 0:
        ax.plot(
            stage2_history.history.get("loss", []),
            label="Stage 2 Train Loss",
            color="#FF5722",
        )
        ax.plot(
            stage2_history.history.get("val_loss", []),
            label="Stage 2 Val Loss",
            color="#FF5722",
            ls="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Training] Curves saved to: {save_path}")


def _best_history_value(history, key):
    values = history.history.get(key, [])
    if isinstance(values, list) and len(values) > 0:
        return float(max(values))
    return 0.0


def save_run_metadata(args, stage1_history, stage2_history, test_results, save_path):
    metadata = {
        "model_type": "transfer_learning",
        "model_name": "EmotionNet_MobileNetV2",
        "backbone": args.backbone,
        "dataset": "FER2013",
        "input_size": [args.input_size, args.input_size],
        "channels": 3,
        "preprocessing": "mobilenet_v2",
        "class_names": EMOTION_LABELS,
        "training": {
            "head_epochs": args.head_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_at": args.fine_tune_at,
            "head_lr": args.head_lr,
            "fine_tune_lr": args.fine_tune_lr,
            "batch_size": args.batch_size,
            "freeze_bn": args.freeze_bn,
            "class_weights_enabled": not args.no_class_weights,
        },
        "stage_1": {
            "epochs_ran": len(stage1_history.history.get("accuracy", [])),
            "best_val_accuracy": _best_history_value(stage1_history, "val_accuracy"),
        },
        "stage_2": {
            "epochs_ran": len(stage2_history.history.get("accuracy", []))
            if stage2_history
            else 0,
            "best_val_accuracy": _best_history_value(stage2_history, "val_accuracy")
            if stage2_history
            else 0.0,
        },
        "test_results": {
            "loss": float(test_results[0]) if len(test_results) > 0 else None,
            "accuracy": float(test_results[1]) if len(test_results) > 1 else None,
            "top_2_accuracy": float(test_results[2]) if len(test_results) > 2 else None,
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=2))
    print(f"[Training] Run metadata saved to: {save_path}")
    return metadata


def build_datasets(args):
    (
        X_train,
        y_train,
        y_train_raw,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
    ) = load_data(args)

    try:
        visualize_samples(
            X_train,
            y_train_raw,
            save_path=os.path.join(args.log_dir, "sample_grid.png"),
        )
    except Exception as exc:
        print(f"[Warning] Could not save sample grid: {exc}")

    train_ds, val_ds = create_transfer_tf_datasets(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )

    X_test_ds = np.asarray(X_test, dtype=np.float32)
    y_test_ds = np.asarray(y_test, dtype=np.float32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_ds, y_test_ds))
    test_ds = (
        test_ds.batch(args.batch_size)
        .map(
            lambda x, y: (mobilenet_v2_preprocess(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    class_weights = (
        None if args.no_class_weights else compute_class_weights(y_train_raw)
    )
    return (
        X_train,
        y_train,
        y_train_raw,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
        train_ds,
        val_ds,
        test_ds,
        class_weights,
    )


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("  FER-2013 Transfer Learning Training")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()

    (
        X_train,
        y_train,
        y_train_raw,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
        train_ds,
        val_ds,
        test_ds,
        class_weights,
    ) = build_datasets(args)

    if args.resume and os.path.exists(args.resume):
        print(f"[Training] Resuming from existing model: {args.resume}")
        model = cast(Model, keras.models.load_model(args.resume))
    else:
        print("[Step 1/2] Building transfer model...")
        model = cast(
            Model,
            build_transfer_emotion_model(
                input_shape=(args.input_size, args.input_size, 3),
                backbone_name=args.backbone,
                weights=args.weights,
                train_backbone=False,
            ),
        )

    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    # Stage 1: train the head only.
    print("\n[Stage 1] Feature extraction: freezing backbone and training head...")
    model = set_backbone_trainable(model, trainable=False)
    model = cast(Model, compile_model(model, learning_rate=args.head_lr))

    stage1_log_dir = os.path.join(args.log_dir, "stage1")
    stage1_callbacks = get_callbacks(
        model_save_path=args.model_path,
        log_dir=stage1_log_dir,
        csv_log_path=os.path.join(stage1_log_dir, "history.csv"),
        patience=args.patience,
        use_tensorboard=(not args.no_tensorboard),
    )

    stage1_start = time.time()
    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=stage1_callbacks,
        class_weight=class_weights,
        verbose="auto",
    )
    stage1_elapsed = time.time() - stage1_start
    print(f"[Stage 1] Completed in {stage1_elapsed / 60:.1f} minutes")

    # Stage 2: fine-tune the top layers of the backbone.
    print("\n[Stage 2] Fine-tuning top backbone layers...")
    backbone = get_backbone_layer(model)
    total_layers = len(backbone.layers)
    fine_tune_at = args.fine_tune_at
    if fine_tune_at < 0:
        fine_tune_at = total_layers + fine_tune_at
    fine_tune_at = max(0, min(fine_tune_at, total_layers))

    model = set_backbone_trainable(
        model,
        trainable=True,
        fine_tune_at=fine_tune_at,
        freeze_batch_norm=args.freeze_bn,
    )
    model = cast(Model, compile_model(model, learning_rate=args.fine_tune_lr))

    stage2_log_dir = os.path.join(args.log_dir, "stage2")
    stage2_callbacks = get_callbacks(
        model_save_path=args.model_path,
        log_dir=stage2_log_dir,
        csv_log_path=os.path.join(stage2_log_dir, "history.csv"),
        patience=args.patience,
        use_tensorboard=(not args.no_tensorboard),
    )

    stage2_start = time.time()
    stage2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=stage2_callbacks,
        class_weight=class_weights,
        verbose="auto",
    )
    stage2_elapsed = time.time() - stage2_start
    print(f"[Stage 2] Completed in {stage2_elapsed / 60:.1f} minutes")

    print("\n[Evaluation] Loading best checkpoint for final test evaluation...")
    if os.path.exists(args.model_path):
        best_model = cast(Model, keras.models.load_model(args.model_path))
    else:
        best_model = model

    raw_test_results = best_model.evaluate(test_ds, verbose="auto")
    test_results = (
        list(raw_test_results)
        if isinstance(raw_test_results, (list, tuple))
        else [float(raw_test_results)]
    )
    metric_names = best_model.metrics_names

    print("\n" + "=" * 50)
    print("  FINAL TEST RESULTS")
    print("=" * 50)
    for name, val in zip(metric_names, test_results):
        print(f"  {name:>20}: {val:.4f}")
    print("=" * 50)

    plot_training_history(
        stage1_history,
        stage2_history,
        os.path.join(args.log_dir, "training_curves.png"),
    )

    metadata = save_run_metadata(
        args,
        stage1_history,
        stage2_history,
        test_results,
        os.path.join(args.log_dir, "training_metadata.json"),
    )
    save_model_metadata(args.model_path, metadata)

    saved_model_path = os.path.join(args.log_dir, "savedmodel")
    export_ok, export_errors = export_saved_model(best_model, saved_model_path)
    if export_ok:
        print(f"[Saving] SavedModel exported to: {saved_model_path}")
    else:
        for err in export_errors:
            print(f"[Warning] {err}")

    print("\n✓ Transfer-learning training complete!")
    print(f"  Best model: {args.model_path}")
    print(f"  Metadata: {os.path.join(args.log_dir, 'training_metadata.json')}")
    print(
        f"  Model metadata: {os.path.splitext(args.model_path)[0] + '.metadata.json'}"
    )


if __name__ == "__main__":
    main()
