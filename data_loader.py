"""
data_loader.py
==============
FER2013 dataset loading, preprocessing, and augmentation utilities.

The baseline CNN still uses 48x48 grayscale inputs. The new transfer-learning
pipeline converts FER images to RGB and resizes them for pretrained backbones
such as MobileNetV2.
"""

from __future__ import annotations

import os
from collections import Counter

import cv2
import keras
import matplotlib
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Class labels exactly as in FER2013
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Angry": "#FF4444",
    "Disgust": "#8B4513",
    "Fear": "#9B59B6",
    "Happy": "#F1C40F",
    "Sad": "#3498DB",
    "Surprise": "#E67E22",
    "Neutral": "#95A5A6",
}

IMG_SIZE = 48
TRANSFER_IMG_SIZE = 96
NUM_CLASSES = 7


def mobilenet_v2_preprocess(x):
    """Apply MobileNetV2 preprocessing without importing the submodule directly."""
    mobilenet_module = getattr(tf.keras.applications, "mobilenet_v2", None)
    if mobilenet_module is None:
        return x
    return mobilenet_module.preprocess_input(x)


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def _read_image(img_path: str, input_size: int, color_mode: str):
    if color_mode == "grayscale":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    elif color_mode == "rgb":
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("color_mode must be 'grayscale' or 'rgb'")

    if img is None:
        return None

    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def load_fer2013_from_dirs(
    dataset_dir: str,
    input_size: int = IMG_SIZE,
    color_mode: str = "grayscale",
):
    """Load the FER2013 dataset from train/ and test/ directory trees."""
    print(f"[DataLoader] Loading FER2013 from directory: {dataset_dir}")

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Could not find train/test directories in {dataset_dir}"
        )

    def load_split(split_dir):
        pixels = []
        labels = []
        for cls_idx, label in enumerate(EMOTION_LABELS):
            class_dir = os.path.join(split_dir, label.lower())
            if not os.path.exists(class_dir):
                print(f"[Warning] Missing directory: {class_dir}")
                continue

            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img = _read_image(
                    img_path, input_size=input_size, color_mode=color_mode
                )
                if img is not None:
                    pixels.append(img)
                    labels.append(cls_idx)

        return np.array(pixels, dtype=np.float32), np.array(labels, dtype=np.int64)

    print("  Loading training set...")
    X_train_full, y_train_full = load_split(train_dir)
    print("  Loading test set...")
    X_test, y_test = load_split(test_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    print("\n[DataLoader] Dataset loaded successfully!")
    print(f"  Train:      {len(X_train):>6} samples")
    print(f"  Validation: {len(X_val):>6} samples")
    print(f"  Test:       {len(X_test):>6} samples")

    counts = Counter(y_train)
    print("\n[DataLoader] Train class distribution:")
    for cls_idx, label in enumerate(EMOTION_LABELS):
        n = counts.get(cls_idx, 0)
        pct = n / len(y_train) * 100 if len(y_train) > 0 else 0
        print(f"  {label:>10}: {n:5d} ({pct:4.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _ensure_grayscale_channels(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X[..., np.newaxis]
    if X.ndim == 4 and X.shape[-1] == 1:
        return X
    if X.ndim == 4 and X.shape[-1] == 3:
        return np.mean(X, axis=-1, keepdims=True)
    raise ValueError(f"Expected grayscale-compatible array, got shape {X.shape}")


def _ensure_rgb_channels(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return np.repeat(X[..., np.newaxis], 3, axis=-1)
    if X.ndim == 4 and X.shape[-1] == 1:
        return np.repeat(X, 3, axis=-1)
    if X.ndim == 4 and X.shape[-1] == 3:
        return X
    raise ValueError(f"Expected RGB-compatible array, got shape {X.shape}")


def preprocess_images(X_train, y_train, X_val, y_val, X_test, y_test):
    """Normalize grayscale inputs to [0,1] and one-hot encode labels."""
    X_train = _ensure_grayscale_channels(X_train).astype(np.float32) / 255.0
    X_val = _ensure_grayscale_channels(X_val).astype(np.float32) / 255.0
    X_test = _ensure_grayscale_channels(X_test).astype(np.float32) / 255.0

    y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_ohe = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_ohe = keras.utils.to_categorical(y_test, NUM_CLASSES)

    print("[DataLoader] Baseline preprocessing complete.")
    print(
        f"  X_train: {X_train.shape}, dtype={X_train.dtype}, range=[{X_train.min():.2f}, {X_train.max():.2f}]"
    )
    print(f"  y_train: {y_train_ohe.shape}")

    return (
        X_train,
        y_train_ohe,
        y_train,
        X_val,
        y_val_ohe,
        y_val,
        X_test,
        y_test_ohe,
        y_test,
    )


def preprocess_transfer_images(X_train, y_train, X_val, y_val, X_test, y_test):
    """Prepare RGB arrays for transfer learning; preprocessing happens in tf.data."""
    X_train = _ensure_rgb_channels(X_train).astype(np.float32)
    X_val = _ensure_rgb_channels(X_val).astype(np.float32)
    X_test = _ensure_rgb_channels(X_test).astype(np.float32)

    y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_ohe = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_ohe = keras.utils.to_categorical(y_test, NUM_CLASSES)

    print("[DataLoader] Transfer-learning preprocessing complete.")
    print(
        f"  X_train: {X_train.shape}, dtype={X_train.dtype}, range=[{X_train.min():.2f}, {X_train.max():.2f}]"
    )
    print(f"  y_train: {y_train_ohe.shape}")

    return (
        X_train,
        y_train_ohe,
        y_train,
        X_val,
        y_val_ohe,
        y_val,
        X_test,
        y_test_ohe,
        y_test,
    )


# ---------------------------------------------------------------------------
# Class weights & augmentation
# ---------------------------------------------------------------------------


def compute_class_weights(y_train_raw):
    """Compute balanced class weights to handle severe class imbalance."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train_raw
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

    print("\n[DataLoader] Class weights (balanced):")
    for i, label in enumerate(EMOTION_LABELS):
        print(f"  {label:>10}: {class_weight_dict[i]:.4f}")

    return class_weight_dict


def build_augmentation_pipeline(include_brightness: bool = False):
    """Baseline augmentation pipeline for normalized grayscale CNN inputs."""
    layers_list = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03, fill_mode="nearest"),
        layers.RandomZoom(height_factor=0.08, width_factor=0.08),
        layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
        layers.RandomContrast(factor=0.06),
    ]

    if include_brightness:
        layers_list.append(layers.RandomBrightness(factor=0.06, value_range=(0.0, 1.0)))

    return keras.Sequential(layers_list, name="augmentation_pipeline")


def build_transfer_augmentation_pipeline():
    """Moderate augmentation for RGB transfer-learning training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03, fill_mode="nearest"),
            layers.RandomZoom(height_factor=0.08, width_factor=0.08),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomBrightness(factor=0.06),
            layers.RandomContrast(factor=0.06),
        ],
        name="transfer_augmentation_pipeline",
    )


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------


def create_tf_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=64,
    augment: bool = True,
    include_brightness: bool = False,
):
    """Create optimized datasets for the baseline grayscale CNN."""
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True).batch(
        batch_size
    )

    if augment:
        augmentation = build_augmentation_pipeline(
            include_brightness=include_brightness
        )
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    print(
        f"\n[DataLoader] tf.data pipelines created (batch_size={batch_size}, augment={augment}, brightness={include_brightness})"
    )
    return train_ds, val_ds


def create_transfer_tf_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32,
    preprocessing="mobilenet_v2",
):
    """Create tf.data pipelines for transfer learning with MobileNetV2 preprocessing."""
    augmentation = build_transfer_augmentation_pipeline()
    preprocess_fn = None
    if preprocessing == "mobilenet_v2":
        preprocess_fn = mobilenet_v2_preprocess

    def _train_map(x, y):
        x = augmentation(x, training=True)
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        return x, y

    def _val_map(x, y):
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(_train_map, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = (
        val_ds.batch(batch_size)
        .map(_val_map, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print(
        f"\n[DataLoader] Transfer tf.data pipelines created (batch_size={batch_size}, preprocessing={preprocessing})"
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Visualization & smoke-test helpers
# ---------------------------------------------------------------------------


def visualize_samples(X, y_raw, n_samples=14, save_path="logs/sample_grid.png"):
    """Visualize one sample per class and the class distribution."""
    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    fig.suptitle("FER2013 Sample Images", fontsize=14, fontweight="bold")

    for cls_idx in range(NUM_CLASSES):
        mask = y_raw == cls_idx
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        sample_idx = np.random.choice(idxs)
        img = X[sample_idx]

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        elif img.ndim == 3 and img.shape[-1] == 3 and img.max() > 1.0:
            img = np.clip(img / 255.0, 0.0, 1.0)

        ax = axes[0, cls_idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(EMOTION_LABELS[cls_idx], fontsize=9)
        ax.axis("off")

    counts = Counter(y_raw)
    ax2 = axes[1, :]
    for a in ax2:
        a.axis("off")

    ax_bar = fig.add_axes((0.1, 0.05, 0.8, 0.35))
    cls_counts = [counts[i] for i in range(NUM_CLASSES)]
    colors = [EMOTION_COLORS[label] for label in EMOTION_LABELS]
    bars = ax_bar.bar(
        EMOTION_LABELS, cls_counts, color=colors, edgecolor="black", linewidth=0.5
    )
    ax_bar.set_title("Class Distribution (Training Set)", fontsize=10)
    ax_bar.set_ylabel("Sample Count")
    for bar, count in zip(bars, cls_counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 50,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_bar.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[DataLoader] Sample grid saved to: {save_path}")
    plt.close()
