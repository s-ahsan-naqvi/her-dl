"""
model.py
========
Emotion recognition models for FER-2013.

This module keeps the existing lightweight CNN baseline intact and adds a
transfer-learning path built around MobileNetV2 for the new workflow.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Model, layers
from keras.regularizers import l2

# FER2013 emotion class labels (7 classes as defined by Goodfellow et al.)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = 7
IMG_SIZE = 48  # Baseline CNN input size
TRANSFER_IMG_SIZE = 96  # Good lightweight transfer-learning input size
TRANSFER_CHANNELS = 3


# ---------------------------------------------------------------------------
# Baseline custom CNN
# ---------------------------------------------------------------------------


def depthwise_separable_block(x, filters, strides=1, dropout_rate=0.1):
    """Depthwise separable convolution block used by the baseline CNN."""
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def standard_conv_block(x, filters, kernel_size=3, strides=1):
    """Standard Conv -> BN -> ReLU block for the initial baseline stem layers."""
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x


def build_emotion_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Build the existing lightweight grayscale CNN baseline."""
    inputs = keras.Input(shape=input_shape, name="image_input")

    x = standard_conv_block(inputs, filters=32, kernel_size=3)
    x = standard_conv_block(x, filters=32, kernel_size=3)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(x, filters=64, dropout_rate=0.1)
    x = depthwise_separable_block(x, filters=64, dropout_rate=0.1)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(x, filters=128, dropout_rate=0.15)
    x = depthwise_separable_block(x, filters=128, dropout_rate=0.15)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(x, filters=256, dropout_rate=0.2)
    x = depthwise_separable_block(x, filters=256, dropout_rate=0.2)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = depthwise_separable_block(x, filters=512, dropout_rate=0.25)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="EmotionNet_MobileInspired")


# ---------------------------------------------------------------------------
# Transfer learning model
# ---------------------------------------------------------------------------


def _build_backbone(
    backbone_name: str, input_shape: Tuple[int, int, int], weights: str
):
    backbone_name = backbone_name.lower()
    if backbone_name != "mobilenetv2":
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Only 'MobileNetV2' is implemented right now."
        )

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    backbone._name = "mobilenetv2_backbone"
    return backbone


def build_transfer_emotion_model(
    input_shape=(TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE, TRANSFER_CHANNELS),
    num_classes=NUM_CLASSES,
    backbone_name: str = "MobileNetV2",
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    train_backbone: bool = False,
) -> Model:
    """Build a MobileNetV2-based transfer-learning model for FER-2013."""
    inputs = keras.Input(shape=input_shape, name="image_input")

    backbone = _build_backbone(backbone_name, input_shape, weights)
    backbone.trainable = train_backbone

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.2, name="head_dropout_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_dense_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EmotionNet_MobileNetV2")
    return model


def get_backbone_layer(model: Model, backbone_layer_name: str = "mobilenetv2_backbone"):
    """Return the nested backbone layer from a transfer-learning model."""
    try:
        return model.get_layer(backbone_layer_name)
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, Model) or "mobilenet" in layer.name.lower():
                return layer
        raise


def set_backbone_trainable(
    model: Model,
    trainable: bool,
    backbone_layer_name: str = "mobilenetv2_backbone",
    fine_tune_at: Optional[int] = None,
    freeze_batch_norm: bool = True,
):
    """Freeze/unfreeze the backbone, optionally keeping early layers frozen."""
    backbone = get_backbone_layer(model, backbone_layer_name=backbone_layer_name)
    backbone.trainable = trainable

    if not trainable:
        for layer in backbone.layers:
            layer.trainable = False
        return model

    layers_in_backbone = backbone.layers
    if fine_tune_at is None:
        fine_tune_at = 0
    fine_tune_at = max(0, min(int(fine_tune_at), len(layers_in_backbone)))

    for idx, layer in enumerate(layers_in_backbone):
        layer.trainable = idx >= fine_tune_at
        if freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    return model


# ---------------------------------------------------------------------------
# Compile / callbacks
# ---------------------------------------------------------------------------


def compile_model(model, learning_rate=1e-3, label_smoothing=0.1):
    """Compile a model with Adam and categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        ],
    )
    return model


def get_callbacks(
    model_save_path: str = "models/best_model.keras",
    log_dir: str = "logs/",
    csv_log_path: Optional[str] = None,
    monitor: str = "val_accuracy",
    patience: int = 15,
    use_tensorboard: bool = True,
):
    """Return a standard callback bundle for training runs.

    The TensorBoard callback is optional; if `use_tensorboard` is True we attempt
    to create it but fall back gracefully if TensorBoard isn't installed.
    """
    if csv_log_path is None:
        csv_log_path = os.path.join(log_dir, "training_history.csv")

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path) or ".", exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_log_path, append=False),
    ]

    if use_tensorboard:
        try:
            tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.insert(-1, tb_cb)  # insert before CSVLogger
        except Exception as exc:  # ImportError or TBNotInstalledError
            print(f"[Warning] TensorBoard callback unavailable: {exc}")

    return callbacks


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_model_metadata_path(model_path: str) -> str:
    """Return the sidecar metadata path for a saved model file."""
    base, _ = os.path.splitext(model_path)
    return f"{base}.metadata.json"


def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> str:
    """Persist model metadata alongside the model file."""
    metadata_path = get_model_metadata_path(model_path)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """Load model metadata if the sidecar JSON exists."""
    metadata_path = get_model_metadata_path(model_path)
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_saved_model(model: Model, export_path: str) -> Tuple[bool, List[str]]:
    """Export a model to SavedModel format with safe fallbacks.

    Returns (success, error_messages).
    """
    errors: List[str] = []
    stripped_model: Optional[Model] = None

    if hasattr(model, "export"):
        try:
            model.export(export_path)
            return True, errors
        except Exception as exc:
            errors.append(f"Keras export failed: {exc}")

    try:
        stripped_model = keras.models.clone_model(model)
        stripped_model.set_weights(model.get_weights())
        if hasattr(stripped_model, "export"):
            stripped_model.export(export_path)
            return True, errors
    except Exception as exc:
        errors.append(f"Keras export (stripped model) failed: {exc}")

    try:
        export_target = stripped_model if stripped_model is not None else model
        tf.saved_model.save(export_target, export_path)
        return True, errors
    except Exception as exc:
        errors.append(f"tf.saved_model.save failed: {exc}")

    return False, errors


def infer_runtime_config(
    model: Model, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Infer runtime input size / channels / preprocessing from metadata and model shape."""
    metadata = metadata or {}

    input_shape_meta = metadata.get("input_shape")
    if isinstance(input_shape_meta, (list, tuple)) and len(input_shape_meta) >= 3:
        input_shape = [
            int(input_shape_meta[0]),
            int(input_shape_meta[1]),
            int(input_shape_meta[2]),
        ]
    else:
        model_shape = getattr(model, "input_shape", None)
        if isinstance(model_shape, list):
            model_shape = model_shape[0]
        if isinstance(model_shape, (list, tuple)) and len(model_shape) >= 4:
            input_shape = [
                int(model_shape[1]),
                int(model_shape[2]),
                int(model_shape[3]),
            ]
        else:
            input_shape = [IMG_SIZE, IMG_SIZE, 1]

    input_size_meta = metadata.get("input_size")
    if isinstance(input_size_meta, (list, tuple)) and len(input_size_meta) >= 2:
        input_size = [int(input_size_meta[0]), int(input_size_meta[1])]
    else:
        input_size = [int(input_shape[0]), int(input_shape[1])]

    channels_meta = metadata.get("channels")
    if channels_meta is None and len(input_shape) >= 3:
        channels = int(input_shape[2])
    elif channels_meta is None:
        channels = 1
    else:
        channels = int(channels_meta)

    preprocessing = metadata.get("preprocessing")
    if preprocessing is None:
        preprocessing = (
            "mobilenet_v2" if channels == TRANSFER_CHANNELS else "grayscale_0_1"
        )

    model_type = metadata.get("model_type")
    if not isinstance(model_type, str):
        model_type = "transfer_learning" if channels == 3 else "baseline_cnn"

    backbone = (
        metadata.get("backbone") if isinstance(metadata.get("backbone"), str) else None
    )

    return {
        "input_shape": input_shape,
        "input_size": input_size,
        "channels": channels,
        "preprocessing": preprocessing,
        "model_type": model_type,
        "backbone": backbone,
    }


if __name__ == "__main__":
    # Quick sanity checks for both model paths.
    model = build_emotion_model()
    model = compile_model(model)
    model.summary()

    total = model.count_params()
    print(f"\nBaseline parameters: {total:,}")
    print(f"Model size estimate: ~{total * 4 / 1e6:.1f} MB (float32)")

    dummy_input = np.random.randn(4, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    dummy_output = model(dummy_input, training=False)
    print(f"\nBaseline input shape:  {dummy_input.shape}")
    print(f"Baseline output shape: {dummy_output.shape}")
    print(f"Output sum (should be ~4.0): {tf.reduce_sum(dummy_output).numpy():.4f}")

    transfer_model = build_transfer_emotion_model()
    transfer_model = compile_model(transfer_model)
    transfer_model.summary()
    transfer_dummy = np.random.randn(2, TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE, 3).astype(
        np.float32
    )
    transfer_out = transfer_model(transfer_dummy, training=False)
    print(f"\nTransfer input shape:  {transfer_dummy.shape}")
    print(f"Transfer output shape: {transfer_out.shape}")
    print(
        f"Transfer output sum (should be ~2.0): {tf.reduce_sum(transfer_out).numpy():.4f}"
    )
    print("\n✓ Model architectures verified.")
