"""
Out-of-Distribution (OOD) Binary Detector Model Architecture.

This module implements an independent binary CNN classifier designed to determine
whether an input image belongs to the supported Fashion-MNIST clothing domain (Label 1)
or an out-of-distribution non-clothing domain (Label 0).

It operates as an isolated pre-classifier gate before the existing 10-class Fashion-MNIST classifier:
Image Upload -> Production Preprocessing -> OOD Detector -> Gate Decision -> Existing CNN (if accepted).

Architecture Specifications:
- Input shape: (28, 28, 1), float32, range [0.0, 1.0]
- Feature Extractor: 2 Convolutional blocks (Conv2D + ReLU + MaxPooling2D)
- Classification Head: Flatten -> Dense (64, ReLU) -> Dropout (0.3) -> Dense (1, Sigmoid)
- Output: Single binary probability P(clothing) in [0.0, 1.0]
- Loss Function: Binary Crossentropy
"""

from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (28, 28, 1)
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_DROPOUT_RATE: float = 0.3
DEFAULT_CONV1_FILTERS: int = 32
DEFAULT_CONV2_FILTERS: int = 64
DEFAULT_DENSE_UNITS: int = 64
DEFAULT_RANDOM_SEED: int = 42


def build_ood_model(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    compile_model: bool = True,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int | None = DEFAULT_RANDOM_SEED,
) -> keras.Model:
    """
    Constructs and optionally compiles the canonical binary Out-of-Distribution (OOD) Detector CNN.

    Parameters:
        input_shape: Input image dimensions (height, width, channels), default (28, 28, 1).
        dropout_rate: Dropout rate for dense classification head, default 0.3.
        compile_model: Whether to compile the model before returning, default True.
        learning_rate: Learning rate for Adam optimizer if compiled, default 0.001.
        seed: Random seed for layer weight initializers, default 42.

    Returns:
        keras.Model: Configured Keras Sequential model for binary OOD classification.
    """
    conv1_init = keras.initializers.HeNormal(seed=seed) if seed is not None else "he_normal"
    conv2_init = keras.initializers.HeNormal(seed=seed + 1 if seed is not None else None) if seed is not None else "he_normal"
    dense_init = keras.initializers.HeNormal(seed=seed + 2 if seed is not None else None) if seed is not None else "he_normal"
    out_init = keras.initializers.GlorotUniform(seed=seed + 3 if seed is not None else None) if seed is not None else "glorot_uniform"

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="ood_input_image", dtype="float32"),
            layers.Conv2D(
                DEFAULT_CONV1_FILTERS,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_initializer=conv1_init,
                name="ood_conv2d_1",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), name="ood_maxpool_1"),
            layers.Conv2D(
                DEFAULT_CONV2_FILTERS,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_initializer=conv2_init,
                name="ood_conv2d_2",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), name="ood_maxpool_2"),
            layers.Flatten(name="ood_flatten"),
            layers.Dense(
                DEFAULT_DENSE_UNITS,
                activation="relu",
                kernel_initializer=dense_init,
                name="ood_dense_features",
            ),
            layers.Dropout(dropout_rate, seed=seed + 4 if seed is not None else None, name="ood_dropout"),
            layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=out_init,
                name="ood_clothing_probability",
            ),
        ],
        name="FashionOOD_BinaryCNN",
    )

    if compile_model:
        model = compile_ood_model(model, learning_rate=learning_rate)

    return model


def compile_ood_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """
    Compiles the binary OOD detector model with Adam optimizer, binary crossentropy loss,
    and binary classification evaluation metrics.

    Parameters:
        model: Uncompiled or compiled Keras model.
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        keras.Model: Compiled Keras model ready for evaluation or training.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy(name="binary_crossentropy")
    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_architecture_layers(model: keras.Model) -> list[dict[str, Any]]:
    """
    Extracts structural metadata for every layer in the OOD model.

    Returns:
        list of dicts containing name, class, output_shape, param_count, and activation.
    """
    layer_info: list[dict[str, Any]] = []
    for layer in model.layers:
        activation_name = None
        if hasattr(layer, "activation") and layer.activation is not None:
            activation_name = getattr(layer.activation, "__name__", str(layer.activation))

        layer_info.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "output_shape": [int(d) if d is not None else None for d in layer.output.shape],
                "param_count": int(layer.count_params()),
                "activation": activation_name,
            }
        )
    return layer_info


def validate_ood_architecture(
    model: keras.Model | None = None,
    dummy_batch_size: int = 16,
) -> dict[str, Any]:
    """
    Performs comprehensive, lightweight architecture checks on the OOD binary CNN.

    Verifications performed:
    1. Input shape matches expected (None, 28, 28, 1).
    2. Input dtype is float32.
    3. Output shape matches (None, 1).
    4. Output activation is sigmoid.
    5. Model is properly compiled with BinaryCrossentropy.
    6. Parameter count is computed (trainable and non-trainable).
    7. Synthetic dummy batches (zeros, ones, uniform random) pass through forward inference.
    8. Dummy outputs are finite, numeric, and strictly bounded within [0.0, 1.0].

    Note:
        Does NOT load, reference, or use any real dataset samples.

    Returns:
        dict[str, Any]: Structured results of all verifications.
    """
    if model is None:
        model = build_ood_model(compile_model=True)

    results: dict[str, Any] = {
        "model_name": model.name,
        "checks": {},
        "summary": {},
        "all_passed": True,
    }

    # 1. Input Shape Check
    expected_input_shape = (None, 28, 28, 1)
    actual_input_shape = tuple(model.input_shape)
    input_shape_ok = actual_input_shape == expected_input_shape
    results["checks"]["input_shape"] = {
        "expected": expected_input_shape,
        "actual": actual_input_shape,
        "passed": input_shape_ok,
    }

    # 2. Input Dtype Check
    input_dtype = str(model.inputs[0].dtype) if model.inputs else "float32"
    input_dtype_ok = "float32" in input_dtype
    results["checks"]["input_dtype"] = {
        "expected": "float32",
        "actual": input_dtype,
        "passed": input_dtype_ok,
    }

    # 3. Output Shape Check
    expected_output_shape = (None, 1)
    actual_output_shape = tuple(model.output_shape)
    output_shape_ok = actual_output_shape == expected_output_shape
    results["checks"]["output_shape"] = {
        "expected": expected_output_shape,
        "actual": actual_output_shape,
        "passed": output_shape_ok,
    }

    # 4. Output Activation Check
    last_layer = model.layers[-1]
    last_activation = getattr(getattr(last_layer, "activation", None), "__name__", None)
    activation_ok = last_activation == "sigmoid"
    results["checks"]["output_activation"] = {
        "expected": "sigmoid",
        "actual": last_activation,
        "passed": activation_ok,
    }

    # 5. Model Compilation Check
    optimizer = getattr(model, "optimizer", None)
    loss = getattr(model, "loss", None)
    compiled_ok = optimizer is not None and loss is not None
    loss_class_name = loss.__class__.__name__ if loss else None
    results["checks"]["compilation"] = {
        "compiled": compiled_ok,
        "loss_function": loss_class_name,
        "optimizer": optimizer.__class__.__name__ if optimizer else None,
        "passed": compiled_ok and "BinaryCrossentropy" in (loss_class_name or ""),
    }

    # 6. Parameter Counts
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(p.shape) for p in model.trainable_weights))
    non_trainable_params = int(sum(np.prod(p.shape) for p in model.non_trainable_weights))
    results["summary"]["total_params"] = total_params
    results["summary"]["trainable_params"] = trainable_params
    results["summary"]["non_trainable_params"] = non_trainable_params
    results["summary"]["layers_count"] = len(model.layers)
    results["summary"]["layers"] = get_architecture_layers(model)

    # 7. Dummy Inference Forward Pass Check (Strictly synthetic data, no real dataset)
    rng = np.random.RandomState(42)
    dummy_zeros = np.zeros((4, 28, 28, 1), dtype=np.float32)
    dummy_ones = np.ones((4, 28, 28, 1), dtype=np.float32)
    dummy_random = rng.uniform(0.0, 1.0, size=(dummy_batch_size, 28, 28, 1)).astype(np.float32)

    # Forward pass on dummy random batch
    preds_zeros = model(dummy_zeros, training=False).numpy()
    preds_ones = model(dummy_ones, training=False).numpy()
    preds_random = model(dummy_random, training=False).numpy()

    # Forward pass validations
    zeros_finite = bool(np.all(np.isfinite(preds_zeros)))
    ones_finite = bool(np.all(np.isfinite(preds_ones)))
    random_finite = bool(np.all(np.isfinite(preds_random)))
    finite_ok = zeros_finite and ones_finite and random_finite

    zeros_in_range = bool(np.all((preds_zeros >= 0.0) & (preds_zeros <= 1.0)))
    ones_in_range = bool(np.all((preds_ones >= 0.0) & (preds_ones <= 1.0)))
    random_in_range = bool(np.all((preds_random >= 0.0) & (preds_random <= 1.0)))
    range_ok = zeros_in_range and ones_in_range and random_in_range

    dummy_shape_ok = preds_random.shape == (dummy_batch_size, 1)

    results["checks"]["dummy_inference"] = {
        "batch_size_tested": dummy_batch_size,
        "output_shape": list(preds_random.shape),
        "outputs_finite": finite_ok,
        "outputs_in_unit_range": range_ok,
        "sample_dummy_predictions": [float(p[0]) for p in preds_random[:5]],
        "passed": finite_ok and range_ok and dummy_shape_ok,
    }

    # Aggregate pass status
    for check_key, check_val in results["checks"].items():
        if not check_val.get("passed", False):
            results["all_passed"] = False
            break

    return results


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("OOD Binary Detector Architecture Validation")
    print("=" * 60)

    model = build_ood_model(compile_model=True)
    model.summary()

    validation = validate_ood_architecture(model)

    print("\n--- Architecture Validation Summary ---")
    for name, chk in validation["checks"].items():
        status = "PASSED" if chk.get("passed") else "FAILED"
        print(f"[{status}] {name}: {chk}")

    print("\n--- Parameter Summary ---")
    print(f"Total Parameters:         {validation['summary']['total_params']:,}")
    print(f"Trainable Parameters:     {validation['summary']['trainable_params']:,}")
    print(f"Non-Trainable Parameters: {validation['summary']['non_trainable_params']:,}")
    print(f"All Checks Passed:        {validation['all_passed']}")
    print("=" * 60)
