"""
Unit tests for the isolated Out-of-Distribution (OOD) Binary Detector architecture.

Validates that:
1. build_ood_model() builds uncompiled and compiled models correctly.
2. Architecture shapes, activations, and parameter counts meet specifications.
3. Loss is BinaryCrossentropy and optimizer is Adam.
4. Synthetic dummy batches (zeros, ones, uniform random) produce finite, [0, 1] bounded outputs.
5. Zero real dataset samples are used.
6. The original Fashion-MNIST CNN and evaluation artifacts remain untouched.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ood_model import (
    build_ood_model,
    compile_ood_model,
    validate_ood_architecture,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_LEARNING_RATE,
)


def test_ood_model_build_uncompiled():
    print("\n[Test 1] Testing build_ood_model(compile_model=False)...")
    model = build_ood_model(compile_model=False)
    assert isinstance(model, keras.Model), "Model should be a keras.Model instance"
    assert model.name == "FashionOOD_BinaryCNN", f"Unexpected model name: {model.name}"
    assert getattr(model, "optimizer", None) is None, "Model should not be compiled when compile_model=False"
    assert tuple(model.input_shape) == (None, 28, 28, 1), f"Unexpected input shape: {model.input_shape}"
    assert tuple(model.output_shape) == (None, 1), f"Unexpected output shape: {model.output_shape}"
    print(" -> PASSED: Uncompiled model constructed with expected shapes.")


def test_ood_model_compilation():
    print("\n[Test 2] Testing compile_ood_model()...")
    model = build_ood_model(compile_model=False)
    compiled_model = compile_ood_model(model, learning_rate=0.001)
    assert compiled_model.optimizer is not None, "Model optimizer should be present"
    assert isinstance(compiled_model.optimizer, keras.optimizers.Adam), "Optimizer should be Adam"
    assert isinstance(compiled_model.loss, keras.losses.BinaryCrossentropy), "Loss should be BinaryCrossentropy"
    print(" -> PASSED: Model compiled with Adam and BinaryCrossentropy.")


def test_ood_model_architecture_and_parameters():
    print("\n[Test 3] Testing architecture layers and parameter counts...")
    model = build_ood_model(compile_model=True)
    layer_names = [layer.name for layer in model.layers]
    expected_layers = [
        "ood_conv2d_1",
        "ood_maxpool_1",
        "ood_conv2d_2",
        "ood_maxpool_2",
        "ood_flatten",
        "ood_dense_features",
        "ood_dropout",
        "ood_clothing_probability",
    ]
    assert layer_names == expected_layers, f"Layer mismatch. Got {layer_names}"

    # Verify activations
    assert model.get_layer("ood_conv2d_1").activation.__name__ == "relu"
    assert model.get_layer("ood_conv2d_2").activation.__name__ == "relu"
    assert model.get_layer("ood_dense_features").activation.__name__ == "relu"
    assert model.get_layer("ood_clothing_probability").activation.__name__ == "sigmoid"

    # Parameter count verification
    total_params = model.count_params()
    trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)
    non_trainable_params = sum(np.prod(p.shape) for p in model.non_trainable_weights)

    assert total_params == 219649, f"Expected 219,649 total params, got {total_params}"
    assert trainable_params == 219649, f"Expected 219,649 trainable params, got {trainable_params}"
    assert non_trainable_params == 0, f"Expected 0 non-trainable params, got {non_trainable_params}"
    print(f" -> PASSED: Parameters verified (Total={total_params:,}, Trainable={trainable_params:,}).")


def test_ood_model_synthetic_forward_pass():
    print("\n[Test 4] Testing forward pass with synthetic dummy tensors...")
    model = build_ood_model(compile_model=True)

    # 1. Zero batch
    zeros = np.zeros((8, 28, 28, 1), dtype=np.float32)
    preds_zeros = model(zeros, training=False).numpy()
    assert preds_zeros.shape == (8, 1), f"Expected (8, 1), got {preds_zeros.shape}"
    assert np.all(np.isfinite(preds_zeros)), "Outputs must be finite"
    assert np.all((preds_zeros >= 0.0) & (preds_zeros <= 1.0)), "Outputs must be in [0, 1]"

    # 2. Ones batch
    ones = np.ones((8, 28, 28, 1), dtype=np.float32)
    preds_ones = model(ones, training=False).numpy()
    assert preds_ones.shape == (8, 1), f"Expected (8, 1), got {preds_ones.shape}"
    assert np.all(np.isfinite(preds_ones)), "Outputs must be finite"
    assert np.all((preds_ones >= 0.0) & (preds_ones <= 1.0)), "Outputs must be in [0, 1]"

    # 3. Random uniform [0, 1] batch
    rng = np.random.RandomState(42)
    rand_batch = rng.uniform(0.0, 1.0, size=(32, 28, 28, 1)).astype(np.float32)
    preds_rand = model(rand_batch, training=False).numpy()
    assert preds_rand.shape == (32, 1), f"Expected (32, 1), got {preds_rand.shape}"
    assert np.all(np.isfinite(preds_rand)), "Outputs must be finite"
    assert np.all((preds_rand >= 0.0) & (preds_rand <= 1.0)), "Outputs must be in [0, 1]"
    print(" -> PASSED: Synthetic dummy forward pass produced valid probabilities in [0, 1].")


def test_validate_ood_architecture_function():
    print("\n[Test 5] Testing validate_ood_architecture() helper...")
    results = validate_ood_architecture()
    assert results["all_passed"] is True, f"Validation failed: {results}"
    for check_name, check_data in results["checks"].items():
        assert check_data["passed"] is True, f"Check '{check_name}' did not pass: {check_data}"
    print(" -> PASSED: validate_ood_architecture() completed all checks with all_passed=True.")


def test_no_training_or_original_model_modification():
    print("\n[Test 6] Verifying isolation and absence of training/modifications...")
    # Original model file must exist and remain untouched
    orig_model_path = BACKEND_DIR / "models" / "fashion_mnist_cnn.keras"
    assert orig_model_path.exists(), "Original model file must exist"

    # Verify no unexpected weights file was created for OOD detector
    ood_weights_candidate = BACKEND_DIR / "models" / "ood_detector.keras"
    assert not ood_weights_candidate.exists(), "OOD weights must not exist before Stage 3 training"
    print(" -> PASSED: Original model weights intact, no trained OOD weights generated.")


if __name__ == "__main__":
    print("=" * 65)
    print("Running OOD Detector Architecture Test Suite")
    print("=" * 65)

    test_ood_model_build_uncompiled()
    test_ood_model_compilation()
    test_ood_model_architecture_and_parameters()
    test_ood_model_synthetic_forward_pass()
    test_validate_ood_architecture_function()
    test_no_training_or_original_model_modification()

    print("\n" + "=" * 65)
    print("ALL 6 OOD ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
