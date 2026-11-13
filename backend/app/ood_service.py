"""
Out-of-Distribution (OOD) Gate Service.

Provides singleton model loading, architecture validation, threshold evaluation,
and fail-closed gate execution for the candidate OOD detector CNN.
"""

from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, status
import numpy as np
import tensorflow as tf

from app.config import (
    DEFAULT_OOD_ACCEPT_THRESHOLD,
    DEFAULT_OOD_REJECT_THRESHOLD,
    settings,
    validate_ood_thresholds,
)

# Cached OOD Detector Model Singleton
_ood_model_instance: Optional[tf.keras.Model] = None


def get_ood_detector(
    model_path: Optional[Path] = None,
    force_reload: bool = False,
) -> tf.keras.Model:
    """
    Retrieves the candidate OOD binary CNN model, loading it into memory on first request.
    Validates structural architecture (input shape, output shape, dummy inference).
    Fails closed with HTTP 503 if the model cannot be loaded.
    """
    global _ood_model_instance
    if _ood_model_instance is not None and not force_reload:
        return _ood_model_instance

    target_path = Path(model_path) if model_path is not None else settings.OOD_MODEL_PATH

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"OOD detector model checkpoint not found at '{target_path}'. "
                "The application is failing closed to protect the downstream classifier."
            ),
        )

    try:
        loaded_model = tf.keras.models.load_model(target_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load OOD detector model from '{target_path}': {e}",
        )

    # 1. Validate Input Shape: Must be (None, 28, 28, 1)
    actual_input_shape = tuple(loaded_model.input_shape)
    expected_input_shape = (None, 28, 28, 1)
    if actual_input_shape != expected_input_shape:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"OOD detector has invalid input shape {actual_input_shape}; "
                f"expected {expected_input_shape}."
            ),
        )

    # 2. Validate Output Shape: Must be (None, 1)
    actual_output_shape = tuple(loaded_model.output_shape)
    expected_output_shape = (None, 1)
    if actual_output_shape != expected_output_shape:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"OOD detector has invalid output shape {actual_output_shape}; "
                f"expected {expected_output_shape}."
            ),
        )

    # 3. Validate Forward Pass with Synthetic Dummy Batch
    try:
        dummy_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        dummy_output = loaded_model(dummy_tensor, training=False).numpy()
        val = float(dummy_output[0, 0])
        if not np.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"Dummy forward test produced invalid output {val}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OOD detector validation forward pass failed: {e}",
        )

    _ood_model_instance = loaded_model
    return _ood_model_instance


def reset_ood_detector() -> None:
    """Resets the cached OOD detector singleton (useful for testing loading failure paths)."""
    global _ood_model_instance
    _ood_model_instance = None


def evaluate_ood_gate(
    input_tensor: np.ndarray,
    model: Optional[tf.keras.Model] = None,
    reject_threshold: Optional[float] = None,
    accept_threshold: Optional[float] = None,
) -> Tuple[str, float, str]:
    """
    Evaluates an input image tensor against the OOD gate.

    Parameters:
        input_tensor: Normalized float32 image array of shape (1, 28, 28, 1).
        model: Optional pre-loaded OOD model (defaults to cached singleton).
        reject_threshold: Low threshold tau_reject (defaults to settings.OOD_REJECT_THRESHOLD).
        accept_threshold: High threshold tau_accept (defaults to settings.OOD_ACCEPT_THRESHOLD).

    Returns:
        Tuple of (decision: str, score: float, message: str)
        Decision is one of 'ACCEPT', 'REJECT', or 'UNCERTAIN'.

    Fail-Closed Policy:
        If the score is missing, non-numeric, NaN, infinite, or out of range,
        the function fails closed (returns 'REJECT' or raises an error) and never allows
        the image to pass to the downstream classifier.
    """
    tau_reject = reject_threshold if reject_threshold is not None else settings.OOD_REJECT_THRESHOLD
    tau_accept = accept_threshold if accept_threshold is not None else settings.OOD_ACCEPT_THRESHOLD

    # Ensure thresholds satisfy 0 <= reject < accept <= 1
    validate_ood_thresholds(tau_reject, tau_accept)

    # Retrieve detector model (raises HTTP 503 if unavailable)
    detector = model if model is not None else get_ood_detector()

    # Forward inference
    try:
        raw_output = detector(input_tensor, training=False)
        output_np = raw_output.numpy() if hasattr(raw_output, "numpy") else np.asarray(raw_output)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OOD detector inference error: {e}",
        )

    # Validate output format and extract scalar score
    try:
        raw_score = output_np[0, 0] if output_np.ndim > 1 else output_np[0]
        score = float(raw_score)
    except Exception:
        # Non-numeric or missing score: fail closed
        return (
            "REJECT",
            0.0,
            "Image rejected: OOD detector returned a missing or non-numeric confidence score. Failing closed.",
        )

    # Check for NaN, Infinite, or out-of-range values: Fail closed!
    if np.isnan(score) or np.isinf(score) or score < 0.0 or score > 1.0:
        return (
            "REJECT",
            0.0 if not np.isnan(score) else float("nan"),
            f"Image rejected: OOD detector returned an invalid score ({score}). Failing closed.",
        )

    # Three-way decision rule
    if score < tau_reject:
        return (
            "REJECT",
            score,
            "Image rejected: Uploaded image is outside the supported Fashion-MNIST clothing domain. The Fashion-MNIST classifier was not executed.",
        )
    elif score < tau_accept:
        return (
            "UNCERTAIN",
            score,
            "Image uncertain: Uploaded image could not be confidently identified as supported clothing. The Fashion-MNIST classifier was not executed.",
        )
    else:
        return (
            "ACCEPT",
            score,
            "Image accepted: Uploaded image verified as supported Fashion-MNIST clothing domain. Proceeding to classification.",
        )
