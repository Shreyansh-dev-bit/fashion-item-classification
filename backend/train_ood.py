"""
Stage 3: Train and Evaluate Independent Out-of-Distribution (OOD) Detector.

This script trains the canonical binary OOD CNN architecture defined in backend/app/ood_model.py
to distinguish supported Fashion-MNIST clothing images (Label 1) from non-clothing images (Label 0).

Strict Operational Rules:
1. Training uses train.npz for gradient updates and val.npz for validation/early stopping.
2. cal.npz is strictly NOT loaded or used during training, early stopping, or model selection.
3. test.npz and test_unseen_ood.npz are used ONLY for final evaluation after training is finalized.
4. All artifacts are saved to a dedicated directory: backend/ood_artifacts/
5. No existing models, artifacts, or API code are modified or overwritten.
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure stdout supports UTF-8 on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.ood_model import build_ood_model, get_architecture_layers


# Default Directory Paths
OOD_DATA_DIR = BACKEND_DIR / "ood_data"
OOD_ARTIFACTS_DIR = BACKEND_DIR / "ood_artifacts"


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_validate_training_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and rigorously validates train.npz and val.npz.
    Strictly verifies class balance, dimensions, dtypes, and value ranges.
    Guarantees cal.npz, test.npz, and test_unseen_ood.npz are NOT accessed here.
    """
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training dataset: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation dataset: {val_path}")

    print("[Data Loading] Ingesting train.npz and val.npz...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    x_train, y_train = train_data["x"], train_data["y"]
    x_val, y_val = val_data["x"], val_data["y"]

    # Verify shapes and types
    assert x_train.shape[1:] == (28, 28, 1), f"Invalid x_train shape: {x_train.shape}"
    assert x_val.shape[1:] == (28, 28, 1), f"Invalid x_val shape: {x_val.shape}"
    assert x_train.dtype == np.float32, f"Invalid x_train dtype: {x_train.dtype}"
    assert x_val.dtype == np.float32, f"Invalid x_val dtype: {x_val.dtype}"

    # Verify pixel ranges
    assert 0.0 <= np.min(x_train) and np.max(x_train) <= 1.0, "x_train pixel values out of [0.0, 1.0]"
    assert 0.0 <= np.min(x_val) and np.max(x_val) <= 1.0, "x_val pixel values out of [0.0, 1.0]"

    # Verify class balance
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    print(f" -> Train dataset: {len(x_train):,} samples (Negatives y=0: {train_counts[0]:,}, Positives y=1: {train_counts[1]:,})")
    print(f" -> Val dataset:   {len(x_val):,} samples (Negatives y=0: {val_counts[0]:,}, Positives y=1: {val_counts[1]:,})")
    assert train_counts[0] == train_counts[1], f"Train class imbalance detected: {train_counts}"
    assert val_counts[0] == val_counts[1], f"Val class imbalance detected: {val_counts}"

    return x_train, y_train, x_val, y_val


def train_ood_detector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    artifacts_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    patience: int = 3,
    seed: int = 42,
) -> tuple[keras.Model, dict[str, list[float]], dict[str, Any], float]:
    """
    Constructs, compiles, and trains the OOD CNN detector with EarlyStopping and ModelCheckpoint.
    Returns:
        (best_model, history_dict, training_metadata, elapsed_seconds)
    """
    set_seed(seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "ood_detector.keras"

    print("\n[Model Setup] Building canonical binary OOD CNN architecture...")
    model = build_ood_model(
        input_shape=(28, 28, 1),
        dropout_rate=0.3,
        compile_model=True,
        learning_rate=learning_rate,
        seed=seed,
    )
    model.summary()

    # Callbacks: EarlyStopping on val_loss and ModelCheckpoint for best weights
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    print(f"\n[Training] Starting gradient updates for up to {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    elapsed_time = time.time() - start_time
    print(f"[Training] Completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")

    # Determine best epoch based on minimum val_loss
    val_losses = history.history["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch = best_epoch_idx + 1
    total_epochs_run = len(val_losses)

    print(f"\n[Training Summary] Best Epoch: {best_epoch}/{total_epochs_run}")
    print(f" -> Best Val Loss:     {val_losses[best_epoch_idx]:.4f}")
    print(f" -> Best Val Accuracy: {history.history['val_accuracy'][best_epoch_idx] * 100:.2f}%")
    print(f" -> Best Val AUC:      {history.history['val_auc'][best_epoch_idx]:.4f}")

    # Format training history for serialization (convert float32 to standard float)
    history_serializable = {
        k: [float(val) for val in v] for k, v in history.history.items()
    }

    # Reload best saved model from disk to guarantee byte-level evaluation fidelity
    print(f"\n[Checkpoint Loading] Reloading best checkpoint from {checkpoint_path}...")
    best_model = keras.models.load_model(checkpoint_path)

    training_metadata = {
        "project_id": "DL-002",
        "stage": "STAGE 3 - Train & Evaluate Independent OOD Detector",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "random_seed": seed,
        "epochs_configured": epochs,
        "epochs_run": total_epochs_run,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss_function": "BinaryCrossentropy",
        "early_stopping_patience": patience,
        "training_duration_seconds": round(elapsed_time, 2),
        "best_val_loss": round(float(val_losses[best_epoch_idx]), 6),
        "best_val_accuracy": round(float(history.history["val_accuracy"][best_epoch_idx]), 6),
        "best_val_auc": round(float(history.history["val_auc"][best_epoch_idx]), 6),
    }

    return best_model, history_serializable, training_metadata, elapsed_time


def evaluate_known_test_set(
    model: keras.Model,
    data_dir: Path,
    illustrative_threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Evaluates the finalized best model on the official held-out test.npz (20,000 samples).
    Reports threshold-independent ROC-AUC and illustrative threshold metrics (default tau=0.5).
    """
    test_path = data_dir / "test.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")

    print("\n" + "=" * 65)
    print("EVALUATION A: Known-Category Held-Out Test Set (test.npz)")
    print("=" * 65)
    test_data = np.load(test_path, allow_pickle=True)
    x_test, y_test = test_data["x"], test_data["y"]
    meta_source = test_data["meta_source"]

    print(f"Loaded test.npz: {len(x_test):,} samples")
    test_counts = np.bincount(y_test)
    print(f" -> Non-clothing negatives (y=0): {test_counts[0]:,}")
    print(f" -> Supported clothing positives (y=1): {test_counts[1]:,}")

    # Forward inference
    print("Generating probability predictions...")
    y_probs = model.predict(x_test, batch_size=256, verbose=0).flatten()

    # Threshold-independent metrics
    roc_auc = float(roc_auc_score(y_test, y_probs))
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_test, y_probs)

    # Subsample ROC curve data for clean JSON serialization (up to 200 points)
    step = max(1, len(fpr_arr) // 200)
    roc_curve_data = {
        "roc_auc": round(roc_auc, 6),
        "fpr": [round(float(x), 6) for x in fpr_arr[::step]],
        "tpr": [round(float(y), 6) for y in tpr_arr[::step]],
        "thresholds": [round(float(t), 6) for t in thresh_arr[::step]],
    }
    # Ensure final point (1.0, 1.0) is included
    if roc_curve_data["fpr"][-1] != round(float(fpr_arr[-1]), 6):
        roc_curve_data["fpr"].append(round(float(fpr_arr[-1]), 6))
        roc_curve_data["tpr"].append(round(float(tpr_arr[-1]), 6))
        roc_curve_data["thresholds"].append(round(float(thresh_arr[-1]), 6))

    # Illustrative Threshold Metrics (Provisional tau = 0.5)
    y_pred_illustrative = (y_probs >= illustrative_threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred_illustrative))
    prec = float(precision_score(y_test, y_pred_illustrative, zero_division=0))
    rec = float(recall_score(y_test, y_pred_illustrative, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_illustrative, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_illustrative)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # False Acceptance Rate: FP / (FP + TN) = FP / total_negatives
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # False Rejection Rate: FN / (TP + FN) = FN / total_positives
    frr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Source dataset breakdown
    source_metrics: dict[str, Any] = {}
    for src in np.unique(meta_source):
        mask = meta_source == src
        src_y = y_test[mask]
        src_probs = y_probs[mask]
        src_preds = y_pred_illustrative[mask]
        src_total = int(len(src_y))
        if src_y[0] == 1:
            # Positive source (Fashion-MNIST)
            correct_acc = int(np.sum(src_preds == 1))
            false_rej = int(np.sum(src_preds == 0))
            source_metrics[str(src)] = {
                "type": "positive (clothing)",
                "total": src_total,
                "correctly_accepted": correct_acc,
                "falsely_rejected": false_rej,
                "acceptance_rate_pct": round(correct_acc / src_total * 100, 2),
                "false_rejection_rate_pct": round(false_rej / src_total * 100, 2),
                "mean_probability": round(float(np.mean(src_probs)), 4),
            }
        else:
            # Negative source (CIFAR-10 or MNIST)
            correct_rej = int(np.sum(src_preds == 0))
            false_acc = int(np.sum(src_preds == 1))
            source_metrics[str(src)] = {
                "type": "negative (non-clothing)",
                "total": src_total,
                "correctly_rejected": correct_rej,
                "falsely_accepted": false_acc,
                "rejection_rate_pct": round(correct_rej / src_total * 100, 2),
                "false_acceptance_rate_pct": round(false_acc / src_total * 100, 2),
                "mean_probability": round(float(np.mean(src_probs)), 4),
            }

    known_metrics = {
        "dataset": "backend/ood_data/test.npz",
        "total_samples": len(x_test),
        "threshold_independent": {
            "roc_auc": round(roc_auc, 6),
        },
        "illustrative_evaluation": {
            "note": "Provisional evaluation at default tau=0.5. Final threshold calibration belongs to Stage 4 using cal.npz.",
            "threshold": illustrative_threshold,
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "false_acceptance_rate": round(far, 6),
            "false_rejection_rate": round(frr, 6),
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            },
        },
        "source_breakdown": source_metrics,
    }

    confusion_dict = {
        "threshold_evaluated": illustrative_threshold,
        "is_illustrative": True,
        "matrix": [[tn, fp], [fn, tp]],
        "labels": ["OOD (Non-Clothing: 0)", "In-Distribution (Clothing: 1)"],
        "counts": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "rates": {
            "false_acceptance_rate": round(far, 6),
            "false_rejection_rate": round(frr, 6),
            "true_negative_rate": round(1.0 - far, 6),
            "true_positive_rate": round(1.0 - frr, 6),
        },
    }

    print(f" -> ROC-AUC:                {roc_auc:.4f}")
    print(f" -> Accuracy (tau=0.5):     {acc * 100:.2f}%")
    print(f" -> Precision:              {prec * 100:.2f}%")
    print(f" -> Recall:                 {rec * 100:.2f}%")
    print(f" -> F1-Score:               {f1 * 100:.2f}%")
    print(f" -> False Acceptance Rate:  {far * 100:.2f}% ({fp:,} false accepts)")
    print(f" -> False Rejection Rate:   {frr * 100:.2f}% ({fn:,} false rejects)")

    return known_metrics, confusion_dict, roc_curve_data


def evaluate_unseen_test_set(
    model: keras.Model,
    data_dir: Path,
    illustrative_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluates the model on test_unseen_ood.npz (9,998 samples across 100 CIFAR-100 categories).
    All samples are strictly negative (non-clothing).
    Measures zero-shot rejection performance, FAR, and category breakdowns.
    """
    unseen_path = data_dir / "test_unseen_ood.npz"
    if not unseen_path.exists():
        raise FileNotFoundError(f"Missing unseen test dataset: {unseen_path}")

    print("\n" + "=" * 65)
    print("EVALUATION B: Unseen-Category Test Set (test_unseen_ood.npz)")
    print("=" * 65)
    unseen_data = np.load(unseen_path, allow_pickle=True)
    x_unseen, y_unseen = unseen_data["x"], unseen_data["y"]
    meta_class = unseen_data["meta_class"]
    meta_super = unseen_data["meta_superclass"]

    total_unseen = len(x_unseen)
    print(f"Loaded test_unseen_ood.npz: {total_unseen:,} samples (all non-clothing negatives)")
    assert np.all(y_unseen == 0), "Integrity failure: test_unseen_ood must contain only y=0"

    # Forward inference
    print("Generating probability predictions for unseen object categories...")
    probs_unseen = model.predict(x_unseen, batch_size=256, verbose=0).flatten()

    # Illustrative Rejection / Acceptance at tau = 0.5
    # Non-clothing should have score < threshold -> REJECTED
    # Score >= threshold -> FALSELY ACCEPTED
    falsely_accepted = int(np.sum(probs_unseen >= illustrative_threshold))
    correctly_rejected = int(np.sum(probs_unseen < illustrative_threshold))
    far_unseen = float(falsely_accepted / total_unseen)
    rejection_rate = float(correctly_rejected / total_unseen)

    # Score distribution statistics
    score_stats = {
        "mean": round(float(np.mean(probs_unseen)), 6),
        "std": round(float(np.std(probs_unseen)), 6),
        "median": round(float(np.median(probs_unseen)), 6),
        "min": round(float(np.min(probs_unseen)), 6),
        "max": round(float(np.max(probs_unseen)), 6),
        "p10": round(float(np.percentile(probs_unseen, 10)), 6),
        "p25": round(float(np.percentile(probs_unseen, 25)), 6),
        "p50": round(float(np.percentile(probs_unseen, 50)), 6),
        "p75": round(float(np.percentile(probs_unseen, 75)), 6),
        "p90": round(float(np.percentile(probs_unseen, 90)), 6),
        "p95": round(float(np.percentile(probs_unseen, 95)), 6),
        "p99": round(float(np.percentile(probs_unseen, 99)), 6),
    }

    # Per-Superclass Analysis (20 superclasses)
    superclass_breakdown: dict[str, Any] = {}
    for sc in sorted(np.unique(meta_super)):
        mask = meta_super == sc
        sc_probs = probs_unseen[mask]
        sc_total = int(len(sc_probs))
        sc_rej = int(np.sum(sc_probs < illustrative_threshold))
        sc_acc = int(np.sum(sc_probs >= illustrative_threshold))
        sc_rej_rate = float(sc_rej / sc_total) * 100
        sc_far = float(sc_acc / sc_total) * 100

        superclass_breakdown[str(sc)] = {
            "total_samples": sc_total,
            "correctly_rejected": sc_rej,
            "falsely_accepted": sc_acc,
            "rejection_rate_pct": round(sc_rej_rate, 2),
            "false_acceptance_rate_pct": round(sc_far, 2),
            "mean_score": round(float(np.mean(sc_probs)), 4),
            "median_score": round(float(np.median(sc_probs)), 4),
        }

    # Highlighted Object Domain Groupings
    domain_groupings = {
        "fruits_and_vegetables": ["fruit_and_vegetables"],
        "trees_and_plants": ["trees", "flowers"],
        "household_furniture": ["household_furniture"],
        "electronics_and_appliances": ["household_electrical_devices"],
        "buildings_and_structures": ["large_man_made_outdoor_things"],
        "food_containers": ["food_containers"],
        "natural_scenes": ["large_natural_outdoor_scenes"],
        "animals": [
            "aquatic_mammals",
            "fish",
            "insects",
            "large_carnivores",
            "large_omnivores_and_herbivores",
            "medium_mammals",
            "non_insect_invertebrates",
            "reptiles",
            "small_mammals",
        ],
        "vehicles": ["vehicles_1", "vehicles_2"],
        "people": ["people"],
    }

    group_breakdown: dict[str, Any] = {}
    for group_name, sc_list in domain_groupings.items():
        mask = np.isin(meta_super, sc_list)
        grp_probs = probs_unseen[mask]
        grp_total = int(len(grp_probs))
        grp_rej = int(np.sum(grp_probs < illustrative_threshold))
        grp_acc = int(np.sum(grp_probs >= illustrative_threshold))
        group_breakdown[group_name] = {
            "superclasses_included": sc_list,
            "total_samples": grp_total,
            "correctly_rejected": grp_rej,
            "falsely_accepted": grp_acc,
            "rejection_rate_pct": round(grp_rej / grp_total * 100, 2),
            "false_acceptance_rate_pct": round(grp_acc / grp_total * 100, 2),
            "mean_score": round(float(np.mean(grp_probs)), 4),
        }

    # Per-Class Analysis (100 fine classes)
    fine_class_breakdown: dict[str, Any] = {}
    for fc in sorted(np.unique(meta_class)):
        mask = meta_class == fc
        fc_probs = probs_unseen[mask]
        fc_total = int(len(fc_probs))
        fc_rej = int(np.sum(fc_probs < illustrative_threshold))
        fc_acc = int(np.sum(fc_probs >= illustrative_threshold))
        fine_class_breakdown[str(fc)] = {
            "total_samples": fc_total,
            "correctly_rejected": fc_rej,
            "falsely_accepted": fc_acc,
            "rejection_rate_pct": round(fc_rej / fc_total * 100, 2),
            "false_acceptance_rate_pct": round(fc_acc / fc_total * 100, 2),
            "mean_score": round(float(np.mean(fc_probs)), 4),
        }

    # Best and lowest performing fine classes
    sorted_classes = sorted(fine_class_breakdown.items(), key=lambda x: x[1]["rejection_rate_pct"], reverse=True)
    top_5_classes = [c[0] for c in sorted_classes[:5]]
    bottom_5_classes = [c[0] for c in sorted_classes[-5:]]

    unseen_metrics = {
        "dataset": "backend/ood_data/test_unseen_ood.npz",
        "total_unseen_samples": total_unseen,
        "illustrative_threshold": illustrative_threshold,
        "overall_summary": {
            "correctly_rejected": correctly_rejected,
            "falsely_accepted": falsely_accepted,
            "rejection_rate_pct": round(rejection_rate * 100, 2),
            "false_acceptance_rate_pct": round(far_unseen * 100, 2),
        },
        "score_distribution": score_stats,
        "domain_groupings": group_breakdown,
        "superclass_breakdown": superclass_breakdown,
        "top_5_rejection_classes": top_5_classes,
        "bottom_5_rejection_classes": bottom_5_classes,
        "fine_class_breakdown": fine_class_breakdown,
    }

    print(f" -> Total Unseen Negative Samples: {total_unseen:,}")
    print(f" -> Correctly Rejected:            {correctly_rejected:,} ({rejection_rate * 100:.2f}%)")
    print(f" -> Falsely Accepted:              {falsely_accepted:,} ({far_unseen * 100:.2f}%)")
    print(f" -> Mean Sigmoid Probability:      {score_stats['mean']:.4f} (median: {score_stats['median']:.4f})")
    print(f" -> 90th percentile score:         {score_stats['p90']:.4f}")

    return unseen_metrics


def generate_roc_plot(roc_curve_data: dict[str, Any], output_path: Path) -> None:
    """Renders and saves a clean high-resolution ROC curve visualization."""
    fpr = roc_curve_data["fpr"]
    tpr = roc_curve_data["tpr"]
    roc_auc = roc_curve_data["roc_auc"]

    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"OOD Detector ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Acceptance Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Acceptance Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("ROC Curve: Fashion-MNIST OOD Gating Detector", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot Generated] ROC curve saved to: {output_path}")


def save_reproducibility_summary(
    artifacts_dir: Path,
    training_meta: dict[str, Any],
    model: keras.Model,
    model_path: Path,
) -> None:
    """Saves complete environment and artifact reproducibility record."""
    model_size_bytes = model_path.stat().st_size if model_path.exists() else 0
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(p.shape) for p in model.trainable_weights))

    summary = {
        "project_id": "DL-002",
        "stage": "STAGE 3 - Train & Evaluate Independent OOD Detector",
        "model_file": model_path.name,
        "model_file_size_bytes": model_size_bytes,
        "model_file_size_mb": round(model_size_bytes / (1024 * 1024), 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "integrated"),
        "hardware": "CPU (Intel Core i5-14450HX, 16 logical threads)",
        "training_metadata": training_meta,
        "layers": get_architecture_layers(model),
    }

    out_file = artifacts_dir / "reproducibility_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary Saved] Reproducibility record written to: {out_file}")


def save_markdown_readme(artifacts_dir: Path, known_metrics: dict[str, Any], unseen_metrics: dict[str, Any]) -> None:
    """Generates an executive README in the OOD artifacts directory."""
    readme_path = artifacts_dir / "README.md"
    km = known_metrics["illustrative_evaluation"]
    um = unseen_metrics["overall_summary"]

    content = f"""# Out-of-Distribution (OOD) Gating Detector Artifacts

- **Project:** Fashion Item Classification with CNN (DL-002)
- **Model:** `FashionOOD_BinaryCNN` (`ood_detector.keras`)
- **Stage:** STAGE 3 — Training & Evaluation (Isolated)

## Known-Category Test Performance (`test.npz` — 20,000 samples)
- **ROC-AUC:** {known_metrics['threshold_independent']['roc_auc']:.4f}
- **Accuracy (Illustrative $\\tau=0.5$):** {km['accuracy'] * 100:.2f}%
- **Precision:** {km['precision'] * 100:.2f}%
- **Recall:** {km['recall'] * 100:.2f}%
- **F1-Score:** {km['f1_score'] * 100:.2f}%
- **False Acceptance Rate (Non-clothing accepted):** {km['false_acceptance_rate'] * 100:.2f}%
- **False Rejection Rate (Clothing rejected):** {km['false_rejection_rate'] * 100:.2f}%

## Unseen-Category Test Performance (`test_unseen_ood.npz` — 9,998 samples)
- **Total Unseen Samples:** {unseen_metrics['total_unseen_samples']:,}
- **Correctly Rejected:** {um['correctly_rejected']:,} ({um['rejection_rate_pct']:.2f}%)
- **Falsely Accepted (Illustrative $\\tau=0.5$):** {um['falsely_accepted']:,} ({um['false_acceptance_rate_pct']:.2f}%)
- **Mean Probability Score:** {unseen_metrics['score_distribution']['mean']:.4f}

## Artifact Directory Contents
1. `ood_detector.keras` — Trained best model weights checkpoint.
2. `training_history.json` — Epoch-by-epoch training and validation metrics.
3. `training_config.json` — Hyperparameters, early-stopping details, and training setup.
4. `known_test_metrics.json` — Full held-out test evaluation on known categories.
5. `confusion_matrix.json` — Known test confusion matrix and rate breakdowns.
6. `roc_curve_data.json` — Numerical ROC curve points (FPR, TPR, thresholds).
7. `roc_curve.png` — Visualization of the ROC curve.
8. `unseen_rejection_metrics.json` — Comprehensive breakdown by domain group and superclass.
9. `reproducibility_summary.json` — Complete environment and parameter specifications.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[README Saved] Summary documentation written to: {readme_path}")


def main() -> None:
    print("=" * 70)
    print("OUT-OF-DISTRIBUTION (OOD) DETECTOR TRAINING PIPELINE")
    print("Project: Fashion Item Classification with CNN (DL-002)")
    print("=" * 70)

    # 1. Load and validate training & validation datasets
    x_train, y_train, x_val, y_val = load_and_validate_training_data(OOD_DATA_DIR)

    # 2. Train model with EarlyStopping and save best checkpoint
    best_model, history, training_meta, elapsed_time = train_ood_detector(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        artifacts_dir=OOD_ARTIFACTS_DIR,
        epochs=15,
        batch_size=128,
        learning_rate=0.001,
        patience=3,
        seed=42,
    )

    # 3. Save training history and configuration
    hist_path = OOD_ARTIFACTS_DIR / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Artifact Saved] Training history written to: {hist_path}")

    cfg_path = OOD_ARTIFACTS_DIR / "training_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)
    print(f"[Artifact Saved] Training configuration written to: {cfg_path}")

    # 4. Final Evaluation A: Known categories (test.npz)
    known_metrics, cm_data, roc_data = evaluate_known_test_set(
        model=best_model,
        data_dir=OOD_DATA_DIR,
        illustrative_threshold=0.5,
    )
    km_path = OOD_ARTIFACTS_DIR / "known_test_metrics.json"
    with open(km_path, "w", encoding="utf-8") as f:
        json.dump(known_metrics, f, indent=2)
    print(f"[Artifact Saved] Known test metrics written to: {km_path}")

    cm_path = OOD_ARTIFACTS_DIR / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[Artifact Saved] Confusion matrix written to: {cm_path}")

    roc_path = OOD_ARTIFACTS_DIR / "roc_curve_data.json"
    with open(roc_path, "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)
    print(f"[Artifact Saved] ROC curve data written to: {roc_path}")

    # Plot ROC curve
    roc_img_path = OOD_ARTIFACTS_DIR / "roc_curve.png"
    generate_roc_plot(roc_data, roc_img_path)

    # 5. Final Evaluation B: Unseen categories (test_unseen_ood.npz)
    unseen_metrics = evaluate_unseen_test_set(
        model=best_model,
        data_dir=OOD_DATA_DIR,
        illustrative_threshold=0.5,
    )
    um_path = OOD_ARTIFACTS_DIR / "unseen_rejection_metrics.json"
    with open(um_path, "w", encoding="utf-8") as f:
        json.dump(unseen_metrics, f, indent=2)
    print(f"[Artifact Saved] Unseen rejection metrics written to: {um_path}")

    # 6. Reproducibility Summary and Documentation
    model_checkpoint_path = OOD_ARTIFACTS_DIR / "ood_detector.keras"
    save_reproducibility_summary(OOD_ARTIFACTS_DIR, training_meta, best_model, model_checkpoint_path)
    save_markdown_readme(OOD_ARTIFACTS_DIR, known_metrics, unseen_metrics)

    print("\n" + "=" * 70)
    print("STAGE 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Artifacts preserved in: {OOD_ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
