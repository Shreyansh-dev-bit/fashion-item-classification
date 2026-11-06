import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf

from app.config import ARTIFACTS_DIR, FASHION_MNIST_CLASSES, MODELS_DIR
from app.dataset import load_and_preprocess_data, set_seed


def array_to_png_base64(img_2d: np.ndarray) -> str:
    """Converts a 28x28 grayscale float array [0, 1] into a PNG base64 data URI."""
    uint8_img = (np.clip(img_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def compute_class_distributions(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    classes: List[str] = FASHION_MNIST_CLASSES,
) -> Dict[str, Any]:
    """Computes genuine class frequencies for train, val, and test splits."""
    num_classes = len(classes)
    train_counts = [int(np.sum(y_train == i)) for i in range(num_classes)]
    val_counts = [int(np.sum(y_val == i)) for i in range(num_classes)]
    test_counts = [int(np.sum(y_test == i)) for i in range(num_classes)]

    return {
        "classes": classes,
        "total_train": int(len(y_train)),
        "total_val": int(len(y_val)),
        "total_test": int(len(y_test)),
        "total_dataset": int(len(y_train) + len(y_val) + len(y_test)),
        "per_class": [
            {
                "class_id": i,
                "class_name": classes[i],
                "train_count": train_counts[i],
                "val_count": val_counts[i],
                "test_count": test_counts[i],
                "total_count": train_counts[i] + val_counts[i] + test_counts[i],
            }
            for i in range(num_classes)
        ],
    }


def plot_accuracy_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plots and saves the training vs validation accuracy curve."""
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(epochs, [a * 100 for a in history["accuracy"]], label="Training Accuracy", color="#4F46E5", linewidth=2.5, marker="o", markersize=5)
    ax.plot(epochs, [a * 100 for a in history["val_accuracy"]], label="Validation Accuracy", color="#06B6D4", linewidth=2.5, marker="s", markersize=5)
    
    ax.set_title("Fashion-MNIST CNN: Training vs Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="medium")
    ax.set_xticks(list(epochs))
    ax.set_ylim([75, 100])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved accuracy curve to: {output_path}")


def plot_loss_curve(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plots and saves the training vs validation loss curve."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(epochs, history["loss"], label="Training Loss", color="#DC2626", linewidth=2.5, marker="o", markersize=5)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#F59E0B", linewidth=2.5, marker="s", markersize=5)
    
    ax.set_title("Fashion-MNIST CNN: Training vs Validation Loss", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="medium")
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved loss curve to: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: Path) -> None:
    """Generates a high-resolution labeled heatmap of the 10x10 confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Fashion-MNIST CNN: 10-Class Test Confusion Matrix (N = 10,000)", fontsize=14, fontweight="bold", pad=15)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="semibold", labelpad=8)

    # Annotate cells with count and percentage
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center", color=color, fontsize=8, fontweight="medium")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved confusion matrix image to: {output_path}")


def plot_misclassified_grid(
    misclassified_samples: List[Dict[str, Any]],
    x_test: np.ndarray,
    output_path: Path,
    num_samples: int = 15,
) -> None:
    """Plots a 3x5 grid of real misclassified Fashion-MNIST test images with true/pred labels and confidence."""
    samples_to_plot = misclassified_samples[:num_samples]
    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8), dpi=300)
    fig.suptitle("Real Misclassified Fashion-MNIST Test Images (Actual vs. CNN Prediction)", fontsize=14, fontweight="bold", y=0.98)

    for i, sample in enumerate(samples_to_plot):
        ax = axes[i // cols, i % cols]
        idx = sample["test_index"]
        img = x_test[idx].squeeze()
        ax.imshow(img, cmap="gray")
        
        true_name = sample["true_label_name"]
        pred_name = sample["predicted_label_name"]
        conf = sample["confidence"] * 100
        
        title_text = f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)"
        ax.set_title(title_text, fontsize=8.5, color="#DC2626", fontweight="semibold")
        ax.axis("off")

    # Turn off any unused axes
    for j in range(len(samples_to_plot), rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved misclassified grid to: {output_path}")


def generate_evaluation_artifacts(
    model_path: Path = MODELS_DIR / "fashion_mnist_cnn.keras",
    artifacts_dir: Path = ARTIFACTS_DIR,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the saved CNN model strictly on the held-out Fashion-MNIST test set (10,000 images).
    Generates structured JSON data and visualization charts.
    Verifies internal mathematical consistency.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_seed)

    # 1. Load trained model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
    
    print(f"[Evaluator] Loading saved CNN model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 2. Ingest authentic Fashion-MNIST dataset splits
    print("[Evaluator] Loading real dataset splits...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
        val_size=6000,
        random_seed=random_seed,
    )
    
    # 3. Predict on Held-Out Test Set (10,000 images)
    print(f"[Evaluator] Computing predictions on {len(x_test)} held-out test images...")
    prob_preds = model.predict(x_test, batch_size=128, verbose=1)  # shape (10000, 10)
    y_pred = np.argmax(prob_preds, axis=-1)  # shape (10000,)
    
    # Evaluate raw test loss and accuracy from Keras
    eval_res = model.evaluate(x_test, y_test, verbose=0)
    test_loss = float(eval_res[0])
    test_acc_keras = float(eval_res[1])

    # 4. Compute Sklearn Classification Metrics
    accuracy_exact = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=FASHION_MNIST_CLASSES,
        digits=4,
        output_dict=True,
    )

    # 5. Full 10x10 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_total = int(np.sum(cm))
    
    # Consistency Verification
    if cm_total != 10000:
        raise ValueError(f"Consistency check failed! Confusion matrix total is {cm_total}, expected 10,000.")
    if int(report_dict["macro avg"]["support"]) != 10000:
        raise ValueError("Consistency check failed! Classification report support does not equal 10,000.")

    print(f"[Evaluator] Consistency verified: Confusion matrix sum = {cm_total} == 10,000.")
    print(f"[Evaluator] Test Accuracy: {accuracy_exact * 100:.2f}%, Macro F1: {macro_f1 * 100:.2f}%")

    # 6. Sample Correct and Misclassified Predictions
    correct_indices = np.where(y_pred == y_test)[0]
    misclassified_indices = np.where(y_pred != y_test)[0]

    print(f"[Evaluator] Correct test count: {len(correct_indices)}, Misclassified count: {len(misclassified_indices)}")

    # Collect sample correct predictions (2 from each of the 10 classes = 20 samples)
    correct_samples: List[Dict[str, Any]] = []
    for class_id in range(len(FASHION_MNIST_CLASSES)):
        class_correct_idx = [idx for idx in correct_indices if y_test[idx] == class_id][:2]
        for idx in class_correct_idx:
            probs = [float(p) for p in prob_preds[idx]]
            correct_samples.append({
                "test_index": int(idx),
                "true_label_id": int(y_test[idx]),
                "true_label_name": FASHION_MNIST_CLASSES[y_test[idx]],
                "predicted_label_id": int(y_pred[idx]),
                "predicted_label_name": FASHION_MNIST_CLASSES[y_pred[idx]],
                "confidence": float(probs[y_pred[idx]]),
                "probabilities": probs,
                "image_base64": array_to_png_base64(x_test[idx].squeeze()),
            })

    # Collect sample misclassified predictions (30 diverse misclassified samples)
    misclassified_samples: List[Dict[str, Any]] = []
    # Sort misclassifications by confidence (highest confidence mistakes are most interesting)
    mis_sorted = sorted(misclassified_indices, key=lambda idx: prob_preds[idx, y_pred[idx]], reverse=True)
    for idx in mis_sorted[:30]:
        probs = [float(p) for p in prob_preds[idx]]
        misclassified_samples.append({
            "test_index": int(idx),
            "true_label_id": int(y_test[idx]),
            "true_label_name": FASHION_MNIST_CLASSES[y_test[idx]],
            "predicted_label_id": int(y_pred[idx]),
            "predicted_label_name": FASHION_MNIST_CLASSES[y_pred[idx]],
            "confidence": float(probs[y_pred[idx]]),
            "true_class_confidence": float(probs[y_test[idx]]),
            "probabilities": probs,
            "image_base64": array_to_png_base64(x_test[idx].squeeze()),
        })

    # 7. Class Distribution Information
    class_dist = compute_class_distributions(y_train, y_val, y_test, FASHION_MNIST_CLASSES)

    # 8. Load or Validate Epoch Training History
    history_file = artifacts_dir / "training_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            epoch_history = json.load(f)
    else:
        raise FileNotFoundError(f"Training history missing at {history_file}.")

    # 9. Save JSON Artifacts
    # Artifact 1: evaluation_metrics.json
    metrics_payload = {
        "project_id": "DL-002",
        "dataset": "Fashion-MNIST",
        "evaluated_set": "Held-Out Test Set",
        "test_loss": test_loss,
        "test_accuracy": accuracy_exact,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_test_samples": cm_total,
        "total_correct": int(len(correct_indices)),
        "total_misclassified": int(len(misclassified_indices)),
        "classes": FASHION_MNIST_CLASSES,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    with open(artifacts_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Artifact 2: training_curves.json
    curves_payload = {
        "epochs": list(range(1, len(epoch_history["accuracy"]) + 1)),
        "training_accuracy": epoch_history["accuracy"],
        "validation_accuracy": epoch_history["val_accuracy"],
        "training_loss": epoch_history["loss"],
        "validation_loss": epoch_history["val_loss"],
    }
    with open(artifacts_dir / "training_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves_payload, f, indent=2)

    # Artifact 3: sample_predictions.json
    predictions_payload = {
        "correct_predictions": correct_samples,
        "misclassified_predictions": misclassified_samples,
    }
    with open(artifacts_dir / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2)

    # Artifact 4: class_distributions.json
    with open(artifacts_dir / "class_distributions.json", "w", encoding="utf-8") as f:
        json.dump(class_dist, f, indent=2)

    # Artifact 5: artifact_metadata.json
    metadata_payload = {
        "project_id": "DL-002",
        "project_name": "Fashion Item Classification with CNN",
        "domain": "Computer Vision / Retail",
        "seed": random_seed,
        "dataset_split_sizes": {
            "training_samples": int(len(x_train)),
            "validation_samples": int(len(x_val)),
            "held_out_test_samples": int(len(x_test)),
            "total_dataset_samples": int(len(x_train) + len(x_val) + len(x_test)),
        },
        "cnn_configuration": {
            "model_name": "FashionMNIST_CNN",
            "input_shape": [28, 28, 1],
            "layers": [
                {"name": "conv2d_1", "type": "Conv2D", "filters": 32, "kernel_size": [3, 3], "activation": "relu"},
                {"name": "maxpool_1", "type": "MaxPooling2D", "pool_size": [2, 2]},
                {"name": "conv2d_2", "type": "Conv2D", "filters": 64, "kernel_size": [3, 3], "activation": "relu"},
                {"name": "maxpool_2", "type": "MaxPooling2D", "pool_size": [2, 2]},
                {"name": "flatten", "type": "Flatten"},
                {"name": "dense_features", "type": "Dense", "units": 128, "activation": "relu"},
                {"name": "dropout", "type": "Dropout", "rate": 0.3},
                {"name": "output_probabilities", "type": "Dense", "units": 10, "activation": "softmax"},
            ],
            "total_trainable_parameters": int(model.count_params()),
        },
        "training_configuration": {
            "epochs": len(epoch_history["accuracy"]),
            "batch_size": 64,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss_function": "SparseCategoricalCrossentropy",
            "metrics": ["accuracy"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file_path": str(model_path.resolve()),
        "consistency_checks": {
            "confusion_matrix_sum_equals_10000": bool(cm_total == 10000),
            "report_support_sum_equals_10000": bool(int(report_dict["macro avg"]["support"]) == 10000),
            "distribution_sum_equals_70000": bool(class_dist["total_dataset"] == 70000),
            "accuracy_matches_confusion_matrix_diagonal": bool(
                np.isclose(accuracy_exact, np.trace(cm) / cm_total)
            ),
        },
    }
    with open(artifacts_dir / "artifact_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    # 10. Generate and Save Chart Images
    plot_accuracy_curve(epoch_history, artifacts_dir / "accuracy_curve.png")
    plot_loss_curve(epoch_history, artifacts_dir / "loss_curve.png")
    plot_confusion_matrix(cm, FASHION_MNIST_CLASSES, artifacts_dir / "confusion_matrix.png")
    plot_misclassified_grid(misclassified_samples, x_test, artifacts_dir / "misclassified_grid.png", num_samples=15)

    print("[Artifacts] All JSON files and high-resolution chart images generated successfully!")
    return {
        "metrics": metrics_payload,
        "metadata": metadata_payload,
    }
