import json
import sys
import time
from pathlib import Path

# Ensure stdout supports UTF-8 on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path so app modules are resolvable
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.config import ARTIFACTS_DIR, FASHION_MNIST_CLASSES, MODELS_DIR
from app.dataset import load_and_preprocess_data, set_seed
from app.evaluator import generate_evaluation_artifacts
from app.model import build_fashion_mnist_cnn, compile_model


def run_training_pipeline(
    epochs: int = 15,
    batch_size: int = 64,
    random_seed: int = 42,
    model_name: str = "fashion_mnist_cnn.keras",
):
    print("=" * 70)
    print("FASHION ITEM CLASSIFICATION WITH CNN (DL-002)")
    print("Training Pipeline on Real Fashion-MNIST Dataset")
    print("=" * 70)

    # 1. Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Set deterministic seeds
    print(f"\n[1/5] Setting random seed to {random_seed} (Python, NumPy, TensorFlow)...")
    set_seed(random_seed)

    # 3. Load & Preprocess Data
    print("\n[2/5] Ingesting and preprocessing Fashion-MNIST dataset...")
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
            val_size=6000,
            random_seed=random_seed,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load/preprocess Fashion-MNIST dataset: {e}", file=sys.stderr)
        raise

    # 4. Construct CNN Architecture
    print("\n[3/5] Building Convolutional Neural Network architecture...")
    try:
        model = build_fashion_mnist_cnn(
            input_shape=(28, 28, 1),
            num_classes=len(FASHION_MNIST_CLASSES),
            dropout_rate=0.3,
        )
        model = compile_model(model, learning_rate=0.001)
        model.summary()
    except Exception as e:
        print(f"\n[ERROR] Failed to construct or compile CNN model: {e}", file=sys.stderr)
        raise

    # 5. Train the Model
    print(f"\n[4/5] Training for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()
    try:
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1,
        )
    except Exception as e:
        print(f"\n[ERROR] Training execution failed: {e}", file=sys.stderr)
        raise

    elapsed_time = time.time() - start_time
    print(f"\n[Info] Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")

    # 6. Evaluate on Held-Out Test Set (10,000 images)
    print("\n[5/5] Evaluating model on official 10,000 held-out test images...")
    try:
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f"\n-> Final Test Loss:     {test_loss:.4f}")
        print(f"-> Final Test Accuracy: {test_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"\n[ERROR] Evaluation on test set failed: {e}", file=sys.stderr)
        raise

    # 7. Save Model & Training Artifacts
    model_save_path = MODELS_DIR / model_name
    print(f"\n[Saving] Saving trained Keras model to: {model_save_path}")
    model.save(model_save_path)

    # Save training history and summary for evaluation metrics
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    history_save_path = ARTIFACTS_DIR / "training_history.json"
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    summary = {
        "project_id": "DL-002",
        "model_architecture": "FashionMNIST_CNN",
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "training_time_seconds": round(elapsed_time, 2),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "model_saved_path": str(model_save_path),
        "classes": FASHION_MNIST_CLASSES,
    }
    summary_save_path = ARTIFACTS_DIR / "training_summary.json"
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saving] Saved training history to: {history_save_path}")
    print(f"[Saving] Saved training summary to: {summary_save_path}")

    # 8. Generate full evaluation artifacts (JSON & charts) on held-out test set
    print("\n[Artifacts] Generating comprehensive evaluation metrics and visualizations...")
    try:
        generate_evaluation_artifacts(
            model_path=model_save_path,
            artifacts_dir=ARTIFACTS_DIR,
            random_seed=random_seed,
        )
    except Exception as e:
        print(f"[WARNING] Post-training artifact generation failed: {e}", file=sys.stderr)

    print("\n[SUCCESS] Real training and evaluation pipeline completed successfully!")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the Fashion Item Classification CNN on real Fashion-MNIST data (DL-002)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fashion_mnist_cnn.keras",
        help="Output model filename inside backend/models/",
    )

    args = parser.parse_args()
    run_training_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        model_name=args.model_name,
    )
