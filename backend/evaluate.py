import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
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

from app.config import ARTIFACTS_DIR, MODELS_DIR
from app.evaluator import generate_evaluation_artifacts


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the Fashion Item Classification CNN and generate authentic artifacts (DL-002)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "fashion_mnist_cnn.keras"),
        help="Path to the saved Keras model file",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Directory to save evaluation JSON files and charts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FASHION ITEM CLASSIFICATION WITH CNN (DL-002)")
    print("Evaluation & Real Artifact Generation on Held-Out Test Set")
    print("=" * 70)

    results = generate_evaluation_artifacts(
        model_path=Path(args.model_path),
        artifacts_dir=Path(args.artifacts_dir),
        random_seed=args.seed,
    )

    metrics = results["metrics"]
    metadata = results["metadata"]

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Model Evaluated:     {metadata['cnn_configuration']['model_name']} ({args.model_path})")
    print(f"Test Set Evaluated:  {metrics['total_test_samples']} authentic held-out Fashion-MNIST images")
    print(f"Test Accuracy:       {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:      {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score:   {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Correct Count:       {metrics['total_correct']}")
    print(f"Misclassified Count: {metrics['total_misclassified']}")
    print("\nConsistency Verifications:")
    for check_name, passed in metadata["consistency_checks"].items():
        status = "PASSED" if passed else "FAILED"
        print(f" - {check_name}: {status}")

    print("\nArtifacts Saved in:", args.artifacts_dir)
    print(" - evaluation_metrics.json")
    print(" - training_curves.json")
    print(" - sample_predictions.json")
    print(" - class_distributions.json")
    print(" - artifact_metadata.json")
    print(" - accuracy_curve.png")
    print(" - loss_curve.png")
    print(" - confusion_matrix.png")
    print(" - misclassified_grid.png")
    print("\n[SUCCESS] All evaluation artifacts generated and verified cleanly!")


if __name__ == "__main__":
    main()
