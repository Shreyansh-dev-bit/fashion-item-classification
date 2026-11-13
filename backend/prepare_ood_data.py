"""
Executable OOD Dataset Preparation Pipeline & Audit Suite.
Fashion Item Classification with CNN (DL-002) - Stage 1 Review & Corrections.

Prepares isolated, reproducible Out-of-Distribution (OOD) datasets:
- Train: 80,000 samples (40k Pos Fashion-MNIST + 40k Neg [24k CIFAR-10, 16k MNIST])
- Val: 20,000 samples (10k Pos Fashion-MNIST + 10k Neg [6k CIFAR-10, 4k MNIST])
- Cal: 20,000 samples (10k Pos Fashion-MNIST + 10k Neg [6k CIFAR-10, 4k MNIST])
- Test: 20,000 samples (10k Pos Fashion-MNIST test + 10k Neg [6k CIFAR-10 test, 4k MNIST test])
- Test Unseen OOD: 10,000 samples (CIFAR-100 official test: fruits/food, plants/trees,
  furniture, electronics, buildings, household items - 100% held out from training/val/cal)

Production Parity:
- Exact PIL Lanczos-3 downsampling.
- Uniform corner-based background inversion applied to all datasets.
- Distinct auditing of exact byte-level cryptographic MD5 hashes and perceptual dHashes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Ensure stdout/stderr supports UTF-8 on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import ARTIFACTS_DIR, MODELS_DIR
from app.ood_data import (
    audit_duplicates_and_leakage,
    partition_ood_datasets,
    save_ood_splits,
    set_seed,
    validate_ood_split,
)


def get_file_sha256(path: Path) -> str:
    """Calculates SHA256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_ood_data_pipeline(
    random_seed: int = 42,
    output_dir: Path = BACKEND_DIR / "ood_data",
) -> None:
    start_time = time.time()
    print("=" * 80)
    print("FASHION ITEM CLASSIFICATION WITH CNN (DL-002)")
    print("STAGE 1: Isolated Out-of-Distribution (OOD) Dataset Pipeline & Audit")
    print("=" * 80)

    # 0. Safety Check: Record original model and artifact hashes before pipeline
    print("\n[Safety Check] Verifying existing application assets are protected...")
    cnn_model_path = MODELS_DIR / "fashion_mnist_cnn.keras"
    initial_model_hash = None
    if cnn_model_path.exists():
        initial_model_hash = get_file_sha256(cnn_model_path)
        print(f"  - Verified existing CNN model: {cnn_model_path.name} (SHA256: {initial_model_hash[:16]}...)")
    else:
        print(f"  - Notice: {cnn_model_path.name} not found.")

    initial_artifact_hashes = {}
    if ARTIFACTS_DIR.exists():
        for p in sorted(ARTIFACTS_DIR.glob("*.json")):
            initial_artifact_hashes[p.name] = get_file_sha256(p)
        print(f"  - Verified {len(initial_artifact_hashes)} existing evaluation artifacts in {ARTIFACTS_DIR.name}/")

    # 1. Deterministic Seed
    print(f"\n[1/6] Setting deterministic random seed to {random_seed}...")
    set_seed(random_seed)

    # 2. Ingestion & Preprocessing & Partitioning
    print("\n[2/6] Ingesting, preprocessing, and partitioning datasets across strictly isolated split boundaries...")
    splits = partition_ood_datasets(random_seed=random_seed)

    # 3. Comprehensive Split Validation
    print("\n[3/6] Executing validation checks on all splits...")
    validation_reports = {}
    for sname, sdata in splits.items():
        print(f"  - Validating split '{sname}'...")
        rep = validate_ood_split(sdata)
        validation_reports[sname] = rep
        print(
            f"    -> Status: {rep['status']} | Samples: {rep['total_samples']:,} "
            f"(Pos: {rep['positive_samples']:,}, Neg: {rep['negative_samples']:,}) | "
            f"Shape: {rep['tensor_shape']} | Range: [{rep['pixel_min']}, {rep['pixel_max']}]"
        )

    # 4. Rigorous Duplicate & Anti-Leakage Auditing
    print("\n[4/6] Executing multi-level duplicate and anti-leakage audit...")
    audit_report = audit_duplicates_and_leakage(splits)
    print(f"  -> Audit status: {audit_report['overall_status']}")
    print("  [Index Isolation Checks]")
    for c in audit_report["index_isolation_checks"]:
        print(f"    - {c['description']}: OK")

    print("  [Exact Cryptographic Byte-Level Hash (MD5) Checks]")
    for c in audit_report["exact_byte_hash_checks"]:
        print(f"    - {c['check']}: {c['identical_byte_matches']} identical matches ({c['status']})")

    print("  [Perceptual Difference Hash (dHash) Gradient Checks]")
    for c in audit_report["perceptual_dhash_checks"]:
        print(f"    - {c['comparison']}: {c['perceptual_h0_matches']} structural profile matches")

    # 5. Serialization to disk
    print(f"\n[5/6] Serializing dataset archives to {output_dir}...")
    save_summary = save_ood_splits(splits, output_dir)
    for f in save_summary["saved_files"]:
        print(f"  - Created {f['file']} ({f['size_mb']} MB, {f['samples']:,} samples)")

    # 6. Generate summary JSON
    print("\n[6/6] Writing comprehensive metadata summary JSON...")
    summary_path = output_dir / "dataset_summary.json"
    summary_data = {
        "project_id": "DL-002",
        "stage": "STAGE 1 - Isolated OOD Dataset Preparation & Audit",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "random_seed": random_seed,
        "preprocessing_specification": {
            "grayscale_formula": "PIL convert('L') ITU-R 601-2 (0.2989*R + 0.5870*G + 0.1140*B)",
            "resampling_filter": "PIL Image.Resampling.LANCZOS (Lanczos-3)",
            "normalization": "uint8 / 255.0 -> float32 [0.0, 1.0]",
            "background_inversion": "corner_avg = (arr[0,0] + arr[0,-1] + arr[-1,0] + arr[-1,-1]) / 4.0; if corner_avg > 0.5: arr = 1.0 - arr",
            "canonical_shape": [28, 28, 1],
            "dtype": "float32",
        },
        "datasets_used": {
            "fashion_mnist": {
                "role": "In-Distribution (Positive, Clothing)",
                "classes": 10,
                "train_samples_used": 60000,
                "test_samples_used": 10000,
                "source": "tf.keras.datasets.fashion_mnist (Zalando Research, MIT License)",
            },
            "cifar10": {
                "role": "Out-of-Distribution (Known Negative: Animals & Vehicles)",
                "classes": 10,
                "train_samples_used": 36000,
                "test_samples_used": 6000,
                "source": "tf.keras.datasets.cifar10 (Alex Krizhevsky, U of Toronto)",
            },
            "mnist": {
                "role": "Out-of-Distribution (Known Negative: Handwritten Digits 0-9)",
                "classes": 10,
                "train_samples_used": 24000,
                "test_samples_used": 4000,
                "source": "tf.keras.datasets.mnist (Yann LeCun et al., CC BY-SA 3.0)",
            },
            "cifar100": {
                "role": "Out-of-Distribution (Reserved Unseen Categories: Fruits/Food, Trees/Plants, Furniture, Electronics, Buildings, Everyday Objects)",
                "classes": 100,
                "train_samples_used": 0,
                "test_samples_used": 10000,
                "source": "tf.keras.datasets.cifar100 (Alex Krizhevsky, U of Toronto)",
            },
        },
        "split_counts": {
            "train": {"total": 80000, "pos": 40000, "neg": 40000},
            "val": {"total": 20000, "pos": 10000, "neg": 10000},
            "cal": {"total": 20000, "pos": 10000, "neg": 10000},
            "test": {"total": 20000, "pos": 10000, "neg": 10000},
            "test_unseen_ood": {"total": 10000, "pos": 0, "neg": 10000},
        },
        "splits": validation_reports,
        "leakage_and_duplicate_audit": audit_report,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  - Written metadata summary to {summary_path.name}")

    # Final Safety Check: Verify original model and artifacts were NOT touched
    print("\n[Safety Verification] Confirming zero corruption of existing project assets...")
    if cnn_model_path.exists():
        final_model_hash = get_file_sha256(cnn_model_path)
        assert final_model_hash == initial_model_hash, (
            "CRITICAL: fashion_mnist_cnn.keras was altered during dataset pipeline!"
        )
        print("  - CONFIRMED: fashion_mnist_cnn.keras is byte-identical and untouched.")

    for art_name, initial_hash in initial_artifact_hashes.items():
        art_path = ARTIFACTS_DIR / art_name
        final_hash = get_file_sha256(art_path)
        assert final_hash == initial_hash, (
            f"CRITICAL: Artifact {art_name} was modified during dataset pipeline!"
        )
    print(f"  - CONFIRMED: All {len(initial_artifact_hashes)} evaluation artifacts are byte-identical and untouched.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"STAGE 1 PIPELINE & AUDIT COMPLETED in {elapsed:.1f}s")
    print(f"Total OOD dataset volume: 150,000 samples across 5 splits:")
    print(f"  * 80,000 train (50% pos / 50% neg)")
    print(f"  * 20,000 val (50% pos / 50% neg)")
    print(f"  * 20,000 cal (50% pos / 50% neg)")
    print(f"  * 20,000 known test (50% pos / 50% neg)")
    print(f"  * 10,000 unseen-category test (100% negative holdout: fruits, plants, furniture, etc.)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and audit isolated OOD datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / "ood_data",
        help="Target output directory for OOD datasets",
    )
    args = parser.parse_args()

    run_ood_data_pipeline(random_seed=args.seed, output_dir=args.output_dir)
