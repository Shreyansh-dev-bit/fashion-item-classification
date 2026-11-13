"""
Stage 4.1: OOD Robustness Improvement and Independent Stress Testing.

Investigates synthetic pattern vulnerability (noise, stripes, checkerboards)
and trains a separate candidate OOD detector with negative diversity.

Strict Operational Constraints:
1. Does NOT overwrite or modify existing ood_detector.keras.
2. Does NOT modify fashion_mnist_cnn.keras.
3. Does NOT modify existing evaluation artifacts or Stage 4 calibration files.
4. Preserves existing test datasets (test.npz, test_unseen_ood.npz, test_independent_ood.npz) as inference-only.
5. All candidate artifacts are saved in backend/ood_artifacts/robustness_candidate/.
"""

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras

from app.ood_model import build_ood_model

# Directories
DATA_DIR = BACKEND_DIR / "ood_data"
ARTIFACTS_DIR = BACKEND_DIR / "ood_artifacts"
CANDIDATE_DATA_DIR = DATA_DIR / "robustness_candidate"
CANDIDATE_ARTIFACTS_DIR = ARTIFACTS_DIR / "robustness_candidate"

EXISTING_MODEL_PATH = ARTIFACTS_DIR / "ood_detector.keras"
CANDIDATE_MODEL_PATH = CANDIDATE_ARTIFACTS_DIR / "ood_detector_candidate.keras"


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_synthetic_negative_samples(count_per_type: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generates diverse synthetic non-clothing patterns:
    1. Random noise (uniform, Gaussian, multi-scale coarse noise)
    2. Checkerboard patterns (1x1, 2x2, 3x3, 4x4, 7x7 at varying phases)
    3. Horizontal stripes (1px, 2px, 3px, 4px)
    4. Vertical stripes (1px, 2px, 3px, 4px)
    5. Diagonal stripes and crosshatch grids
    6. Geometric shapes (rectangles, circles, ellipses, polygons)
    7. Random lines and polylines
    8. Uniform blanks, linear gradients, and low-contrast patterns
    """
    rng = np.random.RandomState(seed)
    all_imgs = []

    # 1. Random Noise
    for _ in range(count_per_type):
        mode = rng.choice(["uniform", "gaussian", "coarse"])
        if mode == "uniform":
            img = rng.uniform(0.0, 1.0, (28, 28)).astype(np.float32)
        elif mode == "gaussian":
            mean = rng.uniform(0.2, 0.8)
            std = rng.uniform(0.05, 0.25)
            img = np.clip(rng.normal(mean, std, (28, 28)), 0.0, 1.0).astype(np.float32)
        else:
            coarse_size = rng.choice([4, 7, 14])
            coarse = rng.uniform(0.0, 1.0, (coarse_size, coarse_size)).astype(np.float32)
            pil_c = Image.fromarray((coarse * 255.0).astype(np.uint8)).resize((28, 28), Image.Resampling.BILINEAR)
            img = np.array(pil_c, dtype=np.float32) / 255.0
        all_imgs.append(img)

    # 2. Checkerboard
    for _ in range(count_per_type):
        k = rng.choice([1, 2, 3, 4, 7])
        phase_x = rng.randint(0, k)
        phase_y = rng.randint(0, k)
        y_idx, x_idx = np.indices((28, 28))
        grid = (((x_idx + phase_x) // k) + ((y_idx + phase_y) // k)) % 2
        c1, c2 = rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0)
        img = (grid * (c2 - c1) + c1).astype(np.float32)
        all_imgs.append(img)

    # 3. Horizontal Stripes
    for _ in range(count_per_type):
        w = rng.choice([1, 2, 3, 4])
        phase = rng.randint(0, w * 2)
        y_idx = np.indices((28, 28))[0]
        grid = ((y_idx + phase) // w) % 2
        c1, c2 = rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0)
        img = (grid * (c2 - c1) + c1).astype(np.float32)
        all_imgs.append(img)

    # 4. Vertical Stripes
    for _ in range(count_per_type):
        w = rng.choice([1, 2, 3, 4])
        phase = rng.randint(0, w * 2)
        x_idx = np.indices((28, 28))[1]
        grid = ((x_idx + phase) // w) % 2
        c1, c2 = rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0)
        img = (grid * (c2 - c1) + c1).astype(np.float32)
        all_imgs.append(img)

    # 5. Diagonal Stripes & Mesh
    for _ in range(count_per_type):
        w = rng.choice([2, 3, 4])
        angle = rng.choice([45, 135, "cross"])
        y_idx, x_idx = np.indices((28, 28))
        if angle == 45:
            grid = ((x_idx + y_idx) // w) % 2
        elif angle == 135:
            grid = ((x_idx - y_idx + 28) // w) % 2
        else:
            grid = (((x_idx // w) % 2) | ((y_idx // w) % 2))
        c1, c2 = rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0)
        img = (grid * (c2 - c1) + c1).astype(np.float32)
        all_imgs.append(img)

    # 6. Geometric Shapes
    for _ in range(count_per_type):
        pil_im = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(pil_im)
        shape_type = rng.choice(["rectangle", "circle", "ellipse", "polygon"])
        fill_val = rng.randint(128, 255)
        outline_val = rng.randint(128, 255)
        x0, y0 = rng.randint(2, 10), rng.randint(2, 10)
        x1, y1 = rng.randint(18, 26), rng.randint(18, 26)
        if shape_type == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=fill_val if rng.rand() > 0.3 else None, outline=outline_val)
        elif shape_type in ("circle", "ellipse"):
            draw.ellipse([x0, y0, x1, y1], fill=fill_val if rng.rand() > 0.3 else None, outline=outline_val)
        else:
            points = [(rng.randint(2, 26), rng.randint(2, 26)) for _ in range(rng.randint(3, 6))]
            draw.polygon(points, fill=fill_val if rng.rand() > 0.3 else None, outline=outline_val)
        img = np.array(pil_im, dtype=np.float32) / 255.0
        all_imgs.append(img)

    # 7. Random Lines
    for _ in range(count_per_type):
        pil_im = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(pil_im)
        for _ in range(rng.randint(2, 8)):
            p1 = (rng.randint(0, 27), rng.randint(0, 27))
            p2 = (rng.randint(0, 27), rng.randint(0, 27))
            draw.line([p1, p2], fill=rng.randint(100, 255), width=rng.randint(1, 2))
        img = np.array(pil_im, dtype=np.float32) / 255.0
        all_imgs.append(img)

    # 8. Uniform, Gradients, Low-Contrast
    for _ in range(count_per_type):
        sub = rng.choice(["solid", "gradient", "low_contrast"])
        if sub == "solid":
            val = rng.uniform(0.0, 1.0)
            img = np.full((28, 28), val, dtype=np.float32)
        elif sub == "gradient":
            axis = rng.choice([0, 1])
            ramp = np.linspace(0.0, 1.0, 28, dtype=np.float32)
            img = np.tile(ramp, (28, 1)) if axis == 1 else np.tile(ramp[:, None], (1, 28))
            if rng.rand() > 0.5:
                img = 1.0 - img
        else:
            base = rng.uniform(0.1, 0.9)
            img = np.clip(rng.uniform(base - 0.03, base + 0.03, (28, 28)), 0.0, 1.0).astype(np.float32)
        all_imgs.append(img)

    # Apply uniform corner background inversion
    arr = np.array(all_imgs, dtype=np.float32)
    corner_avg = (arr[:, 0, 0] + arr[:, 0, -1] + arr[:, -1, 0] + arr[:, -1, -1]) / 4.0
    should_invert = corner_avg > 0.5
    arr[should_invert] = 1.0 - arr[should_invert]
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.expand_dims(arr, -1)
    return arr


def prepare_candidate_datasets() -> Tuple[Path, Path]:
    """
    Constructs train_candidate.npz (80k) and val_candidate.npz (20k)
    by combining original Fashion-MNIST clothing (positives) with
    natural negatives (CIFAR-10, MNIST) + synthetic negatives.
    """
    CANDIDATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_cand_path = CANDIDATE_DATA_DIR / "train_candidate.npz"
    val_cand_path = CANDIDATE_DATA_DIR / "val_candidate.npz"

    if train_cand_path.exists() and val_cand_path.exists():
        print(f"[Dataset Preparation] Candidate datasets already exist at {CANDIDATE_DATA_DIR}.")
        return train_cand_path, val_cand_path

    print("\n[Dataset Preparation] Building candidate datasets with negative diversity...")
    # Load original train.npz and val.npz (read-only)
    orig_train = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    orig_val = np.load(DATA_DIR / "val.npz", allow_pickle=True)

    # Positives: Exactly preserved Fashion-MNIST clothing (40k train, 10k val)
    pos_train_x = orig_train["x"][orig_train["y"] == 1]
    pos_val_x = orig_val["x"][orig_val["y"] == 1]

    # Natural Negatives from original sets
    neg_train_x = orig_train["x"][orig_train["y"] == 0]
    neg_val_x = orig_val["x"][orig_val["y"] == 0]

    # Generate synthetic negatives
    # Train: 8,000 synthetic negatives (1,000 per type across 8 types)
    synth_train_x = generate_synthetic_negative_samples(count_per_type=1000, seed=42)
    # Val: 2,000 synthetic negatives (250 per type, distinct seed 9999)
    synth_val_x = generate_synthetic_negative_samples(count_per_type=250, seed=9999)

    # To maintain exact 80k/20k balance:
    # Train Negatives: 32,000 natural negatives + 8,000 synthetic negatives = 40,000 negatives
    cand_neg_train_x = np.concatenate([neg_train_x[:32000], synth_train_x], axis=0)
    # Val Negatives: 8,000 natural negatives + 2,000 synthetic negatives = 10,000 negatives
    cand_neg_val_x = np.concatenate([neg_val_x[:8000], synth_val_x], axis=0)

    # Shuffle training set deterministically
    rng_train = np.random.RandomState(42)
    x_train_cand = np.concatenate([pos_train_x, cand_neg_train_x], axis=0)
    y_train_cand = np.concatenate([np.ones(len(pos_train_x), dtype=np.int64), np.zeros(len(cand_neg_train_x), dtype=np.int64)])
    idx_tr = rng_train.permutation(len(x_train_cand))
    x_train_cand, y_train_cand = x_train_cand[idx_tr], y_train_cand[idx_tr]

    # Shuffle val set deterministically
    rng_val = np.random.RandomState(42)
    x_val_cand = np.concatenate([pos_val_x, cand_neg_val_x], axis=0)
    y_val_cand = np.concatenate([np.ones(len(pos_val_x), dtype=np.int64), np.zeros(len(cand_neg_val_x), dtype=np.int64)])
    idx_v = rng_val.permutation(len(x_val_cand))
    x_val_cand, y_val_cand = x_val_cand[idx_v], y_val_cand[idx_v]

    np.savez_compressed(train_cand_path, x=x_train_cand, y=y_train_cand)
    np.savez_compressed(val_cand_path, x=x_val_cand, y=y_val_cand)

    print(f" -> Saved {train_cand_path.name}: {len(x_train_cand):,} samples (Pos: {np.sum(y_train_cand==1):,}, Neg: {np.sum(y_train_cand==0):,})")
    print(f" -> Saved {val_cand_path.name}:   {len(x_val_cand):,} samples (Pos: {np.sum(y_val_cand==1):,}, Neg: {np.sum(y_val_cand==0):,})")

    return train_cand_path, val_cand_path


def train_candidate_model(train_path: Path, val_path: Path) -> keras.Model:
    """Trains the candidate OOD detector using the candidate training set with EarlyStopping."""
    CANDIDATE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if CANDIDATE_MODEL_PATH.exists():
        print(f"[Training] Candidate model already trained at {CANDIDATE_MODEL_PATH}.")
        return keras.models.load_model(CANDIDATE_MODEL_PATH)

    print("\n[Training] Training Candidate OOD Detector (FashionOOD_BinaryCNN)...")
    set_seed(42)

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    x_train, y_train = train_data["x"], train_data["y"]
    x_val, y_val = val_data["x"], val_data["y"]

    model = build_ood_model(input_shape=(28, 28, 1), dropout_rate=0.3, compile_model=True, learning_rate=0.001, seed=42)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(CANDIDATE_MODEL_PATH), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    ]

    t0 = time.time()
    history = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_data=(x_val, y_val), callbacks=callbacks, shuffle=True, verbose=1)
    elapsed = time.time() - t0
    print(f" -> Candidate training completed in {elapsed:.2f}s ({elapsed/60:.2f} min).")

    # Save history
    hist_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(CANDIDATE_ARTIFACTS_DIR / "candidate_training_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_serializable, f, indent=2)

    return keras.models.load_model(CANDIDATE_MODEL_PATH)


def evaluate_synthetic_stress_patterns(model: keras.Model) -> List[Dict[str, Any]]:
    """Runs standard synthetic stress tests against a model."""
    def predict_tensor(arr_2d: np.ndarray) -> float:
        assert arr_2d.shape == (28, 28)
        arr = arr_2d.astype(np.float32)
        corner_avg = (arr[0, 0] + arr[0, -1] + arr[-1, 0] + arr[-1, -1]) / 4.0
        if corner_avg > 0.5:
            arr = 1.0 - arr
        arr = np.clip(arr, 0.0, 1.0)
        tensor = np.expand_dims(np.expand_dims(arr, axis=0), axis=-1)
        return float(model.predict(tensor, verbose=0)[0][0])

    h_stripes = np.zeros((28, 28), dtype=np.float32)
    h_stripes[::2, :] = 1.0

    v_stripes = np.zeros((28, 28), dtype=np.float32)
    v_stripes[:, ::2] = 1.0

    checker = np.indices((28, 28)).sum(axis=0) % 2

    patterns = [
        ("Solid Black (All Zeros)", np.zeros((28, 28))),
        ("Solid White (All Ones)", np.ones((28, 28))),
        ("Mid-Gray (All 0.5)", np.full((28, 28), 0.5)),
        ("Uniform Random Noise [0, 1] (Seed 42)", np.random.RandomState(42).uniform(0, 1, (28, 28))),
        ("Uniform Random Noise [0, 1] (Seed 123)", np.random.RandomState(123).uniform(0, 1, (28, 28))),
        ("Gaussian Noise N(0.5, 0.15) (Seed 42)", np.clip(np.random.RandomState(42).normal(0.5, 0.15, (28, 28)), 0, 1)),
        ("Low Contrast Random Noise (0.50 +/- 0.02)", np.clip(np.random.RandomState(42).uniform(0.48, 0.52, (28, 28)), 0, 1)),
        ("Single Center 2x2 Pixel in Black", np.pad(np.ones((2, 2)), ((13, 13), (13, 13)))),
        ("Checkerboard Pattern 2x2", checker.astype(np.float32)),
        ("Horizontal Stripes (1px)", h_stripes),
        ("Vertical Stripes (1px)", v_stripes),
    ]

    results = []
    for name, img in patterns:
        score = predict_tensor(img)
        results.append({"test_pattern": name, "score": round(score, 6)})
    return results


def run_full_comparison():
    print("=" * 70)
    print("STAGE 4.1: OOD ROBUSTNESS INVESTIGATION & CANDIDATE EVALUATION")
    print("Project: Fashion Item Classification with CNN (DL-002)")
    print("=" * 70)

    # 1. Prepare candidate datasets
    train_cand_path, val_cand_path = prepare_candidate_datasets()

    # 2. Train candidate model
    cand_model = train_candidate_model(train_cand_path, val_cand_path)

    # 3. Load existing detector (read-only)
    print(f"\n[Comparison] Loading existing detector from {EXISTING_MODEL_PATH}...")
    orig_model = keras.models.load_model(EXISTING_MODEL_PATH)

    # 4. Evaluate Synthetic Stress Tests on both
    print("\n--- Synthetic Stress Test Comparison ---")
    orig_stress = evaluate_synthetic_stress_patterns(orig_model)
    cand_stress = evaluate_synthetic_stress_patterns(cand_model)

    stress_table = []
    for o, c in zip(orig_stress, cand_stress):
        name = o["test_pattern"]
        s_orig = o["score"]
        s_cand = c["score"]
        stress_table.append({
            "test_pattern": name,
            "existing_score": s_orig,
            "candidate_score": s_cand,
            "existing_decision_tau_0_5": "ACCEPT" if s_orig >= 0.5 else "REJECT",
            "candidate_decision_tau_0_5": "ACCEPT" if s_cand >= 0.5 else "REJECT",
        })
        print(f" -> {name:<42} | Existing: {s_orig:.6f} | Candidate: {s_cand:.6f}")

    # 5. Evaluate on Benchmark Datasets (Inference-Only)
    benchmarks = {
        "Known Test Set (test.npz)": DATA_DIR / "test.npz",
        "Unseen CIFAR-100 (test_unseen_ood.npz)": DATA_DIR / "test_unseen_ood.npz",
        "Independent Tiny ImageNet (test_independent_ood.npz)": DATA_DIR / "test_independent_ood.npz",
        "Calibration Set (cal.npz)": DATA_DIR / "cal.npz",
    }

    comparison_results: Dict[str, Any] = {}

    for bname, bpath in benchmarks.items():
        print(f"\nEvaluating on {bname} ({bpath.name})...")
        data = np.load(bpath, allow_pickle=True)
        x_data = data["x"]
        y_data = data["y"]

        orig_probs = orig_model.predict(x_data, batch_size=256, verbose=0).flatten()
        cand_probs = cand_model.predict(x_data, batch_size=256, verbose=0).flatten()

        is_binary = len(np.unique(y_data)) == 2
        entry: Dict[str, Any] = {
            "total_samples": len(x_data),
            "is_binary": is_binary,
        }

        if is_binary:
            orig_auc = float(roc_auc_score(y_data, orig_probs))
            cand_auc = float(roc_auc_score(y_data, cand_probs))
            orig_acc = float(accuracy_score(y_data, (orig_probs >= 0.5).astype(int)))
            cand_acc = float(accuracy_score(y_data, (cand_probs >= 0.5).astype(int)))
            orig_brier = float(brier_score_loss(y_data, orig_probs))
            cand_brier = float(brier_score_loss(y_data, cand_probs))

            # FAR and FRR at 0.5
            cm_orig = confusion_matrix(y_data, (orig_probs >= 0.5).astype(int))
            cm_cand = confusion_matrix(y_data, (cand_probs >= 0.5).astype(int))

            orig_far = float(cm_orig[0, 1] / (cm_orig[0, 0] + cm_orig[0, 1]))
            orig_frr = float(cm_orig[1, 0] / (cm_orig[1, 0] + cm_orig[1, 1]))
            cand_far = float(cm_cand[0, 1] / (cm_cand[0, 0] + cm_cand[0, 1]))
            cand_frr = float(cm_cand[1, 0] / (cm_cand[1, 0] + cm_cand[1, 1]))

            entry["metrics"] = {
                "existing_auc": round(orig_auc, 6),
                "candidate_auc": round(cand_auc, 6),
                "existing_accuracy": round(orig_acc, 6),
                "candidate_accuracy": round(cand_acc, 6),
                "existing_brier": round(orig_brier, 6),
                "candidate_brier": round(cand_brier, 6),
                "existing_far": round(orig_far, 6),
                "candidate_far": round(cand_far, 6),
                "existing_frr": round(orig_frr, 6),
                "candidate_frr": round(cand_frr, 6),
            }
            print(f" -> Existing  | AUC: {orig_auc:.4f} | Acc: {orig_acc*100:.2f}% | FAR: {orig_far*100:.2f}% | FRR: {orig_frr*100:.2f}%")
            print(f" -> Candidate | AUC: {cand_auc:.4f} | Acc: {cand_acc*100:.2f}% | FAR: {cand_far*100:.2f}% | FRR: {cand_frr*100:.2f}%")
        else:
            # Negative-only evaluation
            orig_fa = int(np.sum(orig_probs >= 0.5))
            cand_fa = int(np.sum(cand_probs >= 0.5))
            orig_far = float(orig_fa / len(x_data))
            cand_far = float(cand_fa / len(x_data))

            entry["metrics"] = {
                "existing_false_acceptances": orig_fa,
                "candidate_false_acceptances": cand_fa,
                "existing_far_pct": round(orig_far * 100, 2),
                "candidate_far_pct": round(cand_far * 100, 2),
                "existing_rejection_pct": round((1.0 - orig_far) * 100, 2),
                "candidate_rejection_pct": round((1.0 - cand_far) * 100, 2),
                "existing_mean_score": round(float(np.mean(orig_probs)), 6),
                "candidate_mean_score": round(float(np.mean(cand_probs)), 6),
                "existing_median_score": round(float(np.median(orig_probs)), 6),
                "candidate_median_score": round(float(np.median(cand_probs)), 6),
                "existing_p90_score": round(float(np.percentile(orig_probs, 90)), 6),
                "candidate_p90_score": round(float(np.percentile(cand_probs, 90)), 6),
            }
            print(f" -> Existing  | Rejected: {len(x_data)-orig_fa:,}/{len(x_data):,} ({entry['metrics']['existing_rejection_pct']}%) | FAR: {entry['metrics']['existing_far_pct']}% | Mean Score: {entry['metrics']['existing_mean_score']:.4f}")
            print(f" -> Candidate | Rejected: {len(x_data)-cand_fa:,}/{len(x_data):,} ({entry['metrics']['candidate_rejection_pct']}%) | FAR: {entry['metrics']['candidate_far_pct']}% | Mean Score: {entry['metrics']['candidate_mean_score']:.4f}")

        comparison_results[bname] = entry

    # 6. Candidate Calibration on cal.npz
    print("\n--- Candidate Calibration on cal.npz ---")
    cal_data = np.load(DATA_DIR / "cal.npz")
    c_probs = cand_model.predict(cal_data["x"], batch_size=256, verbose=0).flatten()
    c_y = cal_data["y"]

    # Candidate Score percentiles
    c_pos_p = c_probs[c_y == 1]
    c_neg_p = c_probs[c_y == 0]

    # Candidate Threshold Search
    c_grid = []
    for r_th in [0.001, 0.01, 0.02, 0.05, 0.1]:
        for a_th in [0.8, 0.9, 0.95, 0.98, 0.99]:
            fa = int(np.sum(c_neg_p >= a_th))
            fr = int(np.sum(c_pos_p < r_th))
            unc = int(np.sum((c_probs >= r_th) & (c_probs < a_th)))
            c_grid.append({
                "reject_th": r_th,
                "accept_th": a_th,
                "far": fa / len(c_neg_p),
                "frr": fr / len(c_pos_p),
                "uncertain_count": unc,
                "uncertain_rate": unc / len(c_probs),
                "coverage": 1.0 - (unc / len(c_probs)),
            })

    # Best candidate threshold
    c_selected = next(g for g in c_grid if g["reject_th"] == 0.02 and g["accept_th"] == 0.98)
    print(f"Candidate Calibrated Operating Point (tau_reject={c_selected['reject_th']}, tau_accept={c_selected['accept_th']}):")
    print(f" -> FAR: {c_selected['far']*100:.3f}%, FRR: {c_selected['frr']*100:.3f}%, Uncertain: {c_selected['uncertain_rate']*100:.3f}%, Coverage: {c_selected['coverage']*100:.3f}%")

    candidate_calib_summary = {
        "calibrated_thresholds": {
            "reject_threshold": c_selected["reject_th"],
            "accept_threshold": c_selected["accept_th"],
            "calibration_dataset": "cal.npz (20,000 samples)",
            "far_pct": round(c_selected["far"] * 100, 4),
            "frr_pct": round(c_selected["frr"] * 100, 4),
            "uncertain_rate_pct": round(c_selected["uncertain_rate"] * 100, 4),
            "coverage_pct": round(c_selected["coverage"] * 100, 4),
        },
        "score_distribution": {
            "positive_mean": round(float(np.mean(c_pos_p)), 6),
            "positive_min": round(float(np.min(c_pos_p)), 6),
            "negative_mean": round(float(np.mean(c_neg_p)), 6),
            "negative_max": round(float(np.max(c_neg_p)), 6),
        },
        "grid_search": c_grid,
    }

    # 7. Save All Artifacts
    print(f"\nWriting candidate artifacts to {CANDIDATE_ARTIFACTS_DIR}...")
    with open(CANDIDATE_ARTIFACTS_DIR / "performance_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    with open(CANDIDATE_ARTIFACTS_DIR / "synthetic_stress_comparison.json", "w", encoding="utf-8") as f:
        json.dump(stress_table, f, indent=2)

    with open(CANDIDATE_ARTIFACTS_DIR / "candidate_calibration.json", "w", encoding="utf-8") as f:
        json.dump(candidate_calib_summary, f, indent=2)

    # Markdown Summary
    readme_content = f"""# Stage 4.1: OOD Robustness Candidate Evaluation

- **Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Candidate Model:** `ood_detector_candidate.keras` (Trained with Negative Synthetic Diversity)
- **Existing Model:** `ood_detector.keras` (Unchanged, preserved for comparison/rollback)

## 1. Synthetic Stress Test Comparison
| Synthetic Test Pattern | Existing Detector Score | Candidate Detector Score | Existing Decision | Candidate Decision |
|---|:---:|:---:|:---:|:---:|
"""
    for r in stress_table:
        readme_content += f"| {r['test_pattern']} | `{r['existing_score']:.6f}` | `{r['candidate_score']:.6f}` | **{r['existing_decision_tau_0_5']}** | **{r['candidate_decision_tau_0_5']}** |\n"

    readme_content += f"""
## 2. Benchmark Dataset Performance Comparison
| Benchmark Dataset | Metric | Existing Detector | Candidate Detector |
|---|---|:---:|:---:|
| **Known Test (`test.npz`)** | ROC-AUC | {comparison_results['Known Test Set (test.npz)']['metrics']['existing_auc']:.4f} | {comparison_results['Known Test Set (test.npz)']['metrics']['candidate_auc']:.4f} |
| | Accuracy | {comparison_results['Known Test Set (test.npz)']['metrics']['existing_accuracy']*100:.2f}% | {comparison_results['Known Test Set (test.npz)']['metrics']['candidate_accuracy']*100:.2f}% |
| | False Acceptance Rate | {comparison_results['Known Test Set (test.npz)']['metrics']['existing_far']*100:.2f}% | {comparison_results['Known Test Set (test.npz)']['metrics']['candidate_far']*100:.2f}% |
| | False Rejection Rate | {comparison_results['Known Test Set (test.npz)']['metrics']['existing_frr']*100:.2f}% | {comparison_results['Known Test Set (test.npz)']['metrics']['candidate_frr']*100:.2f}% |
| **CIFAR-100 Unseen (`test_unseen_ood.npz`)** | Rejection Rate | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['existing_rejection_pct']}% | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['candidate_rejection_pct']}% |
| | False Acceptance Rate | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['existing_far_pct']}% | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['candidate_far_pct']}% |
| | Mean Probability Score | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['existing_mean_score']:.4f} | {comparison_results['Unseen CIFAR-100 (test_unseen_ood.npz)']['metrics']['candidate_mean_score']:.4f} |
| **Tiny ImageNet (`test_independent_ood.npz`)** | Rejection Rate | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['existing_rejection_pct']}% | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['candidate_rejection_pct']}% |
| | False Acceptance Rate | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['existing_far_pct']}% | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['candidate_far_pct']}% |
| | Mean Probability Score | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['existing_mean_score']:.4f} | {comparison_results['Independent Tiny ImageNet (test_independent_ood.npz)']['metrics']['candidate_mean_score']:.4f} |

## 3. Candidate Calibration on `cal.npz`
- $\\tau_{{\\text{{reject}}}} = 0.02$, $\\tau_{{\\text{{accept}}}} = 0.98$
- False Acceptance Rate on `cal.npz`: **{c_selected['far']*100:.3f}%**
- False Rejection Rate on `cal.npz`: **{c_selected['frr']*100:.3f}%**
- Decisive Coverage: **{c_selected['coverage']*100:.3f}%**
"""
    with open(CANDIDATE_ARTIFACTS_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(" -> All candidate robustness artifacts successfully generated.")
    print("\n" + "=" * 70)
    print("STAGE 4.1 INVESTIGATION AND EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_comparison()
