# Out-of-Distribution (OOD) Dataset Documentation & Audit

**Project:** Fashion Item Classification with CNN (`DL-002`)
**Stage:** Stage 1 — Isolated OOD Dataset Preparation & Audit
**Directory:** `backend/ood_data/`
**Generated Files:** `train.npz`, `val.npz`, `cal.npz`, `test.npz`, `test_unseen_ood.npz`, `dataset_summary.json`

---

## 1. Why OOD Detection is Needed

The existing production model is a 10-class Convolutional Neural Network trained exclusively on the **Fashion-MNIST** benchmark (apparel items: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot).

Standard softmax neural networks suffer from overconfident misclassification when given inputs outside their target training domain. When presented with images of dogs, automobiles, handwritten digits, natural landscapes, fruits, or household items, standard softmax outputs still sum to $1.0$. The model will classify an arbitrary non-clothing object into one of the 10 clothing categories with false high confidence (often $>90\%$).

To ensure real-world reliability and trustworthiness, an **Out-of-Distribution (OOD) detector** will act as an upstream gate:
- **In-Distribution (Positive, Clothing):** Accepted and passed to the 10-class CNN classifier.
- **Out-of-Distribution (Negative, Non-clothing):** Rejected with a clean, informative domain rejection message before reaching the core classifier.

---

## 2. Production Preprocessing Audit and Parity

### Comparison: Production vs. OOD Data Pipeline

We audited the exact implementation in `backend/app/main.py` against the OOD dataset preparation pipeline:

| Dimension / Step | Production Endpoint (`predict_image`) | Initial Stage 1 (`ood_data.py`) | Inconsistency Identified | Corrected Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Image Ingestion** | `PIL.Image.open(io.BytesIO(contents))` + `verify()` | Direct uint8 NumPy arrays from `tf.keras.datasets` | None (canonical raw benchmark arrays are already decoded uint8) | Preserved, converted via `Image.fromarray()` when resizing |
| **Color Mode** | `gray_img = pil_img.convert("L")` (ITU-R 601-2 formula: $0.2989R + 0.5870G + 0.1140B$) | `tf.image.rgb_to_grayscale` | TF float rounding slightly diverges from PIL integer luma | Uses PIL `Image.fromarray(img).convert("L")` directly for 100% exact parity with production |
| **Resizing** | `gray_img.resize((28, 28), Image.Resampling.LANCZOS)` | `tf.image.resize(..., (28, 28), method="lanczos3")` | TensorFlow's `lanczos3` kernel has border padding differences from PIL's `Image.Resampling.LANCZOS` (mean abs pixel diff $\approx 0.032$, max $0.144$) | Uses PIL `Image.fromarray(...).resize((28, 28), Image.Resampling.LANCZOS)` to match production down to the exact pixel kernel |
| **Normalization** | `arr = np.array(resized_img, dtype=np.float32) / 255.0` | `cast(float32) / 255.0` | None (identical `/ 255.0` scaling to `[0.0, 1.0]`) | Preserved |
| **Corner-Based Background Inversion** | `corner_avg = (arr[0,0] + arr[0,-1] + arr[-1,0] + arr[-1,-1]) / 4.0; if corner_avg > 0.5: arr = 1.0 - arr` | **Omitted** for CIFAR-10 in initial preparation | **Critical Inconsistency**: 49.9% of natural images in CIFAR-10 have $corner\_avg > 0.5$ (bright skies, daylight, white backgrounds). In production, `predict_image` automatically inverts them before passing to the model. Without this in training, the detector would encounter inverted negative images at runtime that it was never trained on! | Applied the exact production function: `corner_avg = (arr[0,0] + arr[0,-1] + arr[-1,0] + arr[-1,-1]) / 4.0; if corner_avg > 0.5: arr = 1.0 - arr` uniformly across all datasets! |
| **Final Shape** | `(1, 28, 28, 1)` | `(N, 28, 28, 1)` | None (batch dimension vs single sample) | Preserved |

### Universal Transformation Rule
The exact production transformation is applied uniformly across all datasets:
- **Fashion-MNIST:** $corner\_avg \le 0.5$ ($0.0\%$ inverted) -> preserved as canonical dark background.
- **MNIST:** $corner\_avg \le 0.5$ ($0.0\%$ inverted) -> preserved as canonical dark background.
- **CIFAR-10 / CIFAR-100:** Samples with bright sky/daylight/background ($corner\_avg > 0.5$) are inverted to dark background, exactly as production does when a user uploads them. Samples with dark backgrounds remain uninverted.

---

## 3. Datasets Used, Licensing, and Sources

| Dataset | Modality | Semantic Categories | Role | Source & Licensing |
| :--- | :--- | :--- | :--- | :--- |
| **Fashion-MNIST** | $28 \times 28$ Grayscale | 10 clothing categories (`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`) | **In-Distribution (Positive, Label=1)** | `tf.keras.datasets.fashion_mnist` (Zalando Research, MIT License) |
| **CIFAR-10** | $32 \times 32$ RGB Photos | 10 non-clothing categories:<br>• 6 Animals (`bird`, `cat`, `deer`, `dog`, `frog`, `horse`)<br>• 4 Vehicles (`airplane`, `automobile`, `ship`, `truck`) | **Out-of-Distribution (Known Negative, Label=0)** | `tf.keras.datasets.cifar10` (Alex Krizhevsky, U of Toronto, Public/Academic) |
| **MNIST Digits** | $28 \times 28$ Grayscale | 10 handwritten digit categories ($0, 1, 2, \dots, 9$) | **Out-of-Distribution (Known Negative, Label=0)** | `tf.keras.datasets.mnist` (Yann LeCun et al., CC BY-SA 3.0) |
| **CIFAR-100** | $32 \times 32$ RGB Photos | 100 non-clothing fine categories across 20 superclasses: Fruits/Food, Trees/Plants, Furniture, Electronics, Buildings, Everyday Objects | **Out-of-Distribution (Reserved Unseen Categories, Label=0)** | `tf.keras.datasets.cifar100` (Alex Krizhevsky, U of Toronto, Public/Academic) |

---

## 4. Semantic Separation & Unseen-Category Evaluation Design

To prevent shortcut learning and objectively evaluate generalization, negative categories are strictly partitioned into **Known Categories** and **Reserved Unseen Categories**:

```
                                      ┌─────────────────────────────────────────────────────────────┐
                                      │              ALL CANDIDATE NEGATIVE DOMAINS                 │
                                      └──────────────────────────────┬──────────────────────────────┘
                                                                     │
                                     ┌───────────────────────────────┴──────────────────────────────┐
                                     ▼                                                              ▼
    ┌──────────────────────────────────────────────────┐        ┌──────────────────────────────────────────────────┐
    │    KNOWN NEGATIVE CATEGORIES (CIFAR-10 & MNIST)  │        │   RESERVED UNSEEN CATEGORIES (CIFAR-100 TEST)    │
    ├──────────────────────────────────────────────────┤        ├──────────────────────────────────────────────────┤
    │ • CIFAR-10 Animals: bird, cat, deer, dog, frog,  │        │ • Fruits & Food: apple, mushroom, orange, pear,  │
    │   horse (6 classes)                              │        │   sweet_pepper, bottles, bowls, cans, cups,      │
    │ • CIFAR-10 Vehicles: airplane, automobile, ship, │        │   plates (10 classes)                            │
    │   truck (4 classes)                              │        │ • Trees & Plants: maple, oak, palm, pine, willow,│
    │ • MNIST Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9     │        │   orchids, poppies, roses, sunflowers, tulips    │
    │   (10 classes)                                   │        │   (10 classes)                                   │
    │                                                  │        │ • Household Furniture: bed, chair, couch, table, │
    │ ───► Split into TRAIN, VAL, CAL, and TEST ◄───   │        │   wardrobe (5 classes)                           │
    │                                                  │        │ • Household Electronics: clock, keyboard, lamp,  │
    │                                                  │        │   telephone, television (5 classes)              │
    │                                                  │        │ • Buildings & Architecture: bridge, castle,      │
    │                                                  │        │   house, road, skyscraper (5 classes)            │
    │                                                  │        │ • Natural Scenes: cloud, forest, mountain, etc.  │
    │                                                  │        │                                                  │
    │                                                  │        │ ───► 100% RESERVED for test_unseen_ood.npz ◄───  │
    │                                                  │        │ (Zero samples in Training, Val, or Calibration!) │
    └──────────────────────────────────────────────────┘        └──────────────────────────────────────────────────┘
```

---

## 5. Exact Image Counts and Split Breakdown

### Summary by Split

| Split Name | Total Samples | Positive Samples (Fashion-MNIST) | Negative Samples (CIFAR-10) | Negative Samples (MNIST) | Negative Samples (CIFAR-100) | Pos / Neg Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **80,000** | 40,000 | 24,000 | 16,000 | 0 | 50.0% / 50.0% |
| **Validation** | **20,000** | 10,000 | 6,000 | 4,000 | 0 | 50.0% / 50.0% |
| **Calibration** | **20,000** | 10,000 | 6,000 | 4,000 | 0 | 50.0% / 50.0% |
| **Final Test** | **20,000** | 10,000 | 6,000 | 4,000 | 0 | 50.0% / 50.0% |
| **Test Unseen OOD** | **10,000** | 0 | 0 | 0 | 10,000 | 0.0% / 100.0% |
| **Total Pipeline** | **150,000** | **70,000** | **42,000** | **28,000** | **10,000** | — |

---

## 6. Exact vs. Perceptual Hash Auditing

We distinguish between two fundamentally different hashing techniques:

### A. Exact Cryptographic Hash Audit (MD5)
- **Mechanism:** Operates directly on the raw quantized `uint8` pixel byte buffer (`(x * 255).round().astype(uint8).tobytes()`).
- **Property:** Avalanche effect — any difference in a single pixel intensity alters the digest completely.
- **Purpose:** Proves that **zero identical byte-level samples** were accidentally duplicated across splits.
- **Results:**
  - `Train vs Test`: **0** byte-level collisions (PASSED).
  - `Val vs Test`: **0** byte-level collisions (PASSED).
  - `Cal vs Test`: **0** byte-level collisions (PASSED).
  - `Train vs Test Unseen OOD`: **0** byte-level collisions (PASSED).

### B. Perceptual Difference Hash Audit (64-bit dHash)
- **Mechanism:** Downsamples image to an $8 \times 9$ thumbnail and computes horizontal luminance gradients (`diff = thumbnail[:, :-1] > thumbnail[:, 1:]`), packing 64 boolean gradient signs into a 64-bit unsigned integer.
- **Property:** Measures structural visual layout and silhouette orientation. Invariant to subtle compression artifacts or sub-pixel luminance noise.
- **Purpose:** Audits perceptual similarity ($H = 0$ indicates identical gradient layout; $H \le 2$ indicates near-identical visual silhouettes).
- **Findings:**
  - In Fashion-MNIST, standard clothing items of the same category (e.g. basic plain black T-shirts or basic ankle boots) naturally share similar gradient layouts.
  - Across the entire 80,000-sample training pool and 20,000-sample test set, only **`14`** pairs share identical 64-bit dHashes ($H = 0$), all confirmed to be distinct physical scans of canonical apparel items within the benchmark.
  - Between the training pool and the unseen CIFAR-100 test set, **`0`** pairs share identical dHashes.

---

## 7. Limitations and Considerations

1. **Resolution Constraints:** All images are normalized to $28 \times 28$ pixels. Fine-grained texture details in high-resolution photography are compressed into this canonical geometry.
2. **Domain Diversity:** CIFAR-10 and CIFAR-100 are natural photographic scenes downsampled from color, whereas MNIST and Fashion-MNIST are centered scans. This creates valuable stylistic diversity for testing OOD generalization, but requires the detector to learn structural semantics rather than simply background intensity histograms.
3. **Threshold Boundary:** Calibration data (`cal.npz`) must be held strictly separate from `train.npz` and `test.npz` so that decision thresholds can be tuned without data snooping.
