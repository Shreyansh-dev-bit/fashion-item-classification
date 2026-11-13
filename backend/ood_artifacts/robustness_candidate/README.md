# Stage 4.1: OOD Robustness Candidate Evaluation

- **Date:** 2026-11-13 16:16:03 UTC
- **Candidate Model:** `ood_detector_candidate.keras` (Trained with Negative Synthetic Diversity)
- **Existing Model:** `ood_detector.keras` (Unchanged, preserved for comparison/rollback)

## 1. Synthetic Stress Test Comparison
| Synthetic Test Pattern | Existing Detector Score | Candidate Detector Score | Existing Decision | Candidate Decision |
|---|:---:|:---:|:---:|:---:|
| Solid Black (All Zeros) | `0.334697` | `0.005771` | **REJECT** | **REJECT** |
| Solid White (All Ones) | `0.334697` | `0.005771` | **REJECT** | **REJECT** |
| Mid-Gray (All 0.5) | `0.000000` | `0.000000` | **REJECT** | **REJECT** |
| Uniform Random Noise [0, 1] (Seed 42) | `1.000000` | `0.000000` | **ACCEPT** | **REJECT** |
| Uniform Random Noise [0, 1] (Seed 123) | `1.000000` | `0.000000` | **ACCEPT** | **REJECT** |
| Gaussian Noise N(0.5, 0.15) (Seed 42) | `0.000000` | `0.000000` | **REJECT** | **REJECT** |
| Low Contrast Random Noise (0.50 +/- 0.02) | `0.000000` | `0.000000` | **REJECT** | **REJECT** |
| Single Center 2x2 Pixel in Black | `0.040088` | `0.035585` | **REJECT** | **REJECT** |
| Checkerboard Pattern 2x2 | `1.000000` | `0.000000` | **ACCEPT** | **REJECT** |
| Horizontal Stripes (1px) | `1.000000` | `0.000000` | **ACCEPT** | **REJECT** |
| Vertical Stripes (1px) | `1.000000` | `0.000001` | **ACCEPT** | **REJECT** |

## 2. Benchmark Dataset Performance Comparison
| Benchmark Dataset | Metric | Existing Detector | Candidate Detector |
|---|---|:---:|:---:|
| **Known Test (`test.npz`)** | ROC-AUC | 1.0000 | 1.0000 |
| | Accuracy | 99.98% | 99.92% |
| | False Acceptance Rate | 0.01% | 0.14% |
| | False Rejection Rate | 0.03% | 0.02% |
| **CIFAR-100 Unseen (`test_unseen_ood.npz`)** | Rejection Rate | 99.7% | 98.76% |
| | False Acceptance Rate | 0.3% | 1.24% |
| | Mean Probability Score | 0.0033 | 0.0136 |
| **Tiny ImageNet (`test_independent_ood.npz`)** | Rejection Rate | 99.18% | 98.99% |
| | False Acceptance Rate | 0.82% | 1.01% |
| | Mean Probability Score | 0.0083 | 0.0109 |

## 3. Candidate Calibration on `cal.npz`
- $\tau_{\text{reject}} = 0.02$, $\tau_{\text{accept}} = 0.98$
- False Acceptance Rate on `cal.npz`: **0.010%**
- False Rejection Rate on `cal.npz`: **0.000%**
- Decisive Coverage: **99.740%**
