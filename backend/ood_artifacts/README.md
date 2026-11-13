# Out-of-Distribution (OOD) Gating Detector Artifacts

- **Project:** Fashion Item Classification with CNN (DL-002)
- **Model:** `FashionOOD_BinaryCNN` (`ood_detector.keras`)
- **Stage:** STAGE 3 — Training & Evaluation (Isolated)

## Known-Category Test Performance (`test.npz` — 20,000 samples)
- **ROC-AUC:** 1.0000
- **Accuracy (Illustrative $\tau=0.5$):** 99.98%
- **Precision:** 99.99%
- **Recall:** 99.97%
- **F1-Score:** 99.98%
- **False Acceptance Rate (Non-clothing accepted):** 0.01%
- **False Rejection Rate (Clothing rejected):** 0.03%

## Unseen-Category Test Performance (`test_unseen_ood.npz` — 9,998 samples)
- **Total Unseen Samples:** 9,998
- **Correctly Rejected:** 9,968 (99.70%)
- **Falsely Accepted (Illustrative $\tau=0.5$):** 30 (0.30%)
- **Mean Probability Score:** 0.0033

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

## Independent-Source OOD Generalization Test (Tiny ImageNet — 9,050 samples)
- **Dataset:** Tiny ImageNet Validation Set (ImageNet ILSVRC-2012 subset)
- **Classes Retained:** 181 strictly non-clothing categories (19 clothing/apparel classes excluded)
- **Total Samples:** 9,050 (100% negative, $y=0$)
- **Correctly Rejected:** 8,976 (99.18%)
- **Falsely Accepted (Illustrative $\tau=0.5$):** 74 (0.82%)
- **Mean Probability Score:** 0.0083 (Median: 0.0000)
- **90th Percentile Score ($p_{90}$):** 0.0000
- **Exact Pixel Collisions Across All Prior Datasets:** 0 collisions detected.
