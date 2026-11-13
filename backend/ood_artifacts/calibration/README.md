# Out-of-Distribution (OOD) Calibration and Threshold Selection

- **Project:** Fashion Item Classification with CNN (DL-002)
- **Stage:** STAGE 4 — OOD Calibration and Decision Threshold Selection
- **Calibration Split:** `backend/ood_data/cal.npz` (20,000 samples: 10k clothing, 10k non-clothing)

## Selected Three-Way Decision Thresholds
- **Rejection Threshold ($\tau_{\text{reject}}$):** `0.02`
- **Acceptance Threshold ($\tau_{\text{accept}}$):** `0.98`
- **Ambiguous Buffer ($\text{Score} \in [0.02, 0.98)$):** `UNCERTAIN` (Blocked from 10-class CNN)

### Performance on Calibration Data (`cal.npz`)
- **False Acceptance Rate (FAR):** `0.000%` (0 false acceptances out of 10,000 non-clothing samples)
- **False Rejection Rate (FRR):** `0.000%` (0 false rejections out of 10,000 clothing samples)
- **Uncertain Decision Rate:** `0.050%` (10 samples out of 20,000)
- **Decisive Coverage:** `99.950%` (19,990 / 20,000 samples)
- **Decisive Accuracy:** `100.000%`

## Probability Calibration Assessment
- **Raw Sigmoid Brier Score:** `0.000159`
- **Raw Sigmoid ECE (10 bins):** `0.0195%`
- **Temperature Scaling ($T = 1.3781$):** Brier `0.000148`, ECE `0.0235%`
- **Conclusion:** Raw sigmoid outputs are already exceptionally well-calibrated (ECE < 0.02%).

## Fail-Safe Mechanism
- Scores that are NaN, Inf, None, or non-numeric default immediately to `REJECT` (`passes_to_classifier = False`).
- Uncertain images (`0.02 <= Score < 0.98`) are never passed to the 10-class classifier.
