"""
Frontend Contracts & OOD Integration Verification Suite (DL-002 Stage 6).

Verifies the 10 required frontend interaction scenarios:
1. Authentic clothing image -> ACCEPT -> HTTP 200 with full prediction
2. Non-clothing image -> REJECT -> HTTP 422 with no prediction
3. Ambiguous image -> UNCERTAIN -> HTTP 422 with no prediction
4. Invalid upload -> HTTP 400 with helpful validation message
5. Service / model unavailable -> HTTP 503 with service error message
6. Downstream CNN never executed on REJECT or UNCERTAIN
7. Uploading another image after REJECT
8. Uploading another image after UNCERTAIN
9. Reuploading the exact same file
10. Regression verification of all dashboard endpoints
"""

import base64
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

client = TestClient(app)


def get_authentic_clothing_image() -> Image.Image:
    """Loads an authentic Fashion-MNIST Trouser image from sample_predictions.json."""
    samples_file = BACKEND_DIR / "artifacts" / "sample_predictions.json"
    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    trouser = [s for s in data["correct_predictions"] if s["true_label_name"] == "Trouser"][0]
    b64_str = trouser["image_base64"].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def get_checkerboard_image() -> Image.Image:
    """Generates a 28x28 2x2 checkerboard non-clothing pattern."""
    cb = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            if ((i // 2) + (j // 2)) % 2 == 1:
                cb[i, j] = 255
    return Image.fromarray(cb, mode="L")


def run_all_tests():
    print("=" * 70)
    print("RUNNING FRONTEND-BACKEND INTEGRATION CONTRACT SUITE (DL-002 STAGE 6)")
    print("=" * 70)

    # Scenario 1: Authentic clothing image -> ACCEPT -> prediction displayed
    print("\n[Scenario 1] Authentic Clothing Image -> ACCEPT (HTTP 200)...")
    img_c = get_authentic_clothing_image()
    buf_c = io.BytesIO()
    img_c.save(buf_c, format="PNG")
    buf_c.seek(0)
    res1 = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    data1 = res1.json()
    assert data1["ood_decision"] == "ACCEPT"
    assert data1["classifier_executed"] is True
    assert data1["ood_score"] is not None and data1["ood_score"] >= 0.98
    assert data1["predicted_class_name"] == "Trouser"
    assert len(data1["probabilities"]) == 10
    print(f" -> PASSED: Returned HTTP 200 ACCEPT with predicted class '{data1['predicted_class_name']}'")

    # Scenario 2: Non-clothing image -> REJECT -> no prediction displayed
    print("\n[Scenario 2] Non-Clothing Pattern (Checkerboard) -> REJECT (HTTP 422)...")
    img_cb = get_checkerboard_image()
    buf_cb = io.BytesIO()
    img_cb.save(buf_cb, format="PNG")
    buf_cb.seek(0)
    res2 = client.post("/api/predict", files={"file": ("checkerboard.png", buf_cb.getvalue(), "image/png")})
    assert res2.status_code == 422, f"Expected 422, got {res2.status_code}"
    data2 = res2.json()
    assert data2["ood_decision"] == "REJECT"
    assert data2["classifier_executed"] is False
    assert data2["ood_score"] is not None and data2["ood_score"] < 0.02
    assert data2["predicted_class_id"] is None
    assert data2["predicted_class_name"] is None
    assert data2["confidence"] is None
    assert data2["probabilities"] == []
    print(f" -> PASSED: Returned HTTP 422 REJECT (classifier_executed={data2['classifier_executed']}, probabilities=[])")

    # Scenario 3: Ambiguous image -> UNCERTAIN -> no prediction displayed
    print("\n[Scenario 3] Ambiguous Input -> UNCERTAIN (HTTP 422)...")
    with patch("app.main.evaluate_ood_gate", return_value=("UNCERTAIN", 0.450000, "Ambiguous input")):
        res3 = client.post("/api/predict", files={"file": ("ambiguous.png", buf_cb.getvalue(), "image/png")})
        assert res3.status_code == 422, f"Expected 422, got {res3.status_code}"
        data3 = res3.json()
        assert data3["ood_decision"] == "UNCERTAIN"
        assert data3["classifier_executed"] is False
        assert data3["predicted_class_id"] is None
        assert data3["probabilities"] == []
        print(f" -> PASSED: Returned HTTP 422 UNCERTAIN (classifier_executed={data3['classifier_executed']}, probabilities=[])")

    # Scenario 4: Invalid file -> appropriate validation message (HTTP 400)
    print("\n[Scenario 4] Invalid File Types (HTTP 400)...")
    for bad_name in ["doc.pdf", "script.txt", "vector.svg"]:
        res_bad = client.post("/api/predict", files={"file": (bad_name, b"some content", "application/octet-stream")})
        assert res_bad.status_code == 400
        assert "Allowed file types" in res_bad.json()["detail"]
    print(" -> PASSED: Unsupported files correctly return HTTP 400 with helpful validation message")

    # Scenario 5: Service / model unavailable -> HTTP 503
    print("\n[Scenario 5] Backend Service / Model Unavailable (HTTP 503)...")
    from fastapi import HTTPException
    with patch("app.main.evaluate_ood_gate", side_effect=HTTPException(status_code=503, detail="OOD detector model unavailable")):
        res5 = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
        assert res5.status_code == 503
        assert "unavailable" in res5.json()["detail"].lower()
    print(" -> PASSED: Correctly returned HTTP 503 when detector is unavailable")

    # Scenario 6: Mock verification that CNN is NEVER called on REJECT or UNCERTAIN
    print("\n[Scenario 6] Mock Verification: Classifier NEVER Called on REJECT or UNCERTAIN...")
    with patch("app.main.get_trained_model") as mock_cnn:
        # 6a. REJECT
        res_r = client.post("/api/predict", files={"file": ("checkerboard.png", buf_cb.getvalue(), "image/png")})
        assert res_r.status_code == 422
        assert res_r.json()["ood_decision"] == "REJECT"
        mock_cnn.assert_not_called()

        # 6b. UNCERTAIN
        with patch("app.main.evaluate_ood_gate", return_value=("UNCERTAIN", 0.50, "Uncertain")):
            res_u = client.post("/api/predict", files={"file": ("ambiguous.png", buf_cb.getvalue(), "image/png")})
            assert res_u.status_code == 422
            assert res_u.json()["ood_decision"] == "UNCERTAIN"
            mock_cnn.assert_not_called()
    print(" -> PASSED: Verified get_trained_model() was NOT invoked for REJECT or UNCERTAIN")

    # Scenario 7: Uploading another image after REJECT
    print("\n[Scenario 7] Uploading Authentic Garment After REJECT...")
    # Step A: Upload non-clothing image (REJECT)
    res_step_a = client.post("/api/predict", files={"file": ("checkerboard.png", buf_cb.getvalue(), "image/png")})
    assert res_step_a.status_code == 422
    assert res_step_a.json()["ood_decision"] == "REJECT"
    # Step B: Upload authentic clothing image (ACCEPT)
    buf_c.seek(0)
    res_step_b = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
    assert res_step_b.status_code == 200
    assert res_step_b.json()["ood_decision"] == "ACCEPT"
    assert res_step_b.json()["predicted_class_name"] == "Trouser"
    print(" -> PASSED: Pipeline cleanly transitions from REJECT to ACCEPT on subsequent upload")

    # Scenario 8: Uploading another image after UNCERTAIN
    print("\n[Scenario 8] Uploading Authentic Garment After UNCERTAIN...")
    with patch("app.main.evaluate_ood_gate", return_value=("UNCERTAIN", 0.40, "Uncertain")):
        res_step_u = client.post("/api/predict", files={"file": ("ambiguous.png", buf_cb.getvalue(), "image/png")})
        assert res_step_u.status_code == 422
        assert res_step_u.json()["ood_decision"] == "UNCERTAIN"
    buf_c.seek(0)
    res_step_c = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
    assert res_step_c.status_code == 200
    assert res_step_c.json()["ood_decision"] == "ACCEPT"
    print(" -> PASSED: Pipeline cleanly transitions from UNCERTAIN to ACCEPT on subsequent upload")

    # Scenario 9: Reuploading the exact same file
    print("\n[Scenario 9] Reuploading Identical File...")
    buf_c.seek(0)
    res_first = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
    buf_c.seek(0)
    res_second = client.post("/api/predict", files={"file": ("trouser.png", buf_c.getvalue(), "image/png")})
    assert res_first.status_code == 200 and res_second.status_code == 200
    assert res_first.json()["predicted_class_id"] == res_second.json()["predicted_class_id"]
    assert np.isclose(res_first.json()["confidence"], res_second.json()["confidence"], atol=1e-5)
    print(" -> PASSED: Repeated identical uploads produce deterministic, identical predictions")

    # Scenario 10: Regression check of all dashboard endpoints
    print("\n[Scenario 10] All Dashboard Navigation Endpoints Regression Check...")
    endpoints = [
        "/api/health",
        "/api/model-info",
        "/api/dataset-info",
        "/api/metrics",
        "/api/confusion-matrix",
        "/api/samples?type=all&limit=10",
        "/api/samples?type=correct&limit=5",
        "/api/samples?type=misclassified&limit=5",
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Failed GET {ep} with status {resp.status_code}"
    print(f" -> PASSED: Verified all {len(endpoints)} dashboard metadata and evaluation endpoints return 200 OK")

    print("\n" + "=" * 70)
    print("ALL 10 FRONTEND-BACKEND INTEGRATION CONTRACT SCENARIOS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
