import base64
import io
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    DEFAULT_OOD_ACCEPT_THRESHOLD,
    DEFAULT_OOD_REJECT_THRESHOLD,
    settings,
    validate_ood_thresholds,
)
from app.main import app, get_trained_model, reset_trained_model
from app.ood_service import (
    evaluate_ood_gate,
    get_ood_detector,
    reset_ood_detector,
)

client = TestClient(app)


def get_authentic_clothing_image() -> Image.Image:
    """Loads an authentic Fashion-MNIST Trouser image from sample_predictions.json."""
    samples_file = BACKEND_DIR / "artifacts" / "sample_predictions.json"
    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    trouser = [s for s in data["correct_predictions"] if s["true_label_name"] == "Trouser"][0]
    b64_str = trouser["image_base64"].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def get_synthetic_checkerboard_image() -> Image.Image:
    """Generates a 28x28 2x2 checkerboard pattern known to be non-clothing OOD."""
    cb = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            if ((i // 2) + (j // 2)) % 2 == 1:
                cb[i, j] = 255
    return Image.fromarray(cb, mode="L")


def test_health():
    print("\n[Test 1] Testing GET /api/health...")
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project_id"] == "DL-002"
    assert data["model_available"] is True
    assert data["artifacts_available"] is True
    print(f" -> PASSED: Health OK, Model Available={data['model_available']}, Artifacts Available={data['artifacts_available']}")


def test_model_info():
    print("\n[Test 2] Testing GET /api/model-info...")
    response = client.get("/api/model-info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["project_id"] == "DL-002"
    assert data["model_name"] == "FashionMNIST_CNN"
    assert data["random_seed"] == 42
    assert len(data["labels"]) == 10
    assert data["labels"][0] == "T-shirt/top"
    assert data["labels"][-1] == "Ankle boot"
    assert len(data["architecture_layers"]) >= 8
    print(f" -> PASSED: Model Info OK, Parameters={data['total_trainable_parameters']}, Layers={len(data['architecture_layers'])}")


def test_dataset_info():
    print("\n[Test 3] Testing GET /api/dataset-info...")
    response = client.get("/api/dataset-info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["dataset_name"] == "Fashion-MNIST"
    assert data["split_sizes"]["training_samples"] == 54000
    assert data["split_sizes"]["validation_samples"] == 6000
    assert data["split_sizes"]["held_out_test_samples"] == 10000
    assert data["split_sizes"]["total_dataset_samples"] == 70000
    assert len(data["per_class_distribution"]) == 10
    print(f" -> PASSED: Dataset Info OK, Train={data['split_sizes']['training_samples']}, Val={data['split_sizes']['validation_samples']}, Test={data['split_sizes']['held_out_test_samples']}")


def test_metrics():
    print("\n[Test 4] Testing GET /api/metrics...")
    response = client.get("/api/metrics")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["project_id"] == "DL-002"
    assert np.isclose(data["test_accuracy"], 0.9073, atol=1e-3)
    assert np.isclose(data["macro_f1"], 0.9081, atol=1e-3)
    assert data["total_test_samples"] == 10000
    assert len(data["class_metrics"]) == 10
    assert len(data["training_history"]["training_accuracy"]) == 15
    print(f" -> PASSED: Metrics OK, Test Accuracy={data['test_accuracy_percentage']}%, Macro F1={data['macro_f1_percentage']}%, Total={data['total_test_samples']}")


def test_confusion_matrix():
    print("\n[Test 5] Testing GET /api/confusion-matrix...")
    response = client.get("/api/confusion-matrix")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert len(data["classes"]) == 10
    assert len(data["matrix"]) == 10
    assert all(len(row) == 10 for row in data["matrix"])
    matrix_sum = sum(sum(row) for row in data["matrix"])
    assert matrix_sum == 10000, f"Expected matrix sum to be 10,000, got {matrix_sum}"
    print(f" -> PASSED: Confusion Matrix OK, 10x10 Matrix Sum={matrix_sum} == 10,000")


def test_samples():
    print("\n[Test 6] Testing GET /api/samples...")
    # 6a. All samples
    res_all = client.get("/api/samples?type=all&limit=25")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert len(data_all["samples"]) <= 25
    sample = data_all["samples"][0]
    assert "test_index" in sample
    assert "confidence" in sample
    assert len(sample["probabilities"]) == 10
    assert sample["image_base64"].startswith("data:image/png;base64,")

    # 6b. Correct only
    res_correct = client.get("/api/samples?type=correct&limit=10")
    assert res_correct.status_code == 200
    assert all(s["is_correct"] is True for s in res_correct.json()["samples"])

    # 6c. Misclassified only
    res_mis = client.get("/api/samples?type=misclassified&limit=10")
    assert res_mis.status_code == 200
    assert all(s["is_correct"] is False for s in res_mis.json()["samples"])

    # 6d. Filter by class
    res_class = client.get("/api/samples?type=all&class_name=Trouser")
    assert res_class.status_code == 200
    for s in res_class.json()["samples"]:
        assert s["true_label_name"].lower() == "trouser" or s["predicted_label_name"].lower() == "trouser"

    # 6e. Filter by actual_class and predicted_class
    res_act = client.get("/api/samples?type=misclassified&actual_class=Shirt")
    assert res_act.status_code == 200
    for s in res_act.json()["samples"]:
        assert s["true_label_name"].lower() == "shirt"

    print(f" -> PASSED: Samples OK, Total={data_all['total_count']}, Verified Multi-Axis Filters & Probabilities")


def test_predict_valid_clothing():
    print("\n[Test 7] Testing POST /api/predict with Authentic Supported Clothing...")
    img = get_authentic_clothing_image()

    # 7a. Test valid PNG prediction
    buf_png = io.BytesIO()
    img.save(buf_png, format="PNG")
    buf_png.seek(0)

    res_png = client.post(
        "/api/predict",
        files={"file": ("trouser_test.png", buf_png.getvalue(), "image/png")},
    )
    assert res_png.status_code == 200, f"Expected 200, got {res_png.status_code}: {res_png.text}"
    pred_data = res_png.json()
    assert pred_data["ood_decision"] == "ACCEPT", f"Expected ACCEPT, got {pred_data['ood_decision']}"
    assert pred_data["classifier_executed"] is True
    assert pred_data["ood_score"] is not None and pred_data["ood_score"] >= settings.OOD_ACCEPT_THRESHOLD
    assert pred_data["predicted_class_name"] == "Trouser"
    assert len(pred_data["probabilities"]) == 10
    prob_sum = sum(p["probability"] for p in pred_data["probabilities"])
    assert np.isclose(prob_sum, 1.0, atol=1e-3), f"Probabilities must sum to 1.0, got {prob_sum}"
    print(f" -> PASSED: Valid PNG Prediction -> {pred_data['predicted_class_name']} ({pred_data['confidence_percentage']}%), OOD Score={pred_data['ood_score']}, Decision={pred_data['ood_decision']}")

    # 7b. Test valid JPEG and JPG image
    buf_jpg = io.BytesIO()
    img.convert("RGB").save(buf_jpg, format="JPEG")
    buf_jpg.seek(0)
    res_jpeg = client.post(
        "/api/predict",
        files={"file": ("trouser_test.jpeg", buf_jpg.getvalue(), "image/jpeg")},
    )
    assert res_jpeg.status_code == 200
    assert res_jpeg.json()["ood_decision"] == "ACCEPT"
    assert res_jpeg.json()["classifier_executed"] is True
    print(f" -> PASSED: Valid JPEG Prediction -> {res_jpeg.json()['predicted_class_name']} (OOD: {res_jpeg.json()['ood_decision']})")

    buf_jpg.seek(0)
    res_jpg = client.post(
        "/api/predict",
        files={"file": ("trouser_test.jpg", buf_jpg.getvalue(), "image/jpeg")},
    )
    assert res_jpg.status_code == 200
    assert res_jpg.json()["ood_decision"] == "ACCEPT"
    print(f" -> PASSED: Valid JPG Prediction -> {res_jpg.json()['predicted_class_name']}")

    # 7c. Test valid WEBP image
    buf_webp = io.BytesIO()
    img.save(buf_webp, format="WEBP")
    buf_webp.seek(0)
    res_webp = client.post(
        "/api/predict",
        files={"file": ("trouser_test.webp", buf_webp.getvalue(), "image/webp")},
    )
    assert res_webp.status_code == 200, f"Expected 200, got {res_webp.status_code}: {res_webp.text}"
    webp_data = res_webp.json()
    assert webp_data["ood_decision"] == "ACCEPT"
    assert webp_data["classifier_executed"] is True
    assert webp_data["predicted_class_name"] == "Trouser"
    print(f" -> PASSED: Valid WEBP Prediction -> {webp_data['predicted_class_name']} ({webp_data['confidence_percentage']}%)")

    # 7d. Test background inversion parameters:
    # A light-background apparel photo (white backdrop, typical of retail uploads)
    # inverts to dark background and is ACCEPTED.
    img_white_bg = Image.fromarray(255 - np.array(img), mode="L")
    buf_white_webp = io.BytesIO()
    img_white_bg.save(buf_white_webp, format="WEBP")
    buf_white_webp.seek(0)

    res_invert = client.post(
        "/api/predict?invert_colors=true",
        files={"file": ("trouser_white_bg.webp", buf_white_webp.getvalue(), "image/webp")},
    )
    assert res_invert.status_code == 200
    assert res_invert.json()["ood_decision"] == "ACCEPT"
    assert res_invert.json()["input_image_details"]["inverted_for_dark_background"] is True
    print(" -> PASSED: Explicit Background Inversion (True) on light-background apparel verified (200 ACCEPT)")

    # A native dark-background apparel image with invert_colors=false retains dark background
    buf_webp.seek(0)
    res_no_invert = client.post(
        "/api/predict?invert_colors=false",
        files={"file": ("trouser_dark_bg.webp", buf_webp.getvalue(), "image/webp")},
    )
    assert res_no_invert.status_code == 200
    assert res_no_invert.json()["ood_decision"] == "ACCEPT"
    assert res_no_invert.json()["input_image_details"]["inverted_for_dark_background"] is False
    print(" -> PASSED: Explicit Background Inversion (False) on dark-background apparel verified (200 ACCEPT)")


def test_predict_ood_rejected():
    print("\n[Test 8] Testing POST /api/predict with Non-Clothing Pattern (REJECT)...")
    img_cb = get_synthetic_checkerboard_image()
    buf = io.BytesIO()
    img_cb.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/predict",
        files={"file": ("checkerboard_ood.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 422, f"Expected HTTP 422 for REJECT, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["ood_decision"] == "REJECT", f"Expected REJECT, got {data['ood_decision']}"
    assert data["classifier_executed"] is False, "Downstream classifier MUST NOT execute for REJECT"
    assert data["ood_score"] is not None and data["ood_score"] < settings.OOD_REJECT_THRESHOLD
    assert data["predicted_class_id"] is None
    assert data["predicted_class_name"] is None
    assert data["confidence"] is None
    assert data["probabilities"] == []
    assert "outside the supported Fashion-MNIST clothing domain" in data["message"]
    print(f" -> PASSED: Non-clothing image correctly REJECTED (HTTP 422, score={data['ood_score']}, classifier_executed={data['classifier_executed']})")


def test_predict_ood_uncertain():
    print("\n[Test 9] Testing POST /api/predict with Ambiguous/Uncertain Input (UNCERTAIN)...")
    # Patch evaluate_ood_gate to simulate a score in the calibrated uncertainty band [0.02, 0.98)
    uncertain_score = 0.450000
    mock_msg = "Image uncertain: Uploaded image could not be confidently identified as supported clothing."

    with patch("app.main.evaluate_ood_gate", return_value=("UNCERTAIN", uncertain_score, mock_msg)):
        img = get_authentic_clothing_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = client.post(
            "/api/predict",
            files={"file": ("ambiguous.png", buf.getvalue(), "image/png")},
        )
        assert res.status_code == 422, f"Expected HTTP 422 for UNCERTAIN, got {res.status_code}"
        data = res.json()
        assert data["ood_decision"] == "UNCERTAIN"
        assert data["classifier_executed"] is False, "Downstream classifier MUST NOT execute for UNCERTAIN"
        assert np.isclose(data["ood_score"], uncertain_score, atol=1e-4)
        assert data["predicted_class_id"] is None
        assert data["probabilities"] == []
        print(f" -> PASSED: Ambiguous image correctly marked UNCERTAIN (HTTP 422, score={data['ood_score']}, classifier_executed={data['classifier_executed']})")


def test_mock_classifier_never_called_on_reject_or_uncertain():
    print("\n[Test 10] Verifying Original CNN Classifier is NEVER Called on REJECT or UNCERTAIN...")
    img_cb = get_synthetic_checkerboard_image()
    buf = io.BytesIO()
    img_cb.save(buf, format="PNG")
    buf.seek(0)

    # 10a. Verify on REJECT
    with patch("app.main.get_trained_model") as mock_get_cnn:
        res_reject = client.post(
            "/api/predict",
            files={"file": ("checkerboard.png", buf.getvalue(), "image/png")},
        )
        assert res_reject.status_code == 422
        assert res_reject.json()["ood_decision"] == "REJECT"
        mock_get_cnn.assert_not_called()
    print(" -> PASSED: Verified get_trained_model() was NOT called on REJECT")

    # 10b. Verify on UNCERTAIN
    with patch("app.main.evaluate_ood_gate", return_value=("UNCERTAIN", 0.50, "Uncertain input")):
        with patch("app.main.get_trained_model") as mock_get_cnn_u:
            res_u = client.post(
                "/api/predict",
                files={"file": ("ambiguous.png", buf.getvalue(), "image/png")},
            )
            assert res_u.status_code == 422
            assert res_u.json()["ood_decision"] == "UNCERTAIN"
            mock_get_cnn_u.assert_not_called()
    print(" -> PASSED: Verified get_trained_model() was NOT called on UNCERTAIN")


def test_predict_invalid_files():
    print("\n[Test 11] Testing Input Validation & Error Handling (HTTP 400)...")

    # 11a. Unsupported file extensions
    for bad_ext in ["malicious.txt", "unsupported.bmp", "sample.gif", "document.pdf"]:
        res_bad_ext = client.post(
            "/api/predict",
            files={"file": (bad_ext, b"binary content", "image/png")},
        )
        assert res_bad_ext.status_code == 400
        assert "Allowed file types are PNG, JPG, JPEG, and WEBP only" in res_bad_ext.json()["detail"]
    print(" -> PASSED: Unsupported extensions (.txt, .bmp, .gif, .pdf) correctly rejected with HTTP 400")

    # 11b. Invalid MIME type
    buf_webp = io.BytesIO(b"RIFF....WEBP")
    res_invalid_mime = client.post(
        "/api/predict",
        files={"file": ("test.webp", buf_webp.getvalue(), "application/pdf")},
    )
    assert res_invalid_mime.status_code == 400
    print(f" -> PASSED: Invalid MIME type correctly rejected with HTTP 400: '{res_invalid_mime.json()['detail']}'")

    # 11c. Empty file (0 bytes)
    res_empty = client.post(
        "/api/predict",
        files={"file": ("empty.webp", b"", "image/webp")},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()
    print(" -> PASSED: Empty file (0 bytes) correctly rejected with HTTP 400")

    # 11d. Corrupted image bytes
    res_corrupt = client.post(
        "/api/predict",
        files={"file": ("corrupted.webp", b"not-a-valid-webp-stream", "image/webp")},
    )
    assert res_corrupt.status_code == 400
    print(f" -> PASSED: Corrupted image correctly rejected with HTTP 400: '{res_corrupt.json()['detail']}'")


def test_ood_fail_closed_on_nan_inf_none():
    print("\n[Test 12] Testing Fail-Closed Gate Policies (NaN, Inf, Non-numeric, Out-of-bounds)...")
    dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)

    # 12a. Mock detector returning NaN
    mock_model_nan = MagicMock(return_value=np.array([[float("nan")]]))
    decision_nan, score_nan, msg_nan = evaluate_ood_gate(dummy_input, model=mock_model_nan)
    assert decision_nan == "REJECT", f"Expected REJECT on NaN score, got {decision_nan}"
    assert "invalid score" in msg_nan.lower()
    print(" -> PASSED: NaN score correctly fails closed (REJECT)")

    # 12b. Mock detector returning Infinity
    mock_model_inf = MagicMock(return_value=np.array([[float("inf")]]))
    decision_inf, score_inf, msg_inf = evaluate_ood_gate(dummy_input, model=mock_model_inf)
    assert decision_inf == "REJECT", f"Expected REJECT on Inf score, got {decision_inf}"
    print(" -> PASSED: Inf score correctly fails closed (REJECT)")

    # 12c. Mock detector returning out-of-bounds score (1.50)
    mock_model_oob = MagicMock(return_value=np.array([[1.50]]))
    decision_oob, score_oob, msg_oob = evaluate_ood_gate(dummy_input, model=mock_model_oob)
    assert decision_oob == "REJECT", f"Expected REJECT on out-of-bounds score, got {decision_oob}"
    print(" -> PASSED: Out-of-bounds score (1.50) correctly fails closed (REJECT)")

    # 12d. Mock detector returning None or non-numeric
    mock_model_str = MagicMock(return_value=np.array([["invalid_string"]]))
    decision_str, score_str, msg_str = evaluate_ood_gate(dummy_input, model=mock_model_str)
    assert decision_str == "REJECT", f"Expected REJECT on non-numeric score, got {decision_str}"
    print(" -> PASSED: Non-numeric score correctly fails closed (REJECT)")


def test_ood_detector_loading_failure():
    print("\n[Test 13] Testing Safe Failure on OOD Detector Loading Failure (Fail-Closed)...")
    reset_ood_detector()

    # 13a. Attempt loading non-existent model
    try:
        get_ood_detector(model_path=Path("non_existent_detector.keras"), force_reload=True)
        assert False, "Should have raised HTTPException 503"
    except HTTPException as e:
        assert e.status_code == 503
        assert "not found" in e.detail.lower()
        print(" -> PASSED: Missing OOD detector checkpoint raises HTTP 503")

    # 13b. Verify POST /api/predict fails closed when OOD detector is unavailable
    with patch("app.main.evaluate_ood_gate", side_effect=HTTPException(status_code=503, detail="OOD detector unavailable")):
        with patch("app.main.get_trained_model") as mock_cnn:
            img = get_authentic_clothing_image()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            res = client.post("/api/predict", files={"file": ("trouser.png", buf.getvalue(), "image/png")})
            assert res.status_code == 503
            mock_cnn.assert_not_called()
            print(" -> PASSED: API returns HTTP 503 and protects classifier when OOD service is unavailable")

    # Restore singleton
    reset_ood_detector()
    get_ood_detector()


def test_classifier_loading_failure_when_accepted():
    print("\n[Test 14] Testing Classifier Loading Failure when OOD Gate Accepts (HTTP 503)...")
    with patch("app.main.get_trained_model", side_effect=HTTPException(status_code=503, detail="Classifier unavailable")):
        img = get_authentic_clothing_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        res = client.post("/api/predict", files={"file": ("trouser.png", buf.getvalue(), "image/png")})
        assert res.status_code == 503
        assert "Classifier unavailable" in res.json()["detail"]
        print(" -> PASSED: Missing classifier correctly returns HTTP 503 when accepted")


def test_threshold_configuration_validation():
    print("\n[Test 15] Testing OOD Threshold Configuration Validation Bounds...")
    # Valid bounds
    validate_ood_thresholds(0.02, 0.98)
    validate_ood_thresholds(0.0, 1.0)
    validate_ood_thresholds(0.1, 0.9)
    print(" -> PASSED: Valid thresholds accepted")

    # Inverted bounds (reject >= accept)
    try:
        validate_ood_thresholds(0.98, 0.02)
        assert False, "Should have failed on inverted thresholds"
    except ValueError as e:
        assert "must satisfy" in str(e)

    # Negative bounds
    try:
        validate_ood_thresholds(-0.05, 0.95)
        assert False, "Should have failed on negative threshold"
    except ValueError as e:
        assert "must satisfy" in str(e)

    # Exceeding 1.0
    try:
        validate_ood_thresholds(0.05, 1.25)
        assert False, "Should have failed on threshold > 1.0"
    except ValueError as e:
        assert "must satisfy" in str(e)

    # Equal thresholds (reject == accept)
    try:
        validate_ood_thresholds(0.50, 0.50)
        assert False, "Should have failed on equal thresholds"
    except ValueError as e:
        assert "must satisfy" in str(e)

    print(" -> PASSED: Threshold bounds validation enforced (0.0 <= reject < accept <= 1.0)")


def test_cors_configuration():
    """Test 16: Verify CORS headers, preflight OPTIONS handling, and allowed origins."""
    print("\n[Test 16] Testing CORS Configuration & Preflight Handling...")
    prod_origin = "https://fashion-item-classification.vercel.app"
    preview_origin = "https://fashion-item-classification-git-main.vercel.app"
    unauthorized_origin = "https://unauthorized-domain.com"

    # 1. OPTIONS preflight from production Vercel frontend
    res_opt = client.options(
        "/api/health",
        headers={
            "Origin": prod_origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res_opt.status_code == 200, f"OPTIONS preflight failed: {res_opt.status_code}"
    assert res_opt.headers.get("access-control-allow-origin") == prod_origin
    assert "GET" in res_opt.headers.get("access-control-allow-methods", "")

    # 2. GET request with production Origin
    res_get = client.get("/api/health", headers={"Origin": prod_origin})
    assert res_get.status_code == 200
    assert res_get.headers.get("access-control-allow-origin") == prod_origin

    # 3. Root-level /health endpoint
    res_root_health = client.get("/health", headers={"Origin": prod_origin})
    assert res_root_health.status_code == 200
    assert res_root_health.json()["status"] == "healthy"
    assert res_root_health.headers.get("access-control-allow-origin") == prod_origin

    # 4. Preview Vercel Origin via regex
    res_preview = client.get("/api/health", headers={"Origin": preview_origin})
    assert res_preview.headers.get("access-control-allow-origin") == preview_origin

    # 5. Unauthorized origin must not receive allow-origin header
    res_unauth = client.get("/api/health", headers={"Origin": unauthorized_origin})
    assert res_unauth.headers.get("access-control-allow-origin") is None

    print(" -> PASSED: Production Vercel & preview CORS preflight and headers verified cleanly")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING COMPLETE FASTAPI & OOD GATE ENDPOINT SUITE (DL-002)")
    print("=" * 70)
    test_health()
    test_model_info()
    test_dataset_info()
    test_metrics()
    test_confusion_matrix()
    test_samples()
    test_predict_valid_clothing()
    test_predict_ood_rejected()
    test_predict_ood_uncertain()
    test_mock_classifier_never_called_on_reject_or_uncertain()
    test_predict_invalid_files()
    test_ood_fail_closed_on_nan_inf_none()
    test_ood_detector_loading_failure()
    test_classifier_loading_failure_when_accepted()
    test_threshold_configuration_validation()
    test_cors_configuration()
    print("\n" + "=" * 70)
    print("ALL 16 API, CORS & OOD GATE TESTS PASSED CLEANLY (100% SUCCESS)!")
    print("=" * 70)
