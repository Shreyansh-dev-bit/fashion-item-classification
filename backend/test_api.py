import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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


def test_predict():
    print("\n[Test 7] Testing POST /api/predict...")

    # 7a. Test valid PNG prediction using a synthetic trouser-like image
    img = Image.new("L", (100, 100), color=0)
    # Draw two vertical white legs
    for y in range(20, 90):
        for x in range(30, 45):
            img.putpixel((x, y), 255)
        for x in range(55, 70):
            img.putpixel((x, y), 255)
    for y in range(20, 45):
        for x in range(45, 55):
            img.putpixel((x, y), 255)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res_png = client.post(
        "/api/predict",
        files={"file": ("trouser_test.png", buf.getvalue(), "image/png")},
    )
    assert res_png.status_code == 200, f"Expected 200, got {res_png.status_code}: {res_png.text}"
    pred_data = res_png.json()
    assert "predicted_class_name" in pred_data
    assert "confidence" in pred_data
    assert len(pred_data["probabilities"]) == 10
    prob_sum = sum(p["probability"] for p in pred_data["probabilities"])
    assert np.isclose(prob_sum, 1.0, atol=1e-3), f"Probabilities must sum to 1.0, got {prob_sum}"
    print(f" -> PASSED: Valid PNG Prediction -> {pred_data['predicted_class_name']} ({pred_data['confidence_percentage']}%), Prob Sum={prob_sum:.4f}")

    # 7b. Test valid JPEG image
    buf_jpg = io.BytesIO()
    img.save(buf_jpg, format="JPEG")
    buf_jpg.seek(0)
    res_jpg = client.post(
        "/api/predict",
        files={"file": ("trouser_test.jpg", buf_jpg.getvalue(), "image/jpeg")},
    )
    assert res_jpg.status_code == 200
    print(f" -> PASSED: Valid JPEG Prediction -> {res_jpg.json()['predicted_class_name']}")

    # 7c. Test invalid file extension (.txt)
    res_invalid_ext = client.post(
        "/api/predict",
        files={"file": ("malicious.txt", b"plain text content", "text/plain")},
    )
    assert res_invalid_ext.status_code == 400
    print(f" -> PASSED: Invalid Extension correctly rejected with HTTP 400: '{res_invalid_ext.json()['detail']}'")

    # 7d. Test corrupted image bytes
    res_corrupt = client.post(
        "/api/predict",
        files={"file": ("corrupted.png", b"not-a-valid-png-stream", "image/png")},
    )
    assert res_corrupt.status_code == 400
    print(f" -> PASSED: Corrupted image correctly rejected with HTTP 400: '{res_corrupt.json()['detail']}'")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING COMPLETE FASTAPI ENDPOINT SUITE (DL-002)")
    print("=" * 70)
    test_health()
    test_model_info()
    test_dataset_info()
    test_metrics()
    test_confusion_matrix()
    test_samples()
    test_predict()
    print("\n" + "=" * 70)
    print("ALL FASTAPI ENDPOINTS TESTED AND PASSED CLEANLY!")
    print("=" * 70)
