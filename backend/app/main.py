import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import tensorflow as tf

from app.config import ARTIFACTS_DIR, FASHION_MNIST_CLASSES, MODELS_DIR, settings
from app.schemas import (
    ClassMetricItem,
    ClassProbability,
    ConfusionMatrixResponse,
    DatasetInfoResponse,
    HealthResponse,
    LayerInfo,
    MetricsResponse,
    ModelInfoResponse,
    PredictResponse,
    PreprocessingConfig,
    SamplePredictionItem,
    SamplesResponse,
    SplitSizes,
    TrainingConfig,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Authentic Full-Stack Computer Vision Pipeline for Fashion Item Classification with CNN",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached CNN model singleton
_model_instance: Optional[tf.keras.Model] = None


def get_trained_model() -> tf.keras.Model:
    """
    Retrieves the saved CNN model, loading it into memory on first request.
    Raises HTTP 503 with instructions if the model is not trained.
    """
    global _model_instance
    if _model_instance is None:
        model_file = MODELS_DIR / "fashion_mnist_cnn.keras"
        if not model_file.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model not found at models/fashion_mnist_cnn.keras. Please run 'python train.py' in the backend directory first.",
            )
        try:
            _model_instance = tf.keras.models.load_model(model_file)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load trained Keras model: {e}",
            )
    return _model_instance


def _read_artifact(filename: str) -> Dict[str, Any]:
    file_path = ARTIFACTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Evaluation artifact '{filename}' not found. Please run 'python evaluate.py' in the backend directory first.",
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading artifact '{filename}': {e}",
        )


@app.get("/", tags=["General"])
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} ({settings.PROJECT_ID})",
        "domain": settings.DOMAIN,
        "docs": "/docs",
        "endpoints": {
            "health": f"{settings.API_PREFIX}/health",
            "model_info": f"{settings.API_PREFIX}/model-info",
            "dataset_info": f"{settings.API_PREFIX}/dataset-info",
            "metrics": f"{settings.API_PREFIX}/metrics",
            "confusion_matrix": f"{settings.API_PREFIX}/confusion-matrix",
            "samples": f"{settings.API_PREFIX}/samples",
            "predict": f"{settings.API_PREFIX}/predict",
        },
    }


@app.get(
    f"{settings.API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Status"],
    summary="Check API, model, and artifact availability",
)
def get_health():
    """Returns the operational status of the API, saved model file, and evaluation artifacts."""
    model_path = MODELS_DIR / "fashion_mnist_cnn.keras"
    model_exists = model_path.exists()

    required_artifacts = [
        "evaluation_metrics.json",
        "artifact_metadata.json",
        "class_distributions.json",
        "training_curves.json",
        "sample_predictions.json",
    ]
    missing = [name for name in required_artifacts if not (ARTIFACTS_DIR / name).exists()]

    return HealthResponse(
        status="healthy",
        project_id=settings.PROJECT_ID,
        project_name=settings.PROJECT_NAME,
        domain=settings.DOMAIN,
        version=settings.VERSION,
        model_available=model_exists,
        model_path=str(model_path.resolve()),
        artifacts_available=len(missing) == 0,
        missing_artifacts=missing,
    )


@app.get(
    f"{settings.API_PREFIX}/model-info",
    response_model=ModelInfoResponse,
    tags=["Metadata"],
    summary="Model architecture, layers, labels, and training settings",
)
def get_model_info():
    """Returns the CNN architecture specifications, layer details, hyperparameters, and random seed."""
    metadata = _read_artifact("artifact_metadata.json")
    model_path = MODELS_DIR / "fashion_mnist_cnn.keras"

    cnn_cfg = metadata.get("cnn_configuration", {})
    train_cfg = metadata.get("training_configuration", {})

    layers_info = [
        LayerInfo(
            name=l.get("name", ""),
            type=l.get("type", ""),
            filters=l.get("filters"),
            kernel_size=l.get("kernel_size"),
            pool_size=l.get("pool_size"),
            units=l.get("units"),
            rate=l.get("rate"),
            activation=l.get("activation"),
        )
        for l in cnn_cfg.get("layers", [])
    ]

    return ModelInfoResponse(
        project_id=metadata.get("project_id", settings.PROJECT_ID),
        project_name=metadata.get("project_name", settings.PROJECT_NAME),
        model_name=cnn_cfg.get("model_name", "FashionMNIST_CNN"),
        input_shape=cnn_cfg.get("input_shape", [28, 28, 1]),
        labels=FASHION_MNIST_CLASSES,
        random_seed=metadata.get("seed", 42),
        preprocessing=PreprocessingConfig(),
        architecture_layers=layers_info,
        total_trainable_parameters=cnn_cfg.get("total_trainable_parameters", 225034),
        training_configuration=TrainingConfig(
            epochs=train_cfg.get("epochs", 15),
            batch_size=train_cfg.get("batch_size", 64),
            optimizer=train_cfg.get("optimizer", "Adam"),
            learning_rate=train_cfg.get("learning_rate", 0.001),
            loss_function=train_cfg.get("loss_function", "SparseCategoricalCrossentropy"),
            metrics=train_cfg.get("metrics", ["accuracy"]),
        ),
        model_file_exists=model_path.exists(),
        model_file_path=str(model_path.resolve()),
    )


@app.get(
    f"{settings.API_PREFIX}/dataset-info",
    response_model=DatasetInfoResponse,
    tags=["Metadata"],
    summary="Dataset split sizes, class distribution, and preprocessing details",
)
def get_dataset_info():
    """Returns authentic Fashion-MNIST dataset split information and per-category sample counts."""
    dist_data = _read_artifact("class_distributions.json")
    metadata = _read_artifact("artifact_metadata.json")
    split_sizes_dict = metadata.get("dataset_split_sizes", {})

    return DatasetInfoResponse(
        dataset_name="Fashion-MNIST",
        domain=settings.DOMAIN,
        num_classes=len(FASHION_MNIST_CLASSES),
        classes=FASHION_MNIST_CLASSES,
        split_sizes=SplitSizes(
            training_samples=split_sizes_dict.get("training_samples", 54000),
            validation_samples=split_sizes_dict.get("validation_samples", 6000),
            held_out_test_samples=split_sizes_dict.get("held_out_test_samples", 10000),
            total_dataset_samples=split_sizes_dict.get("total_dataset_samples", 70000),
        ),
        per_class_distribution=dist_data.get("per_class", []),
        preprocessing_details=PreprocessingConfig(),
    )


@app.get(
    f"{settings.API_PREFIX}/metrics",
    response_model=MetricsResponse,
    tags=["Evaluation"],
    summary="Real test accuracy, macro F1-score, class metrics, and training history",
)
def get_metrics():
    """Returns measured performance on the 10,000 held-out test images and epoch training history."""
    eval_metrics = _read_artifact("evaluation_metrics.json")
    history_data = _read_artifact("training_curves.json")

    report = eval_metrics.get("classification_report", {})
    class_metrics_list = []
    for idx, class_name in enumerate(FASHION_MNIST_CLASSES):
        if class_name in report:
            c_data = report[class_name]
            class_metrics_list.append(
                ClassMetricItem(
                    class_id=idx,
                    class_name=class_name,
                    precision=float(c_data["precision"]),
                    recall=float(c_data["recall"]),
                    f1_score=float(c_data["f1-score"]),
                    support=int(c_data["support"]),
                )
            )

    test_acc = float(eval_metrics.get("test_accuracy", 0.0))
    macro_f1 = float(eval_metrics.get("macro_f1", 0.0))

    return MetricsResponse(
        project_id=eval_metrics.get("project_id", settings.PROJECT_ID),
        evaluated_set=eval_metrics.get("evaluated_set", "Held-Out Fashion-MNIST Test Set"),
        test_loss=float(eval_metrics.get("test_loss", 0.0)),
        test_accuracy=test_acc,
        test_accuracy_percentage=round(test_acc * 100, 2),
        macro_f1=macro_f1,
        macro_f1_percentage=round(macro_f1 * 100, 2),
        weighted_f1=float(eval_metrics.get("weighted_f1", 0.0)),
        total_test_samples=int(eval_metrics.get("total_test_samples", 10000)),
        total_correct=int(eval_metrics.get("total_correct", 0)),
        total_misclassified=int(eval_metrics.get("total_misclassified", 0)),
        class_metrics=class_metrics_list,
        training_history={
            "epochs": [float(e) for e in history_data.get("epochs", [])],
            "training_accuracy": history_data.get("training_accuracy", []),
            "validation_accuracy": history_data.get("validation_accuracy", []),
            "training_loss": history_data.get("training_loss", []),
            "validation_loss": history_data.get("validation_loss", []),
        },
    )


@app.get(
    f"{settings.API_PREFIX}/confusion-matrix",
    response_model=ConfusionMatrixResponse,
    tags=["Evaluation"],
    summary="Full 10-class confusion matrix values and class labels",
)
def get_confusion_matrix():
    """Returns the authentic 10x10 confusion matrix computed over all 10,000 test set images."""
    eval_metrics = _read_artifact("evaluation_metrics.json")
    matrix = eval_metrics.get("confusion_matrix", [])

    return ConfusionMatrixResponse(
        classes=FASHION_MNIST_CLASSES,
        matrix=matrix,
        total_samples=int(eval_metrics.get("total_test_samples", 10000)),
        num_classes=len(FASHION_MNIST_CLASSES),
    )


@app.get(
    f"{settings.API_PREFIX}/samples",
    response_model=SamplesResponse,
    tags=["Evaluation"],
    summary="Actual test predictions with confidence scores and filtering",
)
def get_samples(
    filter_type: str = Query(
        "all",
        alias="type",
        description="Filter sample set: 'all', 'correct', or 'misclassified'",
        pattern="^(all|correct|misclassified)$",
    ),
    actual_class: Optional[str] = Query(
        None,
        description="Optional filter by actual ground truth class name (e.g. 'Shirt')",
    ),
    predicted_class: Optional[str] = Query(
        None,
        description="Optional filter by predicted class name (e.g. 'T-shirt/top')",
    ),
    class_name: Optional[str] = Query(
        None,
        description="Optional filter matching either actual or predicted class name",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=100,
        description="Maximum number of samples to return",
    ),
):
    """Returns authentic test set predictions (including image data URIs, confidences, and probabilities)."""
    samples_data = _read_artifact("sample_predictions.json")
    correct_raw = samples_data.get("correct_predictions", [])
    misclassified_raw = samples_data.get("misclassified_predictions", [])

    all_items: List[SamplePredictionItem] = []

    if filter_type in ["all", "correct"]:
        for item in correct_raw:
            all_items.append(
                SamplePredictionItem(
                    test_index=item["test_index"],
                    true_label_id=item["true_label_id"],
                    true_label_name=item["true_label_name"],
                    predicted_label_id=item["predicted_label_id"],
                    predicted_label_name=item["predicted_label_name"],
                    confidence=item["confidence"],
                    confidence_percentage=round(item["confidence"] * 100, 2),
                    probabilities=item["probabilities"],
                    image_base64=item["image_base64"],
                    is_correct=True,
                )
            )

    if filter_type in ["all", "misclassified"]:
        for item in misclassified_raw:
            all_items.append(
                SamplePredictionItem(
                    test_index=item["test_index"],
                    true_label_id=item["true_label_id"],
                    true_label_name=item["true_label_name"],
                    predicted_label_id=item["predicted_label_id"],
                    predicted_label_name=item["predicted_label_name"],
                    confidence=item["confidence"],
                    confidence_percentage=round(item["confidence"] * 100, 2),
                    probabilities=item["probabilities"],
                    image_base64=item["image_base64"],
                    is_correct=False,
                )
            )

    # Filter by actual ground truth class
    if actual_class and actual_class.strip().lower() != "all":
        act_target = actual_class.strip().lower()
        all_items = [it for it in all_items if it.true_label_name.lower() == act_target]

    # Filter by predicted class
    if predicted_class and predicted_class.strip().lower() != "all":
        pred_target = predicted_class.strip().lower()
        all_items = [it for it in all_items if it.predicted_label_name.lower() == pred_target]

    # Filter by generic class_name (either actual or predicted)
    if class_name and class_name.strip().lower() != "all":
        target_name = class_name.strip().lower()
        all_items = [
            it
            for it in all_items
            if it.true_label_name.lower() == target_name
            or it.predicted_label_name.lower() == target_name
        ]

    # Limit results
    sliced_items = all_items[:limit]

    return SamplesResponse(
        total_count=len(sliced_items),
        filter_applied=filter_type,
        class_filter=actual_class or predicted_class or class_name,
        samples=sliced_items,
    )


@app.post(
    f"{settings.API_PREFIX}/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Real inference on an uploaded retail apparel image",
)
async def predict_image(
    file: UploadFile = File(..., description="Image file (PNG, JPG, or JPEG)"),
    invert_colors: Optional[bool] = Query(
        None,
        description="Invert pixel intensities (useful for light-background product images). Auto-detected if omitted.",
    ),
):
    """
    Accepts an uploaded PNG, JPG, or JPEG image, validates it, preprocesses it to
    (1, 28, 28, 1) normalized [0, 1] grayscale, and runs the real CNN model.
    """
    # 1. Validate file extension and MIME type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing in upload request.",
        )

    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    allowed_content_types = {"image/png", "image/jpeg", "image/jpg"}

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{file_ext}'. Allowed file types are PNG, JPG, and JPEG only.",
        )

    if file.content_type and file.content_type.lower() not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type '{file.content_type}'. Allowed types are image/png, image/jpeg.",
        )

    # 2. Read image bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {e}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Open image via PIL and validate structure
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()  # Verify file integrity
        # Re-open because verify() closes the stream
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file. Must be a valid PNG or JPEG image: {e}",
        )

    orig_width, orig_height = pil_img.size
    orig_mode = pil_img.mode

    # 4. Convert to Grayscale
    gray_img = pil_img.convert("L")

    # 5. Resize to 28x28 (using Lanczos interpolation)
    resized_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 6. Normalize pixel values to 0.0-1.0
    arr = np.array(resized_img, dtype=np.float32) / 255.0

    # Auto-detect or apply background inversion if necessary
    # Fashion-MNIST dataset is light foreground apparel on dark (black) background
    corner_avg = float((arr[0, 0] + arr[0, -1] + arr[-1, 0] + arr[-1, -1]) / 4.0)
    should_invert = invert_colors if invert_colors is not None else (corner_avg > 0.5)

    if should_invert:
        arr = 1.0 - arr

    # 7. Reshape to (1, 28, 28, 1)
    input_tensor = arr.reshape(1, 28, 28, 1)

    # 8. Load real CNN model (or return clear 503 error if untrained)
    model = get_trained_model()

    # 9. Run real inference
    try:
        raw_preds = model.predict(input_tensor, verbose=0)
        probabilities = raw_preds[0]  # Array of 10 probabilities
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {e}",
        )

    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])

    # 10. Format response with all ten class probabilities
    class_probs: List[ClassProbability] = [
        ClassProbability(
            class_id=i,
            class_name=FASHION_MNIST_CLASSES[i],
            probability=float(probabilities[i]),
            percentage=round(float(probabilities[i]) * 100, 2),
        )
        for i in range(len(FASHION_MNIST_CLASSES))
    ]

    return PredictResponse(
        predicted_class_id=top_idx,
        predicted_class_name=FASHION_MNIST_CLASSES[top_idx],
        confidence=top_confidence,
        confidence_percentage=round(top_confidence * 100, 2),
        probabilities=class_probs,
        input_image_details={
            "filename": file.filename,
            "original_size": [orig_width, orig_height],
            "original_mode": orig_mode,
            "processed_shape": [1, 28, 28, 1],
            "inverted_for_dark_background": should_invert,
        },
        model_version="DL-002 (FashionMNIST_CNN)",
    )


@app.get(f"{settings.API_PREFIX}/artifacts/{{filename}}", tags=["Artifacts"])
def get_artifact_image(filename: str):
    """Serves high-resolution evaluation chart images."""
    allowed_images = {
        "accuracy_curve.png",
        "loss_curve.png",
        "confusion_matrix.png",
        "misclassified_grid.png",
    }
    if filename not in allowed_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact image '{filename}'.",
        )
    image_path = ARTIFACTS_DIR / filename
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested artifact image '{filename}' not found.",
        )
    return FileResponse(image_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
