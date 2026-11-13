from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    project_id: str = Field(..., example="DL-002")
    project_name: str = Field(..., example="Fashion Item Classification with CNN")
    domain: str = Field(..., example="Computer Vision / Retail")
    version: str = Field(..., example="1.0.0")
    model_available: bool
    model_path: str
    artifacts_available: bool
    missing_artifacts: List[str]


class LayerInfo(BaseModel):
    name: str
    type: str
    filters: Optional[int] = None
    kernel_size: Optional[List[int]] = None
    pool_size: Optional[List[int]] = None
    units: Optional[int] = None
    rate: Optional[float] = None
    activation: Optional[str] = None


class PreprocessingConfig(BaseModel):
    color_mode: str = "Grayscale (1 channel)"
    target_size: List[int] = [28, 28]
    normalization_range: List[float] = [0.0, 1.0]
    input_shape: List[int] = [28, 28, 1]
    formula: str = "normalized_pixel = raw_pixel / 255.0"


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    loss_function: str
    metrics: List[str]


class ModelInfoResponse(BaseModel):
    project_id: str
    project_name: str
    model_name: str
    input_shape: List[int]
    labels: List[str]
    random_seed: int
    preprocessing: PreprocessingConfig
    architecture_layers: List[LayerInfo]
    total_trainable_parameters: int
    training_configuration: TrainingConfig
    model_file_exists: bool
    model_file_path: str


class SplitSizes(BaseModel):
    training_samples: int
    validation_samples: int
    held_out_test_samples: int
    total_dataset_samples: int


class ClassDistributionItem(BaseModel):
    class_id: int
    class_name: str
    train_count: int
    val_count: int
    test_count: int
    total_count: int


class DatasetInfoResponse(BaseModel):
    dataset_name: str = "Fashion-MNIST"
    domain: str = "Computer Vision / Retail"
    num_classes: int = 10
    classes: List[str]
    split_sizes: SplitSizes
    per_class_distribution: List[ClassDistributionItem]
    preprocessing_details: PreprocessingConfig


class ClassMetricItem(BaseModel):
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class MetricsResponse(BaseModel):
    project_id: str
    evaluated_set: str = "Held-Out Fashion-MNIST Test Set"
    test_loss: float
    test_accuracy: float
    test_accuracy_percentage: float
    macro_f1: float
    macro_f1_percentage: float
    weighted_f1: float
    total_test_samples: int
    total_correct: int
    total_misclassified: int
    class_metrics: List[ClassMetricItem]
    training_history: Dict[str, List[float]]


class ConfusionMatrixResponse(BaseModel):
    classes: List[str]
    matrix: List[List[int]]
    total_samples: int
    num_classes: int = 10


class SamplePredictionItem(BaseModel):
    test_index: int
    true_label_id: int
    true_label_name: str
    predicted_label_id: int
    predicted_label_name: str
    confidence: float
    confidence_percentage: float
    probabilities: List[float]
    image_base64: str
    is_correct: bool


class SamplesResponse(BaseModel):
    total_count: int
    filter_applied: str
    class_filter: Optional[str] = None
    samples: List[SamplePredictionItem]


class ClassProbability(BaseModel):
    class_id: int
    class_name: str
    probability: float
    percentage: float


class PredictResponse(BaseModel):
    ood_decision: str = Field(..., description="OOD gate decision: 'ACCEPT', 'REJECT', or 'UNCERTAIN'")
    ood_score: Optional[float] = Field(None, description="OOD clothing domain probability score in [0.0, 1.0]")
    classifier_executed: bool = Field(..., description="Whether the original 10-class Fashion-MNIST classifier was executed")
    message: Optional[str] = Field(None, description="User-friendly status explanation or domain warning message")
    detail: Optional[str] = Field(None, description="Detail message for HTTP client compatibility")
    predicted_class_id: Optional[int] = Field(None, description="Predicted class ID if accepted (0-9)")
    predicted_class_name: Optional[str] = Field(None, description="Predicted class name if accepted")
    confidence: Optional[float] = Field(None, description="Top predicted class confidence in [0.0, 1.0] if accepted")
    confidence_percentage: Optional[float] = Field(None, description="Top predicted class confidence percentage (0-100%) if accepted")
    probabilities: Optional[List[ClassProbability]] = Field(default_factory=list, description="All 10 class probabilities if accepted")
    input_image_details: Dict[str, Any]
    model_version: str = "DL-002 (FashionMNIST_CNN)"
