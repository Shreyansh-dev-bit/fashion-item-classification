export interface HealthResponse {
  status: string
  project_id: string
  project_name: string
  domain: string
  version: string
  model_available: boolean
  model_path: string
  artifacts_available: boolean
  missing_artifacts: string[]
}

export interface LayerInfo {
  name: string
  type: string
  filters?: number | null
  kernel_size?: number[] | null
  pool_size?: number[] | null
  units?: number | null
  rate?: number | null
  activation?: string | null
}

export interface PreprocessingConfig {
  color_mode: string
  target_size: number[]
  normalization_range: number[]
  input_shape: number[]
  formula: string
}

export interface TrainingConfig {
  epochs: number
  batch_size: number
  optimizer: string
  learning_rate: number
  loss_function: string
  metrics: string[]
}

export interface ModelInfoResponse {
  project_id: string
  project_name: string
  model_name: string
  input_shape: number[]
  labels: string[]
  random_seed: number
  preprocessing: PreprocessingConfig
  architecture_layers: LayerInfo[]
  total_trainable_parameters: number
  training_configuration: TrainingConfig
  model_file_exists: boolean
  model_file_path: string
}

export interface SplitSizes {
  training_samples: number
  validation_samples: number
  held_out_test_samples: number
  total_dataset_samples: number
}

export interface ClassDistributionItem {
  class_id: number
  class_name: string
  train_count: number
  val_count: number
  test_count: number
  total_count: number
}

export interface DatasetInfoResponse {
  dataset_name: string
  domain: string
  num_classes: number
  classes: string[]
  split_sizes: SplitSizes
  per_class_distribution: ClassDistributionItem[]
  preprocessing_details: PreprocessingConfig
}

export interface ClassMetricItem {
  class_id: number
  class_name: string
  precision: number
  recall: number
  f1_score: number
  support: number
}

export interface MetricsResponse {
  project_id: string
  evaluated_set: string
  test_loss: number
  test_accuracy: number
  test_accuracy_percentage: number
  macro_f1: number
  macro_f1_percentage: number
  weighted_f1: number
  total_test_samples: number
  total_correct: number
  total_misclassified: number
  class_metrics: ClassMetricItem[]
  training_history: {
    epochs: number[]
    training_accuracy: number[]
    validation_accuracy: number[]
    training_loss: number[]
    validation_loss: number[]
  }
}

export interface ConfusionMatrixResponse {
  classes: string[]
  matrix: number[][]
  total_samples: number
  num_classes: number
}

export interface SamplePredictionItem {
  test_index: number
  true_label_id: number
  true_label_name: string
  predicted_label_id: number
  predicted_label_name: string
  confidence: number
  confidence_percentage: number
  probabilities: number[]
  image_base64: string
  is_correct: boolean
}

export interface SamplesResponse {
  total_count: number
  filter_applied: string
  class_filter?: string | null
  samples: SamplePredictionItem[]
}

export interface ClassProbability {
  class_id: number
  class_name: string
  probability: number
  percentage: number
}

export interface PredictResponse {
  predicted_class_id: number
  predicted_class_name: string
  confidence: number
  confidence_percentage: number
  probabilities: ClassProbability[]
  input_image_details: {
    filename: string
    original_size: number[]
    original_mode: string
    processed_shape: number[]
    inverted_for_dark_background?: boolean
  }
  model_version: string
}
