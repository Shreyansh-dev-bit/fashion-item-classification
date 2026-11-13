import {
  ConfusionMatrixResponse,
  DatasetInfoResponse,
  HealthResponse,
  MetricsResponse,
  ModelInfoResponse,
  PredictResponse,
  SamplesResponse,
} from '../types/api'

/**
 * Resolves the backend API base URL:
 * - Uses import.meta.env.VITE_API_URL when configured.
 * - Safely trims whitespace and trailing slashes to avoid double slashes.
 * - Falls back to the local "/api" route when unconfigured, preserving local development.
 */
function resolveApiBase(): string {
  const envUrl = import.meta.env.VITE_API_URL
  if (typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.trim().replace(/\/+$/, '')
  }
  return '/api'
}

const API_BASE = resolveApiBase()

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`
    try {
      const errorJson = await res.json()
      if (errorJson.detail) {
        errorDetail = errorJson.detail
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail)
  }
  return res.json()
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  return handleResponse<HealthResponse>(res)
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model-info`)
  return handleResponse<ModelInfoResponse>(res)
}

export async function fetchDatasetInfo(): Promise<DatasetInfoResponse> {
  const res = await fetch(`${API_BASE}/dataset-info`)
  return handleResponse<DatasetInfoResponse>(res)
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`)
  return handleResponse<MetricsResponse>(res)
}

export async function fetchConfusionMatrix(): Promise<ConfusionMatrixResponse> {
  const res = await fetch(`${API_BASE}/confusion-matrix`)
  return handleResponse<ConfusionMatrixResponse>(res)
}

export async function fetchSamples(
  type: 'all' | 'correct' | 'misclassified' = 'all',
  className?: string,
  actualClass?: string,
  predictedClass?: string,
  limit: number = 50
): Promise<SamplesResponse> {
  const params = new URLSearchParams()
  params.set('type', type)
  params.set('limit', limit.toString())
  if (className && className !== 'all') {
    params.set('class_name', className)
  }
  if (actualClass && actualClass !== 'all') {
    params.set('actual_class', actualClass)
  }
  if (predictedClass && predictedClass !== 'all') {
    params.set('predicted_class', predictedClass)
  }

  const res = await fetch(`${API_BASE}/samples?${params.toString()}`)
  return handleResponse<SamplesResponse>(res)
}

export async function uploadPredictImage(
  file: File,
  invertColors?: boolean,
  signal?: AbortSignal
): Promise<PredictResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const params = new URLSearchParams()
  if (invertColors !== undefined) {
    params.set('invert_colors', invertColors ? 'true' : 'false')
  }

  const url = `${API_BASE}/predict${params.toString() ? `?${params.toString()}` : ''}`

  let res: Response
  try {
    res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal,
    })
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw err
    }
    throw new Error(
      'Network connection error: Unable to reach the inference backend service. Please check your connection and ensure the backend is running.'
    )
  }

  // HTTP 200 (ACCEPT) and HTTP 422 (REJECT / UNCERTAIN) both provide structured PredictResponse JSON
  if (res.status === 200 || res.status === 422) {
    try {
      const data: PredictResponse = await res.json()
      // Normalize probabilities array in case it is omitted
      if (!data.probabilities) {
        data.probabilities = []
      }
      return data
    } catch {
      throw new Error('Received malformed response payload from prediction service.')
    }
  }

  // HTTP 400: Bad Request / Upload validation error
  if (res.status === 400) {
    let errorDetail = 'Invalid image upload. Allowed file types are PNG, JPG, JPEG, and WEBP.'
    try {
      const errorJson = await res.json()
      if (errorJson.detail) {
        errorDetail = errorJson.detail
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail)
  }

  // HTTP 503: Service / Model unavailable
  if (res.status === 503) {
    let errorDetail = 'The prediction service or model checkpoint is currently unavailable.'
    try {
      const errorJson = await res.json()
      if (errorJson.detail) {
        errorDetail = errorJson.detail
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail)
  }

  // Any other unexpected HTTP error (e.g. 500)
  let errorDetail = `Inference request failed with status ${res.status}`
  try {
    const errorJson = await res.json()
    if (errorJson.detail) {
      errorDetail = errorJson.detail
    }
  } catch {
    // ignore
  }
  throw new Error(errorDetail)
}
