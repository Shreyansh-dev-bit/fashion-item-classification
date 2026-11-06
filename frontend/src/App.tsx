import { useState, useEffect } from 'react'
import {
  fetchHealth,
  fetchModelInfo,
  fetchDatasetInfo,
  fetchMetrics,
  fetchConfusionMatrix,
  fetchSamples,
} from './services/api'
import {
  HealthResponse,
  ModelInfoResponse,
  DatasetInfoResponse,
  MetricsResponse,
  ConfusionMatrixResponse,
  SamplePredictionItem,
} from './types/api'
import { Sidebar, PageId } from './components/Sidebar'
import { Header } from './components/Header'
import { LoadingSpinner } from './components/LoadingSpinner'
import { ErrorBanner } from './components/ErrorBanner'
import { EmptyState } from './components/EmptyState'
import { OverviewPage } from './pages/OverviewPage'
import { DatasetExplorerPage } from './pages/DatasetExplorerPage'
import { ArchitecturePage } from './pages/ArchitecturePage'
import { TrainingEvaluationPage } from './pages/TrainingEvaluationPage'
import { RealPredictionPage } from './pages/RealPredictionPage'
import { ErrorAnalysisPage } from './pages/ErrorAnalysisPage'

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Real data state from FastAPI
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixResponse | null>(null)
  const [samples, setSamples] = useState<SamplePredictionItem[]>([])

  // Async states
  const [loading, setLoading] = useState<boolean>(true)
  const [healthLoading, setHealthLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadAllData()
  }, [])

  const loadAllData = async () => {
    try {
      setLoading(true)
      setHealthLoading(true)
      setError(null)

      // 1. Fetch health first
      const healthData = await fetchHealth()
      setHealth(healthData)
      setHealthLoading(false)

      if (!healthData.model_available || !healthData.artifacts_available) {
        // Model or artifacts not ready
        setLoading(false)
        return
      }

      // 2. Fetch all model information and evaluation artifacts in parallel
      const [mInfo, dInfo, met, cm, samp] = await Promise.all([
        fetchModelInfo(),
        fetchDatasetInfo(),
        fetchMetrics(),
        fetchConfusionMatrix(),
        fetchSamples('all', undefined, undefined, undefined, 50),
      ])

      setModelInfo(mInfo)
      setDatasetInfo(dInfo)
      setMetrics(met)
      setConfusionMatrix(cm)
      setSamples(samp.samples)
    } catch (err: any) {
      setError(err.message || 'Failed to connect to FastAPI service at http://127.0.0.1:8000')
    } finally {
      setLoading(false)
      setHealthLoading(false)
    }
  }

  // Titles for active page
  const pageTitles: Record<PageId, string> = {
    overview: 'Project Overview',
    dataset: 'Dataset Explorer',
    architecture: 'CNN Architecture',
    evaluation: 'Training & Evaluation',
    predict: 'Real Prediction',
    errors: 'Error Analysis',
  }

  // Check if model or artifacts are completely missing
  const needsTraining =
    !loading &&
    health &&
    (!health.model_available || !health.artifacts_available)

  return (
    <div className="min-h-screen bg-[#FAF9F6] text-[#1C1917] flex font-sans antialiased">
      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          onOpenSidebar={() => setSidebarOpen(true)}
          health={health}
          healthLoading={healthLoading}
          pageTitle={pageTitles[activePage]}
        />

        <main className="flex-1 p-6 sm:p-12 overflow-y-auto">
          {error && <ErrorBanner message={error} onRetry={loadAllData} />}

          {loading ? (
            <LoadingSpinner message="Querying real model artifacts from FastAPI..." />
          ) : needsTraining ? (
            <EmptyState
              onRefresh={loadAllData}
              missingArtifacts={health?.missing_artifacts}
            />
          ) : (
            <>
              {activePage === 'overview' && (
                <OverviewPage
                  health={health}
                  modelInfo={modelInfo}
                  metrics={metrics}
                  datasetInfo={datasetInfo}
                  onNavigate={setActivePage}
                />
              )}
              {activePage === 'dataset' && (
                <DatasetExplorerPage datasetInfo={datasetInfo} />
              )}
              {activePage === 'architecture' && (
                <ArchitecturePage modelInfo={modelInfo} />
              )}
              {activePage === 'evaluation' && (
                <TrainingEvaluationPage
                  metrics={metrics}
                  confusionMatrix={confusionMatrix}
                />
              )}
              {activePage === 'predict' && (
                <RealPredictionPage samples={samples} />
              )}
              {activePage === 'errors' && (
                <ErrorAnalysisPage
                  initialSamples={samples}
                  confusionMatrix={confusionMatrix}
                />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}
