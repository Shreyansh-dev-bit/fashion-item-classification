import { useState, useEffect } from 'react'
import {
  CheckCircle2,
  XCircle,
  Eye,
  Info,
  X,
  Filter,
} from 'lucide-react'
import { fetchSamples } from '../services/api'
import { ConfusionMatrixResponse, SamplePredictionItem } from '../types/api'

interface ErrorAnalysisPageProps {
  initialSamples: SamplePredictionItem[]
  confusionMatrix: ConfusionMatrixResponse | null
}

const ALL_CLASSES = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot',
]

const RESEARCH_DIRECTIONS = [
  {
    step: '01',
    title: 'Data Augmentation',
    subtitle: 'Spatial Invariance',
    description:
      'Inject random horizontal reflections, slight rotations (±10°), width/height shifts, and minor zoom transforms during training to make convolutional filters invariant to varied garment drape and folds.',
  },
  {
    step: '02',
    title: 'Batch Normalization',
    subtitle: 'Activation Conditioning',
    description:
      'Insert BatchNormalization layers immediately after Conv2D operations to standardize intermediate channel distributions, smoothing the optimization loss landscape and accelerating convergence.',
  },
  {
    step: '03',
    title: 'Learning-Rate Scheduling',
    subtitle: 'Fine-Grained Convergence',
    description:
      'Incorporate ReduceLROnPlateau or Cosine Decay schedules that attenuate the Adam learning rate from 1e-3 down to 1e-5 when validation loss plateaus, preventing oscillation around sharp minima.',
  },
  {
    step: '04',
    title: 'Deeper Residual Architecture',
    subtitle: 'Skip-Connection Feature Re-use',
    description:
      'Adopt a third convolutional stage with residual skip connections (ResNet blocks) to preserve delicate low-level collar and hem geometry through deeper layers without gradient degradation.',
  },
  {
    step: '05',
    title: 'Transfer Learning for Real Retail',
    subtitle: 'Cross-Domain Generalization',
    description:
      'Fine-tune pre-trained ImageNet backbones (e.g. EfficientNetB0) on 224×224 3-channel RGB catalog imagery to overcome the resolution and color limitations inherent to Fashion-MNIST.',
  },
]

export function ErrorAnalysisPage({ initialSamples, confusionMatrix }: ErrorAnalysisPageProps) {
  const [filterType, setFilterType] = useState<'all' | 'misclassified' | 'correct'>('misclassified')
  const [actualClass, setActualClass] = useState<string>('all')
  const [predictedClass, setPredictedClass] = useState<string>('all')
  const [samples, setSamples] = useState<SamplePredictionItem[]>(initialSamples)
  const [loading, setLoading] = useState(false)
  const [selectedModalSample, setSelectedModalSample] = useState<SamplePredictionItem | null>(null)

  useEffect(() => {
    loadFilteredSamples()
  }, [filterType, actualClass, predictedClass])

  const loadFilteredSamples = async () => {
    try {
      setLoading(true)
      const res = await fetchSamples(
        filterType,
        undefined,
        actualClass === 'all' ? undefined : actualClass,
        predictedClass === 'all' ? undefined : predictedClass,
        50
      )
      setSamples(res.samples)
    } catch {
      // keep current samples
    } finally {
      setLoading(false)
    }
  }

  // Extract top confusing class pairs from real confusion matrix
  const topConfusions = (() => {
    if (!confusionMatrix || !confusionMatrix.matrix) return []
    const cm = confusionMatrix.matrix
    const classes = confusionMatrix.classes || ALL_CLASSES
    const pairs: { trueClass: string; predClass: string; count: number; trueIdx: number; predIdx: number }[] = []

    for (let i = 0; i < cm.length; i++) {
      for (let j = 0; j < cm[i].length; j++) {
        if (i !== j && cm[i][j] > 10) {
          pairs.push({
            trueClass: classes[i],
            predClass: classes[j],
            count: cm[i][j],
            trueIdx: i,
            predIdx: j,
          })
        }
      }
    }

    return pairs.sort((a, b) => b.count - a.count).slice(0, 4)
  })()

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Header */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-10">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-600" />
          <span>Misclassification Forensics &bull; Held-Out Test Failures</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-950 tracking-tight leading-tight">
          Error Analysis &amp; Research Directions
        </h1>

        <p className="mt-4 text-base text-stone-600 leading-relaxed max-w-3xl font-normal">
          Rigorous forensic inspection of real test-set failure cases. By cross-referencing 
          the confusion matrix with individual misclassified silhouettes, we isolate the visual 
          ambiguities that limit model performance and formulate concrete improvement directions.
        </p>
      </section>

      {/* Structured "What the Model Finds Difficult" Section */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Diagnostic Findings &bull; Primary Confusion Clusters
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Confusion Matrix Analysis</span>
        </div>

        {/* Top Confusing Pairs Strip */}
        {topConfusions.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {topConfusions.map((pair, idx) => (
              <div
                key={idx}
                className="p-5 rounded-lg bg-white border border-stone-200 shadow-xs flex flex-col justify-between"
              >
                <div>
                  <span className="text-[10px] font-mono uppercase tracking-wider text-rose-700 block">
                    Confusion Cluster #{idx + 1}
                  </span>
                  <div className="mt-2 text-sm font-bold text-stone-950">
                    {pair.trueClass} &rarr; {pair.predClass}
                  </div>
                  <p className="text-xs text-stone-500 mt-1">
                    Misclassified {pair.count} times in the test set.
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-stone-100 flex items-center justify-between text-[11px] font-mono text-stone-400">
                  <span>Actual #{pair.trueIdx}</span>
                  <span>Predicted #{pair.predIdx}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Diagnostic Explanation Card */}
        <div className="p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-indigo-600 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-base font-bold text-stone-900 tracking-tight">
                Morphological Ambiguities in Upper-Body Garments
              </h3>
              <p className="text-xs text-stone-500 mt-0.5">
                Detailed breakdown of root causes driving misclassifications between Shirt, T-shirt/top, Pullover, and Coat
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2 text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
            <div className="space-y-1.5">
              <h4 className="font-semibold text-stone-900">1. Spatial Resolution Barrier</h4>
              <p>
                At 28&times;28 pixels, collar buttons, fine fabric seams, and pocket stitchings are sub-pixel artifacts. 
                A collared dress shirt collapses into the same coarse geometric envelope as a basic crewneck T-shirt.
              </p>
            </div>

            <div className="space-y-1.5">
              <h4 className="font-semibold text-stone-900">2. Torso Silhouette Overlap</h4>
              <p>
                Coats, cardigans, and pullovers share continuous torso outlines with long sleeves. 
                Unless a coat has distinctive lapels or an open zipper, its silhouette closely mimics a heavy pullover knit.
              </p>
            </div>

            <div className="space-y-1.5">
              <h4 className="font-semibold text-stone-900">3. Absence of Chromatic Cues</h4>
              <p>
                Single-channel luminance erases color and contrast boundaries. 
                In commercial retail catalogs, color is often the dominant distinguishing signal between casual tops and formal outerwear.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Editorial Gallery of Real Test Samples */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">02</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Empirical Test Sample Gallery &bull; Individual Predictions
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">
            {samples.length} Samples Displayed
          </span>
        </div>

        {/* Filter Controls Bar */}
        <div className="p-4 rounded-lg bg-white border border-stone-200 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-stone-400 mr-1 flex items-center gap-1 font-semibold">
              <Filter className="w-3 h-3" />
              <span>Filter:</span>
            </span>

            <button
              onClick={() => setFilterType('misclassified')}
              className={`px-3 py-1 rounded font-mono text-[11px] transition-colors ${
                filterType === 'misclassified'
                  ? 'bg-rose-100 text-rose-900 border border-rose-200 font-bold'
                  : 'bg-stone-50 text-stone-600 hover:bg-stone-100 border border-stone-200'
              }`}
            >
              Misclassified Only
            </button>

            <button
              onClick={() => setFilterType('correct')}
              className={`px-3 py-1 rounded font-mono text-[11px] transition-colors ${
                filterType === 'correct'
                  ? 'bg-emerald-100 text-emerald-900 border border-emerald-200 font-bold'
                  : 'bg-stone-50 text-stone-600 hover:bg-stone-100 border border-stone-200'
              }`}
            >
              Correct Only
            </button>

            <button
              onClick={() => setFilterType('all')}
              className={`px-3 py-1 rounded font-mono text-[11px] transition-colors ${
                filterType === 'all'
                  ? 'bg-stone-900 text-white font-bold'
                  : 'bg-stone-50 text-stone-600 hover:bg-stone-100 border border-stone-200'
              }`}
            >
              All Samples
            </button>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="text-stone-400">Actual:</span>
              <select
                value={actualClass}
                onChange={(e) => setActualClass(e.target.value)}
                className="px-2 py-1 rounded bg-stone-50 border border-stone-200 text-stone-800 text-xs focus:outline-hidden"
              >
                <option value="all">All Classes</option>
                {ALL_CLASSES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="text-stone-400">Predicted:</span>
              <select
                value={predictedClass}
                onChange={(e) => setPredictedClass(e.target.value)}
                className="px-2 py-1 rounded bg-stone-50 border border-stone-200 text-stone-800 text-xs focus:outline-hidden"
              >
                <option value="all">All Classes</option>
                {ALL_CLASSES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Gallery Grid */}
        {loading ? (
          <div className="py-16 text-center text-stone-400 text-xs font-mono">
            Querying filtered test set predictions...
          </div>
        ) : samples.length === 0 ? (
          <div className="py-16 text-center text-stone-400 text-xs font-mono border border-stone-200 bg-white rounded-lg">
            No test samples matching the active filter criteria.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3.5">
            {samples.map((item) => {
              const isCorrect = item.is_correct
              return (
                <div
                  key={item.test_index}
                  onClick={() => setSelectedModalSample(item)}
                  className="p-3.5 rounded-lg bg-white border border-stone-200 hover:border-stone-400 hover:shadow-xs transition-all cursor-pointer group flex flex-col justify-between"
                >
                  <div>
                    {/* Image Box */}
                    <div className="w-full aspect-square rounded bg-black flex items-center justify-center overflow-hidden mb-3 relative">
                      <img
                        src={item.image_base64}
                        alt={item.true_label_name}
                        className="w-full h-full object-contain pixelated"
                        style={{ imageRendering: 'pixelated' }}
                      />
                      <div className="absolute inset-0 bg-stone-900/0 group-hover:bg-stone-900/30 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                        <Eye className="w-5 h-5 text-white" />
                      </div>
                    </div>

                    {/* Class Labels */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className="text-stone-400">Actual:</span>
                        <span className="font-semibold text-stone-900 truncate">
                          {item.true_label_name}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className="text-stone-400">Predicted:</span>
                        <span
                          className={`font-semibold truncate ${
                            isCorrect ? 'text-stone-800' : 'text-rose-700'
                          }`}
                        >
                          {item.predicted_label_name}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Card Footer */}
                  <div className="mt-3 pt-2 border-t border-stone-100 flex items-center justify-between text-[10px] font-mono">
                    <span
                      className={`inline-flex items-center gap-1 font-semibold ${
                        isCorrect ? 'text-emerald-700' : 'text-rose-700'
                      }`}
                    >
                      {isCorrect ? (
                        <>
                          <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                          <span>Correct</span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-3 h-3 text-rose-600" />
                          <span>Error</span>
                        </>
                      )}
                    </span>
                    <span className="text-stone-500 font-medium">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* Numbered Research Directions (Improvement Recommendations) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">03</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Research Directions &bull; Recommended Model Improvements
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Five Methodological Tracks</span>
        </div>

        <div className="space-y-3">
          {RESEARCH_DIRECTIONS.map((item) => (
            <div
              key={item.step}
              className="p-5 rounded-lg bg-white border border-stone-200 hover:border-stone-300 transition-colors"
            >
              <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-4">
                <span className="text-xs font-mono font-bold text-stone-400 shrink-0">
                  {item.step}
                </span>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold text-stone-950 tracking-tight">
                      {item.title}
                    </h3>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400">
                      &bull; {item.subtitle}
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-stone-600 mt-1.5 leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Sample Detail Modal */}
      {selectedModalSample && (
        <div className="fixed inset-0 z-50 bg-stone-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl border border-stone-200 max-w-lg w-full p-6 sm:p-8 shadow-2xl space-y-6 relative">
            <button
              onClick={() => setSelectedModalSample(null)}
              className="absolute top-4 right-4 p-1.5 rounded-md text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div>
              <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-stone-400">
                <span>Test Set Sample #{selectedModalSample.test_index}</span>
                <span>&bull;</span>
                <span
                  className={
                    selectedModalSample.is_correct ? 'text-emerald-700 font-bold' : 'text-rose-700 font-bold'
                  }
                >
                  {selectedModalSample.is_correct ? 'Correctly Classified' : 'Misclassified'}
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-950 mt-1">
                Sample Forensics Inspection
              </h3>
            </div>

            <div className="flex items-center gap-6">
              <div className="w-24 h-24 rounded-lg bg-black border border-stone-300 flex items-center justify-center shrink-0 overflow-hidden">
                <img
                  src={selectedModalSample.image_base64}
                  alt={selectedModalSample.true_label_name}
                  className="w-full h-full object-contain pixelated"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>

              <div className="space-y-1.5 text-xs font-mono">
                <div>
                  <span className="text-stone-400 block text-[10px] uppercase">True Ground Truth</span>
                  <span className="text-stone-950 font-bold text-sm">
                    {selectedModalSample.true_label_name} (#{selectedModalSample.true_label_id})
                  </span>
                </div>
                <div>
                  <span className="text-stone-400 block text-[10px] uppercase">Model Prediction</span>
                  <span
                    className={`font-bold text-sm ${
                      selectedModalSample.is_correct ? 'text-emerald-700' : 'text-rose-700'
                    }`}
                  >
                    {selectedModalSample.predicted_label_name} (#{selectedModalSample.predicted_label_id})
                  </span>
                </div>
              </div>
            </div>

            {/* Probability Distribution Breakdown */}
            <div className="space-y-2 pt-2 border-t border-stone-100">
              <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                Full 10-Class Posterior Probabilities
              </span>

              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {selectedModalSample.probabilities.map((prob: number, idx: number) => {
                  const isPredicted = idx === selectedModalSample.predicted_label_id
                  const isTrue = idx === selectedModalSample.true_label_id
                  const percentage = (prob * 100).toFixed(1)

                  return (
                    <div key={idx} className="space-y-0.5 font-mono text-[11px]">
                      <div className="flex justify-between items-center text-stone-700">
                        <span className="flex items-center gap-1.5">
                          <span className="text-stone-400">#{idx}</span>
                          <span
                            className={
                              isTrue
                                ? 'font-bold text-stone-950'
                                : isPredicted
                                ? 'text-rose-700 font-semibold'
                                : 'text-stone-600'
                            }
                          >
                            {ALL_CLASSES[idx]}
                          </span>
                          {isTrue && (
                            <span className="text-[9px] text-emerald-700 bg-emerald-50 px-1 rounded">
                              Truth
                            </span>
                          )}
                          {isPredicted && !isTrue && (
                            <span className="text-[9px] text-rose-700 bg-rose-50 px-1 rounded">
                              Pred
                            </span>
                          )}
                        </span>
                        <span className="font-semibold text-stone-900">{percentage}%</span>
                      </div>

                      <div className="h-1.5 rounded-full bg-stone-100 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isTrue
                              ? 'bg-indigo-600'
                              : isPredicted
                              ? 'bg-rose-500'
                              : 'bg-stone-300'
                          }`}
                          style={{ width: `${Math.max(Number(percentage), 1)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="pt-2 border-t border-stone-100 flex justify-end">
              <button
                onClick={() => setSelectedModalSample(null)}
                className="px-4 py-2 rounded-md bg-stone-900 text-white text-xs font-medium hover:bg-stone-800 transition-colors"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
