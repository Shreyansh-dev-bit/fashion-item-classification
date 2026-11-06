import { useState, useEffect } from 'react'
import {
  AlertOctagon,
  CheckCircle2,
  XCircle,
  Eye,
  Info,
  Lightbulb,
  TrendingDown,
  Layers,
  Sparkles,
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

const IMPROVEMENT_IDEAS = [
  {
    title: 'Data Augmentation',
    icon: Sparkles,
    color: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    description:
      'Inject random horizontal flips, minor rotations (±10°), slight width/height zoom, and subtle shifts during training to teach invariance to varied clothing poses and folds.',
  },
  {
    title: 'Batch Normalization',
    icon: Layers,
    color: 'bg-cyan-50 text-cyan-700 border-cyan-200',
    description:
      'Insert BatchNormalization layers after Conv2D operations to standardize intermediate feature map activations, smoothing the optimization landscape and speeding up convergence.',
  },
  {
    title: 'Learning-Rate Scheduling',
    icon: TrendingDown,
    color: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    description:
      'Implement ReduceLROnPlateau or Cosine Annealing schedules that decay the Adam learning rate from 1e-3 down to 1e-5 when validation loss plateaus, enabling fine-grained convergence.',
  },
  {
    title: 'Deeper Residual Architecture',
    icon: Layers,
    color: 'bg-violet-50 text-violet-700 border-violet-200',
    description:
      'Add a third Conv block (128 filters) with residual skip connections (ResNet blocks) to preserve low-level edge details while abstracting deeper garment contours.',
  },
  {
    title: 'Transfer Learning for Real Photos',
    icon: Lightbulb,
    color: 'bg-amber-50 text-amber-800 border-amber-200',
    description:
      'Fine-tune pre-trained ImageNet backbones (e.g., EfficientNetB0 or ResNet-50) on high-resolution 224×224 RGB e-commerce catalog photos to conquer real retail product variation.',
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
      // keep current samples if error
    } finally {
      setLoading(false)
    }
  }

  // Extract top off-diagonal confusion pairs directly from the real confusion matrix
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
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-100 mb-2">
          <AlertOctagon className="w-3.5 h-3.5" />
          <span>Error Analysis & Misclassification Forensics</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Model Error Analysis & Failure Modes
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Investigation of real test set misclassifications to diagnose visual confusion boundaries and plan targeted model improvements.
        </p>
      </div>

      {/* Deep-Dive: Why Upper-Body Apparel Is Hard */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-amber-50 text-amber-700 shrink-0">
            <Info className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              The Upper-Body Conundrum: T-shirt/top, Shirt, Pullover & Coat
            </h3>
            <p className="text-xs text-slate-500">
              Why these 4 garment categories account for the vast majority of classification errors
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-1">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-[11px] font-bold text-indigo-700 uppercase">1. Overlapping Outlines</span>
            <p className="text-xs text-slate-600 leading-relaxed">
              T-shirts, shirts, pullovers, and coats all feature a torso body flanked by two sleeves. From an outline silhouette standpoint, their structural geometry is nearly identical.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-[11px] font-bold text-indigo-700 uppercase">2. Similar Sleeve Shapes</span>
            <p className="text-xs text-slate-600 leading-relaxed">
              Long-sleeve shirts, coats, and pullovers share virtually identical sleeve lengths. Without color or brand insignias, sleeve silhouettes alone cannot distinguish them.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-[11px] font-bold text-indigo-700 uppercase">3. Low 28×28 Resolution</span>
            <p className="text-xs text-slate-600 leading-relaxed">
              In a 28×28 matrix, a button placket or zipper is only 1–2 pixels wide. Small structural markers that distinguish a button-down shirt from a sweater get lost in downsampling.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-[11px] font-bold text-indigo-700 uppercase">4. Limited Grayscale Texture</span>
            <p className="text-xs text-slate-600 leading-relaxed">
              Single-channel grayscale strips away yarn thickness, knit patterns, and weave textures. Heavy wool coats and cotton pullovers map to similar gray intensity histograms.
            </p>
          </div>
        </div>
      </div>

      {/* Top Real Confusions from Confusion Matrix */}
      {topConfusions.length > 0 && (
        <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Primary Confusion Pairs (Measured from Real Confusion Matrix)
              </h3>
              <p className="text-xs text-slate-500">
                Top off-diagonal misclassification counts observed in the 10,000-image test set
              </p>
            </div>
            <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
              Measured N = 10,000
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {topConfusions.map((item) => (
              <div
                key={`${item.trueClass}-${item.predClass}`}
                className="p-4 rounded-xl border border-rose-200/80 bg-rose-50/20 space-y-2"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-900">
                    True: <span className="text-indigo-900 font-extrabold">{item.trueClass}</span>
                  </span>
                  <span className="font-mono text-xs font-bold text-rose-700 bg-rose-100 px-2 py-0.5 rounded-full">
                    {item.count} errors
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-xs text-slate-600">
                  <span>Predicted as:</span>
                  <span className="font-bold text-rose-700">{item.predClass}</span>
                </div>

                <div className="text-[11px] text-slate-500 leading-snug pt-1 border-t border-rose-100">
                  {((item.count / 1000) * 100).toFixed(1)}% of all true {item.trueClass} test items were misclassified as {item.predClass}.
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sample Gallery with Multi-Axis Filters */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs space-y-6">
        <div className="flex flex-col gap-4 pb-4 border-b border-slate-100">
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Interactive Test Sample Explorer
            </h3>
            <p className="text-xs text-slate-500">
              Filter authentic test images by correctness, actual ground truth label, and predicted class
            </p>
          </div>

          {/* Filter Bar */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            {/* Correctness Toggle */}
            <div className="flex p-1 rounded-xl bg-slate-100 font-semibold">
              <button
                onClick={() => setFilterType('misclassified')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  filterType === 'misclassified'
                    ? 'bg-white text-rose-900 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Misclassified Only
              </button>
              <button
                onClick={() => setFilterType('correct')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  filterType === 'correct'
                    ? 'bg-white text-emerald-900 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Correct Only
              </button>
              <button
                onClick={() => setFilterType('all')}
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  filterType === 'all'
                    ? 'bg-white text-indigo-900 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                All Samples
              </button>
            </div>

            {/* Actual Class Dropdown */}
            <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-1 rounded-xl">
              <span className="text-slate-500 font-medium">Actual Class:</span>
              <select
                value={actualClass}
                onChange={(e) => setActualClass(e.target.value)}
                className="bg-transparent font-bold text-slate-900 focus:outline-hidden"
              >
                <option value="all">All Actual</option>
                {ALL_CLASSES.map((c) => (
                  <option key={`act-${c}`} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {/* Predicted Class Dropdown */}
            <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-1 rounded-xl">
              <span className="text-slate-500 font-medium">Predicted Class:</span>
              <select
                value={predictedClass}
                onChange={(e) => setPredictedClass(e.target.value)}
                className="bg-transparent font-bold text-slate-900 focus:outline-hidden"
              >
                <option value="all">All Predicted</option>
                {ALL_CLASSES.map((c) => (
                  <option key={`pred-${c}`} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {(actualClass !== 'all' || predictedClass !== 'all' || filterType !== 'misclassified') && (
              <button
                onClick={() => {
                  setFilterType('misclassified')
                  setActualClass('all')
                  setPredictedClass('all')
                }}
                className="text-[11px] font-semibold text-indigo-600 hover:underline px-2"
              >
                Reset Filters
              </button>
            )}
          </div>
        </div>

        {/* Gallery Grid */}
        {loading ? (
          <div className="py-20 text-center text-xs text-slate-500">
            Applying filters to authentic test predictions...
          </div>
        ) : samples.length === 0 ? (
          <div className="py-16 text-center text-slate-400">
            <Info className="w-8 h-8 mx-auto mb-2 text-slate-300" />
            <p className="text-xs font-semibold text-slate-600">
              No sample matches this combination of actual and predicted class
            </p>
            <p className="text-[11px] text-slate-400 mt-1">
              Try setting Actual Class or Predicted Class back to "All"
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
            {samples.map((item) => {
              const isCorrect = item.is_correct
              return (
                <div
                  key={`${item.test_index}-${item.true_label_id}`}
                  onClick={() => setSelectedModalSample(item)}
                  className={`p-3.5 rounded-2xl border transition-all cursor-pointer group flex flex-col justify-between ${
                    isCorrect
                      ? 'border-slate-200/90 bg-white hover:border-emerald-300 hover:shadow-sm'
                      : 'border-rose-200/90 bg-rose-50/20 hover:border-rose-400 hover:shadow-sm'
                  }`}
                >
                  <div>
                    {/* Image Box */}
                    <div className="w-full aspect-square rounded-xl bg-black border border-slate-800 flex items-center justify-center p-2 mb-2.5 overflow-hidden">
                      <img
                        src={item.image_base64}
                        alt={item.true_label_name}
                        className="w-full h-full object-contain filter group-hover:scale-105 transition-transform"
                      />
                    </div>

                    {/* Status Pill */}
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-mono text-slate-400 font-semibold">
                        Index #{item.test_index}
                      </span>
                      <span
                        className={`inline-flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          isCorrect
                            ? 'bg-emerald-50 text-emerald-700'
                            : 'bg-rose-100 text-rose-700'
                        }`}
                      >
                        {isCorrect ? (
                          <CheckCircle2 className="w-3 h-3" />
                        ) : (
                          <XCircle className="w-3 h-3" />
                        )}
                        <span>{item.confidence_percentage}%</span>
                      </span>
                    </div>

                    {/* Labels */}
                    <div className="space-y-0.5 text-left">
                      <div className="text-xs font-bold text-slate-900 truncate">
                        <span className="text-[10px] text-slate-400 font-medium">True: </span>
                        {item.true_label_name}
                      </div>

                      {!isCorrect && (
                        <div className="text-xs font-bold text-rose-600 truncate">
                          <span className="text-[10px] text-rose-400 font-medium">Pred: </span>
                          {item.predicted_label_name}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-100/80 flex items-center justify-between text-[10px] text-slate-400">
                    <span>Inspect</span>
                    <Eye className="w-3 h-3 text-slate-400 group-hover:text-indigo-600" />
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Realistic Improvement Strategies */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-indigo-50 text-indigo-600">
            <Lightbulb className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Realistic Machine Learning Improvement Roadmap
            </h3>
            <p className="text-xs text-slate-500">
              Actionable engineering strategies to boost classification performance beyond 90.7%
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-1">
          {IMPROVEMENT_IDEAS.map((idea) => {
            const Icon = idea.icon
            return (
              <div
                key={idea.title}
                className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 space-y-2"
              >
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-lg border ${idea.color}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <h4 className="text-xs font-bold text-slate-900">{idea.title}</h4>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">{idea.description}</p>
              </div>
            )
          })}
        </div>
      </div>

      {/* Modal Inspector for Individual Sample */}
      {selectedModalSample && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4"
          onClick={() => setSelectedModalSample(null)}
        >
          <div
            className="bg-white rounded-3xl p-6 sm:p-8 max-w-lg w-full border border-slate-200 shadow-2xl relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Forensic Inspector &bull; Test Set #{selectedModalSample.test_index}
                </span>
                <h3 className="text-xl font-bold text-slate-900">
                  {selectedModalSample.is_correct ? 'Correct Classification' : 'Misclassified Sample'}
                </h3>
              </div>
              <button
                onClick={() => setSelectedModalSample(null)}
                className="text-slate-400 hover:text-slate-600 text-sm font-bold p-1.5 rounded-lg hover:bg-slate-100"
              >
                ✕
              </button>
            </div>

            <div className="flex items-center gap-5 p-4 rounded-2xl bg-slate-50 border border-slate-100 mb-5">
              <div className="w-20 h-20 rounded-xl bg-black border border-slate-800 flex items-center justify-center p-1.5 shrink-0">
                <img
                  src={selectedModalSample.image_base64}
                  alt={selectedModalSample.true_label_name}
                  className="w-full h-full object-contain"
                />
              </div>
              <div className="space-y-1 text-xs">
                <div>
                  <span className="text-slate-400 font-semibold">True Ground Truth:</span>{' '}
                  <span className="font-bold text-slate-900">
                    {selectedModalSample.true_label_name} (#{selectedModalSample.true_label_id})
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 font-semibold">CNN Prediction:</span>{' '}
                  <span
                    className={`font-bold ${
                      selectedModalSample.is_correct ? 'text-emerald-700' : 'text-rose-600'
                    }`}
                  >
                    {selectedModalSample.predicted_label_name} (#{selectedModalSample.predicted_label_id})
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 font-semibold">Model Confidence:</span>{' '}
                  <span className="font-mono font-bold text-indigo-700">
                    {selectedModalSample.confidence_percentage}%
                  </span>
                </div>
              </div>
            </div>

            {/* Probability vector */}
            <div>
              <h4 className="text-xs font-bold text-slate-700 mb-2">Class Probability Distribution:</h4>
              <div className="space-y-1.5">
                {ALL_CLASSES.map((cName, idx) => {
                  const prob = selectedModalSample.probabilities[idx] || 0
                  const pct = (prob * 100).toFixed(1)
                  const isTrue = idx === selectedModalSample.true_label_id
                  const isPred = idx === selectedModalSample.predicted_label_id

                  return (
                    <div key={cName} className="flex items-center gap-2 text-xs">
                      <span className="w-24 text-[11px] font-semibold text-slate-700 truncate">
                        {cName}
                      </span>
                      <div className="flex-1 h-3 rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            isPred ? (selectedModalSample.is_correct ? 'bg-emerald-500' : 'bg-rose-500') : 'bg-indigo-300'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-12 text-right font-mono text-[10px] text-slate-500">
                        {pct}%
                      </span>
                      {isTrue && (
                        <span className="text-[10px] font-bold text-emerald-600 uppercase">True</span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
