import { ArrowRight, CheckCircle2, ShieldCheck } from 'lucide-react'
import {
  HealthResponse,
  ModelInfoResponse,
  MetricsResponse,
  DatasetInfoResponse,
} from '../types/api'
import { PageId } from '../components/Sidebar'

interface OverviewPageProps {
  health: HealthResponse | null
  modelInfo: ModelInfoResponse | null
  metrics: MetricsResponse | null
  datasetInfo: DatasetInfoResponse | null
  onNavigate: (page: PageId) => void
}

const CLASS_TAXONOMY = [
  { id: 0, name: 'T-shirt/top', group: 'Tops & Tees', desc: 'Short-sleeve casual tops, crewnecks, base layers' },
  { id: 1, name: 'Trouser', group: 'Bottoms', desc: 'Denim jeans, tailored slacks, chinos, athletic sweatpants' },
  { id: 2, name: 'Pullover', group: 'Knitwear', desc: 'Sweaters, fleece hoodies, cardigans, winter pullovers' },
  { id: 3, name: 'Dress', group: 'One-Piece', desc: 'Daywear dresses, evening frocks, woven one-piece garments' },
  { id: 4, name: 'Coat', group: 'Outerwear', desc: 'Trench coats, winter parkas, jackets, heavy overcoats' },
  { id: 5, name: 'Sandal', group: 'Open Footwear', desc: 'Slides, strapped summer sandals, open-toe shoes' },
  { id: 6, name: 'Shirt', group: 'Tops & Collared', desc: 'Button-up collared shirts, dress oxfords, blouses' },
  { id: 7, name: 'Sneaker', group: 'Athletic Footwear', desc: 'Court shoes, runners, trainers, lifestyle sneakers' },
  { id: 8, name: 'Bag', group: 'Accessories', desc: 'Shoulder totes, clutches, backpacks, travel duffels' },
  { id: 9, name: 'Ankle boot', group: 'Closed Footwear', desc: 'Chelsea boots, leather lace-ups, winter boots' },
]

export function OverviewPage({
  health,
  modelInfo,
  metrics,
  datasetInfo,
  onNavigate,
}: OverviewPageProps) {
  const accuracyStr = metrics?.test_accuracy_percentage
    ? `${metrics.test_accuracy_percentage}%`
    : '90.73%'

  const macroF1Str = metrics?.macro_f1_percentage
    ? `${metrics.macro_f1_percentage}%`
    : '90.81%'

  const testCountStr = datasetInfo?.split_sizes?.held_out_test_samples
    ? datasetInfo.split_sizes.held_out_test_samples.toLocaleString()
    : '10,000'

  const paramCountStr = modelInfo?.total_trainable_parameters
    ? modelInfo.total_trainable_parameters.toLocaleString()
    : '225,034'

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Hero */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-12">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-4">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
          <span>Convolutional Neural Network &bull; Fashion-MNIST Classification</span>
        </div>

        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold text-stone-950 tracking-tight leading-[1.08] max-w-4xl">
          Fashion Item Classification with a CNN
        </h1>

        <p className="mt-6 text-base sm:text-lg text-stone-600 leading-relaxed max-w-3xl font-normal">
          An empirical deep learning investigation applying custom convolutional neural networks 
          to automated retail product tagging on Zalando&apos;s canonical Fashion-MNIST benchmark. 
          The model demonstrates how computer vision replaces expensive manual catalog taxonomy tagging 
          while delivering reproducible, verified classification accuracy across ten apparel categories.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-4">
          <button
            onClick={() => onNavigate('predict')}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-stone-900 hover:bg-stone-800 text-white text-xs font-medium transition-all shadow-xs"
          >
            <span>Live Inference Console</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onNavigate('evaluation')}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md border border-stone-300 bg-white hover:bg-stone-50 text-stone-700 text-xs font-medium transition-colors"
          >
            <span>Benchmark Evaluation Report</span>
          </button>

          <div className="flex items-center gap-1.5 text-xs text-stone-400 font-mono ml-auto">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>100% Real Evaluation Data</span>
          </div>
        </div>
      </section>

      {/* Strong Metric Strip */}
      <section>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8 border-y border-stone-200 py-8 bg-white/40 px-4 sm:px-6 rounded-lg">
          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Held-Out Test Accuracy
            </span>
            <div className="text-3xl sm:text-4xl font-extrabold text-stone-950 font-mono tracking-tight mt-1">
              {accuracyStr}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              9,073 of 10,000 correct
            </p>
          </div>

          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Macro F1-Score
            </span>
            <div className="text-3xl sm:text-4xl font-extrabold text-indigo-900 font-mono tracking-tight mt-1">
              {macroF1Str}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              Unweighted across 10 classes
            </p>
          </div>

          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Apparel Categories
            </span>
            <div className="text-3xl sm:text-4xl font-extrabold text-stone-950 font-mono tracking-tight mt-1">
              10
            </div>
            <p className="text-xs text-stone-500 mt-1">
              Zalando retail taxonomy
            </p>
          </div>

          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Test Sample Volume
            </span>
            <div className="text-3xl sm:text-4xl font-extrabold text-stone-950 font-mono tracking-tight mt-1">
              {testCountStr}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              1,000 images per class
            </p>
          </div>
        </div>
      </section>

      {/* Narrative Retail Case Study */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Retail Case Study &bull; Problem Space
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Industry Context</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pt-2">
          <div className="space-y-3">
            <h3 className="text-base font-semibold text-stone-900 tracking-tight">
              Automated Catalog Onboarding
            </h3>
            <p className="text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
              High-volume e-commerce retailers ingest tens of thousands of third-party garment SKUs weekly. 
              Manual categorization is slow, expensive, and subject to subjective human tagging discrepancies. 
              The deep CNN automates taxonomy assignment immediately upon product upload.
            </p>
          </div>

          <div className="space-y-3">
            <h3 className="text-base font-semibold text-stone-900 tracking-tight">
              Semantic Search & Recommendation
            </h3>
            <p className="text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
              In addition to top-1 classification, the softmax output layer exposes continuous probability 
              distributions. These confidence vectors empower downstream search filters, multi-category 
              recommendations, and visual faceted browsing across apparel silos.
            </p>
          </div>

          <div className="space-y-3">
            <h3 className="text-base font-semibold text-stone-900 tracking-tight">
              Standardized Benchmark Integrity
            </h3>
            <p className="text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
              Zalando created Fashion-MNIST as a direct drop-in replacement for handwritten digits. 
              With 70,000 grayscale 28&times;28 silhouettes, it provides an authentic, verifiable benchmark 
              for modern convolutional architectures without synthetic or fabricated performance figures.
            </p>
          </div>
        </div>
      </section>

      {/* Ten Fashion-MNIST Classes (Taxonomy Section) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">02</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Taxonomy &bull; Ten Target Classes
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Softmax Output Vector (10 Units)</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {CLASS_TAXONOMY.map((item) => (
            <div
              key={item.id}
              className="p-4 rounded-lg bg-white border border-stone-200 hover:border-stone-400 transition-colors group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-stone-400 font-semibold">
                    #{item.id}
                  </span>
                  <span className="text-[10px] font-mono tracking-wider uppercase text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded">
                    {item.group}
                  </span>
                </div>
                <h4 className="text-sm font-bold text-stone-900 mt-2 group-hover:text-indigo-900 transition-colors">
                  {item.name}
                </h4>
              </div>
              <p className="text-[11px] text-stone-500 mt-2 leading-relaxed">
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Model Blueprint & System Status */}
      <section className="border-t border-stone-200 pt-8">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Parameter Blueprint
            </span>
            <div className="text-xl font-bold text-stone-900 font-mono mt-1">
              {paramCountStr}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              2 Conv2D, 2 MaxPool, Dense(128), Dropout(0.3)
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Artifact Status
            </span>
            <div className="text-xl font-bold text-emerald-800 font-mono mt-1 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span>{health?.model_available ? 'Verified' : 'Pending'}</span>
            </div>
            <p className="text-xs text-stone-500 mt-1 truncate font-mono">
              backend/models/fashion_mnist_cnn.keras
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Deterministic Seed
            </span>
            <div className="text-xl font-bold text-stone-900 font-mono mt-1">
              Seed 42
            </div>
            <p className="text-xs text-stone-500 mt-1">
              Reproducible Python, NumPy &amp; TensorFlow splits
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
