import { ShieldCheck } from 'lucide-react'
import { ModelInfoResponse } from '../types/api'

interface ArchitecturePageProps {
  modelInfo: ModelInfoResponse | null
}

interface LayerSpec {
  step: number
  title: string
  subspecs: string
  outputShape: string
  params: number
  role: string
  mathExplain: string
}

const PIPELINE_LAYERS: LayerSpec[] = [
  {
    step: 1,
    title: 'Input',
    subspecs: '28×28×1 Grayscale Image',
    outputShape: '(None, 28, 28, 1)',
    params: 0,
    role: 'Sensor Ingestion',
    mathExplain:
      'Receives the single-channel 28×28 image with pixel intensities normalized to [0.0, 1.0]. Zero learnable parameters.',
  },
  {
    step: 2,
    title: 'Conv2D',
    subspecs: '32 Filters, 3×3, ReLU',
    outputShape: '(None, 26, 26, 32)',
    params: 320,
    role: 'Low-Level Feature Extraction',
    mathExplain:
      'Applies 32 parallel 3×3 convolution kernels across the input grid to detect elementary horizontal/vertical edges, fabric folds, and hem contours.',
  },
  {
    step: 3,
    title: 'MaxPooling',
    subspecs: '2×2 Pool Size, Strides 2',
    outputShape: '(None, 13, 13, 32)',
    params: 0,
    role: 'Spatial Downsampling',
    mathExplain:
      'Extracts the peak activation within each non-overlapping 2×2 window, halving spatial dimensions from 26×26 to 13×13 and imparting minor translation invariance.',
  },
  {
    step: 4,
    title: 'Conv2D',
    subspecs: '64 Filters, 3×3, ReLU',
    outputShape: '(None, 11, 11, 64)',
    params: 18496,
    role: 'Mid-Level Part Abstraction',
    mathExplain:
      'Combines 32 low-level feature channels into 64 higher-level composite patterns (collars, sleeves, shoe soles, handbag straps, zipper lines).',
  },
  {
    step: 5,
    title: 'MaxPooling',
    subspecs: '2×2 Pool Size, Strides 2',
    outputShape: '(None, 5, 5, 64)',
    params: 0,
    role: 'Spatial Condensation',
    mathExplain:
      'Downsamples feature grids from 11×11 to 5×5, isolating dominant activation peaks while suppressing redundant background noise.',
  },
  {
    step: 6,
    title: 'Flatten',
    subspecs: '5 × 5 × 64 = 1,600 Values',
    outputShape: '(None, 1600)',
    params: 0,
    role: 'Tensor Reshaping',
    mathExplain:
      'Unrolls the 3D feature tensor into a 1D vector of 1,600 activations, bridging convolutional feature maps to dense classifier neurons.',
  },
  {
    step: 7,
    title: 'Dense',
    subspecs: '128 Units, ReLU',
    outputShape: '(None, 128)',
    params: 204928,
    role: 'Latent Representation',
    mathExplain:
      'Fully-connected neural projection containing 91% of total network parameters (1,600 × 128 weights + 128 biases), synthesizing extracted features into semantic apparel embeddings.',
  },
  {
    step: 8,
    title: 'Dropout',
    subspecs: '0.3 Rate (30% Random Inactivation)',
    outputShape: '(None, 128)',
    params: 0,
    role: 'Stochastic Regularization',
    mathExplain:
      'Randomly zeroes 30% of dense activations during each training step to prevent co-adaptation of features and mitigate overfitting on training garments.',
  },
  {
    step: 9,
    title: 'Softmax',
    subspecs: '10 Units, Normalized Probabilities',
    outputShape: '(None, 10)',
    params: 1290,
    role: 'Class Posterior Distribution',
    mathExplain:
      'Projects 128 latent dimensions into 10 logits normalized via the exponential Softmax operator, generating valid probabilities that sum strictly to 1.0.',
  },
]

export function ArchitecturePage({ modelInfo }: ArchitecturePageProps) {
  const trainConfig = modelInfo?.training_configuration

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Header */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-10">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
          <span>Convolutional Neural Network &bull; Deep Learning Blueprint</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-950 tracking-tight leading-tight">
          CNN Architecture &amp; Hyperparameters
        </h1>

        <p className="mt-4 text-base text-stone-600 leading-relaxed max-w-3xl font-normal">
          The classification model employs a sequential feed-forward convolutional architecture 
          tailored for 28&times;28 grayscale input tensors. 
          Dual convolutional-pooling blocks capture hierarchical apparel contours before a dense 
          representation layer routes activations into a 10-way softmax distribution.
        </p>
      </section>

      {/* Real Training Configuration (Editorial Spec Panel) */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Training Configuration &bull; Deterministic Parameters
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Fixed Hyperparameters</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Random Seed
            </span>
            <div className="text-xl font-bold text-stone-950 font-mono mt-1">
              {modelInfo?.random_seed ?? 42}
            </div>
            <p className="text-[11px] text-stone-500 mt-1">Python, NumPy, TF</p>
          </div>

          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Optimizer
            </span>
            <div className="text-xl font-bold text-stone-950 font-mono mt-1">
              Adam
            </div>
            <p className="text-[11px] text-stone-500 mt-1">&eta; = 0.001</p>
          </div>

          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Loss Function
            </span>
            <div className="text-lg font-bold text-stone-950 font-mono mt-1 truncate">
              Sparse CrossEntropy
            </div>
            <p className="text-[11px] text-stone-500 mt-1">Integer targets</p>
          </div>

          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Batch Size
            </span>
            <div className="text-xl font-bold text-stone-950 font-mono mt-1">
              {trainConfig?.batch_size ?? 64}
            </div>
            <p className="text-[11px] text-stone-500 mt-1">Samples / step</p>
          </div>

          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Training Epochs
            </span>
            <div className="text-xl font-bold text-stone-950 font-mono mt-1">
              {trainConfig?.epochs ?? 15}
            </div>
            <p className="text-[11px] text-stone-500 mt-1">12,660 gradient steps</p>
          </div>

          <div className="p-4 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Trainable Params
            </span>
            <div className="text-xl font-bold text-indigo-900 font-mono mt-1">
              {modelInfo?.total_trainable_parameters.toLocaleString() ?? '225,034'}
            </div>
            <p className="text-[11px] text-stone-500 mt-1">100% learnable</p>
          </div>
        </div>
      </section>

      {/* Vertical, Numbered Model Pipeline (1 to 9) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">02</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Sequential Model Pipeline &bull; Nine Layer Stages
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Tensor Transformation</span>
        </div>

        <div className="space-y-3">
          {PIPELINE_LAYERS.map((layer) => (
            <div
              key={layer.step}
              className="p-5 sm:p-6 rounded-lg bg-white border border-stone-200 hover:border-stone-300 transition-colors"
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <span className="text-xs font-mono font-bold text-stone-400 pt-0.5 shrink-0">
                    {String(layer.step).padStart(2, '0')}
                  </span>

                  <div>
                    <div className="flex flex-wrap items-center gap-2.5">
                      <h3 className="text-base font-bold text-stone-950 tracking-tight">
                        {layer.title}
                      </h3>
                      <span className="text-xs font-mono text-stone-600 bg-stone-100 px-2 py-0.5 rounded border border-stone-200">
                        {layer.subspecs}
                      </span>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                        {layer.role}
                      </span>
                    </div>

                    <p className="text-xs sm:text-sm text-stone-600 mt-2 leading-relaxed max-w-3xl">
                      {layer.mathExplain}
                    </p>
                  </div>
                </div>

                <div className="sm:text-right shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-stone-100 flex sm:flex-col justify-between items-center sm:items-end">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                      Output Shape
                    </span>
                    <span className="text-xs font-mono font-semibold text-stone-800">
                      {layer.outputShape}
                    </span>
                  </div>

                  <div className="sm:mt-2">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                      Parameters
                    </span>
                    <span className="text-xs font-mono font-semibold text-stone-950">
                      {layer.params.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Reproducibility Guarantee Footnote */}
      <section className="p-6 rounded-lg bg-white border border-stone-200 flex items-center justify-between text-xs text-stone-600">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>
            Model serialized in Keras v3 format (`fashion_mnist_cnn.keras`). 
            Trained weights and optimizer state are 100% reproducible via seed 42.
          </span>
        </div>
        <div className="font-mono text-[11px] text-stone-400 hidden sm:block">
          DL-002 Specification
        </div>
      </section>
    </div>
  )
}
