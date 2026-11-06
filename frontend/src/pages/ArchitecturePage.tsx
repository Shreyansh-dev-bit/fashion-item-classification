import { Layers, Cpu, ArrowDown, ShieldCheck, Sparkles, Filter, Grid, Maximize2, GitCommit, Split } from 'lucide-react'
import { ModelInfoResponse } from '../types/api'

interface ArchitecturePageProps {
  modelInfo: ModelInfoResponse | null
}

const LAYER_EXPLANATIONS = [
  {
    step: 1,
    name: 'Input Layer',
    specs: 'Shape: (28, 28, 1)',
    icon: Grid,
    color: 'from-blue-500 to-indigo-600',
    simpleExplain: 'Accepts the 28x28 single-channel grayscale fashion image with pixel values normalized to [0, 1].',
    outputShape: '(None, 28, 28, 1)',
    params: 0,
  },
  {
    step: 2,
    name: 'Conv2D Layer 1',
    specs: '32 Filters, 3×3 Kernel, ReLU',
    icon: Filter,
    color: 'from-indigo-600 to-indigo-700',
    simpleExplain: 'Slides 32 small 3x3 filters across the garment image to detect low-level visual edges, textures, and outlines.',
    outputShape: '(None, 26, 26, 32)',
    params: 320,
  },
  {
    step: 3,
    name: 'MaxPooling2D Layer 1',
    specs: '2×2 Pool Size, Strides 2',
    icon: Maximize2,
    color: 'from-cyan-600 to-teal-600',
    simpleExplain: 'Halves spatial dimensions from 26x26 to 13x13 by picking maximum activations, adding translation invariance.',
    outputShape: '(None, 13, 13, 32)',
    params: 0,
  },
  {
    step: 4,
    name: 'Conv2D Layer 2',
    specs: '64 Filters, 3×3 Kernel, ReLU',
    icon: Filter,
    color: 'from-indigo-700 to-violet-700',
    simpleExplain: 'Extracts 64 higher-level shape patterns (sleeves, collars, zippers, shoe soles, bag straps) from earlier features.',
    outputShape: '(None, 11, 11, 64)',
    params: 18496,
  },
  {
    step: 5,
    name: 'MaxPooling2D Layer 2',
    specs: '2×2 Pool Size, Strides 2',
    icon: Maximize2,
    color: 'from-cyan-600 to-teal-600',
    simpleExplain: 'Downsamples feature maps further from 11x11 to 5x5, condensing spatial patterns into compact descriptors.',
    outputShape: '(None, 5, 5, 64)',
    params: 0,
  },
  {
    step: 6,
    name: 'Flatten Layer',
    specs: '5 × 5 × 64 = 1,600 values',
    icon: Split,
    color: 'from-slate-600 to-slate-700',
    simpleExplain: 'Unrolls 64 2D spatial feature grids of size 5x5 into a continuous 1D feature vector of length 1,600.',
    outputShape: '(None, 1600)',
    params: 0,
  },
  {
    step: 7,
    name: 'Dense Feature Layer',
    specs: '128 Units, ReLU Activation',
    icon: GitCommit,
    color: 'from-purple-600 to-pink-600',
    simpleExplain: 'Fully-connected layer that combines all 1,600 extracted feature signals to formulate abstract apparel concepts.',
    outputShape: '(None, 128)',
    params: 204928,
  },
  {
    step: 8,
    name: 'Dropout Layer',
    specs: 'Rate = 0.3 (30%)',
    icon: ShieldCheck,
    color: 'from-amber-500 to-orange-600',
    simpleExplain: 'Randomly turns off 30% of neurons during training to prevent co-adaptation and overfitting on training garments.',
    outputShape: '(None, 128)',
    params: 0,
  },
  {
    step: 9,
    name: 'Dense Softmax Output',
    specs: '10 Units, Softmax Activation',
    icon: Sparkles,
    color: 'from-emerald-600 to-teal-600',
    simpleExplain: 'Outputs a normalized probability distribution across the 10 fashion classes summing exactly to 100%.',
    outputShape: '(None, 10)',
    params: 1290,
  },
]

export function ArchitecturePage({ modelInfo }: ArchitecturePageProps) {
  const trainConfig = modelInfo?.training_configuration

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 mb-2">
          <Layers className="w-3.5 h-3.5" />
          <span>Deep Convolutional Architecture</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          CNN Network Architecture & Hyperparameters
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Layer-by-layer anatomy of the 225,034 parameter neural network engineered for Fashion-MNIST classification.
        </p>
      </div>

      {/* Hyperparameters Card */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
        <h3 className="text-base font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-600" />
          <span>Training Hyperparameters & Deterministic Settings</span>
        </h3>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Random Seed</span>
            <div className="text-base font-bold text-slate-900 font-mono mt-0.5">
              {modelInfo?.random_seed ?? 42}
            </div>
            <span className="text-[10px] text-slate-500">Python, NumPy, TF</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Epochs</span>
            <div className="text-base font-bold text-slate-900 font-mono mt-0.5">
              {trainConfig?.epochs ?? 15}
            </div>
            <span className="text-[10px] text-slate-500">Full dataset passes</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Batch Size</span>
            <div className="text-base font-bold text-slate-900 font-mono mt-0.5">
              {trainConfig?.batch_size ?? 64}
            </div>
            <span className="text-[10px] text-slate-500">Samples / gradient step</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Optimizer</span>
            <div className="text-base font-bold text-indigo-700 font-mono mt-0.5">
              {trainConfig?.optimizer ?? 'Adam'}
            </div>
            <span className="text-[10px] text-slate-500">LR = {trainConfig?.learning_rate ?? 0.001}</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Loss Function</span>
            <div className="text-xs font-bold text-slate-900 font-mono mt-1 truncate">
              Sparse CrossEntropy
            </div>
            <span className="text-[10px] text-slate-500">Integer class targets</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase">Trainable Params</span>
            <div className="text-base font-bold text-emerald-700 font-mono mt-0.5">
              {modelInfo?.total_trainable_parameters.toLocaleString() ?? '225,034'}
            </div>
            <span className="text-[10px] text-slate-500">879.04 KB weights</span>
          </div>
        </div>
      </div>

      {/* Layer By Layer Breakdown */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs space-y-6">
        <div>
          <h3 className="text-base font-bold text-slate-900">Sequential Layer Execution Flow</h3>
          <p className="text-xs text-slate-500">How raw image tensors transform into 10 class probabilities</p>
        </div>

        <div className="space-y-3">
          {LAYER_EXPLANATIONS.map((layer, idx) => {
            const Icon = layer.icon
            return (
              <div key={layer.name}>
                <div className="p-4 sm:p-5 rounded-2xl border border-slate-200/80 bg-slate-50/40 hover:bg-white hover:border-indigo-200 transition-all flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <div className={`w-10 h-10 rounded-xl bg-gradient-to-tr ${layer.color} text-white flex items-center justify-center shrink-0 shadow-sm`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-400">Step {layer.step}</span>
                        <h4 className="text-sm font-bold text-slate-900">{layer.name}</h4>
                        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-white text-indigo-700 border border-slate-200 font-semibold">
                          {layer.specs}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 mt-1 leading-relaxed max-w-2xl">
                        {layer.simpleExplain}
                      </p>
                    </div>
                  </div>

                  <div className="flex sm:flex-col items-end gap-1 sm:gap-0 shrink-0 border-t sm:border-t-0 pt-2 sm:pt-0 w-full sm:w-auto justify-between sm:justify-start">
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Tensor Output</span>
                    <span className="text-xs font-mono font-bold text-indigo-950">{layer.outputShape}</span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">{layer.params.toLocaleString()} params</span>
                  </div>
                </div>

                {idx < LAYER_EXPLANATIONS.length - 1 && (
                  <div className="flex justify-center py-1">
                    <ArrowDown className="w-4 h-4 text-slate-300" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
