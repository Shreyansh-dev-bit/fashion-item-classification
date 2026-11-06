import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { ArrowRight, CheckCircle2 } from 'lucide-react'
import { DatasetInfoResponse } from '../types/api'

interface DatasetExplorerPageProps {
  datasetInfo: DatasetInfoResponse | null
}

export function DatasetExplorerPage({ datasetInfo }: DatasetExplorerPageProps) {
  if (!datasetInfo) return null

  const { split_sizes, per_class_distribution } = datasetInfo

  // Format data for Recharts
  const chartData = per_class_distribution.map((item) => ({
    name: item.class_name,
    Train: item.train_count,
    Validation: item.val_count,
    Test: item.test_count,
    Total: item.total_count,
  }))

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Header */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-10">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
          <span>Empirical Data Architecture &bull; Zalando Research</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-950 tracking-tight leading-tight">
          Dataset Explorer &amp; Preprocessing
        </h1>

        <p className="mt-4 text-base text-stone-600 leading-relaxed max-w-3xl font-normal">
          Fashion-MNIST consists of 70,000 standardized 28&times;28 grayscale apparel images 
          stratified across ten distinct clothing and footwear classes. 
          The dataset replaces canonical MNIST digit benchmarks with genuine apparel textures, 
          collars, and silhouettes.
        </p>
      </section>

      {/* 01 — DATASET */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Dataset &bull; Benchmark Foundation
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">70,000 Total Images</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Origin &amp; Author
            </span>
            <div className="text-lg font-bold text-stone-900 mt-1">
              Zalando Research
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              Created in 2017 to provide a harder, more relevant benchmark for image classification algorithms 
              than hand-written digits while retaining identical 28&times;28 sizing.
            </p>
          </div>

          <div className="p-6 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Class Distribution
            </span>
            <div className="text-lg font-bold text-stone-900 mt-1">
              Perfect Balance
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              Exactly 7,000 images per class across all 10 categories, eliminating majority-class bias 
              and enabling balanced macro-averaged evaluation.
            </p>
          </div>

          <div className="p-6 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Visual Characteristics
            </span>
            <div className="text-lg font-bold text-stone-900 mt-1">
              Centered Grayscale Silhouettes
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              Single-channel silhouettes centered against dark backgrounds, standardizing spatial orientation 
              for convolutional filter extraction.
            </p>
          </div>
        </div>
      </section>

      {/* 02 — PREPROCESSING */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">02</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Preprocessing &bull; Input Transformation Pipeline
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Mathematical Pipeline</span>
        </div>

        {/* Highlight Pipeline Flow */}
        <div className="p-6 rounded-lg bg-white border border-stone-200 shadow-xs">
          <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block mb-3">
            Tensor Pipeline Sequence
          </span>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs sm:text-sm font-mono text-stone-800">
            <span className="px-3 py-1.5 rounded bg-stone-100 border border-stone-200 font-semibold">
              28&times;28 Grayscale Image
            </span>
            <ArrowRight className="w-4 h-4 text-stone-400 shrink-0" />
            <span className="px-3 py-1.5 rounded bg-stone-100 border border-stone-200 font-semibold">
              Normalize Pixels to [0, 1]
            </span>
            <ArrowRight className="w-4 h-4 text-stone-400 shrink-0" />
            <span className="px-3 py-1.5 rounded bg-stone-100 border border-stone-200 font-semibold">
              Reshape to (28, 28, 1)
            </span>
            <ArrowRight className="w-4 h-4 text-stone-400 shrink-0" />
            <span className="px-3 py-1.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-900 font-bold">
              CNN Input Tensor
            </span>
          </div>
        </div>

        {/* 4 Detail Panels */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Spatial Resolution
            </span>
            <div className="text-xl font-bold text-stone-900 font-mono mt-1">
              28 &times; 28 px
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              784 scalar inputs per image. Compact spatial dimensions maintain fast training and lightweight inference latency.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Color Channels
            </span>
            <div className="text-xl font-bold text-stone-900 font-mono mt-1">
              1 Channel (Grayscale)
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              Luminance-only values focus network learning directly on structural contours, fabric edges, and geometric outlines.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Raw Dynamic Range
            </span>
            <div className="text-xl font-bold text-stone-900 font-mono mt-1">
              [0, 255] uint8
            </div>
            <p className="text-xs text-stone-500 mt-2 leading-relaxed">
              8-bit integers where 0 represents the dark background and 255 represents high-intensity garment fabric.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-indigo-200/80 bg-indigo-50/20">
            <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-700 block">
              Normalized Tensor
            </span>
            <div className="text-xl font-bold text-indigo-950 font-mono mt-1">
              [0.0, 1.0] float32
            </div>
            <p className="text-xs text-stone-600 mt-2 leading-relaxed">
              Scaled by 1/255.0 to condition numerical gradients, preventing exploding activations during backpropagation.
            </p>
          </div>
        </div>
      </section>

      {/* 03 — SPLIT */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">03</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Split &bull; Stratified Partition Breakdown
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Deterministic Seed 42</span>
        </div>

        {/* Split Numbers Strip */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Training Split
            </span>
            <div className="text-2xl sm:text-3xl font-extrabold text-stone-950 font-mono mt-1">
              {split_sizes.training_samples.toLocaleString()}
            </div>
            <span className="text-xs font-mono text-stone-500 mt-1 block">54,000 images &bull; 77.1%</span>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Validation Split
            </span>
            <div className="text-2xl sm:text-3xl font-extrabold text-stone-950 font-mono mt-1">
              {split_sizes.validation_samples.toLocaleString()}
            </div>
            <span className="text-xs font-mono text-stone-500 mt-1 block">6,000 images &bull; 8.6%</span>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Held-Out Test Split
            </span>
            <div className="text-2xl sm:text-3xl font-extrabold text-indigo-900 font-mono mt-1">
              {split_sizes.held_out_test_samples.toLocaleString()}
            </div>
            <span className="text-xs font-mono text-indigo-700 mt-1 block">10,000 images &bull; 14.3%</span>
          </div>

          <div className="p-5 rounded-lg bg-white border border-stone-200">
            <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
              Total Dataset Size
            </span>
            <div className="text-2xl sm:text-3xl font-extrabold text-stone-950 font-mono mt-1">
              {split_sizes.total_dataset_samples.toLocaleString()}
            </div>
            <span className="text-xs font-mono text-stone-500 mt-1 block">70,000 images &bull; 100%</span>
          </div>
        </div>

        {/* Stratified Distribution Chart */}
        <div className="p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-100 pb-4">
            <div>
              <h3 className="text-sm font-bold text-stone-900 tracking-tight">
                Class Distribution Across Partitions
              </h3>
              <p className="text-xs text-stone-500 mt-0.5">
                Exact counts per apparel class verified from real dataset metadata
              </p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Balanced 7,000 / Category</span>
            </div>
          </div>

          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EAE7E1" />
                <XAxis
                  dataKey="name"
                  angle={-30}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 11, fill: '#78716C' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#78716C' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1C1917',
                    borderRadius: '6px',
                    border: 'none',
                    color: '#FAF9F6',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                  cursor={{ fill: '#F5F4F0' }}
                />
                <Legend
                  verticalAlign="top"
                  height={36}
                  wrapperStyle={{ fontSize: '12px', fontFamily: 'monospace' }}
                />
                <Bar dataKey="Train" stackId="a" fill="#1C1917" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Validation" stackId="a" fill="#78716C" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Test" stackId="a" fill="#4F46E5" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  )
}
