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
import { Database, Sliders, CheckCircle2 } from 'lucide-react'
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
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Page Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 mb-2">
          <Database className="w-3.5 h-3.5" />
          <span>Real Dataset Partitions &bull; Zalando Research</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Dataset Explorer & Preprocessing Pipeline
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Inspection of the genuine 70,000 Fashion-MNIST images across stratified train, validation, and test splits.
        </p>
      </div>

      {/* Split Size Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Total Dataset</span>
          <div className="text-2xl font-black text-slate-900 mt-1">
            {split_sizes.total_dataset_samples.toLocaleString()}
          </div>
          <span className="text-xs font-semibold text-slate-500">70,000 images (100%)</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-indigo-100 bg-indigo-50/20 shadow-2xs">
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-600">Training Split</span>
          <div className="text-2xl font-black text-indigo-950 mt-1">
            {split_sizes.training_samples.toLocaleString()}
          </div>
          <span className="text-xs font-semibold text-indigo-700">54,000 images (77.1%)</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-cyan-100 bg-cyan-50/20 shadow-2xs">
          <span className="text-xs font-bold uppercase tracking-wider text-cyan-600">Validation Split</span>
          <div className="text-2xl font-black text-cyan-950 mt-1">
            {split_sizes.validation_samples.toLocaleString()}
          </div>
          <span className="text-xs font-semibold text-cyan-700">6,000 images (8.6%)</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-violet-100 bg-violet-50/20 shadow-2xs">
          <span className="text-xs font-bold uppercase tracking-wider text-violet-600">Held-Out Test Split</span>
          <div className="text-2xl font-black text-violet-950 mt-1">
            {split_sizes.held_out_test_samples.toLocaleString()}
          </div>
          <span className="text-xs font-semibold text-violet-700">10,000 images (14.3%)</span>
        </div>
      </div>

      {/* Preprocessing Specifications Box */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2.5 rounded-xl bg-indigo-50 text-indigo-600">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">Image Input & Normalization Pipeline</h3>
            <p className="text-xs text-slate-500">Deterministic mathematical transformations applied prior to CNN ingestion</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <div className="text-xs font-bold text-slate-400 uppercase">Input Resolution</div>
            <div className="text-base font-bold text-slate-900 font-mono">28 &times; 28 pixels</div>
            <p className="text-xs text-slate-500">784 total individual pixel sensors per apparel item.</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <div className="text-xs font-bold text-slate-400 uppercase">Color Representation</div>
            <div className="text-base font-bold text-slate-900 font-mono">Grayscale (1 Channel)</div>
            <p className="text-xs text-slate-500">Single luminance channel encoding garment contours.</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <div className="text-xs font-bold text-slate-400 uppercase">Raw Dynamic Range</div>
            <div className="text-base font-bold text-slate-900 font-mono">0 &ndash; 255 (uint8)</div>
            <p className="text-xs text-slate-500">8-bit integer grayscale intensities (0 = black, 255 = white).</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1.5">
            <div className="text-xs font-bold text-indigo-600 uppercase">Normalized CNN Tensor</div>
            <div className="text-base font-bold text-indigo-950 font-mono">[0.0, 1.0] (float32)</div>
            <p className="text-xs text-slate-500 font-mono text-[11px]">x / 255.0 &rarr; shape (28, 28, 1)</p>
          </div>
        </div>
      </div>

      {/* Class Distribution Chart */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-6">
          <div>
            <h3 className="text-base font-bold text-slate-900">Stratified Class Distribution Breakdown</h3>
            <p className="text-xs text-slate-500">Exact real counts per apparel class across train, val, and test splits</p>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-100">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Perfect Class Balance (7,000/class)</span>
          </div>
        </div>

        <div className="h-80 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
              <XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} tick={{ fontSize: 11, fill: '#475569' }} />
              <YAxis tick={{ fontSize: 11, fill: '#475569' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0F172A', borderRadius: '12px', border: 'none', color: '#F8FAFC', fontSize: '12px' }}
                cursor={{ fill: '#F1F5F9' }}
              />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
              <Bar dataKey="Train" stackId="a" fill="#4F46E5" radius={[0, 0, 0, 0]} />
              <Bar dataKey="Validation" stackId="a" fill="#06B6D4" radius={[0, 0, 0, 0]} />
              <Bar dataKey="Test" stackId="a" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Class Distribution Table */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs overflow-x-auto">
        <h3 className="text-base font-bold text-slate-900 mb-4">Sample Count Verification Table</h3>
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider">
              <th className="pb-3">Class ID</th>
              <th className="pb-3">Label Name</th>
              <th className="pb-3 text-right">Train (54k)</th>
              <th className="pb-3 text-right">Validation (6k)</th>
              <th className="pb-3 text-right">Held-Out Test (10k)</th>
              <th className="pb-3 text-right">Total Real Samples</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700 font-medium">
            {per_class_distribution.map((row) => (
              <tr key={row.class_id} className="hover:bg-slate-50/80 transition-colors">
                <td className="py-2.5 font-mono text-slate-400">#{row.class_id}</td>
                <td className="py-2.5 font-bold text-slate-900">{row.class_name}</td>
                <td className="py-2.5 text-right font-mono">{row.train_count.toLocaleString()}</td>
                <td className="py-2.5 text-right font-mono">{row.val_count.toLocaleString()}</td>
                <td className="py-2.5 text-right font-mono text-indigo-600 font-semibold">{row.test_count.toLocaleString()}</td>
                <td className="py-2.5 text-right font-mono font-bold text-slate-900">{row.total_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
