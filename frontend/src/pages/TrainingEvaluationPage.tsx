import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import {
  Award,
  TrendingUp,
  Activity,
  CheckCircle2,
  Table,
  HelpCircle,
} from 'lucide-react'
import { MetricsResponse, ConfusionMatrixResponse } from '../types/api'

interface TrainingEvaluationPageProps {
  metrics: MetricsResponse | null
  confusionMatrix: ConfusionMatrixResponse | null
}

export function TrainingEvaluationPage({
  metrics,
  confusionMatrix,
}: TrainingEvaluationPageProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    trueClass: string
    predClass: string
    count: number
    percentage: string
  } | null>(null)

  if (!metrics) return null

  // Format epoch curves for Recharts
  const history = metrics.training_history
  const curveData = history.epochs.map((epoch, idx) => ({
    epoch: `Epoch ${epoch}`,
    TrainAcc: Number((history.training_accuracy[idx] * 100).toFixed(2)),
    ValAcc: Number((history.validation_accuracy[idx] * 100).toFixed(2)),
    TrainLoss: Number(history.training_loss[idx].toFixed(4)),
    ValLoss: Number(history.validation_loss[idx].toFixed(4)),
  }))

  const cmData = confusionMatrix?.matrix || []
  const classes = confusionMatrix?.classes || metrics.class_metrics.map((c) => c.class_name)

  // Maximum value in confusion matrix for color scaling
  const maxCellVal = Math.max(...cmData.flat(), 1)

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-100 mb-2">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Real Evaluation &bull; Held-Out Test Set (N = 10,000)</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Training History & Real Model Evaluation
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Genuine performance benchmarks measured exclusively on the held-out Fashion-MNIST test dataset.
        </p>
      </div>

      {/* Primary Metric Scorecards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white border border-emerald-200/80 bg-emerald-50/20 shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-800">
              Test Accuracy
            </span>
            <Award className="w-5 h-5 text-emerald-600" />
          </div>
          <div className="text-3xl font-black text-emerald-950 font-mono">
            {metrics.test_accuracy_percentage}%
          </div>
          <span className="text-xs font-semibold text-emerald-700 mt-1 block">
            {metrics.total_correct.toLocaleString()} / {metrics.total_test_samples.toLocaleString()} correct
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-indigo-200/80 bg-indigo-50/20 shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-800">
              Macro F1-Score
            </span>
            <TrendingUp className="w-5 h-5 text-indigo-600" />
          </div>
          <div className="text-3xl font-black text-indigo-950 font-mono">
            {metrics.macro_f1_percentage}%
          </div>
          <span className="text-xs font-semibold text-indigo-700 mt-1 block">
            Balanced across all 10 classes
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Weighted F1-Score
            </span>
            <Activity className="w-5 h-5 text-slate-600" />
          </div>
          <div className="text-3xl font-black text-slate-900 font-mono">
            {(metrics.weighted_f1 * 100).toFixed(2)}%
          </div>
          <span className="text-xs font-semibold text-slate-500 mt-1 block">
            Support-weighted average
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Final Test Loss
            </span>
            <span className="text-xs font-mono font-bold text-slate-400">CrossEntropy</span>
          </div>
          <div className="text-3xl font-black text-slate-900 font-mono">
            {metrics.test_loss.toFixed(4)}
          </div>
          <span className="text-xs font-semibold text-slate-500 mt-1 block">
            Held-out loss convergence
          </span>
        </div>
      </div>

      {/* Real Training Curves (Recharts) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Accuracy Curve */}
        <div className="p-6 sm:p-7 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">Training vs Validation Accuracy</h3>
              <p className="text-xs text-slate-500">15-epoch convergence curve</p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-700 font-semibold">
              Final Val: {history.validation_accuracy.slice(-1)[0] ? (history.validation_accuracy.slice(-1)[0] * 100).toFixed(2) : ''}%
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#64748B' }} />
                <YAxis domain={[75, 100]} tick={{ fontSize: 10, fill: '#64748B' }} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderRadius: '12px', border: 'none', color: '#F8FAFC', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: '11px' }} />
                <Line
                  type="monotone"
                  dataKey="TrainAcc"
                  name="Training Accuracy"
                  stroke="#4F46E5"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="ValAcc"
                  name="Validation Accuracy"
                  stroke="#06B6D4"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss Curve */}
        <div className="p-6 sm:p-7 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">Training vs Validation Loss</h3>
              <p className="text-xs text-slate-500">Sparse Categorical Cross-Entropy</p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-rose-50 text-rose-700 font-semibold">
              Final Val: {history.validation_loss.slice(-1)[0]?.toFixed(4)}
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#64748B' }} />
                <YAxis tick={{ fontSize: 10, fill: '#64748B' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderRadius: '12px', border: 'none', color: '#F8FAFC', fontSize: '12px' }}
                />
                <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: '11px' }} />
                <Line
                  type="monotone"
                  dataKey="TrainLoss"
                  name="Training Loss"
                  stroke="#DC2626"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="ValLoss"
                  name="Validation Loss"
                  stroke="#F59E0B"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Ten-Class Confusion Matrix */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900">
              10-Class Confusion Matrix Heatmap (N = 10,000)
            </h3>
            <p className="text-xs text-slate-500">
              Rows represent True Classes; Columns represent CNN Predicted Classes. Hover over any cell to view exact counts.
            </p>
          </div>
          {hoveredCell ? (
            <div className="text-xs px-3 py-1.5 rounded-lg bg-indigo-900 text-white font-mono shadow-sm">
              True: <strong>{hoveredCell.trueClass}</strong> &rarr; Pred: <strong>{hoveredCell.predClass}</strong>: {hoveredCell.count} ({hoveredCell.percentage}%)
            </div>
          ) : (
            <div className="text-xs text-slate-400 flex items-center gap-1">
              <HelpCircle className="w-3.5 h-3.5" />
              <span>Hover cell for inspection</span>
            </div>
          )}
        </div>

        {/* Matrix Grid */}
        <div className="overflow-x-auto pb-2">
          <div className="inline-block min-w-[640px] w-full">
            {/* Column Headers (Predicted) */}
            <div className="flex pl-28 mb-1">
              {classes.map((cName) => (
                <div
                  key={cName}
                  className="flex-1 text-[10px] font-bold text-slate-600 truncate text-center px-0.5"
                  title={cName}
                >
                  {cName.slice(0, 5)}
                </div>
              ))}
            </div>

            {/* Rows (Actual) */}
            <div className="space-y-1">
              {cmData.map((row, rowIdx) => {
                const trueClass = classes[rowIdx] || `Class ${rowIdx}`
                return (
                  <div key={trueClass} className="flex items-center gap-1">
                    <div
                      className="w-28 text-[11px] font-semibold text-slate-700 truncate text-right pr-2"
                      title={trueClass}
                    >
                      {trueClass}
                    </div>

                    <div className="flex-1 grid grid-cols-10 gap-1">
                      {row.map((val, colIdx) => {
                        const predClass = classes[colIdx] || `Class ${colIdx}`
                        const isDiagonal = rowIdx === colIdx
                        const intensity = val / maxCellVal
                        const pct = ((val / 1000) * 100).toFixed(1)

                        let bgColor = 'rgba(241, 245, 249, 0.7)' // default low
                        let textColor = '#475569'

                        if (isDiagonal) {
                          bgColor = `rgba(79, 70, 229, ${Math.max(0.25, intensity)})`
                          textColor = intensity > 0.5 ? '#FFFFFF' : '#1E1B4B'
                        } else if (val > 0) {
                          bgColor = `rgba(239, 68, 68, ${Math.min(0.85, Math.max(0.12, val / 150))})`
                          textColor = val > 80 ? '#FFFFFF' : '#7F1D1D'
                        }

                        return (
                          <div
                            key={colIdx}
                            onMouseEnter={() =>
                              setHoveredCell({
                                trueClass,
                                predClass,
                                count: val,
                                percentage: pct,
                              })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{ backgroundColor: bgColor, color: textColor }}
                            className="h-9 rounded-md flex items-center justify-center font-mono text-[11px] font-bold cursor-pointer transition-transform hover:scale-105 shadow-2xs"
                          >
                            {val}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="flex justify-between items-center text-[11px] text-slate-400 mt-3 pt-2 border-t border-slate-100">
              <span className="font-semibold text-indigo-700">&bull; Purple Diagonal: Correct Predictions</span>
              <span className="font-semibold text-rose-600">&bull; Red Cells: Misclassifications</span>
              <span>Total Test Images: 10,000</span>
            </div>
          </div>
        </div>
      </div>

      {/* Class-wise Metrics Table */}
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-slate-200/90 shadow-2xs">
        <div className="flex items-center gap-2 mb-4">
          <Table className="w-5 h-5 text-indigo-600" />
          <h3 className="text-base font-bold text-slate-900">Class-Wise Classification Report</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider">
                <th className="pb-3">Apparel Category</th>
                <th className="pb-3 text-right">Precision</th>
                <th className="pb-3 text-right">Recall</th>
                <th className="pb-3 text-right">F1-Score</th>
                <th className="pb-3 text-right">Held-Out Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700 font-medium">
              {metrics.class_metrics.map((row) => (
                <tr key={row.class_id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-3">
                    <span className="font-bold text-slate-900">{row.class_name}</span>
                    <span className="text-[10px] text-slate-400 font-mono ml-2">#{row.class_id}</span>
                  </td>
                  <td className="py-3 text-right font-mono">
                    <span className="font-semibold text-slate-900">
                      {(row.precision * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-3 text-right font-mono">
                    <span className="font-semibold text-slate-900">
                      {(row.recall * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-3 text-right font-mono">
                    <span className={`font-bold ${row.f1_score >= 0.9 ? 'text-emerald-700' : 'text-indigo-700'}`}>
                      {(row.f1_score * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-3 text-right font-mono text-slate-500">
                    {row.support.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
