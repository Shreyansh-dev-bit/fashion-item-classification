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
import { Info } from 'lucide-react'
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
    epoch: `Ep ${epoch}`,
    TrainAcc: Number((history.training_accuracy[idx] * 100).toFixed(2)),
    ValAcc: Number((history.validation_accuracy[idx] * 100).toFixed(2)),
    TrainLoss: Number(history.training_loss[idx].toFixed(4)),
    ValLoss: Number(history.validation_loss[idx].toFixed(4)),
  }))

  const cmData = confusionMatrix?.matrix || []
  const classes = confusionMatrix?.classes || metrics.class_metrics.map((c) => c.class_name)

  // Max value in confusion matrix for proportional intensity
  const maxCellVal = Math.max(...cmData.flat(), 1)

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Header */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-10">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
          <span>Held-Out Test Benchmark &bull; 10,000 Unseen Samples</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-950 tracking-tight leading-tight">
          Training History &amp; Model Evaluation
        </h1>

        <p className="mt-4 text-base text-stone-600 leading-relaxed max-w-3xl font-normal">
          Empirical evaluation results produced exclusively from the held-out Fashion-MNIST test partition. 
          The convolutional network achieved strong convergence over 15 training epochs, 
          reaching over 90.7% generalization accuracy with balanced macro performance.
        </p>
      </section>

      {/* Visual Centrepiece: Test Accuracy & Macro F1-Score */}
      <section>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Test Accuracy Primary Panel */}
          <div className="p-8 rounded-lg bg-white border border-stone-200 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase tracking-wider text-stone-400 font-semibold">
                  Primary Benchmark Metric
                </span>
                <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-medium">
                  Verified Generalization
                </span>
              </div>

              <div className="mt-4">
                <div className="text-5xl sm:text-6xl font-black text-stone-950 font-mono tracking-tight">
                  {metrics.test_accuracy_percentage}%
                </div>
                <h3 className="text-base font-bold text-stone-900 mt-2">
                  Overall Test Accuracy
                </h3>
                <p className="text-xs sm:text-sm text-stone-500 mt-1 leading-relaxed">
                  Exactly {metrics.total_correct.toLocaleString()} of {metrics.total_test_samples.toLocaleString()} held-out test images were categorized correctly on first-choice prediction.
                </p>
              </div>
            </div>

            <div className="mt-8 pt-4 border-t border-stone-100 flex items-center justify-between text-xs text-stone-500 font-mono">
              <span>Error Rate: {(100 - metrics.test_accuracy_percentage).toFixed(2)}%</span>
              <span>N = 10,000 test images</span>
            </div>
          </div>

          {/* Macro F1-Score Primary Panel */}
          <div className="p-8 rounded-lg bg-white border border-indigo-200/90 bg-indigo-50/15 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase tracking-wider text-indigo-700 font-semibold">
                  Balanced Harmonic Mean
                </span>
                <span className="text-[11px] font-mono text-indigo-800 bg-indigo-100/70 px-2 py-0.5 rounded font-medium">
                  Ten-Class Balanced
                </span>
              </div>

              <div className="mt-4">
                <div className="text-5xl sm:text-6xl font-black text-indigo-950 font-mono tracking-tight">
                  {metrics.macro_f1_percentage}%
                </div>
                <h3 className="text-base font-bold text-stone-900 mt-2">
                  Macro F1-Score
                </h3>
                <p className="text-xs sm:text-sm text-stone-600 mt-1 leading-relaxed">
                  Unweighted arithmetic average of F1-scores across all ten categories, demonstrating consistent performance without sacrificing minority categories.
                </p>
              </div>
            </div>

            <div className="mt-8 pt-4 border-t border-indigo-100/80 flex items-center justify-between text-xs text-stone-500 font-mono">
              <span>Weighted F1: {(metrics.weighted_f1 * 100).toFixed(2)}%</span>
              <span>Test Loss: {metrics.test_loss.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Wide Editorial Figures: Real Accuracy & Loss Curves */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Training Dynamics &bull; Accuracy &amp; Loss Figures
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">15 Training Epochs</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Accuracy Figure */}
          <div className="p-6 sm:p-7 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-stone-900 tracking-tight">
                  Figure 1.1: Accuracy Convergence
                </h3>
                <p className="text-xs text-stone-500 mt-0.5">
                  Training vs. Validation Accuracy over 15 Epochs
                </p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-700">
                Final Val: {history.validation_accuracy.slice(-1)[0] ? (history.validation_accuracy.slice(-1)[0] * 100).toFixed(2) : ''}%
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EAE7E1" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#78716C' }} />
                  <YAxis domain={[75, 100]} tick={{ fontSize: 10, fill: '#78716C' }} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1C1917',
                      borderRadius: '6px',
                      border: 'none',
                      color: '#FAF9F6',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
                  <Line
                    type="monotone"
                    dataKey="TrainAcc"
                    name="Train Accuracy"
                    stroke="#1C1917"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ValAcc"
                    name="Validation Accuracy"
                    stroke="#4F46E5"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-stone-500 leading-relaxed font-normal pt-2 border-t border-stone-100">
              Rapid initial learning during epochs 1–5 is followed by smooth plateauing near 91% validation accuracy. 
              The dropout layer (0.3) prevents severe overfitting.
            </p>
          </div>

          {/* Loss Figure */}
          <div className="p-6 sm:p-7 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-stone-900 tracking-tight">
                  Figure 1.2: Cross-Entropy Loss
                </h3>
                <p className="text-xs text-stone-500 mt-0.5">
                  Training vs. Validation Loss over 15 Epochs
                </p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-700">
                Final Val: {history.validation_loss.slice(-1)[0] ? history.validation_loss.slice(-1)[0].toFixed(4) : ''}
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curveData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EAE7E1" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#78716C' }} />
                  <YAxis domain={[0, 0.7]} tick={{ fontSize: 10, fill: '#78716C' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1C1917',
                      borderRadius: '6px',
                      border: 'none',
                      color: '#FAF9F6',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
                  <Line
                    type="monotone"
                    dataKey="TrainLoss"
                    name="Train Loss"
                    stroke="#1C1917"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ValLoss"
                    name="Validation Loss"
                    stroke="#4F46E5"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-stone-500 leading-relaxed font-normal pt-2 border-t border-stone-100">
              Cross-entropy loss steadily falls from ~0.55 down below 0.25 on training data, 
              with validation loss stabilising around 0.25–0.26 without divergence.
            </p>
          </div>
        </div>
      </section>

      {/* Ten-Class Confusion Matrix */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">02</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Confusion Matrix &bull; Held-Out Test Evaluation
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">10,000 Predictions</span>
        </div>

        <div className="p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-100 pb-4">
            <div>
              <h3 className="text-sm font-bold text-stone-900 tracking-tight">
                Ten-Class Empirical Confusion Matrix
              </h3>
              <p className="text-xs text-stone-500 mt-0.5">
                Rows indicate true ground-truth classes; columns indicate model predictions
              </p>
            </div>

            {hoveredCell ? (
              <div className="text-xs font-mono bg-stone-100 px-3 py-1 rounded text-stone-800">
                True: <span className="font-bold">{hoveredCell.trueClass}</span> &rarr; Pred:{' '}
                <span className="font-bold">{hoveredCell.predClass}</span> ({hoveredCell.count} images &bull;{' '}
                {hoveredCell.percentage})
              </div>
            ) : (
              <div className="text-xs text-stone-400 font-mono">
                Hover any cell for class pair details
              </div>
            )}
          </div>

          <div className="overflow-x-auto pt-2">
            <div className="min-w-[620px]">
              {/* Column Headers */}
              <div className="grid grid-cols-11 gap-1 mb-1 text-center font-mono">
                <div className="text-[10px] uppercase tracking-wider text-stone-400 text-left pl-1 self-center">
                  True \ Pred
                </div>
                {classes.map((cls, i) => (
                  <div
                    key={cls}
                    className="text-[10px] font-medium text-stone-600 truncate px-1 py-1 bg-stone-50 rounded"
                    title={cls}
                  >
                    #{i}
                  </div>
                ))}
              </div>

              {/* Rows */}
              {cmData.map((row, rowIdx) => {
                const trueClass = classes[rowIdx] || `Class ${rowIdx}`
                const rowSum = row.reduce((a, b) => a + b, 0) || 1

                return (
                  <div key={rowIdx} className="grid grid-cols-11 gap-1 mb-1 text-center font-mono">
                    <div
                      className="text-[11px] text-left font-medium text-stone-700 truncate pr-2 self-center flex items-center justify-between"
                      title={trueClass}
                    >
                      <span className="truncate">{trueClass}</span>
                      <span className="text-[9px] text-stone-400">#{rowIdx}</span>
                    </div>

                    {row.map((val, colIdx) => {
                      const isDiagonal = rowIdx === colIdx
                      const predClass = classes[colIdx] || `Class ${colIdx}`
                      const ratio = val / maxCellVal
                      const percentage = `${((val / rowSum) * 100).toFixed(1)}%`

                      // Editorial subtle heatmap
                      let bgStyle = 'bg-stone-50 text-stone-300'
                      if (isDiagonal) {
                        if (ratio > 0.8) bgStyle = 'bg-indigo-700 text-white font-bold'
                        else if (ratio > 0.6) bgStyle = 'bg-indigo-600 text-white font-bold'
                        else bgStyle = 'bg-indigo-500 text-white font-semibold'
                      } else if (val > 50) {
                        bgStyle = 'bg-rose-100 text-rose-900 font-semibold'
                      } else if (val > 15) {
                        bgStyle = 'bg-rose-50 text-rose-800'
                      } else if (val > 0) {
                        bgStyle = 'bg-stone-100 text-stone-600'
                      }

                      return (
                        <div
                          key={colIdx}
                          onMouseEnter={() =>
                            setHoveredCell({
                              trueClass,
                              predClass,
                              count: val,
                              percentage,
                            })
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                          className={`h-9 flex items-center justify-center text-xs rounded transition-all cursor-crosshair ${bgStyle} hover:ring-1 hover:ring-stone-900`}
                          title={`${trueClass} -> ${predClass}: ${val} (${percentage})`}
                        >
                          {val}
                        </div>
                      )
                    })}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Class-Wise Performance Table */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">03</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Class-Wise Precision, Recall &amp; F1-Score
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Classification Report</span>
        </div>

        <div className="rounded-lg bg-white border border-stone-200 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-stone-50 border-b border-stone-200 text-stone-500 uppercase text-[10px] tracking-wider">
                <tr>
                  <th className="py-3 px-4">#</th>
                  <th className="py-3 px-4">Class Name</th>
                  <th className="py-3 px-4 text-right">Precision</th>
                  <th className="py-3 px-4 text-right">Recall</th>
                  <th className="py-3 px-4 text-right">F1-Score</th>
                  <th className="py-3 px-4 text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 text-stone-800">
                {metrics.class_metrics.map((row) => (
                  <tr key={row.class_id} className="hover:bg-stone-50/70 transition-colors">
                    <td className="py-3 px-4 text-stone-400">{row.class_id}</td>
                    <td className="py-3 px-4 font-sans font-semibold text-stone-900">
                      {row.class_name}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {(row.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right">
                      {(row.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-indigo-950">
                      {(row.f1_score * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right text-stone-500">
                      {row.support.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot className="bg-stone-50 border-t border-stone-200 font-semibold text-stone-900">
                <tr>
                  <td className="py-3 px-4" colSpan={2}>
                    Macro Average
                  </td>
                  <td className="py-3 px-4 text-right">
                    {(
                      (metrics.class_metrics.reduce((a, b) => a + b.precision, 0) /
                        metrics.class_metrics.length) *
                      100
                    ).toFixed(2)}
                    %
                  </td>
                  <td className="py-3 px-4 text-right">
                    {(
                      (metrics.class_metrics.reduce((a, b) => a + b.recall, 0) /
                        metrics.class_metrics.length) *
                      100
                    ).toFixed(2)}
                    %
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-indigo-900">
                    {metrics.macro_f1_percentage}%
                  </td>
                  <td className="py-3 px-4 text-right text-stone-500">
                    {metrics.total_test_samples.toLocaleString()}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </section>

      {/* Editorial Interpretation: Genuine Confusing Categories */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">04</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Error Forensics &bull; Upper-Body Garment Ambiguity
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Case Study Finding</span>
        </div>

        <div className="p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-indigo-600 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-base font-bold text-stone-900 tracking-tight">
                Why Shirt, T-shirt/top, Pullover, and Coat Are Confused
              </h3>
              <p className="text-xs text-stone-500 mt-0.5">
                Detailed inspection of the genuine misclassification clusters in the confusion matrix
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
            <div className="space-y-2">
              <h4 className="font-semibold text-stone-900">Overlapping Silhouette Outlines</h4>
              <p>
                At 28&times;28 pixels, an unbuttoned shirt and a casual cardigan share near-identical 
                T-shaped silhouettes with torso bodies and outward-reaching sleeves. 
                Without high-resolution button or seam details, convolutional filters perceive identical spatial activation bounds.
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-semibold text-stone-900">Lack of Texture &amp; Color Channels</h4>
              <p>
                In grayscale, subtle fabric textures (woven cotton shirt versus knitted wool pullover) 
                are compressed into identical scalar luminance values. Color signals, which strongly discriminate 
                coats from summer tops in real retail photography, are completely absent from Fashion-MNIST.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
