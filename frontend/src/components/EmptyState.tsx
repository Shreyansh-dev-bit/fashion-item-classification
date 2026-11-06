import { Terminal, RefreshCw, AlertTriangle } from 'lucide-react'

interface EmptyStateProps {
  onRefresh?: () => void
  missingArtifacts?: string[]
}

export function EmptyState({ onRefresh, missingArtifacts = [] }: EmptyStateProps) {
  return (
    <div className="max-w-3xl mx-auto py-16 px-4">
      <div className="border border-stone-200 bg-white rounded-xl p-8 sm:p-12 shadow-xs">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-amber-700">
          <AlertTriangle className="w-4 h-4" />
          <span>Artifact Pipeline Notice</span>
        </div>

        <h2 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight mt-3">
          Trained Model & Artifacts Required
        </h2>

        <p className="mt-3 text-sm text-stone-600 leading-relaxed">
          This case study adheres strictly to genuine empirical machine learning results. 
          No synthetic predictions, simulated confusion matrices, or mock training curves are used.
          Please run the training pipeline to generate the required model and evaluation artifacts.
        </p>

        {missingArtifacts.length > 0 && (
          <div className="mt-6 p-4 rounded-lg bg-stone-50 border border-stone-200 text-xs">
            <span className="font-mono text-stone-500 uppercase tracking-wider text-[10px] block mb-2 font-semibold">
              Missing Target Artifacts ({missingArtifacts.length})
            </span>
            <div className="flex flex-wrap gap-1.5">
              {missingArtifacts.map((name) => (
                <span
                  key={name}
                  className="px-2 py-0.5 rounded bg-white border border-stone-200 font-mono text-stone-700 text-[11px]"
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Terminal execution block */}
        <div className="mt-8 rounded-lg bg-stone-900 text-stone-200 p-5 font-mono text-xs border border-stone-800">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-stone-800 text-stone-400">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-stone-400" />
              <span className="text-[11px] uppercase tracking-wider">Execute Training Pipeline</span>
            </div>
            <span className="text-[10px] text-stone-500">PowerShell / Bash</span>
          </div>

          <div className="space-y-3">
            <div>
              <p className="text-stone-500 text-[11px]"># 1. Activate backend environment</p>
              <div className="mt-1 text-emerald-400 select-all">
                cd backend ; .\.venv\Scripts\Activate.ps1
              </div>
            </div>

            <div>
              <p className="text-stone-500 text-[11px]"># 2. Train CNN on Fashion-MNIST (15 epochs, seed 42)</p>
              <div className="mt-1 text-indigo-300 select-all">
                python train.py --epochs 15 --batch-size 64 --seed 42
              </div>
            </div>

            <div>
              <p className="text-stone-500 text-[11px]"># 3. Generate held-out test evaluation artifacts</p>
              <div className="mt-1 text-stone-300 select-all">
                python evaluate.py
              </div>
            </div>
          </div>
        </div>

        {onRefresh && (
          <div className="mt-8 pt-6 border-t border-stone-200 flex justify-end">
            <button
              onClick={onRefresh}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-stone-900 hover:bg-stone-800 text-white text-xs font-medium transition-all"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Verify Artifact Presence</span>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
