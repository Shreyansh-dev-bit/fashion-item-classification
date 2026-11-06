import { Terminal, Cpu, RefreshCw } from 'lucide-react'

interface EmptyStateProps {
  onRefresh?: () => void
  missingArtifacts?: string[]
}

export function EmptyState({ onRefresh, missingArtifacts = [] }: EmptyStateProps) {
  return (
    <div className="max-w-3xl mx-auto py-12 px-4 text-center">
      <div className="w-16 h-16 rounded-3xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto mb-6 shadow-sm">
        <Cpu className="w-8 h-8" />
      </div>

      <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-900 border border-amber-200 mb-3">
        <span>Model or Evaluation Artifacts Not Found</span>
      </div>

      <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
        Authentic Model Training Required
      </h2>
      <p className="mt-3 text-sm text-slate-600 max-w-xl mx-auto leading-relaxed">
        This project strictly requires real machine learning outputs from the Fashion-MNIST dataset. 
        No fabricated accuracy, simulated confusion matrices, or mock predictions are permitted.
      </p>

      {missingArtifacts.length > 0 && (
        <div className="mt-6 p-4 rounded-xl bg-slate-100/80 border border-slate-200 text-left text-xs max-w-md mx-auto">
          <p className="font-semibold text-slate-700 mb-1.5">Missing Required Files:</p>
          <ul className="list-disc list-inside space-y-0.5 text-slate-600 font-mono">
            {missingArtifacts.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Terminal instructions */}
      <div className="mt-8 text-left max-w-lg mx-auto rounded-2xl bg-slate-950 p-5 shadow-xl border border-slate-800 font-mono text-xs text-slate-200">
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800 text-slate-400">
          <div className="flex items-center gap-1.5">
            <Terminal className="w-4 h-4 text-indigo-400" />
            <span className="font-medium text-slate-300">Run Real ML Pipeline</span>
          </div>
          <span className="text-[10px] text-slate-500">PowerShell / Bash</span>
        </div>

        <div className="space-y-2">
          <p className="text-slate-500"># 1. Navigate to backend & activate environment</p>
          <div className="p-2 rounded bg-slate-900 text-emerald-400 select-all">
            cd backend<br />
            .\.venv\Scripts\Activate.ps1
          </div>

          <p className="text-slate-500 pt-1"># 2. Execute authentic CNN training (15 epochs)</p>
          <div className="p-2 rounded bg-slate-900 text-cyan-300 select-all">
            python train.py --epochs 15 --batch-size 64 --seed 42
          </div>

          <p className="text-slate-500 pt-1"># 3. Generate test set evaluation artifacts</p>
          <div className="p-2 rounded bg-slate-900 text-indigo-300 select-all">
            python evaluate.py
          </div>
        </div>
      </div>

      <div className="mt-8 flex justify-center gap-3">
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold transition-all shadow-md shadow-indigo-600/20 active:scale-95"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Recheck API & Artifacts</span>
          </button>
        )}
      </div>
    </div>
  )
}
