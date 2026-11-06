import { Menu, CheckCircle2, AlertTriangle, Cpu } from 'lucide-react'
import { HealthResponse } from '../types/api'

interface HeaderProps {
  onOpenSidebar: () => void
  health: HealthResponse | null
  healthLoading: boolean
  pageTitle: string
}

export function Header({ onOpenSidebar, health, healthLoading, pageTitle }: HeaderProps) {
  return (
    <header className="h-16 bg-white border-b border-slate-200/80 px-4 sm:px-8 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenSidebar}
          className="p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 lg:hidden transition-colors"
          aria-label="Open sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <h2 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">{pageTitle}</h2>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {/* Model presence indicator */}
        <div className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 border border-slate-200 text-slate-700">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          <span>CNN DL-002</span>
        </div>

        {/* Live health pill */}
        <div
          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border transition-all ${
            healthLoading
              ? 'bg-amber-50 text-amber-700 border-amber-200'
              : health?.status === 'healthy' && health?.model_available
              ? 'bg-emerald-50 text-emerald-800 border-emerald-200 shadow-2xs'
              : 'bg-rose-50 text-rose-800 border-rose-200'
          }`}
        >
          {healthLoading ? (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
              <span>Connecting...</span>
            </>
          ) : health?.status === 'healthy' && health?.model_available ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Model & API Online</span>
            </>
          ) : (
            <>
              <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
              <span>Offline / Unready</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
