import { Menu } from 'lucide-react'
import { HealthResponse } from '../types/api'

interface HeaderProps {
  onOpenSidebar: () => void
  health: HealthResponse | null
  healthLoading: boolean
  pageTitle: string
}

export function Header({ onOpenSidebar, health, healthLoading, pageTitle }: HeaderProps) {
  const isHealthy = health?.status === 'healthy' && health?.model_available

  return (
    <header className="h-16 bg-[#FAF9F6]/90 backdrop-blur-md border-b border-stone-200 px-4 sm:px-10 flex items-center justify-between sticky top-0 z-30 transition-colors">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenSidebar}
          className="p-1.5 rounded-md text-stone-500 hover:text-stone-900 hover:bg-stone-200/60 lg:hidden transition-colors"
          aria-label="Open sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono text-stone-400 hidden sm:inline">DL-002</span>
          <span className="text-stone-300 hidden sm:inline">/</span>
          <h2 className="text-sm sm:text-base font-semibold text-stone-900 tracking-tight">
            {pageTitle}
          </h2>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Subtle spec tag */}
        <div className="hidden md:flex items-center gap-1.5 text-[11px] font-mono text-stone-500 border border-stone-200 bg-white/70 px-2.5 py-1 rounded-md">
          <span>Keras 3</span>
          <span className="text-stone-300">&bull;</span>
          <span>TF 2.18</span>
        </div>

        {/* Minimal Editorial Status Pill */}
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-medium bg-white border border-stone-200 text-stone-700 shadow-2xs">
          {healthLoading ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              <span className="text-stone-500">Connecting...</span>
            </>
          ) : isHealthy ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span>Model & API Online</span>
            </>
          ) : (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              <span className="text-rose-700">Service Unready</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
