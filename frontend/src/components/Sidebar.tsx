import {
  Shirt,
  LayoutDashboard,
  Database,
  Layers,
  LineChart,
  Sparkles,
  AlertOctagon,
  X,
  ExternalLink,
} from 'lucide-react'

export type PageId =
  | 'overview'
  | 'dataset'
  | 'architecture'
  | 'evaluation'
  | 'predict'
  | 'errors'

interface SidebarProps {
  activePage: PageId
  onSelectPage: (page: PageId) => void
  isOpen: boolean
  onClose: () => void
}

const NAV_ITEMS: { id: PageId; label: string; icon: typeof LayoutDashboard; description: string }[] = [
  {
    id: 'overview',
    label: 'Project Overview',
    icon: LayoutDashboard,
    description: 'Retail context & objectives',
  },
  {
    id: 'dataset',
    label: 'Dataset Explorer',
    icon: Database,
    description: 'Fashion-MNIST distributions',
  },
  {
    id: 'architecture',
    label: 'CNN Architecture',
    icon: Layers,
    description: 'Network layers & hyperparams',
  },
  {
    id: 'evaluation',
    label: 'Training & Evaluation',
    icon: LineChart,
    description: 'Curves, metrics & confusion matrix',
  },
  {
    id: 'predict',
    label: 'Real Prediction',
    icon: Sparkles,
    description: 'Live CNN image classification',
  },
  {
    id: 'errors',
    label: 'Error Analysis',
    icon: AlertOctagon,
    description: 'Misclassified test patterns',
  },
]

export function Sidebar({ activePage, onSelectPage, isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200/90 flex flex-col transition-transform duration-300 ease-in-out lg:static lg:translate-x-0 ${
          isOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="h-20 px-6 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/25">
              <Shirt className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100">
                  DL-002
                </span>
                <span className="text-[10px] font-semibold text-slate-400">CV &bull; Retail</span>
              </div>
              <h1 className="text-sm font-bold text-slate-900 tracking-tight leading-snug">
                Fashion CNN AI
              </h1>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation List */}
        <nav className="flex-1 px-3 py-6 space-y-1.5 overflow-y-auto">
          <div className="px-3 pb-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Navigation
          </div>

          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const isActive = activePage === item.id
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectPage(item.id)
                  onClose()
                }}
                className={`w-full flex items-center gap-3.5 px-3.5 py-3 rounded-xl text-left transition-all ${
                  isActive
                    ? 'bg-indigo-50/90 text-indigo-950 font-semibold shadow-sm border border-indigo-100/80'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 font-medium'
                }`}
              >
                <div
                  className={`p-2 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                      : 'bg-slate-100 text-slate-500 group-hover:bg-slate-200'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className={`text-xs ${isActive ? 'font-bold text-indigo-950' : 'text-slate-800'}`}>
                    {item.label}
                  </div>
                  <div className="text-[10px] text-slate-400 truncate">{item.description}</div>
                </div>
              </button>
            )
          })}
        </nav>

        {/* Footer Info Box */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/60">
          <div className="p-3 rounded-xl bg-white border border-slate-200/80 shadow-2xs">
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
              <span>FastAPI Backend</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Real-data pipeline only (no mock results)</p>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="mt-2.5 inline-flex items-center gap-1 text-[11px] font-semibold text-indigo-600 hover:text-indigo-700"
            >
              <span>Swagger API Docs</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </aside>
    </>
  )
}
