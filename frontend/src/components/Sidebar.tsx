import {
  Compass,
  Database,
  Layers,
  LineChart,
  Sparkles,
  AlertOctagon,
  X,
  Radio,
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

const NAV_ITEMS: {
  id: PageId
  number: string
  label: string
  icon: typeof Compass
  caption: string
}[] = [
  {
    id: 'overview',
    number: '01',
    label: 'Project Overview',
    icon: Compass,
    caption: 'Context, objectives & taxonomy',
  },
  {
    id: 'dataset',
    number: '02',
    label: 'Dataset Explorer',
    icon: Database,
    caption: 'Fashion-MNIST partitions',
  },
  {
    id: 'architecture',
    number: '03',
    label: 'CNN Architecture',
    icon: Layers,
    caption: 'Layer pipeline & hyperparameters',
  },
  {
    id: 'evaluation',
    number: '04',
    label: 'Training & Evaluation',
    icon: LineChart,
    caption: 'Real curves & confusion matrix',
  },
  {
    id: 'predict',
    number: '05',
    label: 'Real Prediction',
    icon: Sparkles,
    caption: 'Live inference & top-3 ranking',
  },
  {
    id: 'errors',
    number: '06',
    label: 'Error Analysis',
    icon: AlertOctagon,
    caption: 'Misclassifications & research',
  },
]

export function Sidebar({ activePage, onSelectPage, isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-stone-900/40 backdrop-blur-xs z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Editorial Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 bg-[#FAF9F6] border-r border-stone-200 flex flex-col transition-transform duration-200 ease-out lg:static lg:translate-x-0 ${
          isOpen ? 'translate-x-0 shadow-xl' : '-translate-x-full'
        }`}
      >
        {/* Header / Brand */}
        <div className="pt-8 pb-6 px-6 border-b border-stone-200">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest uppercase font-semibold text-stone-500">
              DL-002 &bull; Computer Vision / Retail
            </span>
            <button
              onClick={onClose}
              className="p-1 rounded text-stone-400 hover:text-stone-700 hover:bg-stone-200/50 lg:hidden transition-colors"
              aria-label="Close sidebar"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <h1 className="text-lg font-bold text-stone-900 tracking-tight mt-2">
            Fashion CNN
          </h1>
          <p className="text-xs text-stone-500 mt-0.5 font-normal">
            Machine Learning Case Study
          </p>
        </div>

        {/* Navigation List */}
        <nav className="flex-1 px-3 py-6 space-y-1 overflow-y-auto">
          <div className="px-3 pb-3 text-[10px] font-mono tracking-widest uppercase text-stone-400">
            Case Study Index
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
                className={`w-full flex items-start gap-3 px-3.5 py-3 rounded-lg text-left transition-all group ${
                  isActive
                    ? 'bg-white text-stone-950 font-medium shadow-xs border border-stone-200'
                    : 'text-stone-600 hover:bg-stone-100/70 hover:text-stone-900'
                }`}
              >
                <span
                  className={`text-[11px] font-mono pt-0.5 transition-colors ${
                    isActive ? 'text-indigo-600 font-bold' : 'text-stone-400 group-hover:text-stone-600'
                  }`}
                >
                  {item.number}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs ${isActive ? 'font-semibold text-stone-900' : 'text-stone-700'}`}>
                      {item.label}
                    </span>
                  </div>
                  <p className="text-[11px] text-stone-400 truncate mt-0.5 font-normal">
                    {item.caption}
                  </p>
                </div>

                <Icon
                  className={`w-3.5 h-3.5 mt-0.5 shrink-0 transition-colors ${
                    isActive ? 'text-indigo-600' : 'text-stone-300 group-hover:text-stone-500'
                  }`}
                />
              </button>
            )
          })}
        </nav>

        {/* Editorial Footer Strip */}
        <div className="p-4 border-t border-stone-200 bg-stone-100/40 text-[11px]">
          <div className="flex items-center justify-between text-stone-600">
            <span className="font-mono text-[10px] uppercase tracking-wider text-stone-400">Environment</span>
            <span className="inline-flex items-center gap-1.5 text-stone-700 font-medium">
              <Radio className="w-3 h-3 text-emerald-600 animate-pulse" />
              FastAPI 0.162
            </span>
          </div>
          <div className="mt-1.5 text-[10px] text-stone-400 leading-tight">
            Deterministic seed 42 &bull; 100% verified artifacts
          </div>
        </div>
      </aside>
    </>
  )
}
