import { Loader2 } from 'lucide-react'

interface LoadingSpinnerProps {
  message?: string
}

export function LoadingSpinner({ message = 'Loading case study data...' }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center py-28 px-4 text-center">
      <div className="w-10 h-10 rounded-full border border-stone-200 bg-white flex items-center justify-center text-stone-700 mb-4 shadow-xs">
        <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
      </div>
      <p className="text-sm font-medium text-stone-900 tracking-tight">{message}</p>
      <p className="text-xs text-stone-400 font-mono mt-1">Reading authentic evaluation artifacts from FastAPI</p>
    </div>
  )
}
