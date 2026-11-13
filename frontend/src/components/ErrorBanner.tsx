import { AlertCircle, RefreshCw } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onRetry?: () => void
  isReconnecting?: boolean
}

export function ErrorBanner({ message, onRetry, isReconnecting }: ErrorBannerProps) {
  return (
    <div className="rounded-lg border border-rose-200 bg-white p-5 text-stone-900 shadow-xs my-6">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-rose-600 mt-0.5 shrink-0" />
          <div>
            <h4 className="text-xs font-mono uppercase tracking-wider text-rose-800 font-semibold">
              API Connection Interruption
            </h4>
            <p className="text-xs text-stone-600 mt-1 leading-relaxed max-w-2xl break-words">
              {message}
            </p>
          </div>
        </div>

        {onRetry && (
          <button
            onClick={onRetry}
            disabled={isReconnecting}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-stone-200 bg-stone-50 hover:bg-stone-100 disabled:opacity-60 disabled:cursor-not-allowed text-stone-700 text-xs font-medium transition-colors shrink-0"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isReconnecting ? 'animate-spin text-stone-500' : ''}`} />
            <span>{isReconnecting ? 'Connecting...' : 'Reconnect'}</span>
          </button>
        )}
      </div>
    </div>
  )
}
