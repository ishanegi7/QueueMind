'use client';

import { cn } from '@/lib/utils';

type ErrorType = 'connection' | 'model' | 'validation' | 'generic';

interface ErrorStateProps {
  title?: string;
  message: string;
  type?: ErrorType;
  onRetry?: () => void;
}

interface ErrorConfig {
  defaultTitle: string;
  iconColor: string;
  borderColor: string;
  bgColor: string;
  icon: React.ReactNode;
}

const ERROR_CONFIGS: Record<ErrorType, ErrorConfig> = {
  connection: {
    defaultTitle: 'QueueMind API is unavailable',
    iconColor: 'text-red-400',
    borderColor: 'border-red-500/30',
    bgColor: 'bg-red-500/10',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
      </svg>
    ),
  },
  model: {
    defaultTitle: 'API online, model unavailable',
    iconColor: 'text-amber-400',
    borderColor: 'border-amber-500/30',
    bgColor: 'bg-amber-500/10',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
      </svg>
    ),
  },
  validation: {
    defaultTitle: 'Validation Error',
    iconColor: 'text-orange-400',
    borderColor: 'border-orange-500/30',
    bgColor: 'bg-orange-500/10',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
      </svg>
    ),
  },
  generic: {
    defaultTitle: 'Error',
    iconColor: 'text-slate-400',
    borderColor: 'border-slate-500/30',
    bgColor: 'bg-slate-500/10',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
      </svg>
    ),
  },
};

export function ErrorState({
  title,
  message,
  type = 'generic',
  onRetry,
}: ErrorStateProps) {
  const config = ERROR_CONFIGS[type];
  const displayTitle = title ?? config.defaultTitle;

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 px-6 rounded-xl border text-center gap-3',
        config.borderColor,
        config.bgColor
      )}
      role="alert"
    >
      <div className={config.iconColor}>{config.icon}</div>
      <div>
        <h4 className="text-white font-semibold text-lg">{displayTitle}</h4>
        <p className="text-slate-300 text-sm mt-1 max-w-md">{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-700 text-slate-200 hover:bg-slate-600 transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Retry
        </button>
      )}
    </div>
  );
}
