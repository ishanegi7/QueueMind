'use client';

interface LoadingStateProps {
  message?: string;
}

export function LoadingState({
  message = 'Loading...',
}: LoadingStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <div className="flex items-center gap-1.5" role="status">
        <span
          className="h-2.5 w-2.5 rounded-full bg-sky-400 animate-pulse"
          style={{ animationDelay: '0ms' }}
        />
        <span
          className="h-2.5 w-2.5 rounded-full bg-sky-400 animate-pulse"
          style={{ animationDelay: '150ms' }}
        />
        <span
          className="h-2.5 w-2.5 rounded-full bg-sky-400 animate-pulse"
          style={{ animationDelay: '300ms' }}
        />
        <span className="sr-only">{message}</span>
      </div>
      {message && (
        <p className="text-slate-400 text-sm font-medium">{message}</p>
      )}
    </div>
  );
}
