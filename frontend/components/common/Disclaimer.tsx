'use client';

import { cn } from '@/lib/utils';

interface DisclaimerProps {
  text?: string;
  compact?: boolean;
}

const DEFAULT_TEXT =
  'QueueMind is an operational patient-flow intelligence prototype. It does not provide clinical advice, diagnosis, or treatment recommendations.';

export function Disclaimer({
  text = DEFAULT_TEXT,
  compact = false,
}: DisclaimerProps) {
  if (compact) {
    return (
      <p className="text-xs text-slate-500 italic">
        {text}
      </p>
    );
  }

  return (
    <div
      className={cn(
        'flex items-start gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3'
      )}
      role="note"
    >
      <svg
        className="h-5 w-5 text-amber-400 shrink-0 mt-0.5"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.5}
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"
        />
      </svg>
      <p className="text-sm text-amber-200/80 leading-relaxed">{text}</p>
    </div>
  );
}
