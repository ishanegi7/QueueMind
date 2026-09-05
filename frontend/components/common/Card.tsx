'use client';

import { cn } from '@/lib/utils';

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, subtitle, children, className }: CardProps) {
  return (
    <div
      className={cn(
        'bg-slate-800/50 border border-slate-700/50 rounded-xl p-6',
        className
      )}
    >
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h3 className="text-white font-semibold text-lg">{title}</h3>
          )}
          {subtitle && (
            <p className="text-slate-400 text-sm mt-0.5">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
