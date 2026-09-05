'use client';

import { BottleneckIndicators } from '@/lib/types';
import { cn, formatIndicatorLabel, formatNumber } from '@/lib/utils';
import { Card } from '@/components/common/Card';

interface BottleneckPanelProps {
  data: BottleneckIndicators;
}

interface IndicatorRowProps {
  indicatorKey: string;
  triggered: boolean;
}

function IndicatorRow({ indicatorKey, triggered }: IndicatorRowProps) {
  return (
    <div className="flex items-center gap-3 py-2">
      {/* Status dot */}
      <span
        className={cn(
          'h-2.5 w-2.5 shrink-0 rounded-full',
          triggered ? 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]' : 'bg-emerald-500'
        )}
        role="img"
        aria-label={triggered ? 'Triggered' : 'OK'}
      />
      {/* Label */}
      <span
        className={cn(
          'text-sm',
          triggered ? 'font-medium text-red-300' : 'text-slate-300'
        )}
      >
        {formatIndicatorLabel(indicatorKey)}
      </span>
    </div>
  );
}

export function BottleneckPanel({ data }: BottleneckPanelProps) {
  const { indicators, severity_score, summary } = data;
  const indicatorEntries = Object.entries(indicators);

  return (
    <Card title="Bottleneck Indicators">
      <div className="space-y-2">
        {/* Severity score */}
        <div className="flex items-center justify-between pb-2">
          <span className="text-sm text-slate-400">Severity</span>
          <span
            className={cn(
              'text-lg font-bold',
              severity_score === 0
                ? 'text-emerald-400'
                : severity_score <= 2
                  ? 'text-amber-400'
                  : severity_score <= 3
                    ? 'text-orange-400'
                    : 'text-red-400'
            )}
          >
            {formatNumber(severity_score, 0)}
            <span className="text-sm font-normal text-slate-500"> / 5</span>
          </span>
        </div>

        {/* Indicator rows */}
        <div className="divide-y divide-slate-800">
          {indicatorEntries.map(([key, triggered]) => (
            <IndicatorRow
              key={key}
              indicatorKey={key}
              triggered={triggered}
            />
          ))}
        </div>

        {/* Summary */}
        {summary && (
          <p className="pt-3 text-xs text-slate-500">{summary}</p>
        )}
      </div>
    </Card>
  );
}
