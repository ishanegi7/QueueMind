'use client';

import { QueueHealthResponse } from '@/lib/types';
import { cn, getHealthStateColor, formatNumber, formatDominantFactor } from '@/lib/utils';
import { Card } from '@/components/common/Card';
import { StatusBadge } from '@/components/common/StatusBadge';

interface QueueHealthHeroProps {
  data: QueueHealthResponse;
}

const CIRCLE_RADIUS = 70;
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_RADIUS;

interface PressureBarProps {
  label: string;
  value: number;
  weight: number;
  colorClass: string;
}

function PressureBar({ label, value, weight, colorClass }: PressureBarProps) {
  const clampedValue = Math.max(0, Math.min(100, value));

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-300">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-xs">
            weight {formatNumber(weight * 100, 0)}%
          </span>
          <span className="font-medium text-white">
            {formatNumber(value, 1)}
          </span>
        </div>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-700">
        <div
          className={cn('h-full rounded-full transition-all duration-500', colorClass)}
          style={{ width: `${clampedValue}%` }}
        />
      </div>
    </div>
  );
}

export function QueueHealthHero({ data }: QueueHealthHeroProps) {
  const { score, state, components, weights, dominant_factor } = data;
  const colors = getHealthStateColor(state);

  const strokeColor: Record<string, string> = {
    HEALTHY: '#10b981',
    MODERATE: '#f59e0b',
    BUSY: '#f97316',
    CRITICAL: '#ef4444',
  };

  const barBgColor: Record<string, string> = {
    HEALTHY: 'bg-emerald-500',
    MODERATE: 'bg-amber-500',
    BUSY: 'bg-orange-500',
    CRITICAL: 'bg-red-500',
  };

  const normalizedScore = Math.max(0, Math.min(100, score));
  const strokeDashoffset =
    CIRCLE_CIRCUMFERENCE - (normalizedScore / 100) * CIRCLE_CIRCUMFERENCE;

  return (
    <Card title="Queue Health" className={cn('border', colors.border)}>
      <div className="flex flex-col items-center gap-6 lg:flex-row lg:items-start lg:gap-10">
        {/* Score Gauge */}
        <div className="flex flex-col items-center gap-3">
          <svg
            width="180"
            height="180"
            viewBox="0 0 180 180"
            className="drop-shadow-lg"
            role="img"
            aria-label={`Queue health score: ${formatNumber(score, 0)} out of 100`}
          >
            {/* Background circle */}
            <circle
              cx="90"
              cy="90"
              r={CIRCLE_RADIUS}
              fill="none"
              stroke="#334155"
              strokeWidth="10"
            />
            {/* Score arc */}
            <circle
              cx="90"
              cy="90"
              r={CIRCLE_RADIUS}
              fill="none"
              stroke={strokeColor[state] ?? '#64748b'}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={CIRCLE_CIRCUMFERENCE}
              strokeDashoffset={strokeDashoffset}
              transform="rotate(-90 90 90)"
              className="transition-all duration-700"
            />
            {/* Score text */}
            <text
              x="90"
              y="85"
              textAnchor="middle"
              dominantBaseline="central"
              className="fill-white text-4xl font-bold"
              style={{ fontSize: '2.25rem', fontWeight: 700 }}
            >
              {formatNumber(score, 0)}
            </text>
            <text
              x="90"
              y="115"
              textAnchor="middle"
              dominantBaseline="central"
              className="fill-slate-400"
              style={{ fontSize: '0.75rem' }}
            >
              / 100
            </text>
          </svg>

          <StatusBadge state={state} size="lg" />

          <p className="text-center text-sm text-slate-400">
            Primary pressure:{' '}
            <span className="font-medium text-slate-200">
              {formatDominantFactor(dominant_factor)}
            </span>
          </p>
        </div>

        {/* Pressure Bars */}
        <div className="flex-1 w-full space-y-4">
          <PressureBar
            label="Congestion Pressure"
            value={components.congestion_pressure}
            weight={weights.congestion}
            colorClass={barBgColor[state] ?? 'bg-slate-500'}
          />
          <PressureBar
            label="Arrival Pressure"
            value={components.arrival_pressure}
            weight={weights.arrivals}
            colorClass={barBgColor[state] ?? 'bg-slate-500'}
          />
          <PressureBar
            label="High-Acuity Pressure"
            value={components.high_acuity_pressure}
            weight={weights.acuity}
            colorClass={barBgColor[state] ?? 'bg-slate-500'}
          />
        </div>
      </div>
    </Card>
  );
}
