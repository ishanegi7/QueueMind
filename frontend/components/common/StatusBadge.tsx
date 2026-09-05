'use client';

import { cn, getHealthStateColor, getStabilityColor } from '@/lib/utils';
import type { HealthState, StabilityState } from '@/lib/types';

interface StatusBadgeProps {
  state: HealthState | StabilityState | string;
  size?: 'sm' | 'md' | 'lg';
}

const HEALTH_STATES: ReadonlySet<string> = new Set<HealthState>([
  'HEALTHY',
  'MODERATE',
  'BUSY',
  'CRITICAL',
]);

const STABILITY_STATES: ReadonlySet<string> = new Set<StabilityState>([
  'STABLE',
  'STRAINED',
  'UNSTABLE',
]);

const sizeStyles = {
  sm: { badge: 'px-2 py-0.5 text-xs gap-1', dot: 'h-1.5 w-1.5' },
  md: { badge: 'px-2.5 py-1 text-sm gap-1.5', dot: 'h-2 w-2' },
  lg: { badge: 'px-3 py-1.5 text-base gap-2', dot: 'h-2.5 w-2.5' },
} as const;

function getColors(state: string): { bg: string; text: string; dot?: string } {
  if (HEALTH_STATES.has(state)) {
    const colors = getHealthStateColor(state as HealthState);
    return { bg: colors.bg, text: colors.text, dot: colors.dot };
  }
  if (STABILITY_STATES.has(state)) {
    return getStabilityColor(state as StabilityState);
  }
  return getStabilityColor(state);
}

export function StatusBadge({ state, size = 'md' }: StatusBadgeProps) {
  const colors = getColors(state);
  const { badge, dot } = sizeStyles[size];

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full font-medium',
        badge,
        colors.bg,
        colors.text
      )}
    >
      <span
        className={cn('rounded-full shrink-0', dot, colors.dot ?? 'bg-current')}
        aria-hidden="true"
      />
      {state}
    </span>
  );
}
