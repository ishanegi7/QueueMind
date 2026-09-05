'use client';

import { formatNumber } from '@/lib/utils';
import { Card } from '@/components/common/Card';

interface FlowKPICardsProps {
  activeCensus: number;
  recentArrivals: number;
  recentDepartures: number;
  highAcuityRatio: number;
}

interface KPIItemProps {
  label: string;
  value: string;
  unit: string;
}

function KPIItem({ label, value, unit }: KPIItemProps) {
  return (
    <Card>
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
        <span className="text-3xl font-bold text-white">{value}</span>
        <span className="text-xs text-slate-500">{unit}</span>
      </div>
    </Card>
  );
}

export function FlowKPICards({
  activeCensus,
  recentArrivals,
  recentDepartures,
  highAcuityRatio,
}: FlowKPICardsProps) {
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <KPIItem
        label="Active Census"
        value={formatNumber(activeCensus, 0)}
        unit="patients"
      />
      <KPIItem
        label="Arrivals (60m)"
        value={formatNumber(recentArrivals, 0)}
        unit="patients"
      />
      <KPIItem
        label="Departures (60m)"
        value={formatNumber(recentDepartures, 0)}
        unit="patients"
      />
      <KPIItem
        label="High Acuity"
        value={formatNumber(highAcuityRatio * 100, 1)}
        unit="% of active patients"
      />
    </div>
  );
}
