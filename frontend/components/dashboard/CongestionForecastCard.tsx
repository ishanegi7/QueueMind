'use client';

import { CongestionForecastResponse, HorizonForecast } from '@/lib/types';
import { cn, getHealthStateColor, formatNumber } from '@/lib/utils';
import { Card } from '@/components/common/Card';
import { StatusBadge } from '@/components/common/StatusBadge';

interface CongestionForecastCardProps {
  data: CongestionForecastResponse;
}

interface HorizonMiniCardProps {
  label: string;
  forecast: HorizonForecast;
}

function HorizonMiniCard({ label, forecast }: HorizonMiniCardProps) {
  const interval = forecast.prediction_interval;

  return (
    <div className="flex flex-col items-center gap-1 rounded-lg bg-slate-800/60 px-4 py-3">
      <span className="text-xs font-medium text-slate-400">{label}</span>
      <span className="text-2xl font-bold text-white">
        {formatNumber(forecast.predicted_census, 0)}
      </span>
      {interval && (
        <span className="text-xs text-slate-500">
          {formatNumber(interval.lower_census, 0)} –{' '}
          {formatNumber(interval.upper_census, 0)}
        </span>
      )}
    </div>
  );
}

export function CongestionForecastCard({
  data,
}: CongestionForecastCardProps) {
  const { current_active_census, forecasts, congestion_state } = data;
  const colors = getHealthStateColor(congestion_state.state);

  return (
    <Card title="Congestion Forecast" className={cn('border', colors.border)}>
      <div className="space-y-4">
        {/* State badge */}
        <div className="flex items-center justify-between">
          <StatusBadge state={congestion_state.state} />
          <span className="text-xs text-slate-500">
            {congestion_state.description}
          </span>
        </div>

        {/* Census + Horizons grid */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {/* Current */}
          <div className="flex flex-col items-center gap-1 rounded-lg bg-slate-800/60 px-4 py-3">
            <span className="text-xs font-medium text-slate-400">Now</span>
            <span className="text-2xl font-bold text-white">
              {formatNumber(current_active_census, 0)}
            </span>
            <span className="text-xs text-slate-500">current</span>
          </div>

          {/* Horizon forecasts */}
          <HorizonMiniCard label="+30m" forecast={forecasts['30m']} />
          <HorizonMiniCard label="+60m" forecast={forecasts['60m']} />
          <HorizonMiniCard label="+120m" forecast={forecasts['120m']} />
        </div>

        {/* Disclaimer */}
        <p className="text-xs text-slate-600 italic">
          Forecasts represent model predictions, not guaranteed future occupancy.
        </p>
      </div>
    </Card>
  );
}
