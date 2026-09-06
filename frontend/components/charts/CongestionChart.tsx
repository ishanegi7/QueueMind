"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import type { CongestionForecastResponse } from "@/lib/types";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import { formatTimestamp } from "@/lib/utils";

interface CongestionChartProps {
  data: CongestionForecastResponse;
}

export function CongestionChart({ data: forecastData }: CongestionChartProps) {
  // Construct chart data points (Now, +30m, +60m, +120m)
  const now = new Date();
  
  const chartData = [
    {
      timeLabel: "Now",
      actualTime: formatTimestamp(now.toISOString()),
      census: forecastData.current_active_census,
      lower: forecastData.current_active_census,
      upper: forecastData.current_active_census,
    }
  ];

  const horizons = ["30m", "60m", "120m"];
  
  horizons.forEach(h => {
    if (forecastData.forecasts[h]) {
      const f = forecastData.forecasts[h];
      const futureTime = new Date(now.getTime() + f.horizon_minutes * 60000);
      
      chartData.push({
        timeLabel: `+${h}`,
        actualTime: formatTimestamp(futureTime.toISOString()),
        census: f.predicted_census,
        lower: f.prediction_interval ? f.prediction_interval.lower_census : f.predicted_census,
        upper: f.prediction_interval ? f.prediction_interval.upper_census : f.predicted_census,
      });
    }
  });

  return (
    <Card className="w-full h-[350px] flex flex-col p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">Department Congestion Forecast</h3>
      </div>
      
      <div className="flex-1 min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            
            <XAxis 
              dataKey="timeLabel" 
              stroke="#94a3b8" 
              tick={{ fill: '#94a3b8', fontSize: 12 }} 
              dy={10}
            />
            
            <YAxis 
              stroke="#94a3b8" 
              tick={{ fill: '#94a3b8', fontSize: 12 }} 
              domain={['auto', 'auto']}
            />
            
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '0.5rem' }}
              labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
              formatter={(value: number | string | readonly (number | string)[] | undefined, name: string | number | undefined) => {
                const numericValue = Number(value);
                if (name === "census") return [numericValue.toFixed(1), "Predicted Census"];
                if (name === "range") return [value, "Range"];
                return [numericValue.toFixed(1), name];
              }}
              labelFormatter={(label, payload) => {
                if (payload && payload.length > 0) {
                  return `${label} (${payload[0].payload.actualTime})`;
                }
                return label;
              }}
            />
            
            {/* Confidence interval area */}
            <Area 
              type="monotone" 
              dataKey="upper" 
              stroke="none" 
              fill="#3b82f6" 
              fillOpacity={0.15} 
            />
            <Area 
              type="monotone" 
              dataKey="lower" 
              stroke="none" 
              fill="#0f172a" 
              fillOpacity={1} 
            />
            
            {/* Point prediction line */}
            <Line 
              type="monotone" 
              dataKey="census" 
              stroke="#3b82f6" 
              strokeWidth={3} 
              dot={{ r: 5, fill: '#0f172a', strokeWidth: 2, stroke: '#3b82f6' }} 
              activeDot={{ r: 7 }} 
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-2 text-xs text-slate-500 italic text-center">
        {forecastData.non_clinical_disclaimer}
      </div>
    </Card>
  );
}
