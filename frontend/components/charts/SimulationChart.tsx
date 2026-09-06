"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import type { SimulationResponse } from "@/lib/types";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart
} from "recharts";
import { formatTimestamp } from "@/lib/utils";

interface SimulationChartProps {
  result: SimulationResponse;
}

export function SimulationChart({ result }: SimulationChartProps) {
  // Construct chart data merging baseline and scenario trajectory
  const data = result.time_steps.map((ts, i) => {
    const base = result.baseline_census[i];
    const sim = result.simulated_census[i];
    
    // Create delta area (min to max of the two)
    const range_min = Math.min(base, sim);
    const range_max = Math.max(base, sim);
    
    return {
      time: formatTimestamp(ts),
      baseline: base,
      scenario: sim,
      range_min,
      range_max,
      isWorse: sim > base
    };
  });

  return (
    <Card className="w-full h-full p-2">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          
          <XAxis 
            dataKey="time" 
            stroke="#94a3b8" 
            tick={{ fill: '#94a3b8', fontSize: 12 }} 
            dy={10}
          />
          
          <YAxis 
            stroke="#94a3b8" 
            tick={{ fill: '#94a3b8', fontSize: 12 }} 
            label={{ value: 'Active Census', angle: -90, position: 'insideLeft', fill: '#94a3b8', dy: 50 }}
          />
          
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '0.5rem' }}
            formatter={(value: number | string | readonly (number | string)[] | undefined) => [Number(value).toFixed(1), "Patients"]}
          />
          
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          
          <Line 
            type="monotone" 
            dataKey="baseline" 
            name="Baseline Trajectory" 
            stroke="#94a3b8" 
            strokeWidth={2} 
            strokeDasharray="5 5" 
            dot={{ r: 4, fill: '#0f172a', strokeWidth: 2 }} 
          />
          
          <Line 
            type="monotone" 
            dataKey="scenario" 
            name="Simulated Scenario" 
            stroke="#3b82f6" 
            strokeWidth={3} 
            dot={{ r: 4, fill: '#0f172a', strokeWidth: 2 }} 
            activeDot={{ r: 6 }} 
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  );
}
