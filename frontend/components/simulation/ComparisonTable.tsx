"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import { StatusBadge } from "@/components/common/StatusBadge";
import type { SimulationResponse } from "@/lib/types";

interface ComparisonTableProps {
  result: SimulationResponse;
}

export function ComparisonTable({ result }: ComparisonTableProps) {
  const getDeltaColor = (delta: number, isHigherWorse: boolean = true) => {
    if (delta === 0) return "text-slate-400";
    if (isHigherWorse) {
      return delta > 0 ? "text-red-400" : "text-emerald-400";
    } else {
      return delta > 0 ? "text-emerald-400" : "text-red-400";
    }
  };

  const formatDelta = (delta: number) => {
    const sign = delta > 0 ? "+" : "";
    return `${sign}${delta.toFixed(1)}`;
  };

  const queueScoreDelta = result.simulated_queue_health.score - result.baseline_queue_health.score;

  return (
    <Card title="Scenario Comparison" className="w-full">
      <div className="overflow-x-auto mt-4">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-400">
              <th className="py-3 px-4 font-medium">Metric</th>
              <th className="py-3 px-4 font-medium">Baseline</th>
              <th className="py-3 px-4 font-medium text-blue-300">Scenario</th>
              <th className="py-3 px-4 font-medium">Delta</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {/* Peak Census */}
            <tr className="hover:bg-slate-800/30">
              <td className="py-3 px-4 text-slate-300 font-medium">Peak Census</td>
              <td className="py-3 px-4 text-white">{result.peak_baseline_census.toFixed(1)}</td>
              <td className="py-3 px-4 text-white font-semibold">{result.peak_simulated_census.toFixed(1)}</td>
              <td className={`py-3 px-4 font-semibold ${getDeltaColor(result.peak_delta)}`}>
                {formatDelta(result.peak_delta)}
              </td>
            </tr>
            
            {/* Final Census */}
            <tr className="hover:bg-slate-800/30">
              <td className="py-3 px-4 text-slate-300 font-medium">Final Census</td>
              <td className="py-3 px-4 text-white">{result.final_baseline_census.toFixed(1)}</td>
              <td className="py-3 px-4 text-white font-semibold">{result.final_simulated_census.toFixed(1)}</td>
              <td className={`py-3 px-4 font-semibold ${getDeltaColor(result.final_simulated_census - result.final_baseline_census)}`}>
                {formatDelta(result.final_simulated_census - result.final_baseline_census)}
              </td>
            </tr>

            {/* Queue Health Score */}
            <tr className="hover:bg-slate-800/30">
              <td className="py-3 px-4 text-slate-300 font-medium">Queue Health Score</td>
              <td className="py-3 px-4 text-white">{result.baseline_queue_health.score.toFixed(1)}</td>
              <td className="py-3 px-4 text-white font-semibold">{result.simulated_queue_health.score.toFixed(1)}</td>
              <td className={`py-3 px-4 font-semibold ${getDeltaColor(queueScoreDelta)}`}>
                {formatDelta(queueScoreDelta)}
              </td>
            </tr>

            {/* Operational State */}
            <tr className="hover:bg-slate-800/30">
              <td className="py-3 px-4 text-slate-300 font-medium">Operational State</td>
              <td className="py-3 px-4">
                <StatusBadge state={result.baseline_queue_health.state} size="sm" />
              </td>
              <td className="py-3 px-4">
                <StatusBadge state={result.simulated_queue_health.state} size="sm" />
              </td>
              <td className="py-3 px-4 text-slate-500">—</td>
            </tr>

            {/* Stability */}
            <tr className="hover:bg-slate-800/30">
              <td className="py-3 px-4 text-slate-300 font-medium">System Stability</td>
              <td className="py-3 px-4 text-slate-500">—</td>
              <td className="py-3 px-4">
                <StatusBadge state={result.stability} size="sm" />
              </td>
              <td className="py-3 px-4 text-slate-500">—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  );
}
