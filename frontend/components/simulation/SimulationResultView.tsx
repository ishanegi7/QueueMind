"use client";

import React from "react";
import { ComparisonTable } from "./ComparisonTable";
import { SimulationChart } from "@/components/charts/SimulationChart";
import type { SimulationResponse } from "@/lib/types";

interface SimulationResultViewProps {
  result: SimulationResponse;
}

export function SimulationResultView({ result }: SimulationResultViewProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6 border-b border-slate-700/50 pb-4">
        {result.scenario_name}
      </h2>

      <div className="h-[400px] mb-8">
        <SimulationChart result={result} />
      </div>

      <ComparisonTable result={result} />

      {result.waiting_time_impact?.status === "unavailable" && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 flex items-start space-x-3">
          <div className="text-amber-500 shrink-0 mt-0.5">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
              <path fillRule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm8.706-1.442c1.146-.573 2.437.463 2.126 1.706l-.709 2.836.042-.02a.75.75 0 01.67 1.34l-.04.022c-1.147.573-2.438-.463-2.127-1.706l.71-2.836-.042.02a.75.75 0 11-.671-1.34l.041-.022zM12 9a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-amber-500 mb-1">Waiting Time Impact Unavailable</h4>
            <p className="text-sm text-slate-300">
              Intervention-aware waiting-time impact is not currently supported by the underlying model.
              Reason: {result.waiting_time_impact.reason}
            </p>
          </div>
        </div>
      )}

      {result.limitations && result.limitations.length > 0 && (
        <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">Simulation Limitations</h4>
          <ul className="list-disc pl-5 text-sm text-slate-400 space-y-1">
            {result.limitations.map((limitation, i) => (
              <li key={i}>{limitation}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="text-xs text-slate-500 italic text-center mt-6">
        {result.baseline_queue_health.non_clinical_disclaimer}
      </div>
    </div>
  );
}
