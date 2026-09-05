"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import { formatMinutes } from "@/lib/utils";
import type { PatientFlowResponse } from "@/lib/types";

interface PatientFlowResultProps {
  result: PatientFlowResponse;
}

export function PatientFlowResult({ result }: PatientFlowResultProps) {
  return (
    <Card className="w-full h-full flex flex-col justify-center">
      <div className="text-center">
        <h3 className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-2">
          Predicted remaining ED journey time
        </h3>
        <div className="text-5xl font-bold text-white mb-4">
          {formatMinutes(result.predicted_remaining_time_minutes)}
        </div>
        
        {result.prediction_interval && (
          <div className="bg-slate-900/50 rounded-lg p-3 inline-block border border-slate-700/50">
            <p className="text-sm text-slate-300">
              Likely between <span className="font-semibold text-white">{formatMinutes(result.prediction_interval.lower_minutes)}</span> and <span className="font-semibold text-white">{formatMinutes(result.prediction_interval.upper_minutes)}</span>
            </p>
            <p className="text-xs text-slate-500 mt-1">
              ({result.prediction_interval.coverage_level * 100}% coverage)
            </p>
          </div>
        )}
      </div>

      <div className="mt-8 pt-4 border-t border-slate-800/50 flex flex-col items-center">
        <div className="flex items-center space-x-2 text-xs text-slate-500 mb-2">
          <span className="bg-slate-800 px-2 py-1 rounded text-slate-400">Model: {result.model_name} v{result.model_version}</span>
          {result.prediction_interval && (
            <span className="bg-slate-800 px-2 py-1 rounded text-slate-400">Method: {result.prediction_interval.method}</span>
          )}
        </div>
        <p className="text-xs italic text-slate-500 text-center max-w-sm">
          {result.non_causal_disclaimer}
        </p>
      </div>
    </Card>
  );
}
