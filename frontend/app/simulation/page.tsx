"use client";

import React, { useState } from "react";
import { ScenarioSelector } from "@/components/simulation/ScenarioSelector";
import { BaselineConfigForm } from "@/components/simulation/BaselineConfigForm";
import { SimulationResultView } from "@/components/simulation/SimulationResultView";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { Disclaimer } from "@/components/common/Disclaimer";
import { simulateWhatIf, ApiConnectionError, ApiValidationError } from "@/lib/api";
import type { SimulationScenarioType, SimulationRequest, SimulationResponse } from "@/lib/types";

export default function SimulationPage() {
  const [scenarioType, setScenarioType] = useState<SimulationScenarioType>("discharge_acceleration");
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<{type: "connection"|"validation"|"generic", message: string} | null>(null);

  const handleRunSimulation = async (request: SimulationRequest) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await simulateWhatIf(request);
      setResult(res);
    } catch (err) {
      if (err instanceof ApiConnectionError) {
        setError({ type: "connection", message: err.message });
      } else if (err instanceof ApiValidationError) {
        setError({ type: "validation", message: err.message });
      } else {
        setError({ type: "generic", message: err instanceof Error ? err.message : "Unknown simulation error" });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleScenarioChange = (type: SimulationScenarioType) => {
    setScenarioType(type);
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white tracking-tight">What-If Simulation Engine</h1>
        <p className="text-slate-400 mt-1">Explore operational counterfactuals and stress-test queue stability.</p>
      </div>

      <Disclaimer compact={true} />

      <ScenarioSelector selected={scenarioType} onSelect={handleScenarioChange} />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4">
          <BaselineConfigForm 
            scenarioType={scenarioType} 
            onSubmit={handleRunSimulation} 
            isLoading={isLoading} 
          />
        </div>
        
        <div className="lg:col-span-8">
          {isLoading ? (
            <div className="h-full min-h-[500px] flex items-center justify-center border border-slate-800 rounded-xl bg-slate-900/50">
              <LoadingState message="Executing discrete-time flow simulation..." />
            </div>
          ) : error ? (
            <div className="pt-4">
              <ErrorState type={error.type} message={error.message} />
            </div>
          ) : result ? (
            <SimulationResultView result={result} />
          ) : (
            <div className="h-full min-h-[500px] flex items-center justify-center border border-dashed border-slate-700 rounded-xl bg-slate-800/20">
              <p className="text-slate-500 text-sm">Configure parameters and run a simulation to view results.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
