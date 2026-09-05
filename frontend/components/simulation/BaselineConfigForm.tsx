"use client";

import React, { useState } from "react";
import { Card } from "@/components/common/Card";
import type { SimulationRequest, SimulationScenarioType } from "@/lib/types";

interface BaselineConfigFormProps {
  scenarioType: SimulationScenarioType;
  onSubmit: (request: SimulationRequest) => void;
  isLoading?: boolean;
}

export function BaselineConfigForm({ scenarioType, onSubmit, isLoading }: BaselineConfigFormProps) {
  const [initialCensus, setInitialCensus] = useState(40);
  const [intervalsCount, setIntervalsCount] = useState(4);
  const [intervalMinutes, setIntervalMinutes] = useState(30);
  const [arrivalsStr, setArrivalsStr] = useState("5,6,5,4");
  const [departuresStr, setDeparturesStr] = useState("4,5,4,3");
  const [highAcuityRatio, setHighAcuityRatio] = useState(0.20);
  
  // Scenario params
  const [accelerationRate, setAccelerationRate] = useState(20);
  const [reducedCapacity, setReducedCapacity] = useState(30);
  const [additionalArrivals, setAdditionalArrivals] = useState(10);
  const [surgeDurationSteps, setSurgeDurationSteps] = useState(2);
  const [surgeAcuityRatio, setSurgeAcuityRatio] = useState(0.50);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Parse arrays
    const arrivals = arrivalsStr.split(",").map(s => Number(s.trim()));
    const departures = departuresStr.split(",").map(s => Number(s.trim()));
    
    // Validate arrays length
    if (arrivals.length !== departures.length) {
      alert("Arrivals and departures arrays must be the same length.");
      return;
    }
    
    // Generate time_steps (need N+1 steps for N intervals)
    const time_steps: string[] = [];
    const now = new Date();
    now.setMinutes(Math.round(now.getMinutes() / 30) * 30, 0, 0); // round to nearest 30m
    
    for (let i = 0; i <= arrivals.length; i++) {
      const stepDate = new Date(now.getTime() + i * intervalMinutes * 60000);
      time_steps.push(stepDate.toISOString());
    }

    const request: SimulationRequest = {
      scenario_type: scenarioType,
      time_steps,
      initial_census: initialCensus,
      arrivals,
      departures,
      high_acuity_ratio: highAcuityRatio,
    };

    if (scenarioType === "discharge_acceleration") {
      request.acceleration_rate = accelerationRate / 100.0;
    } else if (scenarioType === "capacity_reduction") {
      request.reduced_capacity = reducedCapacity;
    } else if (scenarioType === "arrival_surge") {
      request.additional_arrivals = additionalArrivals;
      request.surge_duration_steps = surgeDurationSteps;
      request.surge_acuity_ratio = surgeAcuityRatio;
    }

    onSubmit(request);
  };

  return (
    <Card title="Simulation Configuration" className="w-full">
      <form onSubmit={handleSubmit} className="space-y-6 mt-4">
        
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 border-b border-slate-700 pb-2">Baseline Parameters</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Initial Census</label>
              <input type="number" min="0" value={initialCensus} onChange={e => setInitialCensus(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Interval Duration (min)</label>
              <input type="number" min="15" step="15" value={intervalMinutes} onChange={e => setIntervalMinutes(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
            </div>
            <div className="sm:col-span-2">
              <label className="block text-sm text-slate-400 mb-1">Arrivals per interval (comma-separated)</label>
              <input type="text" value={arrivalsStr} onChange={e => setArrivalsStr(e.target.value)} required placeholder="e.g. 5,6,5,4" className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
            </div>
            <div className="sm:col-span-2">
              <label className="block text-sm text-slate-400 mb-1">Departures per interval (comma-separated)</label>
              <input type="text" value={departuresStr} onChange={e => setDeparturesStr(e.target.value)} required placeholder="e.g. 4,5,4,3" className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">High Acuity Ratio (0-1)</label>
              <input type="number" min="0" max="1" step="0.05" value={highAcuityRatio} onChange={e => setHighAcuityRatio(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
            </div>
          </div>
        </div>

        <div className="space-y-4 pt-2">
          <h3 className="text-sm font-semibold text-slate-300 border-b border-slate-700 pb-2">Scenario Specifics</h3>
          
          {scenarioType === "discharge_acceleration" && (
            <div>
              <label className="block text-sm text-slate-400 mb-1">Acceleration Rate: {accelerationRate}%</label>
              <input type="range" min="0" max="100" step="5" value={accelerationRate} onChange={e => setAccelerationRate(Number(e.target.value))} className="w-full" />
              <p className="text-xs text-slate-500 mt-1">Boosts baseline departures by this percentage.</p>
            </div>
          )}

          {scenarioType === "capacity_reduction" && (
            <div>
              <label className="block text-sm text-slate-400 mb-1">Reduced Bed Capacity</label>
              <input type="number" min="1" value={reducedCapacity} onChange={e => setReducedCapacity(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              <p className="text-xs text-slate-500 mt-1">Simulates closing beds (e.g. for staffing limits).</p>
            </div>
          )}

          {scenarioType === "arrival_surge" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Additional Surge Arrivals</label>
                <input type="number" min="1" value={additionalArrivals} onChange={e => setAdditionalArrivals(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Duration (intervals)</label>
                <input type="number" min="1" value={surgeDurationSteps} onChange={e => setSurgeDurationSteps(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-sm text-slate-400 mb-1">Surge Acuity Ratio (0-1)</label>
                <input type="number" min="0" max="1" step="0.05" value={surgeAcuityRatio} onChange={e => setSurgeAcuityRatio(Number(e.target.value))} required className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
            </div>
          )}
        </div>

        <button type="submit" disabled={isLoading} className="w-full py-3 mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md transition disabled:opacity-50">
          {isLoading ? "Running Simulation..." : "Run Simulation"}
        </button>
      </form>
    </Card>
  );
}
