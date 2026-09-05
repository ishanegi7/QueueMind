"use client";

import React from "react";
import type { SimulationScenarioType } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ScenarioSelectorProps {
  selected: SimulationScenarioType;
  onSelect: (type: SimulationScenarioType) => void;
}

export function ScenarioSelector({ selected, onSelect }: ScenarioSelectorProps) {
  const tabs: { id: SimulationScenarioType; label: string }[] = [
    { id: "discharge_acceleration", label: "Discharge Acceleration" },
    { id: "capacity_reduction", label: "Capacity Reduction" },
    { id: "arrival_surge", label: "Arrival Surge" },
  ];

  return (
    <div className="flex space-x-1 border-b border-slate-700/50 mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onSelect(tab.id)}
          className={cn(
            "px-4 py-3 text-sm font-medium transition-colors border-b-2",
            selected === tab.id
              ? "border-blue-500 text-white bg-blue-500/10"
              : "border-transparent text-slate-400 hover:text-slate-300 hover:bg-slate-800/50"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
