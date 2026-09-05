"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import type { QueueHealthResponse } from "@/lib/types";
import { getHealthStateColor } from "@/lib/utils";

interface QueueHealthGaugeProps {
  data: QueueHealthResponse;
}

export function QueueHealthGauge({ data }: QueueHealthGaugeProps) {
  const colors = getHealthStateColor(data.state);
  
  // Convert Tailwind color classes to hex for SVG
  let strokeColor = "#94a3b8"; // slate-400 fallback
  if (data.state === "HEALTHY") strokeColor = "#10b981"; // emerald-500
  if (data.state === "MODERATE") strokeColor = "#f59e0b"; // amber-500
  if (data.state === "BUSY") strokeColor = "#f97316"; // orange-500
  if (data.state === "CRITICAL") strokeColor = "#ef4444"; // red-500

  // SVG Gauge calculations
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  // 75% of circle is the gauge arc (270 degrees)
  const arcLength = circumference * 0.75;
  // The gap at the bottom is 25% of the circle
  const strokeDashoffset = circumference - (data.score / 100) * arcLength;

  return (
    <Card className="flex flex-col items-center justify-center p-6 h-full">
      <div className="relative w-[220px] h-[220px] flex items-center justify-center">
        {/* Background Track Arc */}
        <svg
          className="absolute inset-0 w-full h-full transform rotate-135"
          viewBox="0 0 200 200"
        >
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="transparent"
            stroke="#1e293b" // slate-800
            strokeWidth="16"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * 0.25} // creates the 25% gap
            strokeLinecap="round"
          />
        </svg>

        {/* Foreground Score Arc */}
        <svg
          className="absolute inset-0 w-full h-full transform rotate-135 transition-all duration-1000 ease-out"
          viewBox="0 0 200 200"
        >
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="transparent"
            stroke={strokeColor}
            strokeWidth="16"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset + circumference * 0.25} // offset + the gap
            strokeLinecap="round"
          />
        </svg>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center translate-y-2">
          <span className="text-5xl font-bold text-white tracking-tighter">
            {Math.round(data.score)}
          </span>
          <span className="text-sm font-medium text-slate-400 uppercase tracking-widest mt-1">
            Score
          </span>
          <div className={`mt-3 px-3 py-1 rounded-full text-xs font-bold border ${colors.bg} ${colors.text} ${colors.border}`}>
            {data.state}
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-center">
        <h4 className="text-sm font-medium text-slate-300">Detailed Components</h4>
        <div className="mt-3 flex space-x-6">
          <div className="text-center">
            <div className="text-xs text-slate-500 uppercase">Congestion</div>
            <div className="font-semibold text-white">{data.components.congestion_pressure.toFixed(1)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 uppercase">Arrivals</div>
            <div className="font-semibold text-white">{data.components.arrival_pressure.toFixed(1)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 uppercase">Acuity</div>
            <div className="font-semibold text-white">{data.components.high_acuity_pressure.toFixed(1)}</div>
          </div>
        </div>
      </div>
    </Card>
  );
}
