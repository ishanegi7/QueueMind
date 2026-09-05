"use client";

import React, { useState, useEffect } from "react";
import { QueueHealthGauge } from "@/components/charts/QueueHealthGauge";
import { CongestionChart } from "@/components/charts/CongestionChart";
import { BottleneckPanel } from "@/components/dashboard/BottleneckPanel";
import { ShapExplanation } from "@/components/patient-flow/ShapExplanation";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { Disclaimer } from "@/components/common/Disclaimer";
import { 
  computeQueueHealth, 
  forecastCongestion, 
  predictPatientFlow,
  ApiConnectionError,
  ApiServiceError
} from "@/lib/api";
import type { 
  QueueHealthResponse, 
  CongestionForecastResponse,
  PatientFlowExplanation
} from "@/lib/types";

export default function AnalyticsPage() {
  const [queueHealth, setQueueHealth] = useState<QueueHealthResponse | null>(null);
  const [congestion, setCongestion] = useState<CongestionForecastResponse | null>(null);
  const [shapExp, setShapExp] = useState<PatientFlowExplanation | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<{type: "connection"|"model"|"generic", message: string} | null>(null);

  const fetchAnalyticsData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Fetch comprehensive analytics data from a predefined snapshot
      const qhData = await computeQueueHealth({
        active_census: 45,
        recent_arrivals_60m: 12,
        high_acuity_ratio: 0.25,
      });
      
      const congData = await forecastCongestion({
        current_active_census: 45,
        recent_arrivals_60m: 12,
        recent_departures_60m: 8,
        flow_ratio_60m: 1.5,
        high_acuity_census: 11,
        high_acuity_ratio: 0.24,
      });
      
      // Simulate a generic patient flow prediction to get a SHAP profile
      const pfData = await predictPatientFlow({
        acuity: 2,
        temperature: 98.6,
        heartrate: 90,
        resprate: 18,
        active_census: 45,
        elapsed_time_minutes: 30,
        return_explanation: true
      });

      setQueueHealth(qhData);
      setCongestion(congData);
      setShapExp(pfData.explanation);
    } catch (err) {
      if (err instanceof ApiConnectionError) {
        setError({ type: "connection", message: err.message });
      } else if (err instanceof ApiServiceError && err.statusCode === 503) {
        setError({ type: "model", message: err.message });
      } else {
        setError({ type: "generic", message: err instanceof Error ? err.message : "Unknown error" });
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyticsData();
  }, []);

  if (isLoading) {
    return <div className="pt-20"><LoadingState message="Aggregating operational analytics..." /></div>;
  }

  if (error) {
    return (
      <div className="pt-10 max-w-2xl mx-auto">
        <ErrorState type={error.type} message={error.message} onRetry={fetchAnalyticsData} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="mb-8 flex justify-between items-end border-b border-slate-700/50 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Deep Analytics View</h1>
          <p className="text-slate-400 mt-1">Current Snapshot Analytics</p>
        </div>
        <div className="bg-slate-800/80 px-3 py-1 rounded text-xs text-slate-400">
          Historical data views disabled
        </div>
      </div>
      
      <Disclaimer compact={true} />

      {queueHealth && congestion && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-4 flex flex-col gap-6">
            <QueueHealthGauge data={queueHealth} />
            <BottleneckPanel data={congestion.bottleneck_indicators} />
          </div>
          
          <div className="lg:col-span-8 flex flex-col gap-6">
            <CongestionChart data={congestion} />
            {shapExp && (
              <div className="h-[400px]">
                <ShapExplanation explanation={shapExp} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
