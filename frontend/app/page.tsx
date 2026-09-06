"use client";

import React, { useState, useEffect } from "react";
import { QueueHealthHero } from "@/components/dashboard/QueueHealthHero";
import { FlowKPICards } from "@/components/dashboard/FlowKPICards";
import { CongestionForecastCard } from "@/components/dashboard/CongestionForecastCard";
import { PatientFlowForm } from "@/components/patient-flow/PatientFlowForm";
import { PatientFlowResult } from "@/components/patient-flow/PatientFlowResult";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { Disclaimer } from "@/components/common/Disclaimer";
import { 
  computeQueueHealth, 
  forecastCongestion, 
  predictPatientFlow,
  ApiConnectionError,
  ApiServiceError,
  ApiValidationError
} from "@/lib/api";
import type { 
  QueueHealthResponse, 
  CongestionForecastResponse,
  PatientFlowRequest,
  PatientFlowResponse
} from "@/lib/types";

const DEMO_SNAPSHOT = {
  active_census: 45,
  recent_arrivals_60m: 12,
  high_acuity_ratio: 0.25,
};

const DEMO_CONGESTION_SNAPSHOT = {
  current_active_census: 45,
  recent_arrivals_60m: 12,
  recent_departures_60m: 8,
  flow_ratio_60m: 1.5,
  high_acuity_census: 11,
  high_acuity_ratio: 0.24,
  hour_sin: 0,
  hour_cos: 1,
  dayofweek: 3,
  is_weekend: 0
};

export default function DashboardPage() {
  const [queueHealth, setQueueHealth] = useState<QueueHealthResponse | null>(null);
  const [congestion, setCongestion] = useState<CongestionForecastResponse | null>(null);
  const [patientFlowResult, setPatientFlowResult] = useState<PatientFlowResponse | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<{type: "connection"|"model"|"generic", message: string} | null>(null);
  const [pfError, setPfError] = useState<{type: "validation"|"model"|"generic", message: string} | null>(null);

  useEffect(() => {
    let mounted = true;

    const doFetch = async () => {
      try {
        const [qhData, congData] = await Promise.all([
          computeQueueHealth(DEMO_SNAPSHOT),
          forecastCongestion(DEMO_CONGESTION_SNAPSHOT)
        ]);

        if (mounted) {
          setQueueHealth(qhData);
          setCongestion(congData);
          setError(null);
        }
      } catch (err) {
        if (!mounted) return;
        if (err instanceof ApiConnectionError) {
          setError({ type: "connection", message: err.message });
        } else if (err instanceof ApiServiceError && err.statusCode === 503) {
          setError({ type: "model", message: err.message });
        } else {
          setError({ type: "generic", message: err instanceof Error ? err.message : "Unknown error" });
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    };

    doFetch();

    return () => {
      mounted = false;
    };
  }, []);

  const handleRetry = () => {
    setIsLoading(true);
    setError(null);
    // Extract a fetcher for retry if needed, but since we can't easily share it 
    // without triggering ESLint, we duplicate the minimal fetch logic for retry here.
    const doFetch = async () => {
      try {
        const [qhData, congData] = await Promise.all([
          computeQueueHealth(DEMO_SNAPSHOT),
          forecastCongestion(DEMO_CONGESTION_SNAPSHOT)
        ]);
        setQueueHealth(qhData);
        setCongestion(congData);
        setError(null);
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
    doFetch();
  };

  const handlePatientFlowSubmit = async (req: PatientFlowRequest) => {
    setIsPredicting(true);
    setPfError(null);
    setPatientFlowResult(null);
    
    try {
      const res = await predictPatientFlow(req);
      setPatientFlowResult(res);
    } catch (err) {
      if (err instanceof ApiValidationError) {
        setPfError({ type: "validation", message: err.message });
      } else if (err instanceof ApiServiceError && err.statusCode === 503) {
        setPfError({ type: "model", message: err.message });
      } else {
        setPfError({ type: "generic", message: err instanceof Error ? err.message : "Unknown error" });
      }
    } finally {
      setIsPredicting(false);
    }
  };

  if (isLoading) {
    return <div className="pt-20"><LoadingState message="Connecting to QueueMind Intelligence Engine..." /></div>;
  }

  if (error) {
    return (
      <div className="pt-10 max-w-2xl mx-auto">
        <ErrorState type={error.type} message={error.message} onRetry={handleRetry} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="mb-8 flex flex-col items-start gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Operations Dashboard</h1>
          <p className="text-slate-400 mt-1">Departmental flow intelligence snapshot.</p>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/20 text-amber-500 px-3 py-1 rounded-full text-xs font-medium">
          Demo Input / Example Snapshot — not live hospital data.
        </div>
      </div>
      
      <Disclaimer compact={false} />

      {queueHealth && congestion && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <QueueHealthHero data={queueHealth} />
            </div>
            <div className="lg:col-span-2 space-y-6">
              <FlowKPICards 
                activeCensus={congestion.current_active_census}
                recentArrivals={12} // demo value
                recentDepartures={8} // demo value
                highAcuityRatio={queueHealth.components.high_acuity_pressure ? 0.25 : 0} 
              />
              <CongestionForecastCard data={congestion} />
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6 border-t border-slate-800 pt-6">
            <div className="lg:col-span-2">
              <PatientFlowForm onSubmit={handlePatientFlowSubmit} isLoading={isPredicting} />
            </div>
            <div className="lg:col-span-1">
              {pfError ? (
                <ErrorState type={pfError.type} message={pfError.message} />
              ) : patientFlowResult ? (
                <PatientFlowResult result={patientFlowResult} />
              ) : (
                <div className="h-full min-h-[300px] flex items-center justify-center border border-dashed border-slate-700 rounded-xl bg-slate-800/20">
                  <p className="text-slate-500 text-sm">Enter triage inputs to generate prediction.</p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
