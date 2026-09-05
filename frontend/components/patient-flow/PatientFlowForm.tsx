"use client";

import React, { useState } from "react";
import { Card } from "@/components/common/Card";
import type { PatientFlowRequest } from "@/lib/types";

interface PatientFlowFormProps {
  onSubmit: (request: PatientFlowRequest) => void;
  isLoading?: boolean;
}

export function PatientFlowForm({ onSubmit, isLoading }: PatientFlowFormProps) {
  const [formData, setFormData] = useState<Partial<PatientFlowRequest>>({
    coverage_level: 0.90,
    return_explanation: true,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;

    if (type === "number") {
      parsedValue = value === "" ? null : Number(value);
    } else if (type === "checkbox") {
      parsedValue = (e.target as HTMLInputElement).checked;
    } else if (value === "") {
      parsedValue = null;
    }

    setFormData((prev) => ({ ...prev, [name]: parsedValue }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData as PatientFlowRequest);
  };

  return (
    <Card title="Patient Flow Prediction" className="w-full">
      <form onSubmit={handleSubmit} className="space-y-6 mt-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Vitals Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-slate-200 border-b border-slate-700 pb-2">Vitals</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Temp (°F)</label>
                <input type="number" step="0.1" name="temperature" min="70" max="115" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Heart Rate</label>
                <input type="number" name="heartrate" min="20" max="300" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Resp Rate</label>
                <input type="number" name="resprate" min="4" max="70" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">O2 Sat (%)</label>
                <input type="number" name="o2sat" min="50" max="100" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">SBP (mmHg)</label>
                <input type="number" name="sbp" min="40" max="300" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">DBP (mmHg)</label>
                <input type="number" name="dbp" min="20" max="200" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Pain (0-10)</label>
                <input type="number" step="0.1" name="pain" min="0" max="10" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
              </div>
            </div>
          </div>

          <div className="space-y-6">
            {/* Clinical & Temporal Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-200 border-b border-slate-700 pb-2">Clinical & Temporal</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Acuity (1-5)</label>
                  <select name="acuity" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white">
                    <option value="">--</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Gender</label>
                  <select name="gender" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white">
                    <option value="">--</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Arrival Hour</label>
                  <input type="number" name="arrival_hour" min="0" max="23" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Day of Week</label>
                  <select name="arrival_dayofweek" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white">
                    <option value="">--</option>
                    <option value="0">Monday</option>
                    <option value="1">Tuesday</option>
                    <option value="2">Wednesday</option>
                    <option value="3">Thursday</option>
                    <option value="4">Friday</option>
                    <option value="5">Saturday</option>
                    <option value="6">Sunday</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2 col-span-2">
                  <input type="checkbox" name="is_weekend" value="1" onChange={(e) => setFormData(prev => ({...prev, is_weekend: e.target.checked ? 1 : 0}))} id="is_weekend" />
                  <label htmlFor="is_weekend" className="text-sm text-slate-400">Is Weekend</label>
                </div>
              </div>
            </div>

            {/* ED Context Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-200 border-b border-slate-700 pb-2">ED Context</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Elapsed Time (min)</label>
                  <input type="number" name="elapsed_time_minutes" min="0" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Active Census</label>
                  <input type="number" name="active_census" min="0" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Arrivals (60m)</label>
                  <input type="number" name="recent_arrivals_60m" min="0" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Departures (60m)</label>
                  <input type="number" name="recent_departures_60m" min="0" onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded-md p-2 text-white" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Settings Section */}
        <div className="space-y-4 pt-4 border-t border-slate-700">
           <h3 className="text-lg font-semibold text-slate-200">Settings</h3>
           <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-8">
             <div className="flex-1">
               <label className="block text-sm text-slate-400 mb-1">Coverage Level ({formData.coverage_level})</label>
               <input type="range" name="coverage_level" min="0.50" max="0.99" step="0.01" value={formData.coverage_level || 0.90} onChange={handleChange} className="w-full" />
             </div>
             <div className="flex items-center space-x-2 pt-6">
               <input type="checkbox" name="return_explanation" id="return_explanation" checked={formData.return_explanation} onChange={handleChange} />
               <label htmlFor="return_explanation" className="text-sm text-slate-400">Return SHAP Explanation</label>
             </div>
           </div>
        </div>

        <button type="submit" disabled={isLoading} className="w-full py-3 mt-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md transition disabled:opacity-50">
          {isLoading ? "Predicting..." : "Predict Remaining Journey"}
        </button>
      </form>
    </Card>
  );
}
