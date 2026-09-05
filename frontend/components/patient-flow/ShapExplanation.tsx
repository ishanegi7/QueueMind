"use client";

import React from "react";
import { Card } from "@/components/common/Card";
import type { PatientFlowExplanation } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer
} from "recharts";

interface ShapExplanationProps {
  explanation: PatientFlowExplanation;
}

export function ShapExplanation({ explanation }: ShapExplanationProps) {
  // Sort features by absolute attribution and take top 10
  const sortedFeatures = [...explanation.features]
    .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
    .slice(0, 10)
    .map(f => ({
      ...f,
      displayName: `${f.name} = ${f.value !== null ? f.value : 'N/A'}`
    }));

  return (
    <Card title="Model Explanation — Feature Contributions" className="w-full h-full flex flex-col">
      <div className="text-sm text-slate-400 mb-4">
        Base prediction: <span className="text-white font-medium">{explanation.base_value.toFixed(1)} min</span>
      </div>
      
      <div className="flex-1 min-h-[300px] -ml-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedFeatures}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" stroke="#94a3b8" fontSize={12} tickFormatter={(val) => `${val > 0 ? '+' : ''}${val}`} />
            <YAxis 
              type="category" 
              dataKey="displayName" 
              stroke="#94a3b8" 
              fontSize={11} 
              width={140}
              tick={{fill: '#cbd5e1'}}
            />
            <Tooltip 
              cursor={{fill: '#1e293b'}}
              contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '0.5rem'}}
              formatter={(value: any) => [`${value > 0 ? '+' : ''}${value.toFixed(1)} min`, 'Contribution']}
            />
            <Bar dataKey="attribution" radius={[0, 4, 4, 0]}>
              {sortedFeatures.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.attribution > 0 ? '#ef4444' : '#3b82f6'} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 pt-3 border-t border-slate-800 text-xs italic text-slate-500 text-center">
        {explanation.interpretation || "SHAP explains model behavior; it does not establish causality."}
      </div>
    </Card>
  );
}
