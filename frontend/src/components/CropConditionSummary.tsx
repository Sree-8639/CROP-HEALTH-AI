"use client";

import React from "react";

export interface CropConditionSummaryData {
  dominant_condition: string;
  condition_score: number;
  healthy_canopy_pct: number;
  moderate_canopy_pct: number;
  stressed_canopy_pct: number;
  observation_date: string;
  evaluation_method: string;
}

interface CropConditionSummaryProps {
  condition: CropConditionSummaryData;
  isDemo?: boolean;
}

export default function CropConditionSummary({
  condition,
  isDemo = false,
}: CropConditionSummaryProps) {
  const getConditionColor = (cond: string) => {
    switch (cond.toLowerCase()) {
      case "healthy":
        return {
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/30",
          text: "text-emerald-400",
          ring: "text-emerald-500",
          label: "Optimal Canopy Radiance",
        };
      case "moderate":
        return {
          bg: "bg-amber-500/10",
          border: "border-amber-500/30",
          text: "text-amber-400",
          ring: "text-amber-500",
          label: "Moderate Canopy Variability",
        };
      case "stressed":
      case "elevated stress":
        return {
          bg: "bg-rose-500/10",
          border: "border-rose-500/30",
          text: "text-rose-400",
          ring: "text-rose-500",
          label: "Reduced Canopy Radiance / Stress Signals",
        };
      default:
        return {
          bg: "bg-slate-500/10",
          border: "border-slate-500/30",
          text: "text-slate-400",
          ring: "text-slate-500",
          label: "Insufficient Multispectral Evidence",
        };
    }
  };

  const styling = getConditionColor(condition.dominant_condition);

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl flex flex-col justify-between relative overflow-hidden transition-all duration-300 hover:border-slate-700">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">🌿</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Crop Condition Summary
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Obs: {condition.observation_date}
          </span>
        </div>

        {/* Condition Score & Dominant Category Banner */}
        <div
          className={`rounded-xl p-4 border ${styling.border} ${styling.bg} flex items-center justify-between mb-5`}
        >
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-0.5">
              Dominant Condition
            </span>
            <span className={`text-xl font-black ${styling.text}`}>
              {condition.dominant_condition}
            </span>
            <p className="text-xs text-slate-300 mt-1">{styling.label}</p>
          </div>

          <div className="text-right">
            <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400 block mb-0.5">
              Vigor Score
            </span>
            <div className="flex items-baseline justify-end gap-1">
              <span className={`text-3xl font-black font-mono ${styling.text}`}>
                {condition.condition_score.toFixed(0)}
              </span>
              <span className="text-xs font-semibold text-slate-400">/ 100</span>
            </div>
          </div>
        </div>

        {/* Canopy Vigor Distribution Bar */}
        <div className="space-y-2 mb-5">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-300 font-medium">Canopy Vigor Distribution</span>
            <span className="text-[11px] text-slate-400 font-mono">
              Valid Canopy Area
            </span>
          </div>

          <div className="h-3.5 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800/80 shadow-inner">
            <div
              style={{ width: `${condition.healthy_canopy_pct}%` }}
              className="bg-emerald-500 transition-all duration-500"
              title={`Healthy: ${condition.healthy_canopy_pct.toFixed(1)}%`}
            />
            <div
              style={{ width: `${condition.moderate_canopy_pct}%` }}
              className="bg-amber-400 transition-all duration-500"
              title={`Moderate: ${condition.moderate_canopy_pct.toFixed(1)}%`}
            />
            <div
              style={{ width: `${condition.stressed_canopy_pct}%` }}
              className="bg-rose-500 transition-all duration-500"
              title={`Stressed: ${condition.stressed_canopy_pct.toFixed(1)}%`}
            />
          </div>

          {/* Legend */}
          <div className="grid grid-cols-3 gap-2 pt-1 text-center">
            <div className="bg-slate-950/60 rounded-lg py-1 px-2 border border-slate-800/60">
              <span className="text-[10px] text-slate-400 block">Healthy</span>
              <span className="text-xs font-bold text-emerald-400 font-mono">
                {condition.healthy_canopy_pct.toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-950/60 rounded-lg py-1 px-2 border border-slate-800/60">
              <span className="text-[10px] text-slate-400 block">Moderate</span>
              <span className="text-xs font-bold text-amber-400 font-mono">
                {condition.moderate_canopy_pct.toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-950/60 rounded-lg py-1 px-2 border border-slate-800/60">
              <span className="text-[10px] text-slate-400 block">Stressed</span>
              <span className="text-xs font-bold text-rose-400 font-mono">
                {condition.stressed_canopy_pct.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Scientific Limitation & Provenance Note */}
      <div className="pt-3 border-t border-slate-800/80">
        <p className="text-[11px] text-slate-400 leading-relaxed">
          <strong className="text-slate-300">Method:</strong> {condition.evaluation_method}.{" "}
          <span className="text-slate-400">
            Multispectral vegetation vigor represents surface radiometric reflectance and does not confirm disease, pest, or nutrient etiology without ground-truth physical verification.
          </span>
        </p>
      </div>
    </div>
  );
}
