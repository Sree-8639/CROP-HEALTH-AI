"use client";

import React from "react";

export interface VegetationIndicatorMetricData {
  name: string;
  label: string;
  latest_value: number | null;
  previous_value: number | null;
  delta_absolute: number | null;
  delta_percentage: number | null;
  trend_direction: string;
  observation_date: string;
  previous_observation_date: string | null;
  quality_status: string;
  physical_indicator: string;
  status_color: string;
}

export type VegetationMetricsData = Record<string, VegetationIndicatorMetricData>;

interface VegetationMetricsProps {
  metrics: Record<string, VegetationIndicatorMetricData>;
  isDemo?: boolean;
}

export default function VegetationMetrics({
  metrics,
  isDemo = false,
}: VegetationMetricsProps) {
  const getTrendBadge = (direction: string) => {
    switch (direction.toLowerCase()) {
      case "increasing":
        return {
          icon: "↗",
          bg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          text: "Increasing",
        };
      case "decreasing":
        return {
          icon: "↘",
          bg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          text: "Decreasing",
        };
      case "stable":
        return {
          icon: "→",
          bg: "bg-sky-500/10 text-sky-400 border-sky-500/30",
          text: "Stable",
        };
      default:
        return {
          icon: "⋯",
          bg: "bg-slate-500/10 text-slate-400 border-slate-500/30",
          text: "Insufficient Data",
        };
    }
  };

  const getDeltaBadge = (delta: number | null, pct: number | null) => {
    if (delta === null) return <span className="text-slate-500 text-xs">—</span>;
    const isPositive = delta > 0;
    const isNegative = delta < 0;
    const color = isPositive ? "text-emerald-400" : isNegative ? "text-rose-400" : "text-slate-400";
    const sign = isPositive ? "+" : "";

    return (
      <span className={`font-mono text-xs font-semibold ${color}`}>
        {sign}{delta.toFixed(3)}{pct !== null ? ` (${sign}${pct.toFixed(1)}%)` : ""}
      </span>
    );
  };

  const metricKeys = ["NDVI", "NDMI", "NDRE", "EVI"];

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Title & Explanatory Tag */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg">📈</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Multispectral Vegetation Health Indicators
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Calibrated radiometric canopy indices computed directly from Sentinel-2 MSI surface reflectance.
          </p>
        </div>

        <span className="px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700 self-start sm:self-auto">
          Measured Satellite Reflectance
        </span>
      </div>

      {/* Grid of 4 Indices */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricKeys.map((key) => {
          const item = metrics[key];
          if (!item) {
            return (
              <div
                key={key}
                className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 flex flex-col justify-between"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-white text-sm">{key}</span>
                  <span className="text-[10px] text-slate-500">No Data</span>
                </div>
                <p className="text-xs text-slate-500">Indicator unavailable for this observation.</p>
              </div>
            );
          }

          const trend = getTrendBadge(item.trend_direction);

          return (
            <div
              key={key}
              className="bg-slate-950/70 rounded-xl p-4 border border-slate-800/80 flex flex-col justify-between hover:border-slate-700 transition"
            >
              {/* Header */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="font-black text-white text-base tracking-tight font-mono">
                      {item.name}
                    </span>
                    <span className="text-[11px] text-slate-400 font-medium">
                      ({item.label})
                    </span>
                  </div>

                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border flex items-center gap-1 ${trend.bg}`}
                  >
                    <span>{trend.icon}</span>
                    <span>{trend.text}</span>
                  </span>
                </div>

                {/* Physical meaning */}
                <p className="text-[11px] text-slate-400 mb-3 leading-snug">
                  {item.physical_indicator}
                </p>

                {/* Latest Value */}
                <div className="bg-slate-900/90 rounded-lg p-2.5 border border-slate-800/60 mb-3">
                  <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider block">
                    Latest ({item.observation_date})
                  </span>
                  <div className="flex items-baseline justify-between mt-0.5">
                    <span className="text-2xl font-black text-white font-mono">
                      {item.latest_value !== null ? item.latest_value.toFixed(3) : "N/A"}
                    </span>
                    {getDeltaBadge(item.delta_absolute, item.delta_percentage)}
                  </div>
                </div>

                {/* Previous Observation & Delta */}
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span className="text-[11px]">Previous Value:</span>
                    <span className="font-mono text-slate-300">
                      {item.previous_value !== null ? item.previous_value.toFixed(3) : "—"}
                    </span>
                  </div>
                  {item.previous_observation_date && (
                    <div className="flex justify-between text-slate-500 text-[10px]">
                      <span>Previous Date:</span>
                      <span className="font-mono">{item.previous_observation_date}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Quality & Measurement Classification */}
              <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-[10px]">
                <span className="text-slate-500">Quality: {item.quality_status}</span>
                <span className="text-emerald-400 font-medium">Verified Sensor Index</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Distinction Footer */}
      <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400">
        <span>
          ℹ️ <strong>Physical Validation:</strong> These indices reflect canopy reflectance absorption in specific optical wavelengths and should not be used as single-factor diagnoses.
        </span>
      </div>
    </div>
  );
}
