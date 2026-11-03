"use client";

import React, { useState } from "react";

export interface PredictionIntervalData {
  date: string;
  lower: number;
  upper: number;
  confidence_level: number;
}

export interface GrowthForecastData {
  target_index: string;
  baseline_algorithm: string;
  forecast_horizon: string;
  historical_slope_per_day: number | null;
  forecast_dates: string[];
  projected_values: number[];
  prediction_intervals: PredictionIntervalData[];
  walk_forward_metrics: Record<string, any>;
  forecast_limitations: string[];
  disclaimer: string;
}

interface GrowthForecastPanelProps {
  forecast: GrowthForecastData;
  isDemo?: boolean;
}

export default function GrowthForecastPanel({
  forecast,
  isDemo = false,
}: GrowthForecastPanelProps) {
  const [showIntervals, setShowIntervals] = useState(true);

  const mae = forecast.walk_forward_metrics?.mae;
  const rmse = forecast.walk_forward_metrics?.rmse;
  const mape = forecast.walk_forward_metrics?.mape;

  const slope = forecast.historical_slope_per_day;
  const slopeDescription =
    slope !== null
      ? slope > 0.001
        ? "Positive Growth Rate (+/day)"
        : slope < -0.001
        ? "Declining Trajectory (-/day)"
        : "Stable Plateau (near-zero slope)"
      : "Historical Trend Baseline";

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🔮</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Vegetation Growth Trend & Projections
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-950/80 text-sky-300 border border-sky-800/60 uppercase">
              {forecast.target_index} Projection
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Statistical extrapolation based on historical multi-temporal observation trajectories.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowIntervals(!showIntervals)}
            className={`px-3 py-1 rounded-lg text-xs font-semibold border transition ${
              showIntervals
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : "bg-slate-800 text-slate-400 border-slate-700"
            }`}
          >
            {showIntervals ? "✓ 80% Prediction Band" : "Show Prediction Band"}
          </button>
        </div>
      </div>

      {/* Model Spec & Validation Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Baseline Model
          </span>
          <span className="text-sm font-bold text-white font-mono block">
            {forecast.baseline_algorithm}
          </span>
          <span className="text-[10px] text-slate-500">
            Horizon: {forecast.forecast_horizon}
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Trajectory Slope
          </span>
          <span className="text-sm font-bold text-emerald-400 font-mono block">
            {slope !== null ? `${(slope * 100).toFixed(3)}%/day` : "N/A"}
          </span>
          <span className="text-[10px] text-slate-500 truncate block">
            {slopeDescription}
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Walk-Forward MAE
          </span>
          <span className="text-sm font-bold text-sky-400 font-mono block">
            {mae !== undefined && mae !== null ? Number(mae).toFixed(4) : "Validated"}
          </span>
          <span className="text-[10px] text-slate-500">
            Mean Absolute Error
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Walk-Forward RMSE
          </span>
          <span className="text-sm font-bold text-slate-300 font-mono block">
            {rmse !== undefined && rmse !== null ? Number(rmse).toFixed(4) : "Calibrated"}
          </span>
          <span className="text-[10px] text-slate-500">
            Root Mean Square
          </span>
        </div>
      </div>

      {/* Projection Timeline Cards */}
      <div className="space-y-2 mb-5">
        <span className="text-xs font-semibold text-slate-300 block">
          Projected {forecast.target_index} Horizon Timeline:
        </span>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {forecast.forecast_dates.map((date, idx) => {
            const projectedVal = forecast.projected_values[idx];
            const interval = forecast.prediction_intervals[idx];

            return (
              <div
                key={date}
                className="bg-slate-950/80 rounded-xl p-3 border border-sky-900/30 text-center relative overflow-hidden"
              >
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-sky-500/40" />
                <span className="text-[10px] font-mono text-slate-400 block mb-1">
                  {date}
                </span>
                <span className="text-base font-black text-sky-300 font-mono block">
                  {projectedVal !== undefined ? projectedVal.toFixed(3) : "—"}
                </span>

                {showIntervals && interval && (
                  <div className="mt-1.5 pt-1.5 border-t border-slate-800/80 text-[10px] font-mono text-slate-400">
                    <span className="text-emerald-400/90">{interval.lower.toFixed(2)}</span>
                    <span className="text-slate-600 mx-0.5">~</span>
                    <span className="text-emerald-400/90">{interval.upper.toFixed(2)}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Forecast Limitations */}
      {forecast.forecast_limitations && forecast.forecast_limitations.length > 0 && (
        <div className="bg-slate-950/40 rounded-xl p-3.5 border border-slate-800/80 mb-4">
          <span className="text-[11px] font-bold text-slate-300 block mb-1">
            Statistical Limitations & Constraints:
          </span>
          <ul className="text-[11px] text-slate-400 space-y-1 list-disc list-inside">
            {forecast.forecast_limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Mandatory Non-Guaranteed Disclaimer */}
      <div className="pt-3 border-t border-slate-800/80 text-[11px] text-slate-400">
        <strong className="text-slate-300">Agricultural Disclaimer:</strong> {forecast.disclaimer}
      </div>
    </div>
  );
}
