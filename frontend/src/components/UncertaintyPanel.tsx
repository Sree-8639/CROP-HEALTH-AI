"use client";

import React from "react";

export interface PredictionUncertaintyData {
  growth_prediction_intervals_available: boolean;
  interval_lead_time_scaling: string;
  minimum_observation_requirement: number;
  current_observation_count: number;
  uncertainty_explanation: string;
  stress_class_probability_margin: number | null;
  stress_prediction_certainty_warning?: string | null;
}

export interface UncertaintyPanelProps {
  uncertainty: PredictionUncertaintyData;
  targetIndex?: string;
  forecastDates?: string[];
}

export default function UncertaintyPanel({
  uncertainty,
  targetIndex = "NDVI",
  forecastDates = [],
}: UncertaintyPanelProps) {
  const {
    growth_prediction_intervals_available,
    interval_lead_time_scaling,
    minimum_observation_requirement,
    current_observation_count,
    uncertainty_explanation,
    stress_class_probability_margin,
    stress_prediction_certainty_warning,
  } = uncertainty;

  const hasSufficientObs =
    current_observation_count >= minimum_observation_requirement;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">📐</span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              Prediction Uncertainty & Empirical Intervals
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Separates observational uncertainty and forecasting spread from
            arbitrary single confidence percentages.
          </p>
        </div>

        <div className="px-3.5 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 font-mono">
          Lead-time Scaling:{" "}
          <strong className="text-cyan-400">
            {interval_lead_time_scaling}
          </strong>
        </div>
      </div>

      {/* Two-Column Uncertainty Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Growth Forecast Interval Availability */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Growth Forecast Interval
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                growth_prediction_intervals_available
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
              }`}
            >
              {growth_prediction_intervals_available
                ? "Active (80% Empirical)"
                : "Withheld (Insufficient Evidence)"}
            </span>
          </div>

          <div className="space-y-1.5 text-xs text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Minimum Acquisitions Required:</span>
              <span className="font-mono font-bold text-white">
                {minimum_observation_requirement} observations
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Available In Current AOI:</span>
              <span
                className={`font-mono font-bold ${
                  hasSufficientObs ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {current_observation_count} observations
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Forecasting Horizon Steps:</span>
              <span className="font-mono text-cyan-300">
                {forecastDates.length > 0 ? `${forecastDates.length} steps` : "Medium (3 steps)"}
              </span>
            </div>
          </div>

          {!hasSufficientObs && (
            <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/30 text-[11px] text-amber-200">
              ⚠️ Uncertainty bounds withheld: Fewer than {minimum_observation_requirement}{" "}
              clean acquisitions exist in the time window. Synthesizing intervals without
              historical residual distribution would create false precision.
            </div>
          )}
        </div>

        {/* Stress Detection Decision Margin */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Stress Classifier Separation
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
              Calibrated Margin
            </span>
          </div>

          <div className="space-y-1.5 text-xs text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Top-1 vs Runner-Up Margin:</span>
              <span className="font-mono font-bold text-white">
                {stress_class_probability_margin !== null
                  ? `${(stress_class_probability_margin * 100).toFixed(1)}%`
                  : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Separation Quality:</span>
              <span className="font-semibold text-emerald-400">
                {stress_class_probability_margin !== null &&
                stress_class_probability_margin > 0.3
                  ? "Decisive Margin (>30%)"
                  : stress_class_probability_margin !== null &&
                    stress_class_probability_margin > 0.15
                  ? "Moderate Separation (15-30%)"
                  : "Close Boundary (<15%)"}
              </span>
            </div>
          </div>

          {stress_prediction_certainty_warning && (
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
              {stress_prediction_certainty_warning}
            </div>
          )}
        </div>
      </div>

      {/* Uncertainty Methodological Explanation */}
      <div className="p-4 rounded-xl bg-slate-950/60 border border-cyan-900/40 text-cyan-200/90 text-xs leading-relaxed space-y-2">
        <div className="flex items-center gap-2 font-bold text-cyan-300">
          <span>ℹ️</span>
          <span>Empirical Spread vs Arbitrary Confidence Percentage</span>
        </div>
        <p>{uncertainty_explanation}</p>
        <p className="text-[11px] text-slate-400">
          <strong>Notice:</strong> An empirical prediction interval describes where{" "}
          80% of historical out-of-sample forecast deviations landed during walk-forward
          testing. It is <em>not</em> a guarantee that future satellite passes will stay
          within the interval, especially if abrupt weather or field events occur.
        </p>
      </div>
    </div>
  );
}
