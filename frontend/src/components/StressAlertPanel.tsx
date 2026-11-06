"use client";

import React from "react";

export interface FeatureContributionData {
  feature_name: string;
  feature_label: string;
  measured_value: number | null;
  contribution_weight: number;
  direction: string;
}

export interface StressIntelligenceData {
  is_model_available: boolean;
  status_message: string;
  stress_class: string | null;
  stress_probability: number | null;
  confidence_level: string | null;
  top_contributing_features: FeatureContributionData[];
  missing_sensors: string[];
  model_name: string;
  model_version: string;
  is_demo: boolean;
  ground_truth_advisory: string;
}

interface StressAlertPanelProps {
  stress: StressIntelligenceData;
  isDemo?: boolean;
}

export default function StressAlertPanel({
  stress,
  isDemo = false,
}: StressAlertPanelProps) {
  // Production guardrail: If model is not available or ground truth is required
  if (!stress.is_model_available) {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-rose-900/50 p-6 backdrop-blur-md shadow-xl transition-all">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">⚠️</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Vigour Anomaly Intelligence
            </h3>
          </div>
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">
            MODEL BLOCKED
          </span>
        </div>

        <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-5 text-center my-3">
          <span className="text-2xl block mb-2">🛑</span>
          <h4 className="text-sm font-black text-rose-300 tracking-wide font-mono uppercase mb-2">
            STRESS MODEL UNAVAILABLE — VERIFIED AGRONOMIC GROUND TRUTH REQUIRED
          </h4>
          <p className="text-xs text-rose-200/80 max-w-xl mx-auto leading-relaxed">
            {stress.status_message ||
              "Production stress models are restricted from generating automated diagnoses without calibrated agronomic field ground-truth datasets. No synthetic estimates are substituted in production mode."}
          </p>
          <div className="mt-4 pt-3 border-t border-rose-900/60 text-[11px] text-slate-400">
            Advisory: {stress.ground_truth_advisory}
          </div>
        </div>
      </div>
    );
  }

  // Model is available (Demo mode or calibrated production model)
  const isElevated =
    (stress.stress_probability !== null && stress.stress_probability >= 0.5) ||
    stress.stress_class?.toLowerCase().includes("stress");

  const probPercent =
    stress.stress_probability !== null ? (stress.stress_probability * 100).toFixed(1) : null;

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-4 mb-5">
        <div className="flex items-center gap-2">
          <span className="text-xl">🛰️</span>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Vigour Anomaly Intelligence
            </h3>
            <p className="text-xs text-slate-400">
              Machine learning anomaly detection across combined optical and SAR feature layers.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {stress.is_demo ? (
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
              DEMO PREDICTION
            </span>
          ) : (
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
              CALIBRATED PRODUCTION
            </span>
          )}
          <span className="text-xs font-mono text-slate-500">{stress.model_version}</span>
        </div>
      </div>

      {/* Stress Classification & Probability Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        {/* Classification */}
        <div
          className={`p-4 rounded-xl border ${
            isElevated
              ? "bg-rose-950/30 border-rose-800/60"
              : "bg-emerald-950/30 border-emerald-800/60"
          }`}
        >
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-1">
            Potential Stress Classification
          </span>
          <span
            className={`text-lg font-black tracking-tight ${
              isElevated ? "text-rose-400" : "text-emerald-400"
            }`}
          >
            {stress.stress_class || "No Apparent Stress Detected"}
          </span>
          <p className="text-[11px] text-slate-400 mt-1.5 leading-snug">
            {stress.status_message}
          </p>
        </div>

        {/* Probability Meter */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Stress Probability
              </span>
              <span className="text-[11px] font-mono font-bold text-slate-300">
                {probPercent ? `${probPercent}%` : "N/A"}
              </span>
            </div>
            {/* Progress Bar */}
            <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden mt-2 flex">
              <div
                style={{ width: `${probPercent || 0}%` }}
                className={`transition-all duration-700 ${
                  isElevated ? "bg-rose-500" : "bg-emerald-500"
                }`}
              />
            </div>
          </div>
          <span className="text-[10px] text-slate-500 mt-2 block">
            Confidence: {stress.confidence_level || "Calibrated"}
          </span>
        </div>

        {/* Missing Sensor Warnings */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 flex flex-col justify-between">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-1">
            Sensor Modalities
          </span>
          {stress.missing_sensors && stress.missing_sensors.length > 0 ? (
            <div className="space-y-1">
              <span className="text-xs font-semibold text-amber-400 flex items-center gap-1">
                <span>⚠️</span> Missing Modalities:
              </span>
              <div className="flex flex-wrap gap-1 mt-1">
                {stress.missing_sensors.map((s) => (
                  <span
                    key={s}
                    className="px-2 py-0.5 rounded text-[10px] bg-amber-950/60 text-amber-300 border border-amber-800/60"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-xs text-emerald-400">
              <span>✓</span>
              <span>All required optical & SAR sensors active</span>
            </div>
          )}
          <span className="text-[10px] text-slate-500 mt-2 block">
            Model: {stress.model_name}
          </span>
        </div>
      </div>

      {/* Top Contributing Satellite Features (Explainability) */}
      {stress.top_contributing_features && stress.top_contributing_features.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-800/80">
          <span className="text-xs font-bold text-slate-300 block mb-2.5">
            Key Contributing Satellite Features (Local Attribution)
          </span>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {stress.top_contributing_features.map((feat) => {
              const isStressIncrease = feat.direction === "increases_stress";
              return (
                <div
                  key={feat.feature_name}
                  className="bg-slate-950/60 rounded-lg p-2.5 border border-slate-800/70 text-xs"
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-semibold text-white font-mono text-[11px]">
                      {feat.feature_name}
                    </span>
                    <span
                      className={`text-[10px] font-semibold px-1.5 py-0.2 rounded ${
                        isStressIncrease
                          ? "bg-rose-500/10 text-rose-400"
                          : "bg-emerald-500/10 text-emerald-400"
                      }`}
                    >
                      {isStressIncrease ? "+ Stress" : "- Stress"}
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Measured: {feat.measured_value !== null ? feat.measured_value.toFixed(2) : "—"}</span>
                    <span className="font-mono">w={feat.contribution_weight.toFixed(2)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Agronomic Disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400">
        <strong className="text-slate-300">Ground-Truth Advisory:</strong> {stress.ground_truth_advisory}
      </div>
    </div>
  );
}
