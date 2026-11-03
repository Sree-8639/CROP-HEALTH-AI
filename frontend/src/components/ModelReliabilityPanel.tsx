"use client";

import React from "react";

export interface ModelReliabilityData {
  stress_model: {
    model_name: string;
    model_version: string;
    model_type: string;
    is_demo_model: boolean;
    is_calibrated: boolean;
    calibration_status: string;
    brier_score: number | null;
    expected_calibration_error_ece: number | null;
    calibration_sample_count: number;
    training_dataset_provenance: string;
    note: string;
  };
  growth_model: {
    validation_methodology: string;
    leakage_prevention: string;
    baseline_benchmark: string;
    metrics_available: string[];
    note: string;
  };
}

export interface ModelReliabilityPanelProps {
  reliability: ModelReliabilityData;
  isDemo?: boolean;
}

export default function ModelReliabilityPanel({
  reliability,
  isDemo = false,
}: ModelReliabilityPanelProps) {
  const { stress_model, growth_model } = reliability;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Title & Governance Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🔬</span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              Model Calibration, Validation & Provenance
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Verified statistical error metrics from walk-forward cross-validation
            and isotonic probability calibration.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {stress_model.is_demo_model ? (
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              Demo Mode Model
            </span>
          ) : (
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Production Architecture
            </span>
          )}
        </div>
      </div>

      {/* Two Column Layout: Stress Model vs Growth Forecasting Model */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stress Model Card */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <div>
              <span className="text-xs uppercase font-bold text-slate-400 block">
                Stress Detection Estimator
              </span>
              <span className="text-sm font-bold text-white font-mono">
                {stress_model.model_name}
              </span>
            </div>
            <span className="px-2.5 py-1 rounded-lg text-xs font-mono font-bold bg-slate-800 text-slate-300">
              {stress_model.model_version}
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Architecture / Estimator:</span>
              <span className="font-mono text-white font-semibold">
                {stress_model.model_type}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Calibration Status:</span>
              <span
                className={`font-semibold ${
                  stress_model.is_calibrated
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}
              >
                {stress_model.calibration_status}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Brier Calibration Score:</span>
              <span className="font-mono text-cyan-300 font-bold">
                {stress_model.brier_score !== null
                  ? stress_model.brier_score.toFixed(4)
                  : "N/A"}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Expected Calibration Error (ECE):</span>
              <span className="font-mono text-cyan-300 font-bold">
                {stress_model.expected_calibration_error_ece !== null
                  ? stress_model.expected_calibration_error_ece.toFixed(4)
                  : "N/A"}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Calibration Test Samples:</span>
              <span className="font-mono text-white">
                {stress_model.calibration_sample_count} stratified samples
              </span>
            </div>

            <div className="flex justify-between py-1">
              <span className="text-slate-400">Dataset Provenance:</span>
              <span className="font-medium text-slate-300 text-right max-w-[200px] truncate">
                {stress_model.training_dataset_provenance}
              </span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
            ℹ️ {stress_model.note}
          </div>
        </div>

        {/* Growth Forecast Model Card */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <div>
              <span className="text-xs uppercase font-bold text-slate-400 block">
                Crop Growth Forecasting Engine
              </span>
              <span className="text-sm font-bold text-white font-mono">
                MultimodalGrowthPredictor
              </span>
            </div>
            <span className="px-2.5 py-1 rounded-lg text-xs font-mono font-bold bg-slate-800 text-slate-300">
              v1.0.0
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Validation Methodology:</span>
              <span className="font-semibold text-white text-right max-w-[240px]">
                {growth_model.validation_methodology}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Temporal Leakage Prevention:</span>
              <span className="text-emerald-400 font-semibold">
                Strict (No Lookahead)
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Baseline Benchmark:</span>
              <span className="font-mono text-cyan-300 font-bold">
                {growth_model.baseline_benchmark}
              </span>
            </div>

            <div className="flex justify-between py-1 border-b border-slate-800/40">
              <span className="text-slate-400">Validation Metric Suite:</span>
              <span className="font-mono text-white">
                {growth_model.metrics_available.join(", ")}
              </span>
            </div>

            <div className="flex justify-between py-1">
              <span className="text-slate-400">Selection Policy:</span>
              <span className="font-medium text-slate-300">
                Minimum RMSE during walk-forward validation
              </span>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400">
            ℹ️ {growth_model.note}
          </div>
        </div>
      </div>
    </div>
  );
}
