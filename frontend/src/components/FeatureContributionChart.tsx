"use client";

import React, { useState } from "react";

export interface LocalFeatureContributionItem {
  feature_name: string;
  display_name: string;
  modality: string;
  observed_value: number | null;
  unit: string;
  contribution: number;
  direction: string; // "elevates_stress" | "reduces_stress" | "neutral"
  rank: number;
  explanation: string;
}

export interface GlobalFeatureImportanceItem {
  feature_name: string;
  display_name: string;
  modality: string;
  importance: number;
  rank: number;
}

export interface FeatureContributionChartProps {
  contributions: LocalFeatureContributionItem[];
  predictedClass?: string;
  stressProbability?: number | null;
  method?: string;
  nonCausationDisclaimer?: string;
  globalImportance?: GlobalFeatureImportanceItem[];
  isDemo?: boolean;
}

export default function FeatureContributionChart({
  contributions,
  predictedClass = "Unknown",
  stressProbability,
  method = "SHAP_TreeExplainer",
  nonCausationDisclaimer,
  globalImportance = [],
  isDemo = false,
}: FeatureContributionChartProps) {
  const [viewMode, setViewMode] = useState<"local" | "global">("local");

  const maxLocalContrib = Math.max(
    ...contributions.map((c) => Math.abs(c.contribution)),
    0.001
  );

  const maxGlobalImp = Math.max(
    ...globalImportance.map((g) => g.importance),
    0.001
  );

  const getModalityBadge = (modality: string) => {
    switch (modality) {
      case "sar":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            SAR Radar
          </span>
        );
      case "optical":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            S2 Optical
          </span>
        );
      case "temporal":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
            Rate / Delta
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-700 text-slate-300">
            Quality / Aux
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header and Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🔍</span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              AI Attribution & Feature Explanations
            </h3>
            {isDemo && (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                Synthetic Demo
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Method:{" "}
            <span className="text-emerald-400 font-semibold font-mono">
              {method}
            </span>{" "}
            • Attribution evaluated for condition:{" "}
            <span className="text-white font-semibold">{predictedClass}</span>
            {stressProbability !== null && stressProbability !== undefined && (
              <>
                {" "}
                (Calibrated Probability:{" "}
                <span className="text-amber-400 font-mono font-bold">
                  {(stressProbability * 100).toFixed(1)}%
                </span>
                )
              </>
            )}
          </p>
        </div>

        {/* View Toggle */}
        <div className="inline-flex rounded-xl bg-slate-950 p-1 border border-slate-800 self-start sm:self-auto">
          <button
            onClick={() => setViewMode("local")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
              viewMode === "local"
                ? "bg-emerald-500 text-slate-950 shadow-md font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Observation Attribution (Local)
          </button>
          <button
            onClick={() => setViewMode("global")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
              viewMode === "global"
                ? "bg-indigo-500 text-white shadow-md font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Model Importance (Global)
          </button>
        </div>
      </div>

      {/* Non-Causation Warning Banner */}
      <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-200 text-xs flex items-start gap-2.5">
        <span className="text-base flex-shrink-0">⚠️</span>
        <div className="space-y-0.5">
          <span className="font-bold text-amber-300">
            Scientific Attribution Notice — Correlation Is Not Causation
          </span>
          <p className="text-amber-200/90 leading-relaxed">
            {nonCausationDisclaimer ||
              "Feature contribution values indicate mathematical attribution within the machine-learning estimator based on satellite radiometric signals; they do NOT establish biological or pathological causation. Satellite indices cannot independently diagnose crop disease, pests, or nutrient deficiencies without in-field ground verification."}
          </p>
        </div>
      </div>

      {/* LOCAL VIEW: Instance Contributions */}
      {viewMode === "local" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span>Rank & Satellite Feature</span>
            <span>Observed Value & Directional Contribution</span>
          </div>

          {contributions.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-sm">
              No local feature contributions available for this observation.
            </div>
          ) : (
            <div className="space-y-3">
              {contributions.slice(0, 10).map((c) => {
                const isElevating = c.direction === "elevates_stress";
                const isReducing = c.direction === "reduces_stress";
                const pctWidth = Math.min(
                  100,
                  Math.round((Math.abs(c.contribution) / maxLocalContrib) * 100)
                );

                return (
                  <div
                    key={c.feature_name}
                    className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 transition space-y-2"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-slate-800 text-[11px] font-bold text-slate-300 flex items-center justify-center font-mono">
                          {c.rank}
                        </span>
                        <span className="font-semibold text-white text-sm">
                          {c.display_name}
                        </span>
                        <span className="font-mono text-[11px] text-slate-400">
                          ({c.feature_name})
                        </span>
                        {getModalityBadge(c.modality)}
                      </div>

                      <div className="flex items-center gap-3 self-end sm:self-auto font-mono text-xs">
                        <span className="text-slate-300">
                          Value:{" "}
                          <strong className="text-white">
                            {c.observed_value !== null
                              ? c.observed_value.toFixed(3)
                              : "N/A"}{" "}
                            {c.unit}
                          </strong>
                        </span>

                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold flex items-center gap-1 ${
                            isElevating
                              ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                              : isReducing
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                              : "bg-slate-800 text-slate-400"
                          }`}
                        >
                          {isElevating ? "▲ Elevates Stress" : isReducing ? "▼ Reduces Stress" : "■ Neutral"} (
                          {c.contribution > 0 ? "+" : ""}
                          {c.contribution.toFixed(4)})
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar showing contribution magnitude */}
                    <div className="w-full bg-slate-800/60 rounded-full h-2 overflow-hidden flex">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isElevating
                            ? "bg-gradient-to-r from-rose-500 to-amber-500"
                            : isReducing
                            ? "bg-gradient-to-r from-teal-500 to-emerald-400"
                            : "bg-slate-600"
                        }`}
                        style={{ width: `${pctWidth}%` }}
                      />
                    </div>

                    {/* Explanatory Narrative */}
                    <p className="text-xs text-slate-400 italic pl-1">
                      "{c.explanation}"
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* GLOBAL VIEW: Model Aggregate Importance */}
      {viewMode === "global" && (
        <div className="space-y-4">
          <div className="p-3 rounded-xl bg-slate-950/60 border border-indigo-900/40 text-indigo-200 text-xs flex items-center gap-2">
            <span>ℹ️</span>
            <span>
              <strong>Global Importance:</strong> Represents overall feature
              attribution across the estimator's training reference dataset
              (Random Forest Mean Decrease in Impurity / Gini Importance). This
              reflects general model sensitivity, not this specific observation.
            </span>
          </div>

          {globalImportance.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-sm">
              Global feature importance metadata not available.
            </div>
          ) : (
            <div className="space-y-2.5">
              {globalImportance.map((g) => {
                const pct = Math.round((g.importance / maxGlobalImp) * 100);
                return (
                  <div
                    key={g.feature_name}
                    className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60 flex flex-col gap-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-indigo-900/60 text-indigo-300 text-[10px] font-bold flex items-center justify-center font-mono">
                          {g.rank}
                        </span>
                        <span className="text-white font-medium">
                          {g.display_name}
                        </span>
                        <span className="text-[11px] text-slate-500 font-mono">
                          ({g.feature_name})
                        </span>
                        {getModalityBadge(g.modality)}
                      </div>
                      <span className="font-mono font-bold text-indigo-300">
                        {(g.importance * 100).toFixed(2)}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-800/50 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
