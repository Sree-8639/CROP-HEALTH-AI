"use client";

import React, { useState } from "react";
import FeatureContributionChart, {
  LocalFeatureContributionItem,
  GlobalFeatureImportanceItem,
} from "./FeatureContributionChart";
import DataQualityPanel, { DataQualityAssessmentData } from "./DataQualityPanel";
import UncertaintyPanel, { PredictionUncertaintyData } from "./UncertaintyPanel";
import ModelReliabilityPanel, { ModelReliabilityData } from "./ModelReliabilityPanel";

export interface SingleIndexGrowthExplanationItem {
  index_name: string;
  historical_direction: string;
  trend_slope_per_day: number | null;
  fitted_baseline_model: string;
  selection_rationale: string;
  influencing_recent_observations: Array<{ date: string; value: number }>;
  candidate_model_comparisons: Array<{
    model_name: string;
    mae: number | null;
    rmse: number | null;
    validation_windows: number;
    selected: boolean;
  }>;
  trend_explanation: string;
}

export interface ExplainabilityResponseData {
  status: string;
  is_demo: boolean;
  explanation_status: string;
  explanation_type: string;
  stress_explainability?: {
    method: string;
    predicted_class: string;
    stress_probability: number | null;
    local_feature_contributions: LocalFeatureContributionItem[];
    global_feature_importance?: GlobalFeatureImportanceItem[];
    explanation_limitations: string[];
    non_causation_disclaimer: string;
  } | null;
  growth_forecast_explainability?: {
    index_explanations: Record<string, SingleIndexGrowthExplanationItem>;
    general_selection_rationale: string;
  } | null;
  data_quality_assessment: DataQualityAssessmentData;
  model_reliability: ModelReliabilityData;
  prediction_uncertainty: PredictionUncertaintyData;
  warnings: string[];
  scientific_limitations: string[];
}

export interface ExplainabilityPanelProps {
  data: ExplainabilityResponseData;
  selectedIndex?: string;
  onSelectIndex?: (idx: string) => void;
}

export default function ExplainabilityPanel({
  data,
  selectedIndex = "NDVI",
  onSelectIndex,
}: ExplainabilityPanelProps) {
  const [activeTab, setActiveTab] = useState<
    "stress" | "growth" | "quality" | "uncertainty" | "reliability"
  >("stress");

  const [localIndex, setLocalIndex] = useState<string>(selectedIndex || "NDVI");

  const handleIndexChange = (idx: string) => {
    setLocalIndex(idx);
    if (onSelectIndex) {
      onSelectIndex(idx);
    }
  };

  const {
    is_demo,
    explanation_status,
    stress_explainability,
    growth_forecast_explainability,
    data_quality_assessment,
    model_reliability,
    prediction_uncertainty,
    warnings,
    scientific_limitations,
  } = data;

  const currentGrowthExplanation =
    growth_forecast_explainability?.index_explanations?.[localIndex];

  return (
    <div className="space-y-6">
      {/* Top Banner & Tab Navigation */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400 font-bold text-base">
              🧠
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">
                Explainable AI & Confidence Intelligence
              </h2>
              <span className="text-xs text-slate-400">
                Production-Grade Multi-Perspective Diagnostic System
              </span>
            </div>
          </div>
        </div>

        {/* Tab Selectors */}
        <div className="flex items-center gap-1.5 overflow-x-auto p-1 bg-slate-950 border border-slate-800 rounded-xl">
          <button
            onClick={() => setActiveTab("stress")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
              activeTab === "stress"
                ? "bg-rose-500 text-slate-950 font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            ⚡ Why Stress?
          </button>
          <button
            onClick={() => setActiveTab("growth")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
              activeTab === "growth"
                ? "bg-indigo-500 text-white font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            📈 Why This Growth Trend?
          </button>
          <button
            onClick={() => setActiveTab("quality")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
              activeTab === "quality"
                ? "bg-emerald-500 text-slate-950 font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            📡 Data Quality
          </button>
          <button
            onClick={() => setActiveTab("uncertainty")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
              activeTab === "uncertainty"
                ? "bg-cyan-500 text-slate-950 font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            📐 Uncertainty
          </button>
          <button
            onClick={() => setActiveTab("reliability")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition whitespace-nowrap ${
              activeTab === "reliability"
                ? "bg-purple-500 text-white font-bold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            🔬 Reliability
          </button>
        </div>
      </div>

      {/* Warnings & Notices */}
      {warnings && warnings.length > 0 && (
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-500/50 text-amber-200 text-xs space-y-1">
          <div className="flex items-center gap-2 font-bold text-amber-300 text-sm">
            <span>⚠️</span>
            <span>Assessment Notices & Warnings</span>
          </div>
          {warnings.map((w, i) => (
            <p key={i} className="text-amber-200/90 leading-relaxed pl-5">
              • {w}
            </p>
          ))}
        </div>
      )}

      {/* TAB 1: STRESS DETECTION EXPLAINABILITY */}
      {activeTab === "stress" && (
        <>
          {stress_explainability ? (
            <FeatureContributionChart
              contributions={stress_explainability.local_feature_contributions}
              predictedClass={stress_explainability.predicted_class}
              stressProbability={stress_explainability.stress_probability}
              method={stress_explainability.method}
              nonCausationDisclaimer={stress_explainability.non_causation_disclaimer}
              globalImportance={stress_explainability.global_feature_importance}
              isDemo={is_demo}
            />
          ) : (
            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800 text-center text-slate-400 text-sm">
              Stress model explanation is currently unavailable for this observation.
            </div>
          )}
        </>
      )}

      {/* TAB 2: GROWTH FORECAST EXPLAINABILITY */}
      {activeTab === "growth" && (
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl">📈</span>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  Growth Trend & Forecast Model Selection Explanations
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Explains historical direction, candidate model evaluation, and why a specific forecasting baseline was chosen.
              </p>
            </div>

            {/* Index Selector */}
            <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 self-start sm:self-auto">
              {["NDVI", "NDMI", "NDRE", "EVI", "VV", "VH"].map((idx) => (
                <button
                  key={idx}
                  onClick={() => handleIndexChange(idx)}
                  className={`px-2.5 py-1 text-xs font-mono font-bold rounded-lg transition ${
                    localIndex === idx
                      ? "bg-indigo-500 text-white shadow"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {idx}
                </button>
              ))}
            </div>
          </div>

          {currentGrowthExplanation ? (
            <div className="space-y-6">
              {/* Selected Model & Direction Banner */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs text-slate-400 block">Observed Direction</span>
                  <span
                    className={`text-xl font-bold tracking-tight mt-1 block ${
                      currentGrowthExplanation.historical_direction === "Increasing"
                        ? "text-emerald-400"
                        : currentGrowthExplanation.historical_direction === "Decreasing"
                        ? "text-rose-400"
                        : "text-slate-300"
                    }`}
                  >
                    {currentGrowthExplanation.historical_direction}
                  </span>
                  <span className="text-[11px] text-slate-500 font-mono mt-0.5 block">
                    Slope:{" "}
                    {currentGrowthExplanation.trend_slope_per_day !== null
                      ? `${currentGrowthExplanation.trend_slope_per_day > 0 ? "+" : ""}${currentGrowthExplanation.trend_slope_per_day.toFixed(4)} / day`
                      : "N/A"}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs text-slate-400 block">Selected Forecast Baseline</span>
                  <span className="text-xl font-bold font-mono text-cyan-400 mt-1 block">
                    {currentGrowthExplanation.fitted_baseline_model}
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5 block">
                    Lowest out-of-sample RMSE
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs text-slate-400 block">Recent Anchor Passes</span>
                  <span className="text-xl font-bold font-mono text-white mt-1 block">
                    {currentGrowthExplanation.influencing_recent_observations.length} passes
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5 block">
                    Influencing sequential trajectory
                  </span>
                </div>
              </div>

              {/* Rationale & Narrative */}
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  Selection Rationale & Trend Behavior
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {currentGrowthExplanation.selection_rationale}
                </p>
                <p className="text-xs text-slate-400 leading-relaxed italic border-t border-slate-800/60 pt-2">
                  "{currentGrowthExplanation.trend_explanation}"
                </p>
              </div>

              {/* Candidate Model Comparison Table */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Candidate Baseline Validation Benchmarks (Walk-Forward Validation)
                </h4>
                <div className="overflow-x-auto rounded-xl border border-slate-800">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-2.5">Candidate Model</th>
                        <th className="px-4 py-2.5">Walk-Forward MAE</th>
                        <th className="px-4 py-2.5">Walk-Forward RMSE</th>
                        <th className="px-4 py-2.5">Validation Windows</th>
                        <th className="px-4 py-2.5">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {currentGrowthExplanation.candidate_model_comparisons.map((c) => (
                        <tr
                          key={c.model_name}
                          className={c.selected ? "bg-indigo-950/30" : "hover:bg-slate-950/40"}
                        >
                          <td className="px-4 py-2.5 font-bold text-white flex items-center gap-2">
                            {c.selected && <span className="text-indigo-400 font-sans">✓</span>}
                            <span>{c.model_name}</span>
                          </td>
                          <td className="px-4 py-2.5 text-slate-300">
                            {c.mae !== null ? c.mae.toFixed(4) : "N/A"}
                          </td>
                          <td className="px-4 py-2.5 text-cyan-300 font-bold">
                            {c.rmse !== null ? c.rmse.toFixed(4) : "N/A"}
                          </td>
                          <td className="px-4 py-2.5 text-slate-400">
                            {c.validation_windows} folds
                          </td>
                          <td className="px-4 py-2.5 font-sans">
                            {c.selected ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                                Selected (Best Fit)
                              </span>
                            ) : (
                              <span className="text-slate-500 text-[11px]">
                                Rejected (Higher RMSE)
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-8 text-center text-slate-500 text-sm">
              No growth forecast explanation available for index {localIndex}.
            </div>
          )}
        </div>
      )}

      {/* TAB 3: DATA QUALITY */}
      {activeTab === "quality" && (
        <DataQualityPanel dataQuality={data_quality_assessment} />
      )}

      {/* TAB 4: PREDICTION UNCERTAINTY */}
      {activeTab === "uncertainty" && (
        <UncertaintyPanel
          uncertainty={prediction_uncertainty}
          targetIndex={localIndex}
        />
      )}

      {/* TAB 5: MODEL RELIABILITY */}
      {activeTab === "reliability" && (
        <ModelReliabilityPanel
          reliability={model_reliability}
          isDemo={is_demo}
        />
      )}

      {/* Scientific Limitations Footer */}
      <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          Scientific Guidelines & Diagnostic Safeguards
        </h4>
        <ul className="text-xs text-slate-400 space-y-1 pl-4 list-disc">
          {scientific_limitations.map((lim, i) => (
            <li key={i}>{lim}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
