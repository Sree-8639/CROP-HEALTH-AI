"use client";

import React from "react";

export interface ObservationItem {
  title: string;
  description: string;
  evidence: string[];
}

export interface PatternItem {
  pattern: string;
  evidence: string;
  severity: "low" | "moderate" | "high";
  explanation: string;
}

export interface HistoricalInterpretation {
  summary: string;
  trend: "increasing" | "stable" | "decreasing" | "insufficient_data";
}

export interface AreaToWatch {
  description: string;
  reason: string;
}

export interface DataQualityAssessment {
  assessment: string;
  limitations: string[];
}

export interface AIFieldReportData {
  headline: string;
  summary: string;
  observations: ObservationItem[];
  possible_patterns: PatternItem[];
  historical_interpretation?: HistoricalInterpretation | null;
  areas_to_watch: AreaToWatch[];
  data_quality: DataQualityAssessment;
  recommended_next_steps: string[];
  confidence: "low" | "moderate" | "high";
  is_demo?: boolean;
  generated_at?: string;
  provider_used?: string;
}

interface AIFieldReportProps {
  report: AIFieldReportData | null;
  loading: boolean;
  loadingStage: string;
  error: string | null;
  onGenerate: () => void;
  onRegenerate: () => void;
  hasAnalyticalData: boolean;
  isDemoMode: boolean;
}

export default function AIFieldReport({
  report,
  loading,
  loadingStage,
  error,
  onGenerate,
  onRegenerate,
  hasAnalyticalData,
  isDemoMode,
}: AIFieldReportProps) {
  const scrollToAnalyticalResults = () => {
    const el = document.getElementById("analytical-results");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  const getConfidenceBadge = (confidence: string) => {
    switch (confidence) {
      case "high":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            High Confidence
          </span>
        );
      case "moderate":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 border border-amber-500/30 text-amber-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
            Moderate Confidence
          </span>
        );
      case "low":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 border border-rose-500/30 text-rose-400">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
            Low Confidence
          </span>
        );
    }
  };

  const getSeverityBadge = (severity: "low" | "moderate" | "high") => {
    switch (severity) {
      case "high":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-500/20 text-rose-300 border border-rose-500/30">
            High Severity Signal
          </span>
        );
      case "moderate":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30">
            Moderate Signal
          </span>
        );
      case "low":
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            Low Variation
          </span>
        );
    }
  };

  const getTrendBadge = (trend?: string) => {
    switch (trend) {
      case "increasing":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            Upward / Increasing Signal
          </span>
        );
      case "decreasing":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
            </svg>
            Declining / Decreasing Signal
          </span>
        );
      case "stable":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
            </svg>
            Stable Trajectory
          </span>
        );
      case "insufficient_data":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-400">
            <span>—</span>
            Insufficient Temporal Data
          </span>
        );
    }
  };

  return (
    <section id="field-intelligence" className="mt-12 space-y-6">
      {/* Section Header with farm background image */}
      <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl border border-slate-700 shadow-xl overflow-hidden">
        {/* Background image */}
        <div
          className="absolute inset-0 bg-cover bg-center scale-105"
          style={{ backgroundImage: "url('/bg-field-rows.jpg')" }}
        />
        {/* Dark overlay for readability */}
        <div className="absolute inset-0 bg-gradient-to-r from-slate-950/95 via-slate-900/88 to-indigo-950/75" />
        <div className="relative z-10 space-y-1">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 font-bold text-base shadow-inner">
              ✦
            </div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-3">
              <span>FIELD INTELLIGENCE</span>
              <span className="text-xs uppercase tracking-widest px-2.5 py-0.5 rounded-full font-mono font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Explainable AI Layer
              </span>
            </h2>
          </div>
          <p className="text-sm text-slate-400 max-w-2xl">
            Synthesizes calculated Sentinel-2 multi-spectral indices (NDVI, NDMI, NDRE, EVI), spatial vigor zones, and temporal trajectories into an objective, evidence-grounded agronomic synthesis.
          </p>
        </div>

        {/* Action Button */}
        <div className="relative z-10">
          {!report && !loading && (
            <button
              onClick={onGenerate}
              disabled={!hasAnalyticalData || loading}
              className={`px-5 py-2.5 rounded-xl font-medium text-sm flex items-center gap-2 transition-all shadow-lg ${
                hasAnalyticalData
                  ? "bg-gradient-to-r from-indigo-600 to-emerald-600 hover:from-indigo-500 hover:to-emerald-500 text-white shadow-indigo-500/20 hover:shadow-indigo-500/30 active:scale-95"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
              }`}
            >
              <span>✦</span>
              <span>Generate Field Intelligence</span>
            </button>
          )}

          {report && !loading && (
            <button
              onClick={onRegenerate}
              disabled={loading}
              className="px-4 py-2 rounded-xl text-xs font-medium text-indigo-300 hover:text-white bg-indigo-950/50 hover:bg-indigo-900/60 border border-indigo-800/60 transition-all flex items-center gap-2 shadow"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Regenerate Analysis</span>
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="p-8 rounded-2xl bg-slate-900/90 border border-indigo-500/30 shadow-2xl backdrop-blur-sm flex flex-col items-center justify-center text-center space-y-4">
          <div className="relative">
            <div className="w-14 h-14 rounded-full border-4 border-indigo-500/20 border-t-indigo-400 animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center text-indigo-400 text-sm font-bold">
              ✦
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-base font-semibold text-white tracking-wide">{loadingStage || "Generating Field Intelligence..."}</p>
            <p className="text-xs text-slate-400">
              Correlating calculated spectral bands against verified agronomic principles without fabrication.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] font-mono text-slate-500">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Evidence-based reasoning active</span>
          </div>
        </div>
      )}

      {/* Error / Provider Failure State */}
      {error && !loading && (
        <div className="p-6 rounded-2xl bg-slate-900 border border-rose-500/30 shadow-xl space-y-4">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center justify-center shrink-0">
              ⚠️
            </div>
            <div className="space-y-1">
              <h3 className="text-sm font-bold text-white">AI interpretation is currently unavailable</h3>
              <p className="text-xs text-slate-400">
                {error || "The AI interpretation engine encountered an error or reached its rate limit. All calculated mathematical index values, vegetation maps, and condition scores remain fully operational."}
              </p>
            </div>
          </div>
          <div className="pt-2 flex items-center gap-3">
            <button
              onClick={scrollToAnalyticalResults}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition"
            >
              View Analytical Results
            </button>
            <button
              onClick={onRegenerate}
              className="px-4 py-2 rounded-lg bg-indigo-900/40 hover:bg-indigo-900/60 text-indigo-300 text-xs font-semibold border border-indigo-800 transition"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Rendered Analytical Report */}
      {report && !loading && (
        <div className="space-y-6">
          {/* Demo Mode Notice removed — banner hidden in production UI */}

          {/* 1. FIELD SUMMARY */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
              <div className="space-y-1">
                <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-400 font-mono">
                  FIELD SUMMARY
                </span>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  {report.headline}
                </h3>
              </div>
              <div>{getConfidenceBadge(report.confidence)}</div>
            </div>

            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              {report.summary}
            </p>

            {report.generated_at && (
              <div className="pt-2 text-[11px] text-slate-500 font-mono flex items-center justify-between">
                <span>Generated: {new Date(report.generated_at).toLocaleString()}</span>
                {report.provider_used && (
                  <span className="text-slate-600">Engine: {report.provider_used}</span>
                )}
              </div>
            )}
          </div>

          {/* 2-Column Grid for Observations & Patterns */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 2. KEY OBSERVATIONS */}
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 font-mono">
                  KEY OBSERVATIONS
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {report.observations.length} Identified
                </span>
              </div>

              <div className="space-y-4">
                {report.observations.map((obs, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2.5"
                  >
                    <h4 className="text-sm font-semibold text-slate-200">
                      {obs.title}
                    </h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {obs.description}
                    </p>
                    {obs.evidence && obs.evidence.length > 0 && (
                      <div className="pt-1 flex flex-wrap gap-1.5">
                        {obs.evidence.map((ev, evIdx) => (
                          <span
                            key={evIdx}
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-mono font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/20"
                          >
                            <span className="text-emerald-500">📊</span>
                            <span>{ev}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 3. POSSIBLE PATTERNS */}
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-400 font-mono">
                  POSSIBLE PATTERNS
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Agronomic Hypotheses
                </span>
              </div>

              <div className="space-y-4">
                {report.possible_patterns.map((pat, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="text-sm font-semibold text-slate-200">
                        {pat.pattern}
                      </h4>
                      {getSeverityBadge(pat.severity)}
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed">
                      {pat.explanation}
                    </p>

                    <div className="pt-1 flex items-center gap-2">
                      <span className="text-[10px] uppercase font-bold text-slate-500 font-mono">
                        Evidence:
                      </span>
                      <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                        {pat.evidence}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 3-Column Grid: Historical Signal, Areas To Watch, Limitations */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 4. HISTORICAL SIGNAL */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg flex flex-col justify-between space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-400 font-mono">
                    HISTORICAL SIGNAL
                  </span>
                </div>
                <div>{getTrendBadge(report.historical_interpretation?.trend)}</div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {report.historical_interpretation?.summary ||
                    "Single-date satellite acquisition analyzed. Multi-temporal monitoring recommended to observe historical progression."}
                </p>
              </div>
            </div>

            {/* 5. AREAS TO WATCH */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-3">
              <div className="border-b border-slate-800 pb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-amber-400 font-mono">
                  AREAS TO WATCH
                </span>
              </div>
              <div className="space-y-2.5">
                {report.areas_to_watch.map((item, idx) => (
                  <div key={idx} className="space-y-0.5">
                    <p className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                      <span>{item.description}</span>
                    </p>
                    <p className="text-[11px] text-slate-400 pl-3">
                      {item.reason}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* 6. DATA LIMITATIONS */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-3">
              <div className="border-b border-slate-800 pb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 font-mono">
                  DATA LIMITATIONS
                </span>
              </div>
              <div className="space-y-2 text-xs">
                <p className="text-slate-300 leading-snug">
                  {report.data_quality.assessment}
                </p>
                {report.data_quality.limitations && report.data_quality.limitations.length > 0 && (
                  <ul className="space-y-1 list-disc list-inside text-[11px] text-slate-400">
                    {report.data_quality.limitations.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* 7. NEXT STEPS (Non-prescriptive Ground Verification) */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 font-mono flex items-center gap-2">
                <span>📋</span>
                <span>RECOMMENDED NEXT STEPS (NON-PRESCRIPTIVE GROUND SCOUTING)</span>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
              {report.recommended_next_steps.map((step, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex items-start gap-2.5 text-xs text-slate-300 leading-relaxed"
                >
                  <span className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center justify-center font-mono font-bold shrink-0 text-[11px]">
                    {idx + 1}
                  </span>
                  <span>{step}</span>
                </div>
              ))}
            </div>

            <div className="pt-2 text-[11px] text-slate-500 flex items-center gap-1.5 italic">
              <span>ℹ️</span>
              <span>
                Scientific guardrail: Optical satellite indices evaluate electromagnetic canopy reflectance and do not constitute physical diagnostic lab tests or chemical application prescriptions.
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
