"use client";

import React from "react";

export interface DataQualityAssessmentData {
  quality_tier: string; // "Good" | "Moderate" | "Limited" | "Insufficient"
  cloud_coverage: number;
  valid_pixel_pct: number;
  nodata_pct: number;
  observation_count: number;
  temporal_gap_days: number | null;
  s1_available: boolean;
  s2_available: boolean;
  missing_features_count: number;
  score_factors?: Record<string, any>;
  data_limitations: string[];
}

export interface DataQualityPanelProps {
  dataQuality: DataQualityAssessmentData;
}

export default function DataQualityPanel({ dataQuality }: DataQualityPanelProps) {
  const {
    quality_tier,
    cloud_coverage,
    valid_pixel_pct,
    nodata_pct,
    observation_count,
    temporal_gap_days,
    s1_available,
    s2_available,
    missing_features_count,
    data_limitations,
  } = dataQuality;

  const getTierStyling = (tier: string) => {
    switch (tier.toLowerCase()) {
      case "good":
        return {
          bg: "bg-emerald-500/15",
          border: "border-emerald-500/50",
          text: "text-emerald-400",
          badgeBg: "bg-emerald-500",
          badgeText: "text-slate-950",
          icon: "✓",
          summary: "High radiometric clarity and complete multimodal sensor coverage.",
        };
      case "moderate":
        return {
          bg: "bg-amber-500/15",
          border: "border-amber-500/50",
          text: "text-amber-400",
          badgeBg: "bg-amber-500",
          badgeText: "text-slate-950",
          icon: "!",
          summary: "Acceptable observational evidence with moderate clouds or historical spacing.",
        };
      case "limited":
        return {
          bg: "bg-orange-500/15",
          border: "border-orange-500/50",
          text: "text-orange-400",
          badgeBg: "bg-orange-500",
          badgeText: "text-white",
          icon: "⚠",
          summary: "Degraded evidence; missing SAR, elevated cloud interference, or low sample density.",
        };
      case "insufficient":
      default:
        return {
          bg: "bg-rose-500/15",
          border: "border-rose-500/50",
          text: "text-rose-400",
          badgeBg: "bg-rose-500",
          badgeText: "text-white",
          icon: "✕",
          summary: "Inadequate valid canopy pixels or observations to support reliable inference.",
        };
    }
  };

  const style = getTierStyling(quality_tier);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Title & Quality Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">📡</span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              Observational Evidence & Data Quality
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            4-Tier automated evaluation of satellite sensor fidelity, cloud
            obscuration, and temporal completeness.
          </p>
        </div>

        <div className={`px-4 py-2 rounded-xl border ${style.bg} ${style.border} flex items-center gap-3`}>
          <span className={`w-7 h-7 rounded-lg ${style.badgeBg} ${style.badgeText} font-black flex items-center justify-center text-sm`}>
            {style.icon}
          </span>
          <div>
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block">
              Evidence Quality Tier
            </span>
            <span className={`text-base font-black tracking-tight ${style.text}`}>
              {quality_tier.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Grid of 6 Key Observational Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {/* Valid Pixel Percentage */}
        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between">
          <span className="text-xs text-slate-400">Valid Pixel Area</span>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {valid_pixel_pct.toFixed(1)}%
            </span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1">
            NoData / SCL Mask: {nodata_pct.toFixed(1)}%
          </span>
        </div>

        {/* Cloud Coverage */}
        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between">
          <span className="text-xs text-slate-400">Cloud Coverage</span>
          <div className="mt-1 flex items-baseline gap-1">
            <span
              className={`text-2xl font-bold font-mono ${
                cloud_coverage > 20
                  ? "text-rose-400"
                  : cloud_coverage > 10
                  ? "text-amber-400"
                  : "text-emerald-400"
              }`}
            >
              {cloud_coverage.toFixed(1)}%
            </span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1">
            {cloud_coverage <= 10 ? "Clear atmospheric path" : "Atmospheric attenuation"}
          </span>
        </div>

        {/* Historical Observations */}
        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between">
          <span className="text-xs text-slate-400">Time-Series Depth</span>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-bold font-mono text-cyan-400">
              {observation_count}
            </span>
            <span className="text-xs text-slate-400">passes</span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1">
            {observation_count >= 6
              ? "Dense multi-temporal context"
              : observation_count >= 2
              ? "Minimal sequential baseline"
              : "Single pass / no history"}
          </span>
        </div>

        {/* S1 + S2 Sensor Status */}
        <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col justify-between">
          <span className="text-xs text-slate-400">Sensors Active</span>
          <div className="mt-1 flex items-center gap-1.5 flex-wrap">
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                s2_available
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "bg-slate-800 text-slate-500"
              }`}
            >
              {s2_available ? "✓ Sentinel-2" : "✗ S2 Missing"}
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                s1_available
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                  : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
              }`}
            >
              {s1_available ? "✓ Sentinel-1" : "✗ S1 Missing"}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1">
            {temporal_gap_days !== null
              ? `Temporal gap: ±${temporal_gap_days.toFixed(1)}d`
              : "No SAR temporal match"}
          </span>
        </div>
      </div>

      {/* Feature Completeness & Missing Attributes */}
      <div className="p-3.5 rounded-xl bg-slate-950/50 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Multimodal Feature Completeness:</span>
          <span className="font-mono font-bold text-white">
            {37 - missing_features_count} / 37 features
          </span>
          {missing_features_count > 0 && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              {missing_features_count} missing (imputed / masked)
            </span>
          )}
        </div>

        <span className="text-slate-500 text-[11px]">
          Evaluated via Multimodal Satellite Quality Schema
        </span>
      </div>

      {/* Transparent Data Limitations List */}
      {data_limitations && data_limitations.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-semibold text-slate-300 block">
            Observed Data Limitations in Current Window:
          </span>
          <ul className="space-y-1.5 pl-1">
            {data_limitations.map((lim, idx) => (
              <li
                key={idx}
                className="text-xs text-amber-300/90 flex items-start gap-2 bg-amber-950/20 border border-amber-900/30 p-2 rounded-lg"
              >
                <span className="text-amber-400 font-bold">•</span>
                <span>{lim}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Scientific Clarification Notice */}
      <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
        <strong className="text-slate-300">Important Scientific Clarification:</strong>{" "}
        Data quality evaluates observational integrity, sensor availability, and cloud masking.
        It must <em>never</em> be conflated with or described as machine-learning model accuracy.
        A "Good" data quality status indicates clean satellite evidence, not a guarantee that the underlying model's predictions are correct.
      </div>
    </div>
  );
}
