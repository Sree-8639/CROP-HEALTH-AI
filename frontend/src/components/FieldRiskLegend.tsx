"use client";

import React from "react";

interface FieldRiskLegendProps {
  layerKey: string;
  legendData?: Record<string, any>;
}

export default function FieldRiskLegend({
  layerKey,
  legendData,
}: FieldRiskLegendProps) {
  if (layerKey === "risk_classification") {
    const items = [
      {
        label: "Low Potential Risk",
        color: "#10b981",
        description: "Nominal canopy vigor, stress prob < 35%, stable trajectory",
      },
      {
        label: "Moderate Potential Risk",
        color: "#f59e0b",
        description: "Stress prob 35–65% or moderate moisture/chlorophyll reduction",
      },
      {
        label: "Elevated Potential Risk",
        color: "#e11d48",
        description: "Stress prob ≥ 65%, severe water deficit (NDMI < 0), or sharp drop",
      },
      {
        label: "Insufficient Evidence",
        color: "#64748b",
        description: "Cloud cover, cloud shadow, water, or sensor boundary limits",
      },
    ];

    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>🛡️</span> AI Risk Classification Legend
          </span>
          <span className="text-[10px] text-slate-400">4-Tier System</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
          {items.map((item) => (
            <div
              key={item.label}
              className="flex items-start gap-2.5 p-2 rounded-xl bg-slate-950/50 border border-slate-800/80"
            >
              <div
                className="w-3.5 h-3.5 rounded-md flex-shrink-0 mt-0.5 shadow-sm"
                style={{ backgroundColor: item.color }}
              />
              <div className="min-w-0">
                <span className="font-semibold text-slate-200 block text-[11px]">
                  {item.label}
                </span>
                <span className="text-[10px] text-slate-400 leading-tight block">
                  {item.description}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (layerKey === "potential_stress") {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>⚡</span> Calibrated Stress Probability Scale
          </span>
          <span className="text-[10px] text-slate-400">Model Inference (0%–100%)</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-600 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>0% (Low Stress)</span>
            <span>35% (Moderate)</span>
            <span>65% (Elevated)</span>
            <span>100% (High Stress)</span>
          </div>
        </div>
      </div>
    );
  }

  if (["ndvi", "evi"].includes(layerKey)) {
    const isNdvi = layerKey === "ndvi";
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>{isNdvi ? "🌿" : "🍃"}</span> {isNdvi ? "NDVI" : "EVI"} Canopy Density
          </span>
          <span className="text-[10px] text-slate-400">Index Range [-0.1 to 1.0]</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-emerald-600 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>&lt; 0.1 (Bare/Sparse)</span>
            <span>0.4 (Moderate Biomass)</span>
            <span>&gt; 0.75 (Dense Canopy)</span>
          </div>
        </div>
      </div>
    );
  }

  if (layerKey === "ndmi") {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>💧</span> NDMI Canopy Water Content
          </span>
          <span className="text-[10px] text-slate-400">Moisture Spectrum [-0.2 to 0.6]</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-amber-200 via-sky-400 to-blue-700 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>&lt; 0.0 (Water Stress)</span>
            <span>0.2 (Adequate Moisture)</span>
            <span>&gt; 0.45 (High Hydration)</span>
          </div>
        </div>
      </div>
    );
  }

  if (layerKey === "ndre") {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>🔬</span> NDRE Red-Edge Chlorophyll
          </span>
          <span className="text-[10px] text-slate-400">Chlorophyll / Nitrogen Status</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-yellow-300 via-emerald-400 to-teal-800 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>&lt; 0.15 (Low Nitrogen)</span>
            <span>0.3 (Standard)</span>
            <span>&gt; 0.5 (Optimal Chlorophyll)</span>
          </div>
        </div>
      </div>
    );
  }

  if (layerKey === "historical_change") {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>🔄</span> Temporal Comparison Delta (Δ)
          </span>
          <span className="text-[10px] text-slate-400">Pixel Difference vs Baseline</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-rose-500 via-amber-300 to-emerald-500 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>Δ &lt; -0.10 (Decreasing)</span>
            <span>-0.05 to +0.05 (Stable)</span>
            <span>Δ &gt; +0.10 (Increasing)</span>
          </div>
        </div>
      </div>
    );
  }

  if (layerKey === "growth_trend") {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
            <span>📈</span> Spatial Growth Trajectory
          </span>
          <span className="text-[10px] text-slate-400">Slope Rate per Day</span>
        </div>

        <div className="space-y-1.5">
          <div className="h-3 rounded-full bg-gradient-to-r from-amber-600 via-yellow-300 to-emerald-500 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>Negative Trend</span>
            <span>Steady / Nominal</span>
            <span>Vigorous Expansion</span>
          </div>
        </div>
      </div>
    );
  }

  // Fallback: Data Quality
  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl space-y-2.5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="text-xs font-semibold text-white tracking-wide flex items-center gap-1.5">
          <span>☁️</span> Data Quality & Masking
        </span>
        <span className="text-[10px] text-slate-400">Quality Filter</span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="w-3.5 h-3.5 rounded-md bg-emerald-500 shadow-sm" />
          <span className="text-slate-200 text-[11px] font-medium">Valid Canopy (Quality &gt; 0)</span>
        </div>
        <div className="flex items-center gap-2 p-2 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="w-3.5 h-3.5 rounded-md bg-rose-500 shadow-sm" />
          <span className="text-slate-200 text-[11px] font-medium">Cloud / Shadow / NoData</span>
        </div>
      </div>
    </div>
  );
}
