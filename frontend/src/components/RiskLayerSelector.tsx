"use client";

import React from "react";

export interface LayerOption {
  key: string;
  label: string;
  category: "Risk & Stress" | "Vegetation Indices" | "Temporal & Quality";
  icon: string;
  description: string;
  badge?: string;
}

export const RISK_LAYER_OPTIONS: LayerOption[] = [
  {
    key: "risk_classification",
    label: "AI Risk Classification",
    category: "Risk & Stress",
    icon: "🛡️",
    description: "Multi-criteria spatial risk categories (Low, Moderate, Elevated, Insufficient Evidence)",
    badge: "Core AI",
  },
  {
    key: "potential_stress",
    label: "Stress Probability",
    category: "Risk & Stress",
    icon: "⚡",
    description: "Multimodal calibrated early stress probability model inference",
  },
  {
    key: "ndvi",
    label: "NDVI (Vigor)",
    category: "Vegetation Indices",
    icon: "🌿",
    description: "Normalized Difference Vegetation Index — overall canopy greenness and density",
  },
  {
    key: "ndmi",
    label: "NDMI (Moisture)",
    category: "Vegetation Indices",
    icon: "💧",
    description: "Normalized Difference Moisture Index — canopy water content and hydration",
  },
  {
    key: "ndre",
    label: "NDRE (Chlorophyll)",
    category: "Vegetation Indices",
    icon: "🔬",
    description: "Normalized Difference Red Edge — early nitrogen and chlorophyll status",
  },
  {
    key: "evi",
    label: "EVI (Canopy)",
    category: "Vegetation Indices",
    icon: "🍃",
    description: "Enhanced Vegetation Index — high-biomass structural canopy density",
  },
  {
    key: "historical_change",
    label: "Historical Change (Δ)",
    category: "Temporal & Quality",
    icon: "🔄",
    description: "Pixel-level temporal difference against baseline satellite acquisition",
  },
  {
    key: "growth_trend",
    label: "Growth Trend",
    category: "Temporal & Quality",
    icon: "📈",
    description: "Spatial trajectory rate of change over the observation period",
  },
  {
    key: "data_quality",
    label: "Data Quality & Clouds",
    category: "Temporal & Quality",
    icon: "☁️",
    description: "Valid canopy coverage, cloud contamination, and nodata masking",
  },
];

interface RiskLayerSelectorProps {
  selectedLayer: string;
  onSelectLayer: (layerKey: string) => void;
  availableLayerKeys?: string[];
}

export default function RiskLayerSelector({
  selectedLayer,
  onSelectLayer,
  availableLayerKeys,
}: RiskLayerSelectorProps) {
  const categories = ["Risk & Stress", "Vegetation Indices", "Temporal & Quality"] as const;

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-4 backdrop-blur-md shadow-xl">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🗺️</span>
          <h4 className="text-sm font-semibold text-white tracking-wide">
            Spatial Intelligence Layers
          </h4>
        </div>
        <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
          Canonical 10m UTM
        </span>
      </div>

      <div className="space-y-3.5">
        {categories.map((cat) => {
          const options = RISK_LAYER_OPTIONS.filter((opt) => opt.category === cat);
          return (
            <div key={cat} className="space-y-1.5">
              <span className="text-[10px] font-semibold tracking-wider text-slate-400 uppercase px-1">
                {cat}
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-1.5">
                {options.map((opt) => {
                  const isActive = selectedLayer === opt.key;
                  const isAvailable =
                    !availableLayerKeys || availableLayerKeys.includes(opt.key);

                  return (
                    <button
                      key={opt.key}
                      type="button"
                      disabled={!isAvailable}
                      onClick={() => onSelectLayer(opt.key)}
                      title={opt.description}
                      className={`relative flex items-center gap-2 px-3 py-2 rounded-xl text-left transition-all duration-200 border text-xs ${
                        isActive
                          ? "bg-emerald-500/15 border-emerald-500/50 text-white shadow-lg shadow-emerald-950/40"
                          : isAvailable
                          ? "bg-slate-950/60 border-slate-800 text-slate-300 hover:bg-slate-800/60 hover:border-slate-700"
                          : "bg-slate-950/30 border-slate-900 text-slate-600 cursor-not-allowed opacity-50"
                      }`}
                    >
                      <span className="text-sm flex-shrink-0">{opt.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`font-medium truncate ${
                              isActive ? "text-emerald-300 font-semibold" : "text-slate-200"
                            }`}
                          >
                            {opt.label}
                          </span>
                          {opt.badge && (
                            <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                              {opt.badge}
                            </span>
                          )}
                        </div>
                      </div>
                      {isActive && (
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse flex-shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
