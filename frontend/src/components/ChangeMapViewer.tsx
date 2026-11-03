"use client";

import { useState } from "react";
import Image from "next/image";

export interface ComparisonObservation {
  observation_id: string;
  date: string;
  cloud_coverage: number;
  valid_pixel_percentage: number;
  quality: string;
  preview_map?: string;
}

export interface DeltaInfo {
  value_a: number | null;
  value_b: number | null;
  absolute_change: number;
  percentage_change: number;
}

export interface ChangeBreakdown {
  lower_signal_pct: number;
  minimal_change_pct: number;
  higher_signal_pct: number;
  valid_pixels: number;
}

interface ChangeMapViewerProps {
  observationA: ComparisonObservation;
  observationB: ComparisonObservation;
  deltas: Record<string, DeltaInfo>;
  changeMapUrl: string;
  changeBreakdown: ChangeBreakdown;
  onClose?: () => void;
}

export default function ChangeMapViewer({
  observationA,
  observationB,
  deltas,
  changeMapUrl,
  changeBreakdown,
  onClose,
}: ChangeMapViewerProps) {
  const [activeTab, setActiveTab] = useState<"change_map" | "side_by_side">("change_map");

  return (
    <div className="w-full bg-slate-900/90 rounded-2xl border border-slate-800 p-5 sm:p-7 shadow-2xl backdrop-blur-md space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>🔄 Pairwise Temporal Comparison</span>
          </div>
          <h3 className="text-xl font-bold text-white tracking-tight">
            Observation A ({observationA.date}) vs Observation B ({observationB.date})
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Spatial vegetation index signal difference between the two selected Sentinel-2 acquisitions.
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab("change_map")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              activeTab === "change_map"
                ? "bg-emerald-500 text-slate-950 font-semibold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            🗺️ Spatial Change Map
          </button>
          <button
            onClick={() => setActiveTab("side_by_side")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              activeTab === "side_by_side"
                ? "bg-emerald-500 text-slate-950 font-semibold shadow"
                : "text-slate-400 hover:text-white"
            }`}
          >
            👁️ Side-by-Side
          </button>
        </div>
      </div>

      {/* Numerical Deltas Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {Object.entries(deltas).map(([idxKey, info]) => {
          const isPositive = info.absolute_change > 0;
          const isNegative = info.absolute_change < 0;

          return (
            <div
              key={idxKey}
              className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1 text-center sm:text-left"
            >
              <span className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider">
                Δ {idxKey.toUpperCase()}
              </span>
              <div className="flex items-baseline justify-center sm:justify-start gap-1.5">
                <span
                  className={`text-lg font-bold font-mono ${
                    isPositive
                      ? "text-emerald-400"
                      : isNegative
                      ? "text-rose-400"
                      : "text-slate-300"
                  }`}
                >
                  {info.absolute_change > 0 ? `+${info.absolute_change.toFixed(3)}` : info.absolute_change.toFixed(3)}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  ({info.percentage_change > 0 ? `+${info.percentage_change.toFixed(1)}%` : `${info.percentage_change.toFixed(1)}%`})
                </span>
              </div>
              <div className="text-[11px] text-slate-500 flex justify-between pt-1">
                <span>A: {info.value_a !== null ? info.value_a.toFixed(2) : "-"}</span>
                <span>B: {info.value_b !== null ? info.value_b.toFixed(2) : "-"}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Visual Display */}
      {activeTab === "change_map" ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* Change Map Image Display */}
            <div className="md:col-span-2 relative w-full aspect-square max-h-[460px] rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center shadow-inner">
              {changeMapUrl ? (
                <img
                  src={changeMapUrl}
                  alt="NDVI Spatial Change Map"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-sm text-slate-500">Loading change map...</div>
              )}
            </div>

            {/* Change Legend & Breakdown */}
            <div className="space-y-4 p-5 rounded-xl bg-slate-950/70 border border-slate-800">
              <h4 className="text-sm font-semibold text-white tracking-tight border-b border-slate-800 pb-2">
                Spatial Shift Distribution
              </h4>

              <div className="space-y-3 text-xs">
                {/* Higher Signal */}
                <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-500/20 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 font-medium text-emerald-300">
                      <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                      Higher vegetation index signal
                    </span>
                    <span className="font-mono font-bold text-emerald-400">
                      {changeBreakdown.higher_signal_pct.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Δ NDVI &gt; +0.10 (increased canopy greenness / chlorophyll absorption)
                  </p>
                </div>

                {/* Minimal Change */}
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-700 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 font-medium text-slate-300">
                      <span className="w-2.5 h-2.5 rounded-sm bg-slate-400" />
                      Minimal change
                    </span>
                    <span className="font-mono font-bold text-slate-300">
                      {changeBreakdown.minimal_change_pct.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    -0.10 ≤ Δ NDVI ≤ +0.10 (stable vegetation condition)
                  </p>
                </div>

                {/* Lower Signal */}
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/20 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 font-medium text-rose-300">
                      <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
                      Lower vegetation index signal
                    </span>
                    <span className="font-mono font-bold text-rose-400">
                      {changeBreakdown.lower_signal_pct.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Δ NDVI &lt; -0.10 (reduced canopy greenness / senescence / harvest)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Side by Side Preview */
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-sky-400">Observation A</span>
              <span className="text-slate-400">📅 {observationA.date}</span>
            </div>
            <div className="relative aspect-square max-h-[340px] rounded-lg overflow-hidden bg-slate-900 border border-slate-800 flex items-center justify-center">
              {observationA.preview_map ? (
                <img src={observationA.preview_map} alt="Obs A" className="w-full h-full object-contain" />
              ) : (
                <span className="text-xs text-slate-500">Preview not available</span>
              )}
            </div>
            <div className="flex justify-between text-xs text-slate-400 pt-1">
              <span>Cloud: {observationA.cloud_coverage.toFixed(1)}%</span>
              <span>Valid: {observationA.valid_pixel_percentage.toFixed(1)}%</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-amber-400">Observation B</span>
              <span className="text-slate-400">📅 {observationB.date}</span>
            </div>
            <div className="relative aspect-square max-h-[340px] rounded-lg overflow-hidden bg-slate-900 border border-slate-800 flex items-center justify-center">
              {observationB.preview_map ? (
                <img src={observationB.preview_map} alt="Obs B" className="w-full h-full object-contain" />
              ) : (
                <span className="text-xs text-slate-500">Preview not available</span>
              )}
            </div>
            <div className="flex justify-between text-xs text-slate-400 pt-1">
              <span>Cloud: {observationB.cloud_coverage.toFixed(1)}%</span>
              <span>Valid: {observationB.valid_pixel_percentage.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Cautious Scientific Notice */}
      <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-400 flex items-start gap-2.5 leading-relaxed">
        <span className="text-sm">⚠️</span>
        <div>
          <span className="font-semibold text-slate-300">Scientific Interpretation Notice: </span>
          Vegetation index difference maps depict relative shifts in spectral reflectance between two satellite acquisitions.
          Reductions in signal (&lt; -0.10) do not definitively demonstrate crop damage or mortality, and may stem from
          normal crop maturity, harvesting, tillage, or moisture depletion. Similarly, positive index gains may reflect crop
          biomass accumulation, weed emergence, or recent irrigation/rainfall.
        </div>
      </div>
    </div>
  );
}
