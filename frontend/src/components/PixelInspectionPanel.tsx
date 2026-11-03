"use client";

import React from "react";

export interface PixelInspectionData {
  status: string;
  is_valid_pixel: boolean;
  pixel_coords: { row: number; col: number };
  geo_coords: { latitude: number; longitude: number };
  risk_category: string;
  risk_color_hex: string;
  stress_class?: string | null;
  stress_probability?: number | null;
  vegetation_indices: Record<string, number | null>;
  sar_backscatter: Record<string, number | null>;
  historical_change_delta?: number | null;
  growth_trend_slope?: number | null;
  quality_status: string;
  evidence_summary: string;
  scientific_disclaimer: string;
}

interface PixelInspectionPanelProps {
  inspection: PixelInspectionData | null;
  isLoading?: boolean;
  onClear?: () => void;
}

export default function PixelInspectionPanel({
  inspection,
  isLoading = false,
  onClear,
}: PixelInspectionPanelProps) {
  if (isLoading) {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl flex flex-col items-center justify-center text-center space-y-3 min-h-[220px]">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-slate-300 font-medium">
          Retrieving radiometric pixel data and evaluating spatial evidence...
        </span>
        <span className="text-xs text-slate-400">
          Querying canonical 10m UTM grid & multimodal feature fusion
        </span>
      </div>
    );
  }

  if (!inspection) {
    return (
      <div className="bg-slate-900/90 rounded-2xl border border-dashed border-slate-800 p-6 backdrop-blur-md shadow-xl flex flex-col items-center justify-center text-center space-y-2 min-h-[180px]">
        <span className="text-2xl">🎯</span>
        <h5 className="text-sm font-semibold text-slate-200">
          Interactive Pixel Inspection Crosshair
        </h5>
        <p className="text-xs text-slate-400 max-w-sm">
          Click any point within the field boundary to inspect actual satellite spectral bands,
          radar backscatter, calibrated stress probability, and classification evidence.
        </p>
      </div>
    );
  }

  const indices = inspection.vegetation_indices || {};
  const sar = inspection.sar_backscatter || {};

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 backdrop-blur-md shadow-xl space-y-4">
      {/* Header with Risk Category & Clear Button */}
      <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-3.5 h-3.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: inspection.risk_color_hex }}
            />
            <h4 className="text-base font-bold text-white tracking-tight">
              {inspection.risk_category}
            </h4>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 font-mono">
            <span>
              Lat: {inspection.geo_coords.latitude.toFixed(6)}°, Lon:{" "}
              {inspection.geo_coords.longitude.toFixed(6)}°
            </span>
            <span>•</span>
            <span className="text-slate-300">
              Grid: [{inspection.pixel_coords.row}, {inspection.pixel_coords.col}]
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
              inspection.is_valid_pixel
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                : "bg-slate-800 border-slate-700 text-slate-400"
            }`}
          >
            {inspection.quality_status}
          </span>
          {onClear && (
            <button
              onClick={onClear}
              type="button"
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
              title="Clear selection"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Transparent Evidence Box */}
      <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
        <span className="text-[10px] font-semibold tracking-wider text-emerald-400 uppercase block">
          AI Evidence & Rationale
        </span>
        <p className="text-xs text-slate-300 leading-relaxed font-sans">
          {inspection.evidence_summary}
        </p>
      </div>

      {/* Actual Index Values Grid */}
      <div className="space-y-1.5">
        <span className="text-[11px] font-semibold text-slate-400 block px-0.5">
          Measured Satellite Spectral Indices
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            { label: "NDVI", val: indices.ndvi ?? indices.NDVI, color: "text-emerald-400" },
            { label: "NDMI", val: indices.ndmi ?? indices.NDMI, color: "text-sky-400" },
            { label: "NDRE", val: indices.ndre ?? indices.NDRE, color: "text-teal-400" },
            { label: "EVI", val: indices.evi ?? indices.EVI, color: "text-emerald-300" },
          ].map((item) => (
            <div
              key={item.label}
              className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800 text-center"
            >
              <span className="text-[10px] text-slate-400 block font-medium">
                {item.label}
              </span>
              <span className={`text-base font-bold font-mono ${item.color}`}>
                {item.val !== null && item.val !== undefined ? item.val.toFixed(3) : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Multimodal Radar & Model Insights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
        {/* Sentinel-1 SAR Backscatter */}
        <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800 space-y-1.5">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
            Sentinel-1 SAR Radar
          </span>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>VV Polarized:</span>
            <span className="text-white font-semibold">
              {sar.vv !== null && sar.vv !== undefined ? `${sar.vv.toFixed(1)} dB` : "N/A"}
            </span>
          </div>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>VH Cross-Pol:</span>
            <span className="text-white font-semibold">
              {sar.vh !== null && sar.vh !== undefined ? `${sar.vh.toFixed(1)} dB` : "N/A"}
            </span>
          </div>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>VV/VH Ratio:</span>
            <span className="text-white font-semibold">
              {sar.vv_vh_ratio !== null && sar.vv_vh_ratio !== undefined
                ? `${sar.vv_vh_ratio.toFixed(2)} dB`
                : "N/A"}
            </span>
          </div>
        </div>

        {/* Temporal and Trend Dynamics */}
        <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800 space-y-1.5">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
            Model & Temporal Dynamics
          </span>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>Stress Probability:</span>
            <span className="text-amber-400 font-semibold">
              {inspection.stress_probability !== null && inspection.stress_probability !== undefined
                ? `${(inspection.stress_probability * 100).toFixed(1)}%`
                : "Rule-Based"}
            </span>
          </div>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>Historical Δ:</span>
            <span
              className={`font-semibold ${
                (inspection.historical_change_delta ?? 0) < -0.05
                  ? "text-rose-400"
                  : "text-emerald-400"
              }`}
            >
              {inspection.historical_change_delta !== null &&
              inspection.historical_change_delta !== undefined
                ? inspection.historical_change_delta.toFixed(3)
                : "N/A"}
            </span>
          </div>
          <div className="flex justify-between font-mono text-[11px] text-slate-300">
            <span>Trajectory Slope:</span>
            <span className="text-slate-200 font-semibold">
              {inspection.growth_trend_slope !== null && inspection.growth_trend_slope !== undefined
                ? `${(inspection.growth_trend_slope * 1000).toFixed(2)}/1k d`
                : "N/A"}
            </span>
          </div>
        </div>
      </div>

      {/* Non-causation agricultural warning */}
      <div className="text-[10px] text-slate-400/80 leading-relaxed border-t border-slate-800/80 pt-2 flex items-start gap-1.5">
        <span className="text-amber-400 flex-shrink-0">⚠️</span>
        <span>
          {inspection.scientific_disclaimer}
        </span>
      </div>
    </div>
  );
}
