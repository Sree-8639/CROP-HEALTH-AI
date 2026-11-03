"use client";

import React from "react";
import Link from "next/link";

export interface FieldRiskCategoryItemData {
  category_name: string;
  area_hectares: number;
  percentage_of_field: number;
  pixel_count: number;
  color_hex: string;
  description: string;
}

export interface FieldRiskSummaryData {
  analysis_id?: string | null;
  total_area_hectares: number;
  valid_canopy_hectares: number;
  valid_canopy_percentage: number;
  nodata_percentage: number;
  stress_signal_coverage_pct: number;
  risk_distribution: Record<string, FieldRiskCategoryItemData>;
  canonical_crs: string;
  spatial_resolution_m: number;
  available_layer_keys: string[];
  has_geotiff_export: boolean;
}

interface FieldRiskSummaryProps {
  fieldRisk: FieldRiskSummaryData;
  isDemo?: boolean;
}

export default function FieldRiskSummary({
  fieldRisk,
  isDemo = false,
}: FieldRiskSummaryProps) {
  const dist = fieldRisk.risk_distribution || {};
  const categories = Object.values(dist);

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🗺️</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Field-Level Spatial Risk Distribution
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Spatial categorization based on 10m canonical UTM pixel grid multi-index anomalies.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link
            href="/analyze"
            className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-emerald-600/90 hover:bg-emerald-500 text-white transition flex items-center gap-1.5 shadow-sm"
          >
            <span>🔍</span>
            <span>Open Interactive Map</span>
          </Link>
        </div>
      </div>

      {/* Grid: Spatial Specs & Canopy Coverage */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Spatial Grid & CRS
          </span>
          <span className="text-sm font-bold text-white font-mono block">
            {fieldRisk.spatial_resolution_m}m Resolution
          </span>
          <span className="text-[10px] text-slate-500 font-mono truncate block">
            {fieldRisk.canonical_crs}
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Valid Canopy Area
          </span>
          <span className="text-sm font-bold text-emerald-400 font-mono block">
            {fieldRisk.valid_canopy_percentage.toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            {fieldRisk.valid_canopy_hectares.toFixed(1)} of {fieldRisk.total_area_hectares.toFixed(1)} ha
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            NoData / Masked
          </span>
          <span className="text-sm font-bold text-slate-300 font-mono block">
            {fieldRisk.nodata_percentage.toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-500">
            Water, cloud, shadow
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Stress Signal Coverage
          </span>
          <span
            className={`text-sm font-bold font-mono block ${
              fieldRisk.stress_signal_coverage_pct > 15 ? "text-rose-400" : "text-emerald-400"
            }`}
          >
            {fieldRisk.stress_signal_coverage_pct.toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-500">
            Elevated anomaly zone
          </span>
        </div>
      </div>

      {/* Visual Risk Distribution Bar */}
      <div className="space-y-2 mb-5">
        <div className="flex justify-between items-center text-xs">
          <span className="text-slate-300 font-medium">Risk Category Proportions</span>
          <span className="text-[11px] text-slate-400 font-mono">Total Field Hectares</span>
        </div>

        <div className="h-4 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800/80 shadow-inner">
          {categories.map((cat) => (
            <div
              key={cat.category_name}
              style={{
                width: `${cat.percentage_of_field}%`,
                backgroundColor: cat.color_hex,
              }}
              className="transition-all duration-500"
              title={`${cat.category_name}: ${cat.percentage_of_field.toFixed(1)}% (${cat.area_hectares.toFixed(1)} ha)`}
            />
          ))}
        </div>
      </div>

      {/* Detailed Category Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {categories.map((cat) => (
          <div
            key={cat.category_name}
            className="bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center gap-2 mb-1.5">
                <span
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: cat.color_hex }}
                />
                <span className="font-bold text-xs text-white">
                  {cat.category_name}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-snug mb-2">
                {cat.description}
              </p>
            </div>

            <div className="pt-2 border-t border-slate-800/60 flex items-baseline justify-between font-mono">
              <span className="text-xs font-semibold text-slate-300">
                {cat.area_hectares.toFixed(1)} ha
              </span>
              <span className="text-xs font-bold text-white">
                {cat.percentage_of_field.toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Geodesically Scale-Corrected UTM Hectare Calculation Note */}
      <div className="pt-3 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-400 gap-2">
        <span>
          📐 <strong>Area Calculation Methodology:</strong> Hectare statistics are computed from a 10m canonical Transverse Mercator (UTM) conformal grid, adjusted with point scale factor correction ($1/k^2$, distortion &lt;0.2%) to accurately represent ellipsoidal ground area.
        </span>
        {fieldRisk.has_geotiff_export && (
          <span className="text-emerald-400 font-medium whitespace-nowrap">
            ✓ GeoTIFF Rasters Available
          </span>
        )}
      </div>
    </div>
  );
}
