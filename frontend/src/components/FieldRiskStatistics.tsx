"use client";

import React from "react";

export interface RiskCategoryDetail {
  category_name: string;
  area_hectares: number;
  area_sq_km: number;
  percentage_of_field: number;
  pixel_count: number;
  color_hex: string;
  description: string;
}

export interface FieldRiskStatisticsData {
  total_area_hectares: number;
  total_area_sq_km: number;
  valid_area_hectares: number;
  valid_area_sq_km: number;
  valid_pixel_percentage: number;
  nodata_percentage: number;
  risk_distribution: Record<string, RiskCategoryDetail>;
  vegetation_indices_mean: Record<string, number | null>;
  vegetation_indices_median: Record<string, number | null>;
  vegetation_indices_std: Record<string, number | null>;
  stress_signal_coverage_pct: number;
  observation_count: number;
  observation_date: string;
  model_version: string;
  crs: string;
  spatial_resolution_m: number;
  is_area_approximate: boolean;
  data_quality_limitations: string[];
}

interface FieldRiskStatisticsProps {
  statistics: FieldRiskStatisticsData;
  isDemo?: boolean;
}

export default function FieldRiskStatistics({
  statistics,
  isDemo = false,
}: FieldRiskStatisticsProps) {
  const dist = statistics.risk_distribution || {};

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 backdrop-blur-md shadow-xl space-y-5">
      {/* Header & Badges */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-base">📊</span>
            <h4 className="text-base font-bold text-white tracking-tight">
              Field-Level Risk Summary Statistics
            </h4>
          </div>
          <p className="text-xs text-slate-400">
            Observation: <span className="text-slate-200 font-mono">{statistics.observation_date}</span> • CRS:{" "}
            <span className="text-slate-200 font-mono text-[11px]">{statistics.crs}</span> (
            {statistics.spatial_resolution_m}m)
          </p>
        </div>

        {isDemo && (
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-semibold">
            <span>⚠️</span> DEMO DATA — Deterministic Simulation
          </div>
        )}
      </div>

      {/* Top 4 KPI Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 block mb-0.5">
            Total Analyzed Field
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-white font-mono">
              {statistics.total_area_hectares.toFixed(2)}
            </span>
            <span className="text-xs text-slate-400">ha</span>
          </div>
          <span className="text-[10px] text-slate-400">
            ({statistics.total_area_sq_km.toFixed(2)} km²)
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 block mb-0.5">
            Valid Canopy Area
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-emerald-400 font-mono">
              {statistics.valid_area_hectares.toFixed(2)}
            </span>
            <span className="text-xs text-slate-400">ha</span>
          </div>
          <span className="text-[10px] text-emerald-400/80">
            {statistics.valid_pixel_percentage.toFixed(1)}% valid canopy coverage
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 block mb-0.5">
            Stress Signal Coverage
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-amber-400 font-mono">
              {statistics.stress_signal_coverage_pct.toFixed(1)}%
            </span>
          </div>
          <span className="text-[10px] text-slate-400">
            Field area exhibiting elevated/moderate stress
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 block mb-0.5">
            NoData & Clouds
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-slate-300 font-mono">
              {statistics.nodata_percentage.toFixed(1)}%
            </span>
          </div>
          <span className="text-[10px] text-slate-400">
            Excluded from risk score classification
          </span>
        </div>
      </div>

      {/* Risk Category Distribution Progress Bars */}
      <div className="space-y-3 bg-slate-950/40 rounded-xl p-4 border border-slate-800/80">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-200">
            Spatial Risk Category Breakdown
          </span>
          <span className="text-[11px] text-slate-400">
            Based on multimodal satellite inference
          </span>
        </div>

        {/* Visual Stacked Bar */}
        <div className="w-full h-3 rounded-full overflow-hidden flex bg-slate-900 border border-slate-800">
          {Object.entries(dist).map(([k, cat]) => (
            <div
              key={k}
              style={{
                width: `${cat.percentage_of_field}%`,
                backgroundColor: cat.color_hex,
              }}
              title={`${cat.category_name}: ${cat.percentage_of_field.toFixed(1)}% (${cat.area_hectares.toFixed(2)} ha)`}
              className="h-full transition-all duration-300"
            />
          ))}
        </div>

        {/* Category Details Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 pt-1">
          {Object.entries(dist).map(([k, cat]) => (
            <div
              key={k}
              className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between"
            >
              <div className="flex items-center gap-2 mb-1.5">
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: cat.color_hex }}
                />
                <span className="text-xs font-semibold text-white truncate">
                  {cat.category_name}
                </span>
              </div>
              <div className="flex items-baseline justify-between mt-auto">
                <span className="text-sm font-bold font-mono text-slate-100">
                  {cat.area_hectares.toFixed(2)} ha
                </span>
                <span
                  className="text-xs font-semibold font-mono"
                  style={{ color: cat.color_hex }}
                >
                  {cat.percentage_of_field.toFixed(1)}%
                </span>
              </div>
              <span className="text-[10px] text-slate-400 mt-1 font-mono">
                {cat.pixel_count.toLocaleString()} pixels
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Canopy Mean Vegetation Indices */}
      <div className="space-y-2.5">
        <span className="text-xs font-semibold text-slate-200 block">
          Mean Canopy Indices across Valid Canopy Pixels
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { key: "NDVI", name: "NDVI (Vigor)", color: "text-emerald-400" },
            { key: "NDMI", name: "NDMI (Moisture)", color: "text-sky-400" },
            { key: "NDRE", name: "NDRE (Chlorophyll)", color: "text-teal-400" },
            { key: "EVI", name: "EVI (Canopy)", color: "text-emerald-300" },
          ].map((idx) => {
            const val = statistics.vegetation_indices_mean?.[idx.key];
            const medianVal = statistics.vegetation_indices_median?.[idx.key];
            return (
              <div
                key={idx.key}
                className="bg-slate-950/60 rounded-xl p-3 border border-slate-800"
              >
                <span className="text-[11px] font-medium text-slate-400 block mb-1">
                  {idx.name}
                </span>
                <span className={`text-lg font-bold font-mono ${idx.color}`}>
                  {val !== null && val !== undefined ? val.toFixed(3) : "N/A"}
                </span>
                {medianVal !== null && medianVal !== undefined && (
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Median: {medianVal.toFixed(3)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
