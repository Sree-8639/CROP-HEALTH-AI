"use client";

import React from "react";

export interface FieldOverviewData {
  field_id: string;
  field_name: string;
  crop_type?: string;
  location_name: string;
  latitude: number;
  longitude: number;
  radius_km: number;
  total_area_hectares: number;
  total_area_sq_km: number;
  geometry?: any;
  latest_observation_date: string;
  sensors_available: string[];
  data_quality_status: string;
  last_analysis_status: string;
  is_demo: boolean;
}

interface FieldOverviewCardProps {
  overview: FieldOverviewData;
  onOpenFieldSelector?: () => void;
  onOpenNewAOI?: () => void;
}

export default function FieldOverviewCard({
  overview,
  onOpenFieldSelector,
  onOpenNewAOI,
}: FieldOverviewCardProps) {
  const isDemo = overview.is_demo;

  const getQualityBadgeColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "good":
        return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
      case "moderate":
        return "bg-amber-500/20 text-amber-300 border-amber-500/30";
      case "limited":
        return "bg-orange-500/20 text-orange-300 border-orange-500/30";
      default:
        return "bg-rose-500/20 text-rose-300 border-rose-500/30";
    }
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl relative overflow-hidden transition-all duration-300 hover:border-slate-700">
      {/* Background Glow */}
      <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Top Bar: Title, Crop, Badges, Field Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-1.5">
            <span className="text-xl">🌾</span>
            <h2 className="text-xl font-bold text-white tracking-tight">
              {overview.field_name}
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950/80 border border-emerald-600/40 text-emerald-300">
              {overview.crop_type || "Agricultural Parcel"}
            </span>

            {isDemo ? (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 tracking-wider">
                DEMO DATA
              </span>
            ) : (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 tracking-wider">
                PRODUCTION OBSERVATION
              </span>
            )}
          </div>

          <p className="text-xs text-slate-400 flex items-center gap-1.5">
            <span>📍</span>
            <span className="text-slate-300 font-medium">{overview.location_name}</span>
            <span className="text-slate-600">•</span>
            <span className="font-mono text-[11px] text-slate-400">
              {overview.latitude.toFixed(4)}°N, {overview.longitude.toFixed(4)}°E (r={overview.radius_km}km)
            </span>
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5">
          {onOpenFieldSelector && (
            <button
              onClick={onOpenFieldSelector}
              className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition flex items-center gap-1.5 shadow-sm"
            >
              <span>🔄</span>
              <span>Switch Field</span>
            </button>
          )}
          {onOpenNewAOI && (
            <button
              onClick={onOpenNewAOI}
              className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-emerald-600/90 hover:bg-emerald-500 text-white transition flex items-center gap-1.5 shadow-sm"
            >
              <span>➕</span>
              <span>New Field AOI</span>
            </button>
          )}
        </div>
      </div>

      {/* Grid Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-5">
        {/* Total Area */}
        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800/80">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Total Field Area
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-black text-white font-mono">
              {overview.total_area_hectares.toFixed(1)}
            </span>
            <span className="text-xs font-semibold text-emerald-400">ha</span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            {overview.total_area_sq_km.toFixed(2)} km² canonical grid
          </span>
        </div>

        {/* Latest Observation */}
        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800/80">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Latest Observation
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-lg font-bold text-sky-300 font-mono">
              {overview.latest_observation_date}
            </span>
          </div>
          <span className="text-[11px] text-slate-500">
            Copernicus Sentinel Pass
          </span>
        </div>

        {/* Available Sensors */}
        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800/80">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Active Satellite Sensors
          </span>
          <div className="flex flex-wrap gap-1 mt-1">
            {overview.sensors_available.map((sensor) => (
              <span
                key={sensor}
                className="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-950/80 text-sky-300 border border-sky-800/60"
              >
                {sensor}
              </span>
            ))}
          </div>
        </div>

        {/* Data Quality Status */}
        <div className="bg-slate-950/60 rounded-xl p-3.5 border border-slate-800/80">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Data Quality Status
          </span>
          <div className="flex items-center gap-2 mt-1">
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getQualityBadgeColor(
                overview.data_quality_status
              )}`}
            >
              {overview.data_quality_status} Quality
            </span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1 block">
            Analysis: {overview.last_analysis_status}
          </span>
        </div>
      </div>
    </div>
  );
}
