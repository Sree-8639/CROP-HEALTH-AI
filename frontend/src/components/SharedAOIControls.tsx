"use client";

import React, { useState } from "react";
import { useSharedAOI, PRESET_AOIS, AOIType } from "@/context/SharedAOIContext";

interface SharedAOIControlsProps {
  moduleName?: string;
  onExecuteAnalysis?: () => void;
  isLoading?: boolean;
  executeButtonText?: string;
  themeColor?: "emerald" | "teal" | "amber";
}

export default function SharedAOIControls({
  moduleName = "Shared Multi-Module",
  onExecuteAnalysis,
  isLoading = false,
  executeButtonText = "Execute Analysis",
  themeColor = "emerald",
}: SharedAOIControlsProps) {
  const {
    aoiType,
    setAoiType,
    center,
    setCenter,
    radiusKm,
    setRadiusKm,
    polygonPoints,
    setPolygonPoints,
    addPolygonPoint,
    clearPolygon,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    setDatePreset,
    selectedPresetId,
    applyPreset,
    getBbox,
  } = useSharedAOI();

  const [isExpanded, setIsExpanded] = useState(true);
  const [newLat, setNewLat] = useState("");
  const [newLng, setNewLng] = useState("");

  const bbox = getBbox();
  const areaHa =
    aoiType === "radius"
      ? Math.round(Math.PI * radiusKm * radiusKm * 100)
      : Math.round(((bbox[2] - bbox[0]) * 111 * (bbox[3] - bbox[1]) * 111 * 100) * 0.7);

  const themeClasses = {
    emerald: {
      border: "border-emerald-600/40",
      bgHeader: "bg-emerald-950/40",
      accent: "text-emerald-400",
      btn: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30",
      badge: "bg-emerald-900/60 border-emerald-500/40 text-emerald-300",
    },
    teal: {
      border: "border-teal-600/40",
      bgHeader: "bg-teal-950/40",
      accent: "text-teal-400",
      btn: "bg-teal-600 hover:bg-teal-500 text-white shadow-teal-900/30",
      badge: "bg-teal-900/60 border-teal-500/40 text-teal-300",
    },
    amber: {
      border: "border-amber-600/40",
      bgHeader: "bg-amber-950/40",
      accent: "text-amber-400",
      btn: "bg-amber-600 hover:bg-amber-500 text-white shadow-amber-900/30",
      badge: "bg-amber-900/60 border-amber-500/40 text-amber-300",
    },
  }[themeColor];

  const handleAddPoint = (e: React.FormEvent) => {
    e.preventDefault();
    const lat = parseFloat(newLat);
    const lng = parseFloat(newLng);
    if (!isNaN(lat) && !isNaN(lng)) {
      addPolygonPoint({ lat, lng });
      setNewLat("");
      setNewLng("");
    }
  };

  return (
    <div className={`bg-slate-900/90 border ${themeClasses.border} rounded-2xl shadow-2xl backdrop-blur-md overflow-hidden transition-all mb-8`}>
      {/* Header bar */}
      <div className={`px-5 py-3.5 ${themeClasses.bgHeader} border-b border-slate-800 flex items-center justify-between`}>
        <div className="flex items-center gap-2.5">
          <span className="text-xl">🌐</span>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-bold text-white tracking-tight">
                Shared Spatial AOI & Observation Window
              </h2>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${themeClasses.badge}`}>
                Synced Across All Modules
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              AOI geometry and temporal bounds defined here apply seamlessly to /crop, /deforestation, and /heat.
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition font-mono"
        >
          {isExpanded ? "Collapse ▲" : "Configure ▼"}
        </button>
      </div>

      {isExpanded && (
        <div className="p-5 space-y-6">
          {/* Quick Preset Selector */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                1. Select Curated Benchmark AOI (or Custom)
              </label>
              <span className="text-[11px] text-slate-400 font-mono">
                {PRESET_AOIS.length} Regional Benchmark Presets
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
              {PRESET_AOIS.map((p) => {
                const isSelected = selectedPresetId === p.id;
                return (
                  <button
                    key={p.id}
                    onClick={() => applyPreset(p.id)}
                    className={`p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                      isSelected
                        ? `bg-slate-800 border-white/60 shadow-md ring-1 ring-white/30`
                        : "bg-slate-950/60 border-slate-800 hover:bg-slate-800/60 hover:border-slate-700"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="text-xs font-bold text-white truncate">{p.name}</span>
                        <span className="text-[10px]">
                          {p.module === "crop" ? "🌱" : p.module === "deforestation" ? "🌲" : "🔥"}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 line-clamp-1">{p.region}</p>
                    </div>
                    <span className="text-[9px] font-mono text-slate-500 mt-2 block">
                      {p.radiusKm} km radius
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* AOI Mode & Geometry Controls (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  2. AOI Geometry Definition
                </label>
                {/* Switch between Radius and Polygon */}
                <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                  <button
                    onClick={() => setAoiType("radius")}
                    className={`px-3 py-1 rounded-lg font-bold transition ${
                      aoiType === "radius" ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Radius Mode
                  </button>
                  <button
                    onClick={() => setAoiType("polygon")}
                    className={`px-3 py-1 rounded-lg font-bold transition ${
                      aoiType === "polygon" ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Draw / Polygon Mode
                  </button>
                </div>
              </div>

              {aoiType === "radius" ? (
                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[11px] text-slate-400 font-medium block mb-1">
                        Center Latitude (°N)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={center.lat}
                        onChange={(e) => setCenter({ ...center, lat: parseFloat(e.target.value) || 0 })}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-slate-400 font-medium block mb-1">
                        Center Longitude (°E)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={center.lng}
                        onChange={(e) => setCenter({ ...center, lng: parseFloat(e.target.value) || 0 })}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500"
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] text-slate-400 font-medium">
                        Radius from Center: <span className="font-mono text-white font-bold">{radiusKm} km</span>
                      </label>
                      <span className="text-[10px] text-slate-400 font-mono">
                        ~{areaHa.toLocaleString()} hectares ({Math.round(areaHa / 100)} km²)
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="15.0"
                      step="0.5"
                      value={radiusKm}
                      onChange={(e) => setRadiusKm(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>
                </div>
              ) : (
                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium">
                      Vertices in Polygon: <span className="font-bold text-white">{polygonPoints.length}</span>
                    </span>
                    <button
                      onClick={clearPolygon}
                      className="text-[11px] text-rose-400 hover:text-rose-300 font-mono"
                    >
                      Clear Points
                    </button>
                  </div>

                  {/* Vertices chips */}
                  <div className="max-h-24 overflow-y-auto space-y-1 pr-1 font-mono text-[10px]">
                    {polygonPoints.map((pt, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between bg-slate-900 px-2.5 py-1 rounded border border-slate-800"
                      >
                        <span className="text-slate-400">Vertex {idx + 1}</span>
                        <span className="text-emerald-300">
                          [{pt.lat.toFixed(4)}°N, {pt.lng.toFixed(4)}°E]
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Add Vertex form */}
                  <form onSubmit={handleAddPoint} className="flex gap-2 pt-1">
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="Lat (e.g. 16.81)"
                      value={newLat}
                      onChange={(e) => setNewLat(e.target.value)}
                      className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-mono"
                    />
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="Lng (e.g. 81.52)"
                      value={newLng}
                      onChange={(e) => setNewLng(e.target.value)}
                      className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-mono"
                    />
                    <button
                      type="submit"
                      className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-white font-mono"
                    >
                      + Add
                    </button>
                  </form>
                </div>
              )}
            </div>

            {/* Shared Date Range Controls (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                3. Shared Observation Date Range
              </label>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] text-slate-400 font-medium block mb-1">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] text-slate-400 font-medium block mb-1">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                </div>

                {/* Date Presets */}
                <div>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1 font-mono">
                    Quick Temporal Windows
                  </span>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button
                      type="button"
                      onClick={() => setDatePreset("30d")}
                      className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[10px] text-slate-300 font-mono text-left"
                    >
                      📅 Last 30 Days
                    </button>
                    <button
                      type="button"
                      onClick={() => setDatePreset("90d")}
                      className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[10px] text-slate-300 font-mono text-left"
                    >
                      📅 Last 90 Days
                    </button>
                    <button
                      type="button"
                      onClick={() => setDatePreset("monsoon")}
                      className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[10px] text-slate-300 font-mono text-left"
                    >
                      🌧️ Monsoon 2024
                    </button>
                    <button
                      type="button"
                      onClick={() => setDatePreset("pre-monsoon")}
                      className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-[10px] text-slate-300 font-mono text-left"
                    >
                      ☀️ Summer 2024
                    </button>
                  </div>
                </div>

                {/* Bounding box display */}
                <div className="pt-2 border-t border-slate-800 text-[10px] text-slate-400 font-mono flex items-center justify-between">
                  <span>BBox [minX, minY, maxX, maxY]:</span>
                  <span className="text-slate-300">
                    {bbox[0].toFixed(2)}, {bbox[1].toFixed(2)}, {bbox[2].toFixed(2)}, {bbox[3].toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Footer */}
          {onExecuteAnalysis && (
            <div className="pt-4 border-t border-slate-800 flex items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
                <span>Ready to dispatch to module pipeline with synchronized spatial parameters.</span>
              </div>
              <button
                onClick={onExecuteAnalysis}
                disabled={isLoading}
                className={`px-6 py-2.5 rounded-xl font-bold text-xs shadow-lg transition-all flex items-center gap-2 ${themeClasses.btn} ${
                  isLoading ? "opacity-60 cursor-not-allowed" : ""
                }`}
              >
                {isLoading ? (
                  <>
                    <span className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Executing Pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>🚀</span>
                    <span>{executeButtonText}</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

