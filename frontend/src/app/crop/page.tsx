"use client";

import React, { useState } from "react";
import Link from "next/link";
import ModuleNavbar from "@/components/ModuleNavbar";
import SharedAOIControls from "@/components/SharedAOIControls";
import { useSharedAOI } from "@/context/SharedAOIContext";
import { API_BASE } from "@/config/api";

export default function CropModulePage() {
  const { getGeoJsonGeometry, startDate, endDate, aoiType, radiusKm, center, selectedPresetId } = useSharedAOI();
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleRunCropAnalysis = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const geometry = getGeoJsonGeometry();
      const payload = {
        geometry,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: 30.0,
        temporal_window_days: 3,
        normalization: "raw",
      };

      const res = await fetch(`${API_BASE}/api/crop/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server responded with status ${res.status}`);
      }

      const data = await res.json();
      setAnalysisResult(data);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to execute crop analysis pipeline.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <ModuleNavbar currentModule="crop" />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* Module Header */}
        <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-2xl">🌱</span>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                Crop Vigour Anomaly & Agricultural Intelligence
              </h1>
            </div>
            <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
              Multimodal satellite feature fusion combining Sentinel-2 MSI multispectral reflectance (NDVI, NDMI, NDRE, EVI)
              and Sentinel-1 C-band SAR radar backscatter (VV, VH, VV-VH dB difference) via the Common Data-Access Layer.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <Link
              href="/farmer-dashboard"
              className="px-4 py-2 rounded-xl bg-amber-600/90 hover:bg-amber-500 text-white font-bold text-xs shadow-lg transition flex items-center gap-1.5"
            >
              <span>🌾</span>
              <span>Open Farmer Hub</span>
            </Link>
            <Link
              href="/analyze"
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs border border-slate-700 transition flex items-center gap-1.5"
            >
              <span>🔬</span>
              <span>Advanced Studio</span>
            </Link>
          </div>
        </div>

        {/* Shared AOI and Date Controls */}
        <SharedAOIControls
          moduleName="Crop Module"
          themeColor="emerald"
          executeButtonText="Run Crop Analysis Pipeline"
          isLoading={isLoading}
          onExecuteAnalysis={handleRunCropAnalysis}
        />

        {/* Error Notice */}
        {errorMessage && (
          <div className="bg-rose-950/40 border border-rose-800 rounded-2xl p-5 mb-8 text-rose-200 text-xs shadow-xl flex items-start gap-3">
            <span className="text-xl">⚠️</span>
            <div>
              <h3 className="font-bold text-sm text-white mb-0.5">Crop Pipeline Notice</h3>
              <p>{errorMessage}</p>
            </div>
          </div>
        )}

        {/* Results Container */}
        {analysisResult ? (
          <div className="space-y-6">
            {/* Real Data Provenance Badge */}
            {analysisResult.data_source === "REAL" && (
              <div className="p-4 rounded-2xl bg-emerald-950/80 border-2 border-emerald-500 text-emerald-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xl">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🛰️</span>
                  <div>
                    <span className="text-xs font-black uppercase tracking-wider text-emerald-200 block">
                      DATA PROVENANCE: REAL SATELLITE DATA (SENTINEL-1 &amp; SENTINEL-2)
                    </span>
                    <span className="text-[11px] text-emerald-300">
                      Validated on authentic Copernicus Sentinel-2 L2A optical and Sentinel-1 C-SAR radar acquisitions.
                    </span>
                  </div>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-mono font-black bg-emerald-600 text-white uppercase tracking-wider shrink-0 self-start sm:self-auto">
                  REAL
                </span>
              </div>
            )}

            {/* Simulated Data Source Alert Badge */}
            {(analysisResult.data_source === "SIMULATED" || analysisResult.provenance?.mode === "DEMO") && (
              <div className="p-4 rounded-2xl bg-rose-950/80 border-2 border-rose-600 text-rose-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xl">
                <div className="flex items-center gap-2.5">
                  <span className="text-xl">🚨</span>
                  <div>
                    <span className="text-xs font-black uppercase tracking-wider text-rose-200 block">
                      DATA PROVENANCE: SIMULATED / BENCHMARK DATA
                    </span>
                    <span className="text-[11px] text-rose-300">
                      Results generated using offline benchmark simulation. Connect CDSE Copernicus API credentials for live satellite data.
                    </span>
                  </div>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-mono font-black bg-rose-600 text-white uppercase tracking-wider shrink-0 self-start sm:self-auto">
                  SIMULATED
                </span>
              </div>
            )}

            {/* Provenance Banner */}
            {analysisResult.provenance && (
              <div className="bg-slate-900/90 border border-emerald-500/30 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
                <div className="flex items-center gap-3">
                  <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                    analysisResult.provenance.mode === "LIVE"
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-500/50"
                      : "bg-amber-950 text-amber-300 border border-amber-500/50"
                  }`}>
                    {analysisResult.provenance.badge_label}
                  </span>
                  <span className="text-slate-300 font-medium">
                    Sensor: <strong className="text-white">{analysisResult.provenance.sensor}</strong>
                  </span>
                </div>
                <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
                  <span>Cloud: {analysisResult.provenance.cloud_coverage_percentage ?? 0}%</span>
                  <span>Valid: {analysisResult.provenance.valid_pixel_percentage ?? 100}%</span>
                  {analysisResult.provenance.acquisition_dates?.temporal_gap_days !== undefined && (
                    <span>S1-S2 Gap: {analysisResult.provenance.acquisition_dates.temporal_gap_days}d</span>
                  )}
                </div>
              </div>
            )}

            {/* Quality Summary Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Optical Valid %</span>
                <span className="text-xl font-mono font-bold text-emerald-400">
                  {analysisResult.quality_summary?.optical_valid_percentage ?? 100}%
                </span>
              </div>
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">SAR Valid %</span>
                <span className="text-xl font-mono font-bold text-teal-400">
                  {analysisResult.quality_summary?.sar_valid_percentage ?? 100}%
                </span>
              </div>
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Fused Quality</span>
                <span className="text-xl font-mono font-bold text-blue-400">
                  {analysisResult.quality_summary?.fused_valid_percentage ?? 100}%
                </span>
              </div>
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">AI Readiness</span>
                <span className="text-xs font-mono font-bold text-amber-300 block truncate">
                  {analysisResult.fusion_readiness || "READY"}
                </span>
              </div>
            </div>

            {/* Feature Statistics Previews */}
            {analysisResult.feature_statistics && (
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                  <span>📊</span>
                  <span>Extracted Multimodal Crop Feature Metrics</span>
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                  {Object.entries(analysisResult.feature_statistics).map(([feat, stat]: [string, any]) => (
                    <div key={feat} className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 font-mono text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-emerald-300">{feat}</span>
                        <span className="text-[10px] text-slate-400">{stat.unit || ""}</span>
                      </div>
                      <div className="text-slate-300 text-sm font-bold">
                        {stat.mean !== null && stat.mean !== undefined ? Number(stat.mean).toFixed(3) : "N/A"}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-1 flex justify-between">
                        <span>Min: {stat.min !== null ? Number(stat.min).toFixed(2) : "-"}</span>
                        <span>Max: {stat.max !== null ? Number(stat.max).toFixed(2) : "-"}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-3xl p-12 text-center space-y-3">
            <span className="text-4xl block">🌾</span>
            <h3 className="text-base font-bold text-white">Select Shared AOI & Execute Crop Pipeline</h3>
            <p className="text-xs text-slate-400 max-w-lg mx-auto">
              Configure your AOI (radius or polygon) and date range above, then click &quot;Run Crop Analysis Pipeline&quot; to fetch Sentinel-2 + Sentinel-1 observations via the common data-access layer.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

