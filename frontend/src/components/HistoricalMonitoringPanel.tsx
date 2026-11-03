"use client";

import React, { useState } from "react";
import Link from "next/link";

export interface HistoricalObservationSummaryData {
  date: string;
  cloud_coverage: number;
  valid_pixel_pct: number;
  ndvi: number | null;
  ndmi: number | null;
  ndre: number | null;
  evi: number | null;
  quality_verdict: string;
}

export interface HistoricalMonitoringData {
  observation_count: number;
  time_span_days: number;
  date_range: string[];
  timeline: HistoricalObservationSummaryData[];
  overall_trend_direction: string;
  median_cloud_coverage: number;
}

interface HistoricalMonitoringPanelProps {
  historical: HistoricalMonitoringData;
  isDemo?: boolean;
}

export default function HistoricalMonitoringPanel({
  historical,
  isDemo = false,
}: HistoricalMonitoringPanelProps) {
  const [activeMetric, setActiveMetric] = useState<"ndvi" | "ndmi" | "ndre" | "evi">("ndvi");

  const timeline = historical.timeline || [];

  const getMetricValue = (obs: HistoricalObservationSummaryData, metric: string): number | null => {
    switch (metric) {
      case "ndvi":
        return obs.ndvi;
      case "ndmi":
        return obs.ndmi;
      case "ndre":
        return obs.ndre;
      case "evi":
        return obs.evi;
      default:
        return null;
    }
  };

  return (
    <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 backdrop-blur-md shadow-xl transition-all duration-300 hover:border-slate-700">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">📅</span>
            <h3 className="text-base font-bold text-white tracking-tight">
              Historical Observation Timeline & Multi-Temporal Trends
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {historical.observation_count} satellite passes over {historical.time_span_days} days (
            {historical.date_range.join(" to ")})
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Metric Selector Tabs */}
          <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex gap-1">
            {(["ndvi", "ndmi", "ndre", "evi"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setActiveMetric(m)}
                className={`px-2.5 py-1 rounded-lg text-xs font-bold uppercase transition ${
                  activeMetric === m
                    ? "bg-emerald-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          <Link
            href="/history"
            className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition flex items-center gap-1.5 shadow-sm"
          >
            <span>📊</span>
            <span>Detailed History</span>
          </Link>
        </div>
      </div>

      {/* Summary KPI Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Total Observations
          </span>
          <span className="text-sm font-bold text-white font-mono block">
            {historical.observation_count} Sentinel Passes
          </span>
          <span className="text-[10px] text-slate-500">
            {historical.time_span_days} Days Monitored
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Overall Trajectory
          </span>
          <span className="text-sm font-bold text-emerald-400 font-mono block">
            {historical.overall_trend_direction}
          </span>
          <span className="text-[10px] text-slate-500">
            Multi-temporal slope
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Median Cloud Cover
          </span>
          <span className="text-sm font-bold text-sky-400 font-mono block">
            {historical.median_cloud_coverage.toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-500">
            Copernicus SCL Mask
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block mb-0.5">
            Temporal Cadence
          </span>
          <span className="text-sm font-bold text-slate-300 font-mono block">
            ~5-12 Days
          </span>
          <span className="text-[10px] text-slate-500">
            Sentinel-2 Constellation
          </span>
        </div>
      </div>

      {/* Observation Timeline Table / Cards */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
              <th className="py-2.5 px-3">Date</th>
              <th className="py-2.5 px-3">Cloud %</th>
              <th className="py-2.5 px-3">Valid Canopy %</th>
              <th className="py-2.5 px-3 font-mono">NDVI</th>
              <th className="py-2.5 px-3 font-mono">NDMI</th>
              <th className="py-2.5 px-3 font-mono">NDRE</th>
              <th className="py-2.5 px-3 font-mono">EVI</th>
              <th className="py-2.5 px-3">Quality</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {timeline.map((obs, idx) => (
              <tr
                key={obs.date}
                className={`hover:bg-slate-800/40 transition font-mono ${
                  idx === timeline.length - 1 ? "bg-emerald-950/20" : ""
                }`}
              >
                <td className="py-2.5 px-3 font-bold text-slate-200">
                  {obs.date}
                  {idx === timeline.length - 1 && (
                    <span className="ml-1.5 px-1.5 py-0.2 rounded text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 uppercase">
                      Latest
                    </span>
                  )}
                </td>
                <td className="py-2.5 px-3 text-slate-400">
                  {obs.cloud_coverage.toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-slate-300">
                  {obs.valid_pixel_pct.toFixed(1)}%
                </td>
                <td className={`py-2.5 px-3 ${activeMetric === "ndvi" ? "font-bold text-emerald-400" : "text-slate-400"}`}>
                  {obs.ndvi !== null ? obs.ndvi.toFixed(3) : "—"}
                </td>
                <td className={`py-2.5 px-3 ${activeMetric === "ndmi" ? "font-bold text-sky-400" : "text-slate-400"}`}>
                  {obs.ndmi !== null ? obs.ndmi.toFixed(3) : "—"}
                </td>
                <td className={`py-2.5 px-3 ${activeMetric === "ndre" ? "font-bold text-amber-400" : "text-slate-400"}`}>
                  {obs.ndre !== null ? obs.ndre.toFixed(3) : "—"}
                </td>
                <td className={`py-2.5 px-3 ${activeMetric === "evi" ? "font-bold text-emerald-300" : "text-slate-400"}`}>
                  {obs.evi !== null ? obs.evi.toFixed(3) : "—"}
                </td>
                <td className="py-2.5 px-3">
                  <span className="px-2 py-0.5 rounded text-[10px] font-sans font-medium bg-slate-800 text-slate-300 border border-slate-700">
                    {obs.quality_verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Agronomic Disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400">
        <strong className="text-slate-300">Agronomic Scientific Notice:</strong> Index variations across time reflect combined effects of solar illumination geometry, phenological maturity, soil moisture, and cloud shadows. Temporal index changes must not be interpreted as definitive proof of a single agricultural cause.
      </div>
    </div>
  );
}
