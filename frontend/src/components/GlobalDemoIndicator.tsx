"use client";

import React from "react";

export interface GlobalDemoIndicatorProps {
  isDemo?: boolean;
  dataSource?: string;
  provenance?: string;
  acquisitionDate?: string;
  cloudCoverage?: number;
  isMixed?: boolean;
  unverified?: boolean;
  className?: string;
}

/**
 * Universal Demo & Data Provenance Indicator Component (Phase 20)
 *
 * Clearly distinguishes between:
 * 1. Synthetic demonstration data (DEMO MODE)
 * 2. Verified live Copernicus Sentinel acquisitions
 * 3. Mixed data (verified optical + synthetic SAR fallback)
 * 4. Unverified sources
 *
 * Ensures complete scientific transparency without exposing credentials.
 */
export default function GlobalDemoIndicator({
  isDemo = false,
  dataSource,
  provenance,
  acquisitionDate,
  cloudCoverage,
  isMixed = false,
  unverified = false,
  className = "",
}: GlobalDemoIndicatorProps) {
  const effectiveDataSource = dataSource || provenance;
  if (unverified) {
    return (
      <div
        className={`px-3 py-1.5 rounded-xl bg-amber-950/60 border border-amber-600/50 text-amber-200 text-xs flex items-center gap-2 backdrop-blur-md shadow-md ${className}`}
      >
        <span className="text-amber-400">⚠️</span>
        <div className="flex flex-wrap items-center gap-1.5 font-medium">
          <span className="font-bold uppercase tracking-wider text-amber-300">
            DATA SOURCE UNVERIFIED
          </span>
          <span className="text-slate-400">•</span>
          <span className="text-slate-300">
            Observation provenance could not be validated against Copernicus catalog records.
          </span>
        </div>
      </div>
    );
  }

  if (isMixed) {
    return (
      <div
        className={`px-3 py-1.5 rounded-xl bg-indigo-950/60 border border-indigo-500/40 text-indigo-200 text-xs flex items-center gap-2 backdrop-blur-md shadow-md ${className}`}
      >
        <span className="text-indigo-400">🔄</span>
        <div className="flex flex-wrap items-center gap-1.5 font-medium">
          <span className="font-bold uppercase tracking-wider text-indigo-300">
            MIXED DATA DISCLOSURE
          </span>
          <span className="text-slate-400">•</span>
          <span className="text-slate-300">
            Combines verified optical reflectance with synthetic demonstration SAR arrays.
          </span>
        </div>
      </div>
    );
  }

  if (isDemo) {
    return (
      <div
        className={`px-3.5 py-2 rounded-xl bg-amber-950/70 border border-amber-500/50 text-amber-200 text-xs flex items-center justify-between gap-3 backdrop-blur-md shadow-lg ${className}`}
      >
        <div className="flex items-center gap-2.5">
          <span className="text-base">🧪</span>
          <div className="leading-snug">
            <span className="font-extrabold text-amber-300 uppercase tracking-wider mr-2">
              DEMO MODE
            </span>
            <span className="text-amber-100/90 font-medium">
              Demonstration data is being used. Results do not represent live field analysis.
            </span>
          </div>
        </div>
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 whitespace-nowrap hidden sm:inline-block">
          SYNTHETIC REFLECTANCE
        </span>
      </div>
    );
  }

  return (
    <div
      className={`px-3.5 py-1.5 rounded-xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-200 text-xs flex items-center gap-2 backdrop-blur-md shadow-md ${className}`}
    >
      <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
      <div className="flex flex-wrap items-center gap-1.5 font-medium">
        <span className="font-bold uppercase tracking-wider text-emerald-300">
          LIVE SATELLITE DATA
        </span>
        <span className="text-slate-500">•</span>
        <span className="text-slate-300">
          Source: {effectiveDataSource || "Copernicus Sentinel-2 L2A"}
        </span>
        {acquisitionDate && (
          <>
            <span className="text-slate-500">•</span>
            <span className="text-slate-300">Acquired: {acquisitionDate}</span>
          </>
        )}
        {typeof cloudCoverage === "number" && (
          <>
            <span className="text-slate-500">•</span>
            <span className="text-slate-300">Cloud: {cloudCoverage.toFixed(1)}%</span>
          </>
        )}
      </div>
    </div>
  );
}
