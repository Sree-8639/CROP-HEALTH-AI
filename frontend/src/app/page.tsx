"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import GlobalDemoIndicator from "@/components/GlobalDemoIndicator";

export default function Home() {
  const [apiStatus, setApiStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [isDemo, setIsDemo] = useState<boolean>(false);

  useEffect(() => {
    async function checkBackend() {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      try {
        const res = await fetch(`${baseUrl}/api/health`, {
          signal: AbortSignal.timeout(4000),
        });
        if (res.ok) {
          setApiStatus("connected");
          const data = await res.json().catch(() => ({}));
          const demoActive = data.demo_mode ?? (process.env.NEXT_PUBLIC_DEMO_MODE === "true");
          setIsDemo(Boolean(demoActive));
        } else {
          setApiStatus("offline");
          setIsDemo(process.env.NEXT_PUBLIC_DEMO_MODE === "true");
        }
      } catch {
        setApiStatus("offline");
        setIsDemo(process.env.NEXT_PUBLIC_DEMO_MODE === "true");
      }
    }
    checkBackend();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 relative">
      {/* Global Demo Indicator — visible strictly only in demo mode */}
      {isDemo && (
        <GlobalDemoIndicator
          isDemo={true}
          provenance="Sentinel-2 L2A (Copernicus) & Sentinel-1 GRD (ESA) Archive"
        />
      )}

      {/* Top Header / Navigation Bar */}
      <header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl overflow-hidden border border-emerald-400/50 bg-emerald-950/70 shadow-md flex items-center justify-center p-0.5">
              <img src="/logo.png" alt="Crop Health AI Logo" className="h-full w-full object-cover rounded-lg" />
            </div>
            <div>
              <span className="font-extrabold text-lg text-white tracking-tight">CropHealth AI</span>
              <span className="text-[11px] text-emerald-300 ml-2 font-medium px-2 py-0.5 rounded-full bg-emerald-900/60 border border-emerald-500/30 hidden sm:inline-block">
                Precision Agriculture Intelligence
              </span>
            </div>
          </div>

          {/* Standardized Navigation Links with Three Core Modules */}
          <nav className="flex items-center gap-3 sm:gap-5 text-sm">
            <Link href="/" className="font-bold text-white px-2 py-1 border-b-2 border-amber-300">
              HOME
            </Link>
            <Link href="/crop" className="font-bold text-emerald-300 hover:text-white transition flex items-center gap-1">
              <span>🌱</span>
              <span>CROP</span>
            </Link>
            <Link href="/deforestation" className="font-bold text-teal-300 hover:text-white transition flex items-center gap-1">
              <span>🌲</span>
              <span>DEFORESTATION</span>
            </Link>
            <Link href="/heat" className="font-bold text-amber-300 hover:text-white transition flex items-center gap-1">
              <span>🔥</span>
              <span>URBAN HEAT</span>
            </Link>
            <Link href="/farmer-dashboard" className="font-semibold text-amber-200 hover:text-amber-100 transition hidden sm:inline flex items-center gap-1">
              <span>🌾</span>
              <span>FARMER HUB</span>
            </Link>
            <Link href="/analyze" className="font-semibold text-emerald-100 hover:text-white transition hidden md:inline">
              ANALYZE
            </Link>
            <Link href="/model-performance" className="font-semibold text-emerald-100 hover:text-white transition hidden lg:inline">
              METRICS
            </Link>
            <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-emerald-950/70 border-emerald-600/40">
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "connected"
                    ? "bg-emerald-400 animate-pulse"
                    : apiStatus === "checking"
                    ? "bg-amber-400 animate-pulse"
                    : "bg-rose-400"
                }`}
              />
              <span className="text-emerald-100 text-[11px]">
                {apiStatus === "connected" ? "Engine Active" : apiStatus === "checking" ? "Connecting..." : "Offline"}
              </span>
            </div>
          </nav>
        </div>
      </header>

      {/* Hero Container with Bright Farmland Background */}
      <section className="relative min-h-[78vh] sm:min-h-[82vh] flex flex-col justify-between overflow-visible">
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat transition-transform duration-1000 scale-100"
          style={{ backgroundImage: "url('/bg-farmland-hero.jpg')" }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/55 via-black/35 to-black/70 pointer-events-none" />

        {/* Center Hero Content (Issue 8: Headline and Subtitle) */}
        <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 pt-16 sm:pt-24 text-center space-y-5">
          <div className="inline-block">
            <span className="text-xs sm:text-sm font-extrabold uppercase tracking-[0.25em] text-emerald-200 drop-shadow-[0_2px_4px_rgba(0,0,0,0.9)] px-4 py-1.5 rounded-full bg-black/40 border border-emerald-400/30 backdrop-blur-sm">
              SATELLITE EARTH OBSERVATION · MULTIMODAL FUSION
            </span>
          </div>

          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black text-white tracking-tight leading-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.95)]">
            Monitor Crop Health From Space
          </h1>

          <p className="text-sm sm:text-lg text-emerald-50 max-w-2xl mx-auto leading-relaxed drop-shadow-[0_2px_8px_rgba(0,0,0,0.95)] font-medium">
            Turn Sentinel-1 and Sentinel-2 satellite observations into understandable crop monitoring insights, historical trends and field-level intelligence.
          </p>

          {/* Action CTAs (Issue 8) */}
          <div className="pt-3 flex flex-wrap items-center justify-center gap-3.5">
            <Link
              href="/analyze"
              className="px-7 py-3 rounded-lg bg-[#387038] hover:bg-[#2d5d2d] text-white font-bold text-sm tracking-wide shadow-xl transition transform active:scale-95 border border-emerald-400/40 flex items-center gap-2 cursor-pointer"
            >
              <span>Explore Platform</span>
              <span>→</span>
            </Link>
            <a
              href="#sample-analysis"
              className="px-7 py-3 rounded-lg bg-[#d9c480] hover:bg-[#c9b26e] text-slate-950 font-extrabold text-sm tracking-wide shadow-xl transition transform active:scale-95 border border-amber-300/60 flex items-center gap-2 cursor-pointer"
            >
              <span>View Sample Analysis</span>
              <span>↓</span>
            </a>
          </div>
        </div>

        {/* 4 Overlapping Floating Cards at Bottom of Hero (Issue 2: Correct AI Feature Claims) */}
        <div className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full transform translate-y-10 sm:translate-y-14">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Card 1: Vegetation Index Analysis */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider block mb-1">
                  Optical Pipeline
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Vegetation Index Analysis
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Calculates vegetation and moisture-related spectral indices from satellite observations.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-emerald-800">
                <span>NDVI · NDMI · NDRE · EVI</span>
                <span>10m Resolution</span>
              </div>
            </div>

            {/* Card 2: Multimodal Data Fusion */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-sky-700 uppercase tracking-wider block mb-1">
                  Radar + Optical
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Multimodal Data Fusion
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Aligns available Sentinel-1 SAR and Sentinel-2 optical observations into a common spatial feature grid.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-sky-800">
                <span>±3 Days Temporal Window</span>
                <span>VV / VH / Ratio</span>
              </div>
            </div>

            {/* Card 3: Early Crop Stress Detection (Calibrated Claim) */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-amber-700 uppercase tracking-wider block mb-1">
                  Research Status
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Early Crop Stress Detection
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Supervised production stress detection is currently unavailable pending independently verified agronomic ground-truth data.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-amber-800">
                <span>Under Validation</span>
                <span>Rule-Based Preview</span>
              </div>
            </div>

            {/* Card 4: Field Risk Mapping & Farmer Hub */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-teal-700 uppercase tracking-wider block mb-1">
                  Field Risk Mapping
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Heuristic Indicator Maps
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Visualizes spatial indicators and heuristic risk categories. These categories are not validated agronomic diagnoses.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-teal-800">
                <span>Engineering Heuristics</span>
                <span>Farmer Advisory</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 sm:pt-28 pb-16 space-y-16">
        
        {/* ISSUE 6: EXPLORE A SAMPLE ANALYSIS SECTION */}
        <section id="sample-analysis" className="scroll-mt-20 bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-10 shadow-2xl space-y-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wider uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  DEMO DATA · PRE-COMPUTED SAMPLE ANALYSIS
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
                  Copernicus Test AOI
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center gap-2">
                <span>📍 Explore a Sample Analysis: Tadepalligudem Paddy</span>
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
                Demonstrating multimodal satellite processing capabilities on an active paddy field in Andhra Pradesh, India.
                Demonstration data is being used. Results do not represent live field analysis or validated diagnostic outcomes.
              </p>
            </div>
            <Link
              href="/analyze"
              className="self-start md:self-center px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs shadow-lg transition flex items-center gap-1.5 whitespace-nowrap"
            >
              <span>Explore Sample Analysis in App</span>
              <span>→</span>
            </Link>
          </div>

          {/* Sample AOI & Observation Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">Target Coordinates</span>
              <span className="text-sm font-bold font-mono text-emerald-300">16.8102° N, 81.5266° E</span>
              <span className="text-[10px] text-slate-500 block mt-1">AOI Radius: 2.0 km</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">Optical Acquisition</span>
              <span className="text-sm font-bold text-slate-200">Sentinel-2 MSI (L2A)</span>
              <span className="text-[10px] text-emerald-400 block mt-1">Cloud Coverage: 1.2%</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">SAR Acquisition</span>
              <span className="text-sm font-bold text-slate-200">Sentinel-1 C-SAR (GRD)</span>
              <span className="text-[10px] text-sky-400 block mt-1">IW VV+VH Polarized</span>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
              <span className="text-[11px] font-medium text-slate-400 block mb-1">Temporal Window</span>
              <span className="text-sm font-bold font-mono text-amber-300">Δt = +1.4 days</span>
              <span className="text-[10px] text-slate-500 block mt-1">Within default ±3 days</span>
            </div>
          </div>

          {/* Spectral Index & Field Risk Showcase */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* NDVI Card */}
            <div className="p-5 rounded-2xl bg-slate-950/80 border border-emerald-500/30 flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-emerald-400">Vegetation Vigor</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    NDVI 0.68
                  </span>
                </div>
                <h4 className="text-base font-bold text-white mb-1">Healthy Canopy Density</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Normalized Difference Vegetation Index indicates vigorous tillering/vegetative growth phase with strong near-infrared reflectance.
                </p>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div className="bg-gradient-to-r from-amber-500 via-emerald-500 to-emerald-400 h-full w-[68%]" />
              </div>
            </div>

            {/* NDMI Card */}
            <div className="p-5 rounded-2xl bg-slate-950/80 border border-cyan-500/30 flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-cyan-400">Canopy Moisture</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    NDMI 0.42
                  </span>
                </div>
                <h4 className="text-base font-bold text-white mb-1">Adequate Hydration</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Normalized Difference Moisture Index reflects saturated soil and healthy cellular water content in irrigated paddy basins.
                </p>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-cyan-400 h-full w-[58%]" />
              </div>
            </div>

            {/* Field Risk Classification Card */}
            <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-700 flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-amber-300">Heuristic Risk Zoning</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950 text-emerald-300 border border-emerald-500/30">
                    Low Moisture Risk
                  </span>
                </div>
                <h4 className="text-base font-bold text-white mb-1">Engineering Assessment</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Rule-based classification combining optical reflectance and radar roughness. These categories are not validated agronomic diagnoses.
                </p>
              </div>
              <div className="text-[11px] text-slate-500 font-mono">
                Model: Rule-Based Spectral Engine v1.0
              </div>
            </div>
          </div>

          {/* Historical Trend Preview */}
          <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Multi-Temporal Trajectory Preview (Sample Growth Cycle)
              </span>
              <span className="text-xs text-slate-500">6 Historical Observations</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 text-center">
              {[
                { date: "Oct 12", ndvi: "0.35", stage: "Transplanting" },
                { date: "Oct 27", ndvi: "0.48", stage: "Early Tillering" },
                { date: "Nov 11", ndvi: "0.62", stage: "Active Tillering" },
                { date: "Nov 26", ndvi: "0.68", stage: "Panicle Initiation" },
                { date: "Dec 11", ndvi: "0.65", stage: "Heading" },
                { date: "Dec 26", ndvi: "0.70", stage: "Grain Filling" },
              ].map((step, idx) => (
                <div key={idx} className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">{step.date}</span>
                  <span className="text-sm font-bold font-mono text-emerald-400 block my-0.5">{step.ndvi}</span>
                  <span className="text-[10px] text-slate-300 truncate block">{step.stage}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Transparency and Limitations Footer Note */}
          <div className="p-4 rounded-xl bg-slate-950/90 border border-amber-500/20 text-xs text-amber-200/80 leading-relaxed flex items-start gap-2.5">
            <span className="text-base text-amber-400">ℹ️</span>
            <div>
              <strong className="text-amber-300 font-semibold">Data Provenance & Scientific Limitation:</strong> Pre-computed sample metrics demonstrate the platform's multi-sensor extraction pipelines. Growth trajectory projections are subject to observation availability and validation limitations.
            </div>
          </div>
        </section>

        {/* Constellation Comparison Section */}
        <section className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-10 shadow-2xl space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
              Dual-Sensor Satellite Architecture
            </span>
            <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight mt-1 flex items-center gap-2">
              <span>🛰️ Complementary Constellations</span>
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
              Fusing Sentinel-2 optical spectral reflectance with Sentinel-1 active C-band synthetic aperture radar backscatter ensures all-weather observation continuity across crop growth cycles.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Card 1: Sentinel-2 */}
            <div className="p-6 rounded-2xl bg-slate-950/80 border border-emerald-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-2xl">🌿</span>
                  <div>
                    <h3 className="font-bold text-white text-lg">Sentinel-2 MSI Optical</h3>
                    <span className="text-xs font-semibold text-emerald-400">Multispectral Reflectance</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                  10m - 20m GSD
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Direct integration with Copernicus Data Space Ecosystem for calibrated surface reflectance across 13 spectral bands. Accurately evaluates vegetative vigor (NDVI), canopy moisture (NDMI), red-edge chlorophyll absorption (NDRE), and aerosol-adjusted biomass (EVI).
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {["B02 Blue", "B03 Green", "B04 Red", "B05 Red-Edge", "B08 NIR", "B11 SWIR", "NDVI", "NDMI", "NDRE", "EVI"].map((b) => (
                  <span key={b} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-300">
                    {b}
                  </span>
                ))}
              </div>
            </div>

            {/* Card 2: Sentinel-1 */}
            <div className="p-6 rounded-2xl bg-slate-950/80 border border-sky-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="text-2xl">📡</span>
                  <div>
                    <h3 className="font-bold text-white text-lg">Sentinel-1 C-SAR Radar</h3>
                    <span className="text-xs font-semibold text-sky-400">All-Weather Active Microwave</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-sky-500/10 text-sky-300 border border-sky-500/20">
                  10m GRD (IW)
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Sentinel-1 transmits active 5.405 GHz microwave pulses, penetrating clouds and darkness. Evaluates crop physical stalk structure, surface roughness, volume scattering, and root-zone moisture anomalies.
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {["VV Co-Pol", "VH Cross-Pol", "VV (dB)", "VH (dB)", "VV/VH Cross-Ratio", "VV-VH Diff (dB)"].map((b) => (
                  <span key={b} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-sky-300">
                    {b}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Action Links */}
          <div className="pt-2 flex flex-wrap items-center justify-between gap-4 border-t border-slate-800">
            <div className="flex items-center gap-3">
              <Link
                href="/analyze"
                className="px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs shadow-lg transition"
              >
                Launch Field Analysis
              </Link>
              <Link
                href="/farmer-dashboard"
                className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition"
              >
                Open Farmer Dashboard
              </Link>
            </div>
            <Link
              href="/history"
              className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition"
            >
              <span>Explore Multi-Temporal Satellite History</span>
              <span>→</span>
            </Link>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Crop Health AI. Multimodal Satellite Agricultural Intelligence Platform.</p>
          <div className="flex items-center gap-4 text-slate-400">
            <Link href="/analyze" className="hover:text-emerald-400 transition">Analyze Field</Link>
            <span>•</span>
            <Link href="/farmer-dashboard" className="hover:text-emerald-400 transition">Farmer Dashboard</Link>
            <span>•</span>
            <Link href="/history" className="hover:text-emerald-400 transition">History</Link>
            <span>•</span>
            <Link href="/model-performance" className="hover:text-emerald-400 transition">Architecture & Model Targets</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}