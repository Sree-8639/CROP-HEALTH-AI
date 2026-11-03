"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";

export default function Home() {
  const [apiStatus, setApiStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [alertMessage, setAlertMessage] = useState<string | null>(null);

  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/health", {
          signal: AbortSignal.timeout(4000),
        });
        if (res.ok) {
          setApiStatus("connected");
        } else {
          setApiStatus("offline");
        }
      } catch (err) {
        setApiStatus("offline");
      }
    }
    checkBackend();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 relative">
      {/* Top Header / Navigation Bar styled like the reference */}
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

          {/* Navigation Links */}
          <nav className="flex items-center gap-4 sm:gap-6 text-sm">
            <Link href="/" className="font-bold text-white px-2 py-1 border-b-2 border-amber-300">
              HOME
            </Link>
            <Link href="/analyze" className="font-semibold text-emerald-100 hover:text-white transition">
              ANALYZE
            </Link>
            <Link href="/farmer-dashboard" className="font-semibold text-amber-200 hover:text-amber-100 transition flex items-center gap-1">
              <span>🌾</span>
              <span>FARMER HUB</span>
            </Link>
            <Link href="/history" className="font-semibold text-emerald-100 hover:text-white transition hidden md:inline">
              HISTORY
            </Link>
            <Link href="/model-performance" className="font-semibold text-emerald-100 hover:text-white transition hidden lg:inline">
              ARCHITECTURE
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

      {/* Hero Container with Bright Farmland Background (Matching Reference) */}
      <section className="relative min-h-[78vh] sm:min-h-[82vh] flex flex-col justify-between overflow-visible">
        {/* Full-bleed Farmland Background Image */}
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat transition-transform duration-1000 scale-100"
          style={{ backgroundImage: "url('/bg-farmland-hero.jpg')" }}
        />

        {/* Soft, natural light gradient overlay so the green landscape shines through vibrantly while text is crisp */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/45 via-black/25 to-black/60 pointer-events-none" />

        {/* Center Hero Content */}
        <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 pt-16 sm:pt-24 text-center space-y-5">
          {/* Subtitle Badge matching "WELCOME TO FARMLAND" in reference */}
          <div className="inline-block">
            <span className="text-xs sm:text-sm font-extrabold uppercase tracking-[0.25em] text-emerald-200 drop-shadow-[0_2px_4px_rgba(0,0,0,0.9)] px-4 py-1.5 rounded-full bg-black/30 border border-emerald-400/30 backdrop-blur-sm">
              WELCOME TO CROP HEALTH AI · PRECISION AGRICULTURE
            </span>
          </div>

          {/* Main Headline matching reference */}
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black text-white tracking-tight uppercase leading-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.9)]">
            AGRICULTURE IS THE MOST HEALTHFUL
          </h1>

          {/* Descriptive Subtitle */}
          <p className="text-sm sm:text-lg text-emerald-50 max-w-2xl mx-auto leading-relaxed drop-shadow-[0_2px_8px_rgba(0,0,0,0.95)] font-medium">
            Fusing Sentinel-2 optical spectral reflectance with Sentinel-1 C-band synthetic aperture radar backscatter for proactive, field-level crop health monitoring.
          </p>

          {/* Two Action Buttons matching reference style */}
          <div className="pt-3 flex flex-wrap items-center justify-center gap-3.5">
            <Link
              href="/analyze"
              className="px-7 py-3 rounded-lg bg-[#387038] hover:bg-[#2d5d2d] text-white font-bold text-sm tracking-wide shadow-xl transition transform active:scale-95 border border-emerald-400/40 flex items-center gap-2 cursor-pointer"
            >
              <span>Our Services · Analyze Field</span>
              <span>→</span>
            </Link>
            <Link
              href="/farmer-dashboard"
              className="px-7 py-3 rounded-lg bg-[#d9c480] hover:bg-[#c9b26e] text-slate-950 font-extrabold text-sm tracking-wide shadow-xl transition transform active:scale-95 border border-amber-300/60 flex items-center gap-2 cursor-pointer"
            >
              <span>🌾 Farmer Dashboard</span>
              <span>→</span>
            </Link>
          </div>
        </div>

        {/* 4 Overlapping Floating Cards at Bottom of Hero (Matching Reference) */}
        <div className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full transform translate-y-10 sm:translate-y-14">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Card 1: Multispectral Optics */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider block mb-1">
                  Optical Pipeline
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Multispectral Vegetation
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Sentinel-2 calibrated 10m surface reflectance across NDVI, NDMI, NDRE, and EVI canopy vigor.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-emerald-800">
                <span>13 Spectral Bands</span>
                <span>10m Resolution</span>
              </div>
            </div>

            {/* Card 2: SAR Radar */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-sky-700 uppercase tracking-wider block mb-1">
                  Radar Pipeline
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  All-Weather C-Band SAR
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Sentinel-1 active microwave penetration assessing stalk structure, moisture &amp; surface roughness.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-sky-800">
                <span>VV &amp; VH Polarized</span>
                <span>Cloud Penetrating</span>
              </div>
            </div>

            {/* Card 3: AI Stress Diagnostics */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-amber-700 uppercase tracking-wider block mb-1">
                  Intelligence Layer
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Early Crop Stress
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Calibrated detection identifying canopy moisture stress, chlorophyll loss, and micro-climate anomalies.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-amber-800">
                <span>Early Warning</span>
                <span>Field-Level</span>
              </div>
            </div>

            {/* Card 4: Farmer Intelligence */}
            <div className="bg-white/95 hover:bg-white text-slate-900 rounded-xl p-5 shadow-2xl border border-white/60 backdrop-blur-md transition-all duration-300 hover:-translate-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-teal-700 uppercase tracking-wider block mb-1">
                  Farmer Hub
                </span>
                <h3 className="font-extrabold text-base text-slate-900 mb-2">
                  Actionable Operations
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Aggregated field intelligence, multi-temporal trajectories, and localized agronomic management.
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-teal-800">
                <span>Decision Support</span>
                <span>Trend Analytics</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Area (padding-top accounts for the overlapping floating cards) */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 sm:pt-28 pb-16 space-y-12">
        {/* Constellation Comparison Section */}
        <section className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-10 shadow-2xl space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
              Dual-Sensor Satellite Fusion Architecture
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
          </div>
        </div>
      </footer>
    </div>
  );
}
