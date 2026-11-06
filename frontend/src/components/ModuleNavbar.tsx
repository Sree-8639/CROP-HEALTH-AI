"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSharedAOI, PRESET_AOIS } from "@/context/SharedAOIContext";
import { API_BASE } from "@/config/api";

interface ModuleNavbarProps {
  currentModule?: "crop" | "deforestation" | "heat" | "other";
}

export default function ModuleNavbar({ currentModule }: ModuleNavbarProps) {
  const pathname = usePathname();
  const { aoiType, center, radiusKm, polygonPoints, startDate, endDate, selectedPresetId } = useSharedAOI();
  const [apiStatus, setApiStatus] = useState<"connected" | "checking" | "offline">("checking");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
        if (res.ok && isMounted) {
          setApiStatus("connected");
        } else if (isMounted) {
          setApiStatus("offline");
        }
      } catch {
        if (isMounted) setApiStatus("offline");
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const activeMod =
    currentModule ||
    (pathname.startsWith("/crop")
      ? "crop"
      : pathname.startsWith("/deforestation") || pathname.startsWith("/forest")
      ? "deforestation"
      : pathname.startsWith("/heat")
      ? "heat"
      : "other");

  const currentPreset = PRESET_AOIS.find((p) => p.id === selectedPresetId);

  return (
    <header className="bg-slate-950/95 border-b border-slate-800/80 backdrop-blur-md sticky top-0 z-50 shadow-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="h-16 flex items-center justify-between gap-4">
          {/* Logo & Main Title */}
          <Link href="/" className="flex items-center gap-3 group shrink-0">
            <div className="h-10 w-10 rounded-xl overflow-hidden border border-emerald-500/40 bg-emerald-950/70 shadow-md flex items-center justify-center p-0.5 group-hover:border-emerald-400 transition">
              <img src="/logo.png" alt="CropHealth AI Logo" className="h-full w-full object-cover rounded-lg" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base sm:text-lg text-white tracking-tight group-hover:text-emerald-300 transition">
                  CropHealth AI
                </span>
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-900/60 border border-emerald-500/40 text-emerald-300 hidden md:inline-block">
                  Earth Observation
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Multimodal Satellite Intelligence Suite
              </p>
            </div>
          </Link>

          {/* Core Module Switcher (The 3 Modules) */}
          <nav className="flex items-center bg-slate-900/90 p-1 rounded-2xl border border-slate-800 shadow-inner">
            {/* Module 1: Crop */}
            <Link
              href="/crop"
              className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${
                activeMod === "crop"
                  ? "bg-emerald-600 text-white shadow-lg shadow-emerald-900/40"
                  : "text-slate-300 hover:text-white hover:bg-slate-800/60"
              }`}
            >
              <span>🌱</span>
              <span>Crop</span>
            </Link>

            {/* Module 2: Deforestation */}
            <Link
              href="/deforestation"
              className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${
                activeMod === "deforestation"
                  ? "bg-teal-600 text-white shadow-lg shadow-teal-900/40"
                  : "text-slate-300 hover:text-white hover:bg-slate-800/60"
              }`}
            >
              <span>🌲</span>
              <span>Deforestation</span>
              <span className="text-[10px] bg-teal-950/80 border border-teal-400/40 text-teal-300 px-1.5 py-0.2 rounded font-mono hidden lg:inline">
                501
              </span>
            </Link>

            {/* Module 3: Urban Heat */}
            <Link
              href="/heat"
              className={`flex items-center gap-1.5 px-3 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${
                activeMod === "heat"
                  ? "bg-amber-600 text-white shadow-lg shadow-amber-900/40"
                  : "text-slate-300 hover:text-white hover:bg-slate-800/60"
              }`}
            >
              <span>🔥</span>
              <span>Urban Heat</span>
              <span className="text-[10px] bg-amber-950/80 border border-amber-400/40 text-amber-300 px-1.5 py-0.2 rounded font-mono hidden lg:inline">
                501
              </span>
            </Link>
          </nav>

          {/* Right Action Items: Shared Context Pill & Secondary Links */}
          <div className="flex items-center gap-3">
            {/* Shared AOI & Date Snapshot Pill */}
            <div className="hidden xl:flex items-center gap-2 px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-semibold text-slate-200">
                {currentPreset ? currentPreset.name : aoiType === "radius" ? `${radiusKm}km Radius` : `${polygonPoints.length}-pt Polygon`}
              </span>
              <span className="text-slate-500">|</span>
              <span className="font-mono text-slate-400">{startDate} → {endDate}</span>
            </div>

            {/* Legacy Crop Tools Dropdown / Links */}
            <div className="hidden md:flex items-center gap-2 text-xs">
              <Link
                href="/farmer-dashboard"
                className="px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-amber-300 hover:bg-slate-800/70 font-medium transition flex items-center gap-1"
                title="Direct Farmer Intelligence Hub"
              >
                <span>🌾</span>
                <span>Farmer Hub</span>
              </Link>
              <Link
                href="/analyze"
                className="px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-emerald-300 hover:bg-slate-800/70 font-medium transition"
                title="Full Spectral Index Studio"
              >
                Analyze
              </Link>
              <Link
                href="/model-performance"
                className="px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-blue-300 hover:bg-slate-800/70 font-medium transition"
                title="Model Performance & Benchmarks"
              >
                Metrics
              </Link>
            </div>

            {/* API Status Badge */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border bg-slate-900 border-slate-800">
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "connected"
                    ? "bg-emerald-400"
                    : apiStatus === "checking"
                    ? "bg-amber-400 animate-pulse"
                    : "bg-rose-400"
                }`}
              />
              <span className="text-slate-300 font-mono text-[10px]">
                {apiStatus === "connected" ? "Live API" : apiStatus === "checking" ? "Checking" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

