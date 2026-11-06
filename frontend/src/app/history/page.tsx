"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import TimeSeriesChart, { ObservationDataPoint, IndexType } from "@/components/TimeSeriesChart";
import ChangeMapViewer, { ComparisonObservation, DeltaInfo, ChangeBreakdown } from "@/components/ChangeMapViewer";
import { API_ROUTES } from "@/config/api";
import GlobalDemoIndicator from "@/components/GlobalDemoIndicator";

interface LocationData {
  query: string;
  display_name: string;
  latitude: number;
  longitude: number;
}

interface TrendSummary {
  observation_count: number;
  tolerance: number;
  start_date: string;
  latest_date: string;
  indices: Record<
    string,
    {
      start_value: number | null;
      latest_value: number | null;
      absolute_change: number;
      percentage_change: number;
      trend: "Increasing" | "Stable" | "Decreasing" | "Insufficient Data";
    }
  >;
  scientific_interpretation: string;
  disclaimer: string;
}

interface HistoryResponse {
  status: string;
  session_id: string;
  is_demo: boolean;
  observation_count: number;
  date_range: {
    start_date: string;
    end_date: string;
    total_days?: number;
  };
  trend_summary: TrendSummary;
  observations: ObservationDataPoint[];
  sampling_logic?: string;
}

interface CompareResponse {
  status: string;
  observation_a: ComparisonObservation;
  observation_b: ComparisonObservation;
  deltas: Record<string, DeltaInfo>;
  change_map: string;
  change_breakdown: ChangeBreakdown;
}

export default function HistoryPage() {
  // Field Location state (Default: agricultural region in Andhra Pradesh)
  const [locationName, setLocationName] = useState("Tadepalligudem, Andhra Pradesh");
  const [latitude, setLatitude] = useState(16.8102);
  const [longitude, setLongitude] = useState(81.5266);
  const [radiusKm, setRadiusKm] = useState(2.0);

  // Date range presets
  const [datePreset, setDatePreset] = useState<"30days" | "3months" | "6months" | "custom">("3months");
  const [startDate, setStartDate] = useState("2026-07-27");
  const [endDate, setEndDate] = useState("2026-10-27");
  const [cloudThreshold, setCloudThreshold] = useState(30.0);
  const [tolerance, setTolerance] = useState(0.03);

  // Processing state
  const [isSearching, setIsSearching] = useState(false);
  const [searchStep, setSearchStep] = useState<string | null>(null);
  const [historyData, setHistoryData] = useState<HistoryResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Chart configuration
  const [selectedIndex, setSelectedIndex] = useState<IndexType>("ndvi");
  const [multiIndexMode, setMultiIndexMode] = useState(false);
  const [selectedMultiIndices, setSelectedMultiIndices] = useState<IndexType[]>(["ndvi", "ndmi"]);

  // Pairwise Comparison selection
  const [obsAId, setObsAId] = useState<string | null>(null);
  const [obsBId, setObsBId] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<CompareResponse | null>(null);

  // Initialize date preset on mount
  useEffect(() => {
    applyPreset("3months");
  }, []);

  const applyPreset = (preset: "30days" | "3months" | "6months" | "custom") => {
    setDatePreset(preset);
    const end = new Date("2026-10-27T00:00:00");
    const endStr = end.toISOString().split("T")[0];
    setEndDate(endStr);

    let start = new Date(end);
    if (preset === "30days") {
      start.setDate(end.getDate() - 30);
    } else if (preset === "3months") {
      start.setMonth(end.getMonth() - 3);
    } else if (preset === "6months") {
      start.setMonth(end.getMonth() - 6);
    } else {
      return; // custom
    }
    setStartDate(start.toISOString().split("T")[0]);
  };

  const handleSearchLocation = async () => {
    if (!locationName.trim()) return;
    try {
      const res = await fetch(`${API_ROUTES.GEOCODE}?query=${encodeURIComponent(locationName)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.results && data.results.length > 0) {
          const first = data.results[0];
          setLatitude(first.latitude);
          setLongitude(first.longitude);
        }
      }
    } catch (e) {
      console.warn("Geocoding failed, keeping current coordinates:", e);
    }
  };

  const handleFetchHistory = async () => {
    setIsSearching(true);
    setErrorMessage(null);
    setComparisonResult(null);

    // Progress step simulation
    setSearchStep("🛰️ Querying Copernicus Sentinel Hub Catalog...");
    const stepTimer1 = setTimeout(() => {
      setSearchStep("🔍 Filtering clouds & temporal sampling non-redundant scenes...");
    }, 900);
    const stepTimer2 = setTimeout(() => {
      setSearchStep("🌿 Computing NDVI, NDMI, NDRE & EVI spectral time-series...");
    }, 2200);

    try {
      const payload = {
        latitude,
        longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: cloudThreshold,
        tolerance,
      };

      const res = await fetch(API_ROUTES.HISTORY, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with status ${res.status}`);
      }

      const data: HistoryResponse = await res.json();
      setHistoryData(data);

      // Pre-select first and latest observations for quick comparison
      if (data.observations && data.observations.length >= 2) {
        setObsAId(data.observations[0].observation_id);
        setObsBId(data.observations[data.observations.length - 1].observation_id);
      } else if (data.observations && data.observations.length === 1) {
        setObsAId(data.observations[0].observation_id);
        setObsBId(null);
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to retrieve historical observations.");
    } finally {
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      setIsSearching(false);
      setSearchStep(null);
    }
  };

  const handleRunComparison = async () => {
    if (!historyData || !obsAId || !obsBId || obsAId === obsBId) return;

    setIsComparing(true);
    try {
      const res = await fetch(API_ROUTES.HISTORY_COMPARE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: historyData.session_id,
          observation_a_id: obsAId,
          observation_b_id: obsBId,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Pairwise comparison failed.");
      }

      const compData: CompareResponse = await res.json();
      setComparisonResult(compData);
    } catch (err: any) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      setIsComparing(false);
    }
  };

  const toggleMultiIndex = (idx: IndexType) => {
    if (selectedMultiIndices.includes(idx)) {
      if (selectedMultiIndices.length > 1) {
        setSelectedMultiIndices(selectedMultiIndices.filter((i) => i !== idx));
      }
    } else {
      setSelectedMultiIndices([...selectedMultiIndices, idx]);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white relative">
      {/* Full-Page Background: Lush Aerial Paddy Fields */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none -z-20 transform scale-105"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      {/* Ambient glassmorphic tint so the green fields are vibrant while all UI components remain clear */}
      <div className="fixed inset-0 bg-gradient-to-b from-black/45 via-slate-950/60 to-black/75 pointer-events-none -z-10 backdrop-blur-[1px]" />
      {/* Global Demo Indicator — visible strictly only in demo mode */}
      {historyData?.is_demo && (
        <GlobalDemoIndicator
          isDemo={true}
          provenance="Multi-temporal Sentinel-1 SAR & Sentinel-2 Optical Observations"
        />
      )}
      {/* Navigation Header */}
      <header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-xl overflow-hidden border border-emerald-500/40 bg-emerald-950/60 shadow-sm flex items-center justify-center">
                <img src="/logo.png" alt="Crop Health AI Logo" className="h-full w-full object-cover" />
              </div>
              <span className="font-bold text-lg text-white tracking-tight">CropHealth</span>
            </Link>
            <span className="text-xs text-emerald-400 font-medium px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 hidden sm:inline-block">
              Multimodal History & Trends
            </span>
          </div>

          <nav className="flex items-center gap-4 sm:gap-6 text-sm">
            <Link href="/" className="text-slate-400 hover:text-white transition">
              Home
            </Link>
            <Link href="/analyze" className="text-slate-400 hover:text-white transition">
              Analyze
            </Link>
            <Link href="/farmer-dashboard" className="text-emerald-400 hover:text-emerald-300 font-semibold transition flex items-center gap-1">
              <span>🌾</span>
              <span>Farmer Dashboard</span>
            </Link>
            <Link href="/history" className="text-emerald-400 font-bold transition flex items-center gap-1">
              <span>📈</span>
              <span>History</span>
            </Link>
            <Link href="/model-performance" className="text-slate-400 hover:text-white transition hidden md:inline">
              Architecture & Model Targets
            </Link>
            <a
              href={API_ROUTES.DOCS}
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono text-slate-400 hover:text-emerald-400 transition hidden lg:inline"
            >
              API Docs ↗
            </a>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 relative z-10">
        {/* Title & Introduction (High-Contrast Floating Header) */}
        <div className="text-center space-y-3 pt-2 pb-1">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950/85 text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-lg">
            <span>🛰️</span>
            <span>Temporal Change &amp; Satellite Time-Series</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.95)]">
            Historical Field Monitoring
          </h1>
          <p className="text-slate-200 text-sm sm:text-base max-w-3xl mx-auto drop-shadow-[0_2px_8px_rgba(0,0,0,0.95)] font-medium leading-relaxed">
            Track how vegetation signals change across satellite observations. Evaluate seasonal progression, detect shifts
            in canopy vigor, and perform pairwise spatial change detection across chronological Sentinel-2 acquisitions.
          </p>
        </div>

        {/* Demo Data Banner if in demo mode */}
        {historyData?.is_demo && (
          <div className="p-4 rounded-2xl bg-amber-950/40 border border-amber-500/30 text-amber-200 text-xs flex items-center justify-between gap-4 backdrop-blur-sm">
            <div className="flex items-center gap-2.5">
              <span className="text-lg">ℹ️</span>
              <div>
                <span className="font-bold text-amber-300 uppercase tracking-wider">DEMO DATA: </span>
                Copernicus credentials are not configured or live satellite search fallback was engaged. Historical
                observations are generated deterministically using verified synthetic spectral reflectance arrays.
              </div>
            </div>
            <span className="px-2.5 py-1 rounded-md bg-amber-500/20 border border-amber-500/40 font-mono text-[11px] font-bold">
              REPRODUCIBLE
            </span>
          </div>
        )}

        {/* Control Dashboard: Field Selection & Date Range */}
        <section className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-5 sm:p-7 shadow-2xl backdrop-blur-md space-y-6">
          <h2 className="text-base font-semibold text-white tracking-tight flex items-center gap-2 border-b border-slate-800 pb-3">
            <span>📍 Field & Observation Parameters</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Location & AOI */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Search Location / Field</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={locationName}
                    onChange={(e) => setLocationName(e.target.value)}
                    placeholder="e.g. Tadepalligudem, AP"
                    className="flex-1 px-3.5 py-2.5 bg-slate-950 border border-slate-700/80 rounded-xl text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition"
                  />
                  <button
                    onClick={handleSearchLocation}
                    className="px-3.5 py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-xs font-semibold text-slate-200 transition"
                  >
                    Locate
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Latitude</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={latitude}
                    onChange={(e) => setLatitude(parseFloat(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white font-mono text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Longitude</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={longitude}
                    onChange={(e) => setLongitude(parseFloat(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white font-mono text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400">AOI Radius</span>
                  <span className="font-mono text-emerald-400 font-semibold">{radiusKm.toFixed(1)} km</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="10.0"
                  step="0.5"
                  value={radiusKm}
                  onChange={(e) => setRadiusKm(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>
            </div>

            {/* Date Range Selection */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Observation Window</label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => applyPreset("30days")}
                    className={`py-2 rounded-lg text-xs font-medium transition ${
                      datePreset === "30days"
                        ? "bg-emerald-500 text-slate-950 font-bold"
                        : "bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800"
                    }`}
                  >
                    Last 30 Days
                  </button>
                  <button
                    onClick={() => applyPreset("3months")}
                    className={`py-2 rounded-lg text-xs font-medium transition ${
                      datePreset === "3months"
                        ? "bg-emerald-500 text-slate-950 font-bold"
                        : "bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800"
                    }`}
                  >
                    Last 3 Months
                  </button>
                  <button
                    onClick={() => applyPreset("6months")}
                    className={`py-2 rounded-lg text-xs font-medium transition ${
                      datePreset === "6months"
                        ? "bg-emerald-500 text-slate-950 font-bold"
                        : "bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800"
                    }`}
                  >
                    Last 6 Months
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Start Date</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => {
                      setStartDate(e.target.value);
                      setDatePreset("custom");
                    }}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white font-mono text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">End Date</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => {
                      setEndDate(e.target.value);
                      setDatePreset("custom");
                    }}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white font-mono text-xs focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400">Trend Tolerance (±)</span>
                  <span className="font-mono text-emerald-400 font-semibold">{tolerance.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.10"
                  step="0.01"
                  value={tolerance}
                  onChange={(e) => setTolerance(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>
            </div>

            {/* Cloud Threshold & Action */}
            <div className="flex flex-col justify-between space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Max Cloud Coverage</span>
                  <span className="font-mono text-emerald-400 font-semibold">≤ {cloudThreshold.toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="60"
                  step="5"
                  value={cloudThreshold}
                  onChange={(e) => setCloudThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <p className="text-[11px] text-slate-500 mt-2">
                  Observations with cloud cover exceeding this threshold are excluded to preserve optical spectral purity.
                </p>
              </div>

              {/* Submit Button */}
              <div className="pt-2">
                <button
                  onClick={handleFetchHistory}
                  disabled={isSearching}
                  className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 text-slate-950 font-bold text-sm rounded-xl shadow-lg shadow-emerald-500/20 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
                >
                  {isSearching ? (
                    <>
                      <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                      <span>Processing Observations...</span>
                    </>
                  ) : (
                    <>
                      <span>📊 Load History</span>
                      <span>→</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Progress Step Indicator */}
          {searchStep && (
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-emerald-300 flex items-center gap-3 animate-pulse">
              <div className="w-3.5 h-3.5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin shrink-0" />
              <span>{searchStep}</span>
            </div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-500/40 text-rose-200 text-xs space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <span>⚠️</span>
                <span>No Suitable Observations Found</span>
              </div>
              <p>{errorMessage}</p>
              <div className="pt-2 text-[11px] text-rose-300/80">
                Suggestions: (1) Expand the date range window, (2) Increase the maximum cloud coverage threshold, or (3) Select an alternative field coordinate.
              </div>
            </div>
          )}
        </section>

        {/* Historical Analysis Dashboard (rendered once data is loaded) */}
        {historyData && historyData.observations && historyData.observations.length > 0 && (
          <div className="space-y-8">
            {/* Index Selector Bar */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-900/60 p-3 rounded-2xl border border-slate-800">
              {/* Single Index Tabs */}
              {!multiIndexMode ? (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium px-2">Index:</span>
                  {(["ndvi", "ndmi", "ndre", "evi"] as IndexType[]).map((idx) => {
                    const isActive = selectedIndex === idx;
                    const labels: Record<IndexType, string> = {
                      ndvi: "🌿 NDVI (Greenness)",
                      ndmi: "💧 NDMI (Moisture)",
                      ndre: "🍂 NDRE (Chlorophyll)",
                      evi: "🌾 EVI (Enhanced)",
                    };
                    return (
                      <button
                        key={idx}
                        onClick={() => setSelectedIndex(idx)}
                        className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition ${
                          isActive
                            ? "bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20"
                            : "bg-slate-950 text-slate-300 hover:text-white border border-slate-800"
                        }`}
                      >
                        {labels[idx]}
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium px-2">Multi-Series:</span>
                  {(["ndvi", "ndmi", "ndre", "evi"] as IndexType[]).map((idx) => {
                    const isChecked = selectedMultiIndices.includes(idx);
                    return (
                      <button
                        key={idx}
                        onClick={() => toggleMultiIndex(idx)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition border ${
                          isChecked
                            ? "bg-slate-800 text-white border-slate-600"
                            : "bg-slate-950 text-slate-500 border-slate-800"
                        }`}
                      >
                        {isChecked ? "✓ " : "+ "}
                        {idx.toUpperCase()}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Multi-Index Toggle */}
              <button
                onClick={() => setMultiIndexMode(!multiIndexMode)}
                className="px-3.5 py-1.5 rounded-xl text-xs font-medium bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-700 transition"
              >
                {multiIndexMode ? "Switch to Single Index" : "Multi-Index Overlay"}
              </button>
            </div>

            {/* Time-Series Chart */}
            <TimeSeriesChart
              observations={historyData.observations}
              selectedIndex={selectedIndex}
              multiIndexMode={multiIndexMode}
              selectedIndices={selectedMultiIndices}
              selectedObsAId={obsAId}
              selectedObsBId={obsBId}
              onSelectObservation={(obs) => {
                if (!obsAId) {
                  setObsAId(obs.observation_id);
                } else if (!obsBId) {
                  setObsBId(obs.observation_id);
                } else {
                  setObsAId(obs.observation_id);
                }
              }}
            />

            {/* Trend Summary Section */}
            {historyData.trend_summary && (
              <section className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-5 sm:p-7 shadow-2xl backdrop-blur-md space-y-5">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
                      <span>📊 TREND SUMMARY</span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Evaluated between first observation ({historyData.trend_summary.start_date}) and latest observation ({historyData.trend_summary.latest_date}).
                    </p>
                  </div>

                  <div className="text-xs text-slate-400">
                    Classification tolerance: <span className="font-mono text-white">±{historyData.trend_summary.tolerance}</span>
                  </div>
                </div>

                {/* Index Trend Badges Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {(["ndvi", "ndmi", "ndre", "evi"] as IndexType[]).map((idxKey) => {
                    const trendInfo = historyData.trend_summary.indices[idxKey];
                    if (!trendInfo) return null;

                    const trend = trendInfo.trend;
                    const isInc = trend === "Increasing";
                    const isDec = trend === "Decreasing";
                    const isStable = trend === "Stable";

                    return (
                      <div
                        key={idxKey}
                        className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold uppercase text-slate-300 tracking-wider">
                            {idxKey.toUpperCase()}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              isInc
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : isDec
                                ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                                : isStable
                                ? "bg-slate-800 text-slate-300 border border-slate-700"
                                : "bg-slate-800 text-slate-500"
                            }`}
                          >
                            {trend}
                          </span>
                        </div>

                        <div className="space-y-0.5">
                          <div className="text-lg font-bold font-mono text-white">
                            {trendInfo.latest_value !== null ? trendInfo.latest_value.toFixed(3) : "N/A"}
                          </div>
                          <div className="text-xs font-mono text-slate-400 flex items-center gap-1">
                            <span>Δ {trendInfo.absolute_change > 0 ? `+${trendInfo.absolute_change.toFixed(3)}` : trendInfo.absolute_change.toFixed(3)}</span>
                            <span className="text-[11px] text-slate-500">
                              ({trendInfo.percentage_change > 0 ? `+${trendInfo.percentage_change.toFixed(1)}%` : `${trendInfo.percentage_change.toFixed(1)}%`})
                            </span>
                          </div>
                        </div>

                        <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-900 flex justify-between">
                          <span>Start: {trendInfo.start_value !== null ? trendInfo.start_value.toFixed(2) : "-"}</span>
                          <span>Latest: {trendInfo.latest_value !== null ? trendInfo.latest_value.toFixed(2) : "-"}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Scientific Statement Card */}
                <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 text-xs text-slate-300 space-y-1.5 leading-relaxed">
                  <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
                    <span>🔬</span>
                    <span>Observation Trend Analysis:</span>
                  </div>
                  <p>{historyData.trend_summary.scientific_interpretation}</p>
                  <p className="text-[11px] text-slate-500 pt-1">
                    {historyData.trend_summary.disclaimer}
                  </p>
                </div>
              </section>
            )}

            {/* Observation Table */}
            <section className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-5 sm:p-7 shadow-2xl backdrop-blur-md space-y-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
                    <span>📋 Observation Log</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Chronological record of processed Sentinel-2 acquisitions. Select any two observations to compare.
                  </p>
                </div>

                {/* Compare Action Button */}
                <button
                  onClick={handleRunComparison}
                  disabled={!obsAId || !obsBId || obsAId === obsBId || isComparing}
                  className="px-4 py-2 bg-gradient-to-r from-amber-500 to-emerald-500 hover:from-amber-400 hover:to-emerald-400 disabled:from-slate-800 disabled:to-slate-800 text-slate-950 font-bold text-xs rounded-xl shadow-md transition disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isComparing ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                      <span>Computing Map...</span>
                    </>
                  ) : (
                    <>
                      <span>🔄 Compare Observation A & B</span>
                    </>
                  )}
                </button>
              </div>

              {/* Table with responsive horizontal scroll */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs whitespace-nowrap">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                      <th className="py-3 px-3">Date</th>
                      <th className="py-3 px-3">Satellite</th>
                      <th className="py-3 px-3 text-emerald-400">S2 Avail</th>
                      <th className="py-3 px-3 text-sky-400">S1 Avail</th>
                      <th className="py-3 px-3">Cloud %</th>
                      <th className="py-3 px-3 text-emerald-400">NDVI</th>
                      <th className="py-3 px-3 text-cyan-400">NDMI</th>
                      <th className="py-3 px-3 text-amber-400">NDRE</th>
                      <th className="py-3 px-3 text-purple-400">EVI</th>
                      <th className="py-3 px-3 text-sky-400">VV (dB)</th>
                      <th className="py-3 px-3 text-violet-400">VH (dB)</th>
                      <th className="py-3 px-3 text-amber-400">VV/VH</th>
                      <th className="py-3 px-3">Valid %</th>
                      <th className="py-3 px-3">Quality</th>
                      <th className="py-3 px-3 text-center">Compare Selection</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {historyData.observations.map((obs) => {
                      const isObsA = obs.observation_id === obsAId;
                      const isObsB = obs.observation_id === obsBId;
                      const hasS1 = Boolean(obs.sar && obs.sar.vv && obs.sar.vv.mean !== null);

                      return (
                        <tr
                          key={obs.observation_id}
                          className={`hover:bg-slate-800/40 transition ${
                            isObsA
                              ? "bg-sky-950/30"
                              : isObsB
                              ? "bg-amber-950/30"
                              : ""
                          }`}
                        >
                          <td className="py-3 px-3 font-sans font-medium text-white">
                            <div className="flex flex-col">
                              <span>📅 {obs.date}</span>
                              {obs.sar?.date && obs.sar.date !== obs.date && (
                                <span className="text-[10px] text-sky-400 font-mono">S1: {obs.sar.date}</span>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-3 font-sans">
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 border border-slate-700 text-slate-300">
                              {hasS1 ? "S2 + S1" : "S2 Only"}
                            </span>
                          </td>
                          <td className="py-3 px-3 font-sans">
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-400">
                              ✓ Available
                            </span>
                          </td>
                          <td className="py-3 px-3 font-sans">
                            {hasS1 ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-500/20 text-sky-400">
                                ✓ Available
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300">
                                ⚠️ Unavailable
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-3 text-slate-300">{obs.cloud_coverage.toFixed(1)}%</td>
                          <td className="py-3 px-3 text-emerald-400 font-bold">
                            {obs.ndvi.mean !== null ? obs.ndvi.mean.toFixed(3) : "—"}
                          </td>
                          <td className="py-3 px-3 text-cyan-400">
                            {obs.ndmi.mean !== null ? obs.ndmi.mean.toFixed(3) : "—"}
                          </td>
                          <td className="py-3 px-3 text-amber-400">
                            {obs.ndre.mean !== null ? obs.ndre.mean.toFixed(3) : "—"}
                          </td>
                          <td className="py-3 px-3 text-purple-400">
                            {obs.evi.mean !== null ? obs.evi.mean.toFixed(3) : "—"}
                          </td>
                          <td className="py-3 px-3 text-sky-400 font-bold">
                            {obs.sar?.vv?.mean !== null && obs.sar?.vv?.mean !== undefined
                              ? `${obs.sar.vv.mean.toFixed(2)}`
                              : "—"}
                          </td>
                          <td className="py-3 px-3 text-violet-400 font-bold">
                            {obs.sar?.vh?.mean !== null && obs.sar?.vh?.mean !== undefined
                              ? `${obs.sar.vh.mean.toFixed(2)}`
                              : "—"}
                          </td>
                          <td className="py-3 px-3 text-amber-400 font-bold">
                            {obs.sar?.vv_vh_ratio?.mean !== null && obs.sar?.vv_vh_ratio?.mean !== undefined
                              ? `${obs.sar.vv_vh_ratio.mean.toFixed(2)}`
                              : "—"}
                          </td>
                          <td className="py-3 px-3 text-slate-300">{obs.valid_pixel_percentage.toFixed(1)}%</td>
                          <td className="py-3 px-3 font-sans">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                obs.quality === "Excellent"
                                  ? "bg-emerald-500/20 text-emerald-400"
                                  : obs.quality === "Good"
                                  ? "bg-teal-500/20 text-teal-300"
                                  : obs.quality === "Fair"
                                  ? "bg-amber-500/20 text-amber-300"
                                  : "bg-rose-500/20 text-rose-300"
                              }`}
                            >
                              {obs.quality}
                            </span>
                          </td>
                          <td className="py-3 px-3 font-sans text-center">
                            <div className="inline-flex items-center gap-1.5">
                              <button
                                onClick={() => setObsAId(obs.observation_id)}
                                className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                                  isObsA
                                    ? "bg-sky-500 text-slate-950 font-bold"
                                    : "bg-slate-800 text-slate-400 hover:text-white"
                                }`}
                              >
                                {isObsA ? "✓ Obs A" : "Set A"}
                              </button>
                              <button
                                onClick={() => setObsBId(obs.observation_id)}
                                className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                                  isObsB
                                    ? "bg-amber-500 text-slate-950 font-bold"
                                    : "bg-slate-800 text-slate-400 hover:text-white"
                                }`}
                              >
                                {isObsB ? "✓ Obs B" : "Set B"}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            {/* MULTIMODAL OBSERVATION TIMELINE */}
            <section className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-5 sm:p-7 shadow-2xl backdrop-blur-md space-y-5">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
                    <span>⏱️ MULTIMODAL OBSERVATION TIMELINE</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Unified temporal alignment tracking Sentinel-2 optical passes and matched Sentinel-1 SAR microwave acquisitions.
                  </p>
                </div>
                <span className="text-xs font-mono text-sky-400 bg-sky-500/10 px-2.5 py-1 rounded-full border border-sky-500/20">
                  C-band SAR + Multispectral
                </span>
              </div>

              <div className="space-y-4 pt-2">
                {historyData.observations.map((obs) => {
                  const hasS1 = Boolean(obs.sar && obs.sar.vv && obs.sar.vv.mean !== null);

                  return (
                    <div key={`timeline-${obs.observation_id}`} className="relative pl-6 border-l-2 border-slate-800 space-y-3 pb-3">
                      <div className="absolute -left-[9px] top-0.5 w-4 h-4 rounded-full bg-slate-900 border-2 border-emerald-400 flex items-center justify-center">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      </div>

                      <div className="flex flex-wrap items-center gap-3">
                        <span className="font-mono font-bold text-sm text-white">📅 {obs.date}</span>
                        <span className="text-[11px] text-slate-400">Quality: <strong className="text-slate-200">{obs.quality}</strong> ({obs.valid_pixel_percentage.toFixed(1)}% valid)</span>
                        {obs.sar?.date && obs.sar.date !== obs.date && (
                          <span className="text-[10px] text-sky-400 font-mono bg-sky-950/60 px-2 py-0.5 rounded border border-sky-500/30">
                            S1 Pass: {obs.sar.date}
                          </span>
                        )}
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {/* Sentinel-2 Branch */}
                        <div className="p-3 rounded-xl bg-slate-950/80 border border-emerald-500/30 space-y-2">
                          <div className="flex items-center justify-between text-xs font-semibold">
                            <span className="text-emerald-400 flex items-center gap-1.5">
                              <span>🌿</span> Sentinel-2 (Optical)
                            </span>
                            <span className="text-[10px] text-slate-400 font-mono">
                              Cloud: {obs.cloud_coverage.toFixed(1)}%
                            </span>
                          </div>
                          <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                            <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                              <span className="text-[10px] text-slate-400 block">NDVI</span>
                              <span className="text-emerald-400 font-bold">{obs.ndvi.mean !== null ? obs.ndvi.mean.toFixed(3) : "—"}</span>
                            </div>
                            <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                              <span className="text-[10px] text-slate-400 block">NDMI</span>
                              <span className="text-cyan-400 font-bold">{obs.ndmi.mean !== null ? obs.ndmi.mean.toFixed(3) : "—"}</span>
                            </div>
                            <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                              <span className="text-[10px] text-slate-400 block">NDRE</span>
                              <span className="text-amber-400 font-bold">{obs.ndre.mean !== null ? obs.ndre.mean.toFixed(3) : "—"}</span>
                            </div>
                            <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                              <span className="text-[10px] text-slate-400 block">EVI</span>
                              <span className="text-purple-400 font-bold">{obs.evi.mean !== null ? obs.evi.mean.toFixed(3) : "—"}</span>
                            </div>
                          </div>
                        </div>

                        {/* Sentinel-1 Branch */}
                        <div className={`p-3 rounded-xl bg-slate-950/80 border space-y-2 ${
                          hasS1 ? "border-sky-500/30" : "border-slate-800 opacity-80"
                        }`}>
                          <div className="flex items-center justify-between text-xs font-semibold">
                            <span className="text-sky-400 flex items-center gap-1.5">
                              <span>📡</span> Sentinel-1 (C-Band SAR)
                            </span>
                            <span className="text-[10px] text-slate-400 font-mono">
                              {hasS1 ? (obs.sar?.product_type || "Sentinel-1 GRD") : "Unavailable"}
                            </span>
                          </div>
                          {hasS1 ? (
                            <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                              <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                                <span className="text-[10px] text-slate-400 block">VV Backscatter</span>
                                <span className="text-sky-400 font-bold">{obs.sar?.vv?.mean !== null ? `${obs.sar?.vv?.mean?.toFixed(2)} dB` : "—"}</span>
                              </div>
                              <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                                <span className="text-[10px] text-slate-400 block">VH Backscatter</span>
                                <span className="text-violet-400 font-bold">{obs.sar?.vh?.mean !== null ? `${obs.sar?.vh?.mean?.toFixed(2)} dB` : "—"}</span>
                              </div>
                              <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                                <span className="text-[10px] text-slate-400 block">VV/VH Ratio</span>
                                <span className="text-amber-400 font-bold">{obs.sar?.vv_vh_ratio?.mean !== null ? obs.sar?.vv_vh_ratio?.mean?.toFixed(2) : "—"}</span>
                              </div>
                            </div>
                          ) : (
                            <div className="py-2.5 px-3 rounded-lg bg-amber-950/20 border border-amber-500/20 text-[11px] text-amber-300 flex items-center gap-2">
                              <span>⚠️</span>
                              <span>Sentinel-1 observation unavailable for this time window.</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Pairwise Comparison & Change Map Section */}
            {comparisonResult && (
              <ChangeMapViewer
                observationA={comparisonResult.observation_a}
                observationB={comparisonResult.observation_b}
                deltas={comparisonResult.deltas}
                changeMapUrl={comparisonResult.change_map}
                changeBreakdown={comparisonResult.change_breakdown}
              />
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-emerald-900/40 bg-slate-950/90 backdrop-blur-md py-6 text-center text-xs text-slate-400 mt-16 relative z-10">
        <div className="max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Crop Health AI. Multi-Temporal Satellite Vegetation Intelligence.</p>
          <p className="text-slate-400">Powered by Copernicus Sentinel-2 L2A & Temporal Change Detection</p>
        </div>
      </footer>
    </div>
  );
}
