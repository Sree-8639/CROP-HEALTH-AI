"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { API_BASE } from "@/config/api";
import FieldOverviewCard, { FieldOverviewData } from "./FieldOverviewCard";
import CropConditionSummary, { CropConditionSummaryData } from "./CropConditionSummary";
import VegetationMetrics, { VegetationMetricsData } from "./VegetationMetrics";
import StressAlertPanel, { StressIntelligenceData } from "./StressAlertPanel";
import GrowthForecastPanel, { GrowthForecastData } from "./GrowthForecastPanel";
import FieldRiskSummary, { FieldRiskSummaryData } from "./FieldRiskSummary";
import AgriculturalRecommendations, { AgriculturalRecommendationData } from "./AgriculturalRecommendations";
import FieldComparisonPanel, { FieldComparisonResponseData } from "./FieldComparisonPanel";
import GlobalDemoIndicator from "./GlobalDemoIndicator";

export interface FieldCatalogItem {
  field_id: string;
  field_name: string;
  crop_type: string;
  latitude: number;
  longitude: number;
  radius_km: number;
  last_analyzed: string;
}

export interface FarmerDashboardFullData {
  status: string;
  field_overview: FieldOverviewData;
  crop_condition: CropConditionSummaryData;
  vegetation_metrics: VegetationMetricsData;
  stress_intelligence: StressIntelligenceData;
  growth_forecast: GrowthForecastData;
  field_risk: FieldRiskSummaryData;
  recommendations: AgriculturalRecommendationData[];
  generated_at: string;
}

export type DashboardState =
  | "INITIAL_LOADING"
  | "SERVER_WAKING"
  | "LOADING_DATA"
  | "SUCCESS"
  | "EMPTY"
  | "ERROR"
  | "TIMEOUT";

const REQUEST_TIMEOUT_MS = 60000;
const WAKE_DETECTION_DELAY_MS = 3500;

export default function FarmerDashboard() {
  const [fieldsList, setFieldsList] = useState<FieldCatalogItem[]>([]);
  const [activeFieldId, setActiveFieldId] = useState<string>("");
  const [dashboardData, setDashboardData] = useState<FarmerDashboardFullData | null>(null);

  // 7-State Lifecycle Management (Phase 20 Issue 4)
  const [dashboardState, setDashboardState] = useState<DashboardState>("INITIAL_LOADING");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [wakeSecondsElapsed, setWakeSecondsElapsed] = useState<number>(0);

  // In-flight fetch tracking and timers
  const activeFetchAbortRef = useRef<AbortController | null>(null);
  const wakeTimerRef = useRef<NodeJS.Timeout | null>(null);
  const timeoutTimerRef = useRef<NodeJS.Timeout | null>(null);
  const wakeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isFetchingRef = useRef<boolean>(false);

  // Comparison & New AOI Modal State
  const [showComparisonView, setShowComparisonView] = useState<boolean>(false);
  const [comparisonData, setComparisonData] = useState<FieldComparisonResponseData | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [isFieldSelectorOpen, setIsFieldSelectorOpen] = useState(false);
  const [isNewAOIOpen, setIsNewAOIOpen] = useState(false);

  // New Field AOI form state
  const [newFieldName, setNewFieldName] = useState("");
  const [newCropType, setNewCropType] = useState("Paddy / Rice");
  const [newLat, setNewLat] = useState("16.8150");
  const [newLon, setNewLon] = useState("81.5280");
  const [newRadius, setNewRadius] = useState("2.0");
  const [isSubmittingField, setIsSubmittingField] = useState(false);

  const cleanupTimers = () => {
    if (wakeTimerRef.current) {
      clearTimeout(wakeTimerRef.current);
      wakeTimerRef.current = null;
    }
    if (timeoutTimerRef.current) {
      clearTimeout(timeoutTimerRef.current);
      timeoutTimerRef.current = null;
    }
    if (wakeIntervalRef.current) {
      clearInterval(wakeIntervalRef.current);
      wakeIntervalRef.current = null;
    }
  };

  // Load catalog on initial mount
  useEffect(() => {
    fetchCatalog();
    return () => {
      cleanupTimers();
      if (activeFetchAbortRef.current) {
        activeFetchAbortRef.current.abort();
      }
    };
  }, []);

  // Fetch field details whenever activeFieldId changes
  useEffect(() => {
    if (activeFieldId) {
      fetchFieldData(activeFieldId);
    }
  }, [activeFieldId]);

  const fetchCatalog = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/farmer-dashboard/overview`);
      if (res.ok) {
        const data = await res.json();
        const catalogFields: FieldCatalogItem[] = data.fields || [];
        setFieldsList(catalogFields);
        const targetId = data.active_field_id || catalogFields[0]?.field_id || "field_1";
        setActiveFieldId((prev) => prev || targetId);
      } else {
        const errJson = await res.json().catch(() => ({}));
        setDashboardState("ERROR");
        setErrorMessage(errJson.detail || `Server returned status ${res.status} when fetching field catalog.`);
      }
    } catch (err: any) {
      console.warn("Catalog fetch fallback engaged:", err);
      setDashboardState("ERROR");
      setErrorMessage(err?.message || "Failed to reach Crop Health AI server.");
    }
  };

  const fetchFieldData = async (fieldId: string) => {
    if (!fieldId) return;

    // Prevent duplicate in-flight requests
    if (isFetchingRef.current) {
      activeFetchAbortRef.current?.abort();
    }

    cleanupTimers();
    isFetchingRef.current = true;
    setErrorMessage(null);
    setWakeSecondsElapsed(0);
    setDashboardState("INITIAL_LOADING");

    const controller = new AbortController();
    activeFetchAbortRef.current = controller;

    // Detect server cold-start on Render (free tier spins down after inactivity)
    wakeTimerRef.current = setTimeout(() => {
      if (isFetchingRef.current) {
        setDashboardState("SERVER_WAKING");
        wakeIntervalRef.current = setInterval(() => {
          setWakeSecondsElapsed((prev) => prev + 1);
        }, 1000);
      }
    }, WAKE_DETECTION_DELAY_MS);

    // Timeout guardrail (60 seconds)
    timeoutTimerRef.current = setTimeout(() => {
      if (isFetchingRef.current) {
        controller.abort("TIMEOUT");
        cleanupTimers();
        isFetchingRef.current = false;
        setDashboardState("TIMEOUT");
        setErrorMessage("Request timed out after 60 seconds. The satellite processing pipeline or cloud server may still be busy.");
      }
    }, REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch(`${API_BASE}/api/v1/farmer-dashboard/field/${encodeURIComponent(fieldId)}`, {
        signal: controller.signal,
      });

      cleanupTimers();
      isFetchingRef.current = false;

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned status ${res.status}`);
      }

      setDashboardState("LOADING_DATA");
      const data: FarmerDashboardFullData = await res.json();

      if (!data || !data.field_overview) {
        setDashboardState("EMPTY");
      } else {
        setDashboardData(data);
        setDashboardState("SUCCESS");
      }
    } catch (err: any) {
      cleanupTimers();
      isFetchingRef.current = false;

      if (err.name === "AbortError" || controller.signal.aborted) {
        setDashboardState("TIMEOUT");
        setErrorMessage("Request timed out. The cloud server may still be initializing or cold-starting.");
      } else {
        setDashboardState("ERROR");
        setErrorMessage(err.message || "Unable to connect to the Crop Health AI server. Please try again.");
      }
    }
  };

  const handleRunComparison = async (fieldIds: string[]) => {
    setIsComparing(true);
    try {
      const query = `fields=${encodeURIComponent(fieldIds.join(","))}`;
      const res = await fetch(`${API_BASE}/api/v1/farmer-dashboard/compare?${query}`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Comparison failed: ${res.status}`);
      }
      const data: FieldComparisonResponseData = await res.json();
      setComparisonData(data);
    } catch (err: any) {
      alert(`Comparison error: ${err.message}`);
    } finally {
      setIsComparing(false);
    }
  };

  const handleRegisterField = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmittingField(true);
    try {
      const payload = {
        field_name: newFieldName || "Custom Agricultural Field",
        crop_type: newCropType,
        latitude: parseFloat(newLat),
        longitude: parseFloat(newLon),
        radius_km: parseFloat(newRadius),
      };

      const res = await fetch(`${API_BASE}/api/v1/farmer-dashboard/field`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Field registration failed.");
      }

      const created = await res.json();
      await fetchCatalog();
      setActiveFieldId(created.field_overview.field_id);
      setIsNewAOIOpen(false);
    } catch (err: any) {
      alert(`Registration failed: ${err.message}`);
    } finally {
      setIsSubmittingField(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-emerald-500 selection:text-white">
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
            <span className="text-xs text-emerald-300 font-medium px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              Farmer Intelligence Hub
            </span>
          </div>

          <nav className="flex items-center gap-4 sm:gap-6 text-sm">
            <Link href="/" className="text-slate-400 hover:text-white transition">
              Home
            </Link>
            <Link href="/analyze" className="text-slate-400 hover:text-white transition">
              Analyze
            </Link>
            <Link href="/history" className="text-slate-400 hover:text-white transition flex items-center gap-1">
              <span>📈</span>
              <span>History</span>
            </Link>
            <span className="text-emerald-300 font-bold border-b-2 border-emerald-400 pb-0.5 flex items-center gap-1">
              <span>🌾</span>
              <span>Dashboard</span>
            </span>
            <Link href="/model-performance" className="text-slate-400 hover:text-white transition hidden md:inline">
              Architecture &amp; Model Targets
            </Link>
            <a
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noreferrer"
              className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            >
              API Docs ↗
            </a>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 space-y-6 w-full relative z-10">
        {/* Scenic Agricultural Banner */}
        <div className="relative rounded-3xl overflow-hidden border border-emerald-500/20 shadow-2xl p-6 sm:p-8 bg-slate-900">
          <div
            className="absolute inset-0 bg-cover bg-center opacity-55 scale-105 transition-transform duration-1000"
            style={{ backgroundImage: "url('/bg-farm.jpg')" }}
          />
          <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-slate-950/80 to-transparent" />
          <div className="relative z-10 max-w-2xl">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 mb-3">
              {dashboardData?.field_overview?.is_demo === false ? "🌱 Live Field Intelligence Hub" : "🌱 Satellite Field Intelligence Hub"}
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Farmer Agricultural Operations
            </h1>
            <p className="mt-2 text-sm text-slate-300 leading-relaxed">
              Consolidated satellite monitoring, vegetation index analysis, transparent condition scoring, and actionable agronomic management.
            </p>
          </div>
        </div>

        {/* Global Demo Indicator — Rendered strictly only when in demo mode */}
        {dashboardData?.field_overview?.is_demo && (
          <GlobalDemoIndicator
            isDemo={true}
            dataSource="Demonstration Archive"
            acquisitionDate={dashboardData?.crop_condition?.observation_date}
          />
        )}

        {/* Top Controls Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-900/60 p-4 rounded-2xl border border-slate-800 backdrop-blur-md">
          {/* Field Selector Dropdown Button */}
          <div className="relative">
            <button
              onClick={() => setIsFieldSelectorOpen(!isFieldSelectorOpen)}
              className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-white font-semibold text-sm flex items-center gap-2.5 shadow-sm transition"
            >
              <span className="text-emerald-400">🌾</span>
              <span>
                {dashboardData?.field_overview?.field_name || "Select Agricultural Field"}
              </span>
              <span className="text-xs text-slate-400">▼</span>
            </button>

            {/* Field Dropdown Popover */}
            {isFieldSelectorOpen && (
              <div className="absolute left-0 mt-2 w-80 bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-2 z-50 backdrop-blur-xl">
                <div className="p-2 text-xs font-semibold text-slate-400 border-b border-slate-800">
                  Registered Farm Plots
                </div>
                <div className="max-h-60 overflow-y-auto divide-y divide-slate-800/60 py-1">
                  {fieldsList.map((f) => (
                    <button
                      key={f.field_id}
                      onClick={() => {
                        setActiveFieldId(f.field_id);
                        setIsFieldSelectorOpen(false);
                      }}
                      className={`w-full text-left p-2.5 rounded-xl text-xs transition flex items-center justify-between ${f.field_id === activeFieldId
                        ? "bg-emerald-950 text-emerald-300 font-bold border border-emerald-500/30"
                        : "text-slate-300 hover:bg-slate-800"
                        }`}
                    >
                      <div>
                        <div className="font-semibold text-white">{f.field_name}</div>
                        <div className="text-[11px] text-slate-400">{f.crop_type}</div>
                      </div>
                      <span className="font-mono text-[10px] text-slate-500">
                        {f.radius_km} km
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowComparisonView(!showComparisonView)}
              className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-slate-200 font-semibold text-xs transition flex items-center gap-2"
            >
              <span>📊</span>
              <span>{showComparisonView ? "Hide Comparison" : "Compare Fields"}</span>
            </button>
            <button
              onClick={() => setIsNewAOIOpen(true)}
              className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs transition flex items-center gap-2 shadow-lg shadow-emerald-600/20"
            >
              <span>+</span>
              <span>Register New Field</span>
            </button>
          </div>
        </div>

        {/* Modal: Register New Agricultural Field */}
        {isNewAOIOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 max-w-md w-full shadow-2xl relative">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <h3 className="font-bold text-base text-white">Register Agricultural AOI</h3>
                <button
                  onClick={() => setIsNewAOIOpen(false)}
                  className="text-slate-400 hover:text-white"
                >
                  ✕
                </button>
              </div>
              <form onSubmit={handleRegisterField} className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1 font-medium">Field Name</label>
                  <input
                    type="text"
                    required
                    value={newFieldName}
                    onChange={(e) => setNewFieldName(e.target.value)}
                    placeholder="e.g. West Godavari Rice Parcel B"
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1 font-medium">Crop Type</label>
                  <input
                    type="text"
                    value={newCropType}
                    onChange={(e) => setNewCropType(e.target.value)}
                    placeholder="e.g. Paddy / Rice, Corn, Wheat"
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-400 mb-1 font-medium">Latitude</label>
                    <input
                      type="number"
                      step="any"
                      required
                      value={newLat}
                      onChange={(e) => setNewLat(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white font-mono focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1 font-medium">Longitude</label>
                    <input
                      type="number"
                      step="any"
                      required
                      value={newLon}
                      onChange={(e) => setNewLon(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white font-mono focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-slate-400 mb-1 font-medium">AOI Radius (km)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="10.0"
                    required
                    value={newRadius}
                    onChange={(e) => setNewRadius(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-white font-mono focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div className="flex justify-end gap-3 pt-3 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={() => setIsNewAOIOpen(false)}
                    className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isSubmittingField}
                    className="px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50"
                  >
                    {isSubmittingField ? "Registering..." : "Register Field"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* 7-STATE RENDERING LOGIC (Issue 4) */}

        {/* State 1: INITIAL_LOADING */}
        {dashboardState === "INITIAL_LOADING" && !dashboardData && (
          <div className="bg-slate-900/60 rounded-3xl border border-slate-800 p-16 text-center space-y-4 shadow-xl">
            <div className="inline-block animate-spin text-4xl">🌱</div>
            <h3 className="text-lg font-bold text-white">
              Connecting to Agricultural Intelligence Service...
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Requesting field observations, spectral indices, and radar telemetry from server.
            </p>
          </div>
        )}

        {/* State 2: SERVER_WAKING (Cold-start detection on Render free tier) */}
        {dashboardState === "SERVER_WAKING" && !dashboardData && (
          <div className="bg-slate-900/80 rounded-3xl border-2 border-emerald-500/40 p-12 text-center space-y-5 shadow-2xl backdrop-blur-md max-w-2xl mx-auto">
            <div className="inline-flex p-4 rounded-2xl bg-emerald-950/80 border border-emerald-500/30 text-3xl">
              🛰️
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-black text-white tracking-tight">
                Waking up the server. This may take approximately 30 seconds.
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed max-w-lg mx-auto">
                Cloud instances spin down during periods of inactivity to conserve resources. Your container is currently booting and initializing the multispectral analysis engine.
              </p>
            </div>

            {/* Progress indicator */}
            <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-emerald-500 to-lime-500 h-2.5 rounded-full transition-all duration-1000"
                style={{ width: `${Math.min(100, Math.max(15, (wakeSecondsElapsed / 30) * 100))}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Status: Initializing ASGI worker</span>
              <span>{wakeSecondsElapsed}s elapsed</span>
            </div>
          </div>
        )}

        {/* State 3: LOADING_DATA */}
        {dashboardState === "LOADING_DATA" && !dashboardData && (
          <div className="bg-slate-900/60 rounded-3xl border border-slate-800 p-16 text-center space-y-4 shadow-xl">
            <div className="inline-block animate-pulse text-4xl">🌾</div>
            <h3 className="text-lg font-bold text-white">
              Synthesizing Multi-Sensor Observations...
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Assembling NDVI, NDMI, NDRE, EVI, and SAR backscatter features for field review.
            </p>
          </div>
        )}

        {/* State 4 & 5: ERROR or TIMEOUT (with clear message & Retry button) */}
        {(dashboardState === "ERROR" || dashboardState === "TIMEOUT") && !dashboardData && (
          <div className="bg-rose-950/30 border-2 border-rose-800/60 rounded-3xl p-10 text-center space-y-4 shadow-2xl max-w-xl mx-auto">
            <span className="text-4xl block">🛑</span>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-white">
                {dashboardState === "TIMEOUT" ? "Connection Timed Out" : "Connection Notice"}
              </h3>
              <p className="text-xs text-rose-200/90 leading-relaxed">
                {errorMessage || "Unable to connect to the Crop Health AI server. Please try again."}
              </p>
            </div>
            <div className="pt-2">
              <button
                onClick={() => {
                  if (activeFieldId) {
                    fetchFieldData(activeFieldId);
                  } else {
                    fetchCatalog();
                  }
                }}
                className="px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs shadow-lg transition"
              >
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {/* State 6: EMPTY */}
        {dashboardState === "EMPTY" && !dashboardData && (
          <div className="bg-slate-900/60 rounded-3xl border border-slate-800 p-12 text-center space-y-4 shadow-xl">
            <span className="text-4xl block">📭</span>
            <h3 className="text-lg font-bold text-white">No Field Intelligence Found</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              No agricultural data exists for the selected field identifier. Register a new field to initiate monitoring.
            </p>
            <button
              onClick={() => setIsNewAOIOpen(true)}
              className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs"
            >
              Register Field
            </button>
          </div>
        )}

        {/* Non-destructive notification if error occurred while previous data is preserved */}
        {(dashboardState === "ERROR" || dashboardState === "TIMEOUT") && dashboardData && (
          <div className="bg-amber-950/50 border border-amber-600/60 rounded-2xl p-4 text-amber-200 text-xs flex items-center justify-between gap-3 shadow-lg">
            <div className="flex items-center gap-2">
              <span className="text-base">⚠️</span>
              <span>Unable to connect to the Crop Health AI server. Displaying previously loaded field.</span>
            </div>
            <button
              onClick={() => fetchFieldData(activeFieldId)}
              className="px-3 py-1 rounded-lg bg-amber-600 hover:bg-amber-500 text-slate-950 font-bold text-xs"
            >
              Retry
            </button>
          </div>
        )}

        {/* Multi-Field Comparison Panel (Collapsible) */}
        {showComparisonView && (
          <FieldComparisonPanel
            comparisonData={comparisonData}
            availableFieldIds={fieldsList.map((f) => ({ id: f.field_id, name: f.field_name }))}
            onRunComparison={handleRunComparison}
            isLoading={isComparing}
          />
        )}

        {/* State 7: SUCCESS — Main Dashboard Content */}
        {dashboardData && (
          <div className="space-y-6">
            {/* 1. Field Overview */}
            <FieldOverviewCard
              overview={dashboardData.field_overview}
              onOpenFieldSelector={() => setIsFieldSelectorOpen(true)}
              onOpenNewAOI={() => setIsNewAOIOpen(true)}
            />

            {/* 2 & 3: Crop Condition Summary & Multispectral Vegetation Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <CropConditionSummary
                  condition={dashboardData.crop_condition}
                  isDemo={dashboardData.field_overview.is_demo}
                />
              </div>
              <div className="lg:col-span-2">
                <VegetationMetrics
                  metrics={dashboardData.vegetation_metrics}
                  isDemo={dashboardData.field_overview.is_demo}
                />
              </div>
            </div>

            {/* 4 & 5: Vigour Anomaly Intelligence & Growth Forecast */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <StressAlertPanel
                stress={dashboardData.stress_intelligence}
                isDemo={dashboardData.field_overview.is_demo}
              />
              <GrowthForecastPanel
                forecast={dashboardData.growth_forecast}
                isDemo={dashboardData.field_overview.is_demo}
              />
            </div>

            {/* 6. Field-Level Spatial Risk Distribution */}
            <FieldRiskSummary
              fieldRisk={dashboardData.field_risk}
              isDemo={dashboardData.field_overview.is_demo}
            />

            {/* 7. Actionable Agricultural Recommendations */}
            <AgriculturalRecommendations
              recommendations={dashboardData.recommendations}
              cropType={dashboardData.field_overview.crop_type}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-6 text-center text-xs text-slate-500 mt-12">
        <div className="max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Crop Health AI. Farmer Agricultural Operations Platform.</p>
          <div className="flex items-center gap-4 text-slate-400">
            <Link href="/" className="hover:text-emerald-400 transition">Home</Link>
            <span>•</span>
            <Link href="/analyze" className="hover:text-emerald-400 transition">Analyze</Link>
            <span>•</span>
            <Link href="/model-performance" className="hover:text-emerald-400 transition">Architecture &amp; Model Targets</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
