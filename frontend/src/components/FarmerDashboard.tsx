"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import FieldOverviewCard, { FieldOverviewData } from "./FieldOverviewCard";
import CropConditionSummary, { CropConditionSummaryData } from "./CropConditionSummary";
import VegetationMetrics, { VegetationIndicatorMetricData } from "./VegetationMetrics";
import StressAlertPanel, { StressIntelligenceData } from "./StressAlertPanel";
import GrowthForecastPanel, { GrowthForecastData } from "./GrowthForecastPanel";
import FieldRiskSummary, { FieldRiskSummaryData } from "./FieldRiskSummary";
import HistoricalMonitoringPanel, { HistoricalMonitoringData } from "./HistoricalMonitoringPanel";
import AgriculturalRecommendations, { AgriculturalRecommendationData } from "./AgriculturalRecommendations";
import DataReliabilitySummary, { DataReliabilitySummaryData } from "./DataReliabilitySummary";
import FieldComparisonPanel, {
  FieldComparisonResponseData,
} from "./FieldComparisonPanel";
import { API_BASE_URL } from "@/config/api";

export interface DashboardOverviewItem {
  field_id: string;
  field_name: string;
  crop_type: string;
  location_name: string;
  area_hectares: number;
  latest_observation_date: string;
  overall_risk_level: string;
  data_quality_tier: string;
  mean_ndvi: number | null;
  is_demo: boolean;
}

export interface FarmerDashboardFullData {
  status: string;
  field_overview: FieldOverviewData;
  crop_condition: CropConditionSummaryData;
  vegetation_metrics: Record<string, VegetationIndicatorMetricData>;
  stress_intelligence: StressIntelligenceData;
  growth_forecast: GrowthForecastData;
  field_risk: FieldRiskSummaryData;
  historical_monitoring: HistoricalMonitoringData;
  recommendations: AgriculturalRecommendationData[];
  data_reliability: DataReliabilitySummaryData;
  warnings: string[];
  scientific_disclaimers: string[];
}

const API_BASE = API_BASE_URL;

export default function FarmerDashboard() {
  const [fieldsList, setFieldsList] = useState<DashboardOverviewItem[]>([]);
  const [activeFieldId, setActiveFieldId] = useState<string>("field_1");
  const [dashboardData, setDashboardData] = useState<FarmerDashboardFullData | null>(null);
  const [comparisonData, setComparisonData] = useState<FieldComparisonResponseData | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isFieldSelectorOpen, setIsFieldSelectorOpen] = useState<boolean>(false);
  const [isNewAOIOpen, setIsNewAOIOpen] = useState<boolean>(false);
  const [showComparisonView, setShowComparisonView] = useState<boolean>(false);

  // New Field AOI form state
  const [newFieldName, setNewFieldName] = useState("");
  const [newCropType, setNewCropType] = useState("Paddy / Rice");
  const [newLat, setNewLat] = useState("16.8150");
  const [newLon, setNewLon] = useState("81.5280");
  const [newRadius, setNewRadius] = useState("2.0");
  const [isSubmittingField, setIsSubmittingField] = useState(false);

  // Load catalog on initial mount
  useEffect(() => {
    fetchCatalog();
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
      if (!res.ok) {
        throw new Error(`Catalog fetch failed: HTTP ${res.status}`);
      }
      const data = await res.json();
      setFieldsList(data.fields || []);
      if (data.active_field_id && !activeFieldId) {
        setActiveFieldId(data.active_field_id);
      }
    } catch (err: any) {
      console.error("Failed to load catalog:", err);
    }
  };

  const fetchFieldData = async (fieldId: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/farmer-dashboard/field/${fieldId}`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Failed to fetch field: ${res.status}`);
      }
      const data: FarmerDashboardFullData = await res.json();
      setDashboardData(data);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load agricultural intelligence data.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunComparison = async (fieldIds: string[]) => {
    setIsComparing(true);
    try {
      const query = fieldIds.map((id) => `field_ids=${encodeURIComponent(id)}`).join("&");
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
            <span className="text-xs text-emerald-400 font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              Farmer Intelligence Hub
            </span>
          </div>

          <nav className="flex items-center gap-5">
            <Link href="/" className="text-sm font-medium text-slate-400 hover:text-white transition">
              Home
            </Link>
            <Link href="/analyze" className="text-sm font-medium text-slate-400 hover:text-white transition">
              Analyze
            </Link>
            <Link href="/history" className="text-sm font-medium text-slate-400 hover:text-white transition">
              History
            </Link>
            <Link href="/farmer-dashboard" className="text-sm font-bold text-emerald-400 transition flex items-center gap-1">
              <span>🌾</span>
              <span>Dashboard</span>
            </Link>
            <Link href="/model-performance" className="text-sm font-medium text-slate-400 hover:text-white transition hidden md:inline">
              Model Performance
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
              🌱 Real-Time Field Intelligence Hub
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Farmer Agricultural Operations
            </h1>
            <p className="mt-2 text-sm text-slate-300 leading-relaxed">
              Consolidated satellite monitoring, crop stress detection, SAR soil moisture index, and actionable field-level management recommendations.
            </p>
          </div>
        </div>

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

            {/* Field Dropdown Menu */}
            {isFieldSelectorOpen && (
              <div className="absolute top-full left-0 mt-2 w-80 bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl z-50 p-2 space-y-1">
                <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800">
                  Registered Agricultural Fields
                </div>
                {fieldsList.map((f) => (
                  <button
                    key={f.field_id}
                    onClick={() => {
                      setActiveFieldId(f.field_id);
                      setIsFieldSelectorOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-xl text-xs transition flex flex-col ${
                      f.field_id === activeFieldId
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        : "hover:bg-slate-800 text-slate-200"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold">{f.field_name}</span>
                      <span className="font-mono text-[10px] text-slate-400">
                        {f.area_hectares.toFixed(1)} ha
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-[10px] text-slate-400 mt-0.5">
                      <span>{f.crop_type}</span>
                      <span>{f.latest_observation_date}</span>
                    </div>
                  </button>
                ))}

                <button
                  onClick={() => {
                    setIsFieldSelectorOpen(false);
                    setIsNewAOIOpen(true);
                  }}
                  className="w-full text-center px-3 py-2 mt-2 rounded-xl text-xs font-semibold bg-emerald-600/30 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/30 transition flex items-center justify-center gap-1.5"
                >
                  <span>➕</span>
                  <span>Register New Field AOI</span>
                </button>
              </div>
            )}
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => setShowComparisonView(!showComparisonView)}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold border transition flex items-center gap-1.5 ${
                showComparisonView
                  ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700"
              }`}
            >
              <span>⚖️</span>
              <span>{showComparisonView ? "Hide Comparison" : "Compare Fields"}</span>
            </button>

            <button
              onClick={() => fetchFieldData(activeFieldId)}
              disabled={isLoading}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition flex items-center gap-1.5"
            >
              <span className={isLoading ? "animate-spin" : ""}>🔄</span>
              <span>Refresh</span>
            </button>

            <Link
              href="/analyze"
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition flex items-center gap-1.5 shadow-sm"
            >
              <span>🛰️</span>
              <span>Open Spatial Analysis</span>
            </Link>
          </div>
        </div>

        {/* Modal: Register New Field AOI */}
        {isNewAOIOpen && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 max-w-md w-full shadow-2xl relative">
              <button
                onClick={() => setIsNewAOIOpen(false)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>

              <h3 className="text-base font-bold text-white mb-1 flex items-center gap-2">
                <span>➕</span>
                <span>Register New Field AOI</span>
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Submit agricultural boundaries for automated Sentinel-1/2 acquisition.
              </p>

              <form onSubmit={handleRegisterField} className="space-y-3.5">
                <div>
                  <label className="text-[11px] font-semibold text-slate-300 uppercase block mb-1">
                    Field Name
                  </label>
                  <input
                    type="text"
                    value={newFieldName}
                    onChange={(e) => setNewFieldName(e.target.value)}
                    placeholder="e.g. West Godavari Paddy Parcel"
                    required
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-300 uppercase block mb-1">
                    Crop Type
                  </label>
                  <input
                    type="text"
                    value={newCropType}
                    onChange={(e) => setNewCropType(e.target.value)}
                    placeholder="e.g. Paddy / Rice, Maize, Cotton"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-semibold text-slate-300 uppercase block mb-1">
                      Latitude (°N)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={newLat}
                      onChange={(e) => setNewLat(e.target.value)}
                      required
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-slate-300 uppercase block mb-1">
                      Longitude (°E)
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={newLon}
                      onChange={(e) => setNewLon(e.target.value)}
                      required
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-300 uppercase block mb-1">
                    Radius (km)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    min="0.5"
                    max="10.0"
                    value={newRadius}
                    onChange={(e) => setNewRadius(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>

                <div className="pt-2 flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setIsNewAOIOpen(false)}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300"
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

        {/* Loading State */}
        {isLoading && !dashboardData && (
          <div className="bg-slate-900/60 rounded-3xl border border-slate-800 p-16 text-center space-y-4">
            <div className="inline-block animate-spin text-4xl">🌱</div>
            <h3 className="text-lg font-bold text-white">
              Aggregating Agricultural Intelligence...
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Synthesizing Sentinel-2 multispectral indices, Sentinel-1 SAR backscatter, machine learning stress indicators, and empirical growth trajectories.
            </p>
          </div>
        )}

        {/* Error State */}
        {errorMessage && (
          <div className="bg-rose-950/40 border border-rose-800/60 rounded-2xl p-5 text-rose-300 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-lg">🛑</span>
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => fetchFieldData(activeFieldId)}
              className="px-3 py-1 rounded-lg bg-rose-900/60 hover:bg-rose-800 text-white font-semibold"
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

        {/* Main Dashboard Content */}
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

            {/* 4 & 5: Early Stress Intelligence & Growth Forecast */}
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

            {/* 7. Historical Monitoring Timeline */}
            <HistoricalMonitoringPanel
              historical={dashboardData.historical_monitoring}
              isDemo={dashboardData.field_overview.is_demo}
            />

            {/* 8. Evidence-Based Agricultural Recommendations */}
            <AgriculturalRecommendations
              recommendations={dashboardData.recommendations}
              isDemo={dashboardData.field_overview.is_demo}
            />

            {/* 9. Multi-Dimensional Data Reliability */}
            <DataReliabilitySummary
              reliability={dashboardData.data_reliability}
              isDemo={dashboardData.field_overview.is_demo}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500 bg-slate-950 mt-12">
        <p>
          Crop Health AI Platform • Advanced Farmer Intelligence Dashboard • Optical & SAR Constellations
        </p>
      </footer>
    </div>
  );
}
