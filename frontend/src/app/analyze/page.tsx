"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { LayerType } from "@/components/FieldMap";
import AIFieldReport, { AIFieldReportData } from "@/components/AIFieldReport";
import GrowthTrendChart from "@/components/GrowthTrendChart";
import ExplainabilityPanel, { ExplainabilityResponseData } from "@/components/ExplainabilityPanel";
import RiskLayerSelector from "@/components/RiskLayerSelector";
import FieldRiskLegend from "@/components/FieldRiskLegend";
import FieldRiskStatistics from "@/components/FieldRiskStatistics";
import PixelInspectionPanel, { PixelInspectionData } from "@/components/PixelInspectionPanel";
import RiskEvidencePanel from "@/components/RiskEvidencePanel";
import { API_BASE_URL } from "@/config/api";
import GlobalDemoIndicator from "@/components/GlobalDemoIndicator";

// Dynamically import Leaflet FieldMap with SSR disabled (Leaflet requires window object)
const FieldMap = dynamic(() => import("@/components/FieldMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[480px] rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center text-slate-400 gap-3">
      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm">Loading interactive satellite map...</span>
    </div>
  ),
});

// Dynamically import Leaflet FieldRiskMap with SSR disabled
const FieldRiskMap = dynamic(() => import("@/components/FieldRiskMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[520px] rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center text-slate-400 gap-3">
      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm">Loading field-level risk map & spatial overlays...</span>
    </div>
  ),
});

interface LocationData {
  query: string;
  display_name: string;
  latitude: number;
  longitude: number;
}

interface AOIData {
  latitude: number;
  longitude: number;
  radius_km: number;
  radius_meters: number;
  bbox: number[];
  area_sq_km: number;
  geometry?: any;
}

interface SatelliteImageData {
  image_base64: string;
  scene_id: string;
  acquisition_date: string;
  cloud_coverage: number;
  data_source: string;
  processing_level: string;
  bands: string[];
  is_demo: boolean;
  metadata?: any;
}

interface BreakdownCategory {
  label: string;
  pixels: number;
  percentage: number;
}

interface IndexStatistics {
  valid_pixels: number;
  total_pixels: number;
  valid_pixel_percentage: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  std_dev: number | null;
  area_represented_km2: number;
  [key: string]: any;
}

interface SingleIndexData {
  name: string;
  description: string;
  physical_indicator?: string;
  bands_used: string[];
  statistics: IndexStatistics;
  map: string;
}

interface CombinedIndicesResponse {
  status: string;
  scene: {
    scene_id: string;
    acquisition_date: string;
    cloud_coverage: number;
    data_source: string;
    processing_level: string;
    is_demo: boolean;
    date_range: { start_date: string; end_date: string };
  };
  ndvi: SingleIndexData;
  ndmi: SingleIndexData;
  ndre: SingleIndexData;
  evi: SingleIndexData;
}

interface StressPattern {
  type: string;
  label: string;
  severity: "low" | "moderate" | "high";
  percentage: number;
  explanation: string;
}

interface CropConditionResponse {
  status: string;
  overall_condition: string;
  condition_score: number;
  condition_score_mean: number | null;
  condition_score_median: number | null;
  condition_score_std: number | null;
  dominant_zone: string;
  valid_pixel_percentage: number;
  area_represented_km2: number;
  zones: {
    very_low: number;
    low: number;
    moderate: number;
    good: number;
    very_good: number;
  };
  zone_counts: { [key: string]: number };
  indices: {
    ndvi: SingleIndexData;
    ndmi: SingleIndexData;
    ndre: SingleIndexData;
    evi: SingleIndexData;
  };
  patterns: StressPattern[];
  quality: {
    valid_pixels: number;
    invalid_pixels: number;
    valid_percentage: number;
    quality_level: "Excellent" | "Good" | "Fair" | "Poor";
  };
  warnings: string[];
  condition_map: string;
  scene: {
    scene_id: string;
    acquisition_date: string;
    cloud_coverage: number;
    data_source: string;
    processing_level: string;
    is_demo: boolean;
    date_range: { start_date: string; end_date: string };
  };
  analysis_metadata?: any;
}

interface SarChannelStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
}

interface Sentinel1AnalysisResponse {
  status: string;
  is_demo: boolean;
  scene_id: string;
  acquisition_date: string;
  product_type: string;
  polarization: string[];
  orbit_direction?: string;
  valid_pixel_percentage: number;
  vv: SarChannelStats;
  vh: SarChannelStats;
  vv_vh_ratio: {
    mean: number;
    median: number;
    min?: number;
    max?: number;
    std?: number;
  };
  vv_vh_diff_db: {
    mean: number;
    median: number;
  };
  matched_sentinel2_date?: string;
  date_difference_days?: number;
  map_layers?: {
    vv_png?: string;
    vh_png?: string;
    ratio_png?: string;
  };
  physical_interpretation?: Record<string, string>;
  quality?: Record<string, any>;
}

interface MultimodalFusionResponse {
  status: string;
  is_demo: boolean;
  fusion_readiness: "READY_FOR_AI" | "PARTIAL_OPTICAL_ONLY" | "INSUFFICIENT_QUALITY";
  temporal_matching: {
    optical_scene_id: string;
    optical_date: string;
    sar_scene_id: string | null;
    sar_date: string | null;
    date_difference_days: number | null;
    matched: boolean;
    max_temporal_window_days: number;
    strategy: string;
  };
  spatial_alignment: {
    canonical_crs: string;
    utm_zone: number;
    resolution_meters: number;
    grid_shape: [number, number];
    aoi_bounds_utm: [number, number, number, number];
    aoi_bounds_wgs84: [number, number, number, number];
    aoi_area_sq_km: number;
  };
  quality: {
    valid_pixels: number;
    partial_pixels: number;
    invalid_pixels: number;
    total_pixels: number;
    valid_percentage: number;
    partial_percentage: number;
    invalid_percentage: number;
    overall_quality_tier: string;
  };
  features: {
    feature_names: string[];
    feature_count: number;
    optical_features: string[];
    sar_features: string[];
    normalization_applied: string;
    statistics: Record<
      string,
      {
        min: number | null;
        max: number | null;
        mean: number | null;
        median: number | null;
        std: number | null;
        valid_pixels: number;
        description: string;
      }
    >;
  };
  previews?: {
    quality_mask?: string;
    optical_rgb?: string;
    sar_vv?: string;
    sar_vh?: string;
  };
  export_available_formats: string[];
}

interface StressDetectionResponse {
  status: string;
  is_demo: boolean;
  assessment_status: string;
  stress_probability: number | null;
  predicted_class: string;
  model_name: string;
  model_version: string;
  observation_dates: {
    current_optical_date: string | null;
    current_sar_date: string | null;
    prediction_date: string;
    prior_observations_count: number;
    prior_date_range?: [string, string];
  };
  features_used: Record<string, number | null>;
  features_used_count: number;
  data_quality: {
    cloud_coverage: number;
    valid_pixel_pct: number;
    nodata_pct: number;
    s1_s2_temporal_gap_days: number | null;
    missing_sar: boolean;
    quality_verdict: string;
  };
  missing_feature_warnings: string[];
  prediction_limitations: string[];
  field_summary: {
    dominant_condition: string;
    mean_ndvi: number | null;
    mean_ndmi: number | null;
    mean_ndre: number | null;
    mean_evi: number | null;
    mean_sar_vv: number | null;
    mean_sar_vh: number | null;
    sar_ratio: number | null;
  };
  spatial_stress_map?: {
    stress_map_png?: string;
    resolution_meters: number;
    assessable_area_pct: number;
    unassessable_pixels_masked: boolean;
  };
  explainability?: {
    method: string;
    global_top_features?: Array<{ feature: string; importance: number }>;
    local_contributions?: Array<{
      feature: string;
      value: number | null;
      contribution: number;
      direction: string;
    }>;
    disclaimer: string;
  };
  recommended_actions: string[];
}

interface GrowthPredictionResponse {
  status: string;
  assessment_status: string;
  is_demo: boolean;
  model_name: string;
  model_version: string;
  forecast_horizon: string;
  forecast_horizon_label: string;
  forecast_dates: string[];
  forecast_values: Record<string, number[]>;
  prediction_intervals: Record<
    string,
    Array<{ lower: number; upper: number; confidence_level: number }>
  >;
  trend_statistics: Record<
    string,
    {
      index_name: string;
      current_value: number | null;
      first_value: number | null;
      absolute_change: number | null;
      relative_change_pct: number | null;
      trend_slope_per_day: number | null;
      trend_direction: string;
      observation_count: number;
      time_span_days: number;
      description: string;
    }
  >;
  historical_observations: Array<{
    date: string;
    indices: Record<string, number | null>;
    cloud_coverage: number;
    valid_pixel_percentage: number;
    nodata_pct: number;
    quality_status: string;
    sequential_features?: Record<string, any>;
  }>;
  validation_metrics: Record<
    string,
    {
      selected_model: string;
      all_models_evaluated: Record<string, any>;
      residual_std: number;
      prediction_intervals_available: boolean;
    }
  >;
  data_quality: {
    total_observations: number;
    valid_optical_count: number;
    valid_sar_count: number;
    median_cloud_coverage: number;
    median_valid_pixel_pct: number;
    quality_status: string;
  };
  observation_count: number;
  historical_comparison: {
    prior_period: {
      period_label: string;
      date_range: string[];
      observation_count: number;
      mean_values: Record<string, number | null>;
    };
    current_period: {
      period_label: string;
      date_range: string[];
      observation_count: number;
      mean_values: Record<string, number | null>;
    };
    period_to_period_change: Record<string, number | null>;
    forecast_continuation: Record<
      string,
      { first_forecast: number | null; terminal_forecast: number | null }
    >;
  };
  warnings: string[];
  limitations: string[];
  scientific_disclaimer: string;
}

type SatelliteLoadingStep =
  | null
  | "searching"
  | "selecting"
  | "requesting"
  | "loading";

type IndicesLoadingStep =
  | null
  | "requesting_bands"
  | "aligning_resampling"
  | "applying_cloud_mask"
  | "calculating_indices"
  | "generating_colormaps"
  | "complete";

type ConditionLoadingStep =
  | null
  | "fetching_spectral_data"
  | "evaluating_all_indices"
  | "synthesizing_condition_score"
  | "classifying_zones"
  | "detecting_stress_patterns"
  | "complete";

export default function AnalyzePage() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Default initial location: Tadepalligudem, Andhra Pradesh
  const [currentLocation, setCurrentLocation] = useState<LocationData>({
    query: "Tadepalligudem, Andhra Pradesh",
    display_name: "Tadepalligudem, West Godavari District, Andhra Pradesh, India",
    latitude: 16.810244,
    longitude: 81.526592,
  });

  const [radiusKm, setRadiusKm] = useState<number>(2);
  const [confirmedAOI, setConfirmedAOI] = useState<AOIData | null>(null);

  // Date range and Cloud coverage configuration
  const [startDate, setStartDate] = useState("2026-10-01");
  const [endDate, setEndDate] = useState("2026-10-26");
  const [maxCloudCoverage, setMaxCloudCoverage] = useState<number>(30);

  // Phase 3: Satellite loading steps & response data
  const [satelliteLoadingStep, setSatelliteLoadingStep] = useState<SatelliteLoadingStep>(null);
  const [satelliteImage, setSatelliteImage] = useState<SatelliteImageData | null>(null);
  const [satelliteError, setSatelliteError] = useState<string | null>(null);

  // Phase 5: Multispectral Indices loading & state
  const [indicesLoadingStep, setIndicesLoadingStep] = useState<IndicesLoadingStep>(null);
  const [indicesResult, setIndicesResult] = useState<CombinedIndicesResponse | null>(null);
  const [indicesError, setIndicesError] = useState<string | null>(null);

  // Phase 6: Crop Condition Intelligence loading & state
  const [conditionLoadingStep, setConditionLoadingStep] = useState<ConditionLoadingStep>(null);
  const [conditionResult, setConditionResult] = useState<CropConditionResponse | null>(null);
  const [conditionError, setConditionError] = useState<string | null>(null);

  // Phase 11: Sentinel-1 SAR loading & state
  const [sentinel1Result, setSentinel1Result] = useState<Sentinel1AnalysisResponse | null>(null);
  const [sentinel1Loading, setSentinel1Loading] = useState<boolean>(false);
  const [sentinel1Error, setSentinel1Error] = useState<string | null>(null);

  // Phase 12: Multimodal Satellite Fusion loading & state
  const [fusionResult, setFusionResult] = useState<MultimodalFusionResponse | null>(null);
  const [fusionLoading, setFusionLoading] = useState<boolean>(false);
  const [fusionError, setFusionError] = useState<string | null>(null);
  const [temporalWindowDays, setTemporalWindowDays] = useState<number>(3);
  const [normalizationMethod, setNormalizationMethod] = useState<"raw" | "min-max" | "z-score">("raw");

  // Phase 13: AI-Powered Early Crop Stress Detection state
  const [stressResult, setStressResult] = useState<StressDetectionResponse | null>(null);
  const [stressLoading, setStressLoading] = useState<boolean>(false);
  const [stressError, setStressError] = useState<string | null>(null);

  // Phase 14: AI-Powered Crop Growth & Trend Prediction state
  const [growthResult, setGrowthResult] = useState<GrowthPredictionResponse | null>(null);
  const [growthLoading, setGrowthLoading] = useState<boolean>(false);
  const [growthError, setGrowthError] = useState<string | null>(null);
  const [forecastHorizon, setForecastHorizon] = useState<"short" | "medium" | "long">("medium");
  const [forecastModelType, setForecastModelType] = useState<string>("linear_trend");
  const [activeGrowthIndex, setActiveGrowthIndex] = useState<string>("NDVI");
  
  // Phase 15: Explainable AI & Uncertainty Intelligence state
  const [explainabilityResult, setExplainabilityResult] = useState<ExplainabilityResponseData | null>(null);
  const [explainabilityLoading, setExplainabilityLoading] = useState<boolean>(false);
  const [explainabilityError, setExplainabilityError] = useState<string | null>(null);

  // Phase 16: Field-Level AI Risk Mapping & Spatial Intelligence state
  const [fieldRiskResult, setFieldRiskResult] = useState<any | null>(null);
  const [fieldRiskLoading, setFieldRiskLoading] = useState<boolean>(false);
  const [fieldRiskError, setFieldRiskError] = useState<string | null>(null);
  const [selectedRiskLayer, setSelectedRiskLayer] = useState<string>("risk_classification");
  const [inspectedPixel, setInspectedPixel] = useState<PixelInspectionData | null>(null);
  const [pixelLoading, setPixelLoading] = useState<boolean>(false);

  // Active Map Overlay Layer
  const [selectedLayer, setSelectedLayer] = useState<LayerType>("satellite");

  // Active Tab in the Vegetation Indices Inspection View
  const [activeInspectionIndex, setActiveInspectionIndex] = useState<"ndvi" | "ndmi" | "ndre" | "evi">("ndvi");

  // Phase 9: AI Field Intelligence state
  const [aiReport, setAiReport] = useState<AIFieldReportData | null>(null);
  const [aiLoading, setAiLoading] = useState<boolean>(false);
  const [aiLoadingStage, setAiLoadingStage] = useState<string>("");
  const [aiError, setAiError] = useState<string | null>(null);

  const handleGenerateAIReport = async () => {
    if (aiLoading) return;
    if (!conditionResult && !indicesResult) {
      setAiError("Please run Crop Condition analysis before generating Field Intelligence.");
      return;
    }

    setAiLoading(true);
    setAiError(null);

    // Sequential loading stage messages
    const stages = [
      "Preparing field evidence...",
      "Reviewing vegetation indices...",
      "Checking historical patterns...",
      "Generating field interpretation...",
      "Validating report...",
    ];

    let currentStageIndex = 0;
    setAiLoadingStage(stages[0]);
    const stageInterval = setInterval(() => {
      currentStageIndex = (currentStageIndex + 1) % stages.length;
      setAiLoadingStage(stages[currentStageIndex]);
    }, 1100);

    try {
      const activeIndices = conditionResult?.indices || indicesResult;
      const sceneData = conditionResult?.scene || indicesResult?.scene || satelliteImage;

      const payload = {
        location: {
          latitude: confirmedAOI?.latitude ?? currentLocation.latitude,
          longitude: confirmedAOI?.longitude ?? currentLocation.longitude,
          name: currentLocation.display_name,
          radius_km: confirmedAOI?.radius_km ?? radiusKm,
          area_sq_km: confirmedAOI?.area_sq_km,
        },
        observation: {
          date: sceneData?.acquisition_date ?? endDate,
          cloud_coverage: sceneData?.cloud_coverage ?? 0,
          valid_pixel_percentage:
            conditionResult?.valid_pixel_percentage ??
            activeIndices?.ndvi?.statistics?.valid_pixel_percentage ??
            100,
          scene_id: sceneData?.scene_id,
          data_source: sceneData?.data_source ?? "Sentinel-2 L2A",
        },
        indices: activeIndices
          ? {
              ndvi: activeIndices.ndvi?.statistics
                ? {
                    mean: activeIndices.ndvi.statistics.mean,
                    median: activeIndices.ndvi.statistics.median,
                    min: activeIndices.ndvi.statistics.min,
                    max: activeIndices.ndvi.statistics.max,
                  }
                : undefined,
              ndmi: activeIndices.ndmi?.statistics
                ? {
                    mean: activeIndices.ndmi.statistics.mean,
                    median: activeIndices.ndmi.statistics.median,
                    min: activeIndices.ndmi.statistics.min,
                    max: activeIndices.ndmi.statistics.max,
                  }
                : undefined,
              ndre: activeIndices.ndre?.statistics
                ? {
                    mean: activeIndices.ndre.statistics.mean,
                    median: activeIndices.ndre.statistics.median,
                    min: activeIndices.ndre.statistics.min,
                    max: activeIndices.ndre.statistics.max,
                  }
                : undefined,
              evi: activeIndices.evi?.statistics
                ? {
                    mean: activeIndices.evi.statistics.mean,
                    median: activeIndices.evi.statistics.median,
                    min: activeIndices.evi.statistics.min,
                    max: activeIndices.evi.statistics.max,
                  }
                : undefined,
            }
          : undefined,
        condition: conditionResult
          ? {
              score: conditionResult.condition_score,
              category: conditionResult.overall_condition,
              dominant_zone: conditionResult.dominant_zone,
              zones: conditionResult.zones,
            }
          : undefined,
        quality: {
          valid_percentage:
            conditionResult?.quality?.valid_percentage ??
            activeIndices?.ndvi?.statistics?.valid_pixel_percentage ??
            100,
          quality_level: conditionResult?.quality?.quality_level ?? "Good",
          limitations: conditionResult?.warnings ?? [],
        },
        is_demo: sceneData?.is_demo ?? false,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/ai-report`,  {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate AI Field Intelligence report.");
      }

      setAiReport(data);
    } catch (err: any) {
      setAiError(err.message || "Failed to generate AI Field Intelligence report.");
    } finally {
      clearInterval(stageInterval);
      setAiLoading(false);
    }
  };

  const handleSearch = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const cleanQuery = (customQuery !== undefined ? customQuery : query).trim();

    if (!cleanQuery) {
      setErrorMessage("Please enter a location name (city, village, district, or place).");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setSatelliteImage(null);
    setSatelliteError(null);
    setIndicesResult(null);
    setIndicesError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/geocode?query=${encodeURIComponent(cleanQuery)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Location not found. Please try another location.");
      }

      setCurrentLocation({
        query: cleanQuery,
        display_name: data.display_name,
        latitude: data.latitude,
        longitude: data.longitude,
      });

      // Keep user's search query visible so they clearly see what was found
      setQuery(cleanQuery);

      // Automatically configure the field AOI so satellite acquisition is immediately active
      try {
        const aoiResponse = await fetch(`${API_BASE_URL}/api/v1/aoi`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: data.latitude,
            longitude: data.longitude,
            radius_km: radiusKm,
          }),
        });

        if (aoiResponse.ok) {
          const aoiData = await aoiResponse.json();
          setConfirmedAOI(aoiData.aoi);
        }
      } catch (aoiErr) {
        console.warn("Auto-AOI setup warning", aoiErr);
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to search location.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyRadius = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: currentLocation.latitude,
          longitude: currentLocation.longitude,
          radius_km: radiusKm,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to configure Area of Interest.");
      }

      setConfirmedAOI(data.aoi);
      setSatelliteImage(null);
      setSatelliteError(null);
      setIndicesResult(null);
      setIndicesError(null);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to create AOI.");
    }
  };

  // Phase 3: Satellite imagery preview acquisition
  const handleFetchSatelliteImage = async () => {
    setSatelliteError(null);
    setSatelliteImage(null);

    if (new Date(startDate) > new Date(endDate)) {
      setSatelliteError("Start date cannot be after end date.");
      return;
    }

    setSatelliteLoadingStep("searching");

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        const aoiData = await aoiRes.json();
        if (!aoiRes.ok || !aoiData.aoi) {
          throw new Error(aoiData.detail || "Failed to calculate AOI geometry.");
        }
        setConfirmedAOI(aoiData.aoi);
        geometry = aoiData.aoi.geometry;
      }

      await new Promise((resolve) => setTimeout(resolve, 400));
      setSatelliteLoadingStep("selecting");

      await new Promise((resolve) => setTimeout(resolve, 400));
      setSatelliteLoadingStep("requesting");

      const response = await fetch(`${API_BASE_URL}/api/v1/satellite/preview`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          geometry,
          start_date: startDate,
          end_date: endDate,
          max_cloud_coverage: Number(maxCloudCoverage),
        }),
      });

      setSatelliteLoadingStep("loading");
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to retrieve satellite imagery.");
      }

      setSatelliteImage(data);
      setSelectedLayer("satellite");
    } catch (err: any) {
      setSatelliteError(err.message || "Satellite service is temporarily unavailable.");
    } finally {
      setSatelliteLoadingStep(null);
    }
  };

  // Phase 5: Unified Multispectral Indices Calculation (NDVI, NDMI, NDRE, EVI)
  const handleCalculateAllIndices = async () => {
    setIndicesError(null);
    setIndicesResult(null);

    if (new Date(startDate) > new Date(endDate)) {
      setIndicesError("Start date cannot be after end date.");
      return;
    }

    setIndicesLoadingStep("requesting_bands");

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        const aoiData = await aoiRes.json();
        if (!aoiRes.ok || !aoiData.aoi) {
          throw new Error(aoiData.detail || "Failed to calculate AOI geometry.");
        }
        setConfirmedAOI(aoiData.aoi);
        geometry = aoiData.aoi.geometry;
      }

      await new Promise((r) => setTimeout(r, 450));
      setIndicesLoadingStep("aligning_resampling");

      await new Promise((r) => setTimeout(r, 450));
      setIndicesLoadingStep("applying_cloud_mask");

      await new Promise((r) => setTimeout(r, 450));
      setIndicesLoadingStep("calculating_indices");

      await new Promise((r) => setTimeout(r, 400));
      setIndicesLoadingStep("generating_colormaps");

      const aoiArea = confirmedAOI?.area_sq_km || Math.PI * radiusKm * radiusKm;
      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/indices`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          geometry,
          start_date: startDate,
          end_date: endDate,
          max_cloud_coverage: Number(maxCloudCoverage),
          aoi_area_km2: aoiArea,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to compute vegetation indices.");
      }

      setIndicesLoadingStep("complete");
      setIndicesResult(data);
      // Automatically switch map layer to NDVI upon completion
      setSelectedLayer("ndvi");
      setActiveInspectionIndex("ndvi");
    } catch (err: any) {
      setIndicesError(err.message || "Vegetation index analysis failed. Please verify criteria.");
    } finally {
      setIndicesLoadingStep(null);
    }
  };

  // Phase 6: Crop Condition Intelligence Calculation (NDVI + NDMI + NDRE + EVI + Quality + Zones)
  const handleCalculateCropCondition = async () => {
    setConditionError(null);
    setConditionResult(null);

    if (new Date(startDate) > new Date(endDate)) {
      setConditionError("Start date cannot be after end date.");
      return;
    }

    setConditionLoadingStep("fetching_spectral_data");

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        const aoiData = await aoiRes.json();
        if (!aoiRes.ok || !aoiData.aoi) {
          throw new Error(aoiData.detail || "Failed to calculate AOI geometry.");
        }
        setConfirmedAOI(aoiData.aoi);
        geometry = aoiData.aoi.geometry;
      }

      await new Promise((r) => setTimeout(r, 400));
      setConditionLoadingStep("evaluating_all_indices");

      await new Promise((r) => setTimeout(r, 400));
      setConditionLoadingStep("synthesizing_condition_score");

      await new Promise((r) => setTimeout(r, 400));
      setConditionLoadingStep("classifying_zones");

      await new Promise((r) => setTimeout(r, 350));
      setConditionLoadingStep("detecting_stress_patterns");

      const aoiArea = confirmedAOI?.area_sq_km || Math.PI * radiusKm * radiusKm;
      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/crop-condition`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          geometry,
          start_date: startDate,
          end_date: endDate,
          max_cloud_coverage: Number(maxCloudCoverage),
          aoi_area_km2: aoiArea,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to compute crop condition analysis.");
      }

      setConditionLoadingStep("complete");
      setConditionResult(data);
      // Automatically switch map overlay to Crop Condition
      setSelectedLayer("condition");
    } catch (err: any) {
      setConditionError(err.message || "Crop condition analysis failed. Please verify criteria.");
    } finally {
      setConditionLoadingStep(null);
    }
  };

  // Phase 11: Sentinel-1 SAR Backscatter & Structure Analysis
  const handleFetchSentinel1Analysis = async () => {
    setSentinel1Loading(true);
    setSentinel1Error(null);
    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson);
          geometry = aoiJson.geometry;
        }
      }

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        reference_s2_date: conditionResult?.scene?.acquisition_date || indicesResult?.scene?.acquisition_date || null,
        max_temporal_window_days: 7,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/sentinel1`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to retrieve Sentinel-1 SAR observations.");
      }

      setSentinel1Result(data);
      setSelectedLayer("sentinel1_vv");
    } catch (err: any) {
      setSentinel1Error(err.message || "Failed to retrieve Sentinel-1 SAR observations.");
    } finally {
      setSentinel1Loading(false);
    }
  };

  // Phase 12: End-to-End Multimodal Satellite Data Fusion (Sentinel-1 + Sentinel-2)
  const handleRunMultimodalFusion = async () => {
    setFusionLoading(true);
    setFusionError(null);

    if (new Date(startDate) > new Date(endDate)) {
      setFusionError("Start date cannot be after end date.");
      setFusionLoading(false);
      return;
    }

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson.aoi || aoiJson);
          geometry = (aoiJson.aoi || aoiJson).geometry;
        }
      }

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: Number(maxCloudCoverage),
        temporal_window_days: temporalWindowDays,
        normalization: normalizationMethod,
        aoi_area_km2: confirmedAOI?.area_sq_km || Math.PI * radiusKm * radiusKm,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/fusion`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Multimodal data fusion failed.");
      }

      setFusionResult(data);
      setSelectedLayer("quality_mask");
    } catch (err: any) {
      setFusionError(err.message || "Failed to execute multimodal satellite data fusion.");
    } finally {
      setFusionLoading(false);
    }
  };

  // Phase 12: Export Standardized Multimodal Dataset (.npz, .json, .tif)
  const handleExportFusionDataset = async (format: "npz" | "json" | "geotiff") => {
    try {
      let geometry = confirmedAOI?.geometry;
      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: Number(maxCloudCoverage),
        temporal_window_days: temporalWindowDays,
        normalization: normalizationMethod,
        export_format: format,
        aoi_area_km2: confirmedAOI?.area_sq_km || Math.PI * radiusKm * radiusKm,
      };

      const res = await fetch(`${API_BASE_URL}/api/v1/analysis/fusion/export`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Export failed" }));
        throw new Error(err.detail || `Export failed with status ${res.status}`);
      }

      const safeLocName = currentLocation.query.replace(/[^a-zA-Z0-9_-]/g, "_");

      if (format === "json") {
        const jsonData = await res.json();
        const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `multimodal_fusion_${safeLocName}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `multimodal_fusion_${safeLocName}.${format === "geotiff" ? "tif" : "npz"}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (err: any) {
      alert(err.message || "Failed to export multimodal dataset.");
    }
  };

  // Phase 13: AI-Powered Early Crop Stress Detection Handler
  const handleRunEarlyStressDetection = async () => {
    setStressLoading(true);
    setStressError(null);

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson.aoi || aoiJson);
          geometry = (aoiJson.aoi || aoiJson).geometry;
        }
      }

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        max_cloud_coverage: Number(maxCloudCoverage),
        include_spatial_inference: true,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/stress-detection`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Early crop stress detection failed.");
      }

      setStressResult(data);
      if (data.spatial_stress_map?.stress_map_png) {
        setSelectedLayer("stress");
      }
    } catch (err: any) {
      setStressError(err.message || "Failed to execute AI early crop stress detection.");
    } finally {
      setStressLoading(false);
    }
  };

  // Phase 14: AI-Powered Crop Growth & Trend Prediction Handler
  const handleRunCropGrowthPrediction = async () => {
    setGrowthLoading(true);
    setGrowthError(null);

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson.aoi || aoiJson);
          geometry = (aoiJson.aoi || aoiJson).geometry;
        }
      }

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        forecast_horizon: forecastHorizon,
        model_type: forecastModelType,
        selected_indices: ["NDVI", "NDMI", "NDRE", "EVI", "VV", "VH"],
        max_cloud_coverage: Number(maxCloudCoverage),
        aoi_area_km2: confirmedAOI?.area_sq_km || Math.PI * radiusKm * radiusKm,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/growth-prediction`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Crop growth & trend prediction failed.");
      }

      setGrowthResult(data);
    } catch (err: any) {
      setGrowthError(err.message || "Failed to execute AI crop growth trend forecasting.");
    } finally {
      setGrowthLoading(false);
    }
  };

  // Phase 15: Explainable AI & Uncertainty Intelligence Handler
  const handleRunExplainabilityAnalysis = async () => {
    setExplainabilityLoading(true);
    setExplainabilityError(null);

    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson.aoi || aoiJson);
          geometry = (aoiJson.aoi || aoiJson).geometry;
        }
      }

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: endDate,
        explanation_type: "all",
        selected_index: activeGrowthIndex || "NDVI",
        max_cloud_coverage: Number(maxCloudCoverage),
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/explainability`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Explainable AI analysis failed.");
      }

      setExplainabilityResult(data);
    } catch (err: any) {
      setExplainabilityError(err.message || "Failed to execute Explainable AI intelligence.");
    } finally {
      setExplainabilityLoading(false);
    }
  };

  const handleRunFieldRiskMapping = async () => {
    setFieldRiskLoading(true);
    setFieldRiskError(null);
    try {
      let geometry = confirmedAOI?.geometry;
      if (!geometry) {
        const aoiRes = await fetch(`${API_BASE_URL}/api/v1/spatial/calculate-aoi`,  {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: currentLocation.latitude,
            longitude: currentLocation.longitude,
            radius_km: radiusKm,
          }),
        });
        if (aoiRes.ok) {
          const aoiJson = await aoiRes.json();
          setConfirmedAOI(aoiJson.aoi || aoiJson);
          geometry = (aoiJson.aoi || aoiJson).geometry;
        }
      }

      const activeIndices = conditionResult?.indices || indicesResult;
      const sceneData = conditionResult?.scene || indicesResult?.scene || satelliteImage;

      const payload = {
        geometry,
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        radius_km: radiusKm,
        start_date: startDate,
        end_date: sceneData?.acquisition_date || endDate,
        selected_layer: selectedRiskLayer,
        max_cloud_coverage: Number(maxCloudCoverage),
        aoi_area_km2: confirmedAOI?.area_sq_km,
      };

      const res = await fetch(`${API_BASE_URL}/api/v1/analysis/field-risk-map`,  {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Field risk mapping failed.");
      }

      setFieldRiskResult(data);
      setInspectedPixel(null);

      // Scroll to section
      setTimeout(() => {
        document.getElementById("field-risk-map-section")?.scrollIntoView({ behavior: "smooth" });
      }, 150);
    } catch (err: any) {
      setFieldRiskError(err.message || "Failed to generate field-level risk map.");
    } finally {
      setFieldRiskLoading(false);
    }
  };

  const setPresetRange = (days: number) => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - days);

    const fmt = (d: Date) => d.toISOString().split("T")[0];
    setStartDate(fmt(start));
    setEndDate(fmt(end));
  };

  // Convert bbox to Leaflet LatLngBoundsExpression if available
  const overlayBounds: [[number, number], [number, number]] | null = confirmedAOI
    ? [
        [confirmedAOI.bbox[1], confirmedAOI.bbox[0]],
        [confirmedAOI.bbox[3], confirmedAOI.bbox[2]],
      ]
    : null;

  return (
    <div className="min-h-screen text-slate-100 flex flex-col relative overflow-x-hidden">
      {/* Full-Page High-Definition Aerial Farmland Background */}
      <div
        className="fixed inset-0 bg-cover bg-center pointer-events-none -z-10"
        style={{ backgroundImage: "url('/bg-paddy-aerial.jpg')" }}
      />
      {/* Ambient glassmorphic tint so the green fields are vibrant while all UI components remain clear */}
      <div className="fixed inset-0 bg-gradient-to-b from-black/45 via-slate-950/60 to-black/75 pointer-events-none -z-10 backdrop-blur-[1px]" />
      {/* Global Demo Indicator — Rendered strictly only in demo mode */}
      {satelliteImage?.is_demo && (
        <GlobalDemoIndicator
          isDemo={true}
          provenance={satelliteImage?.data_source ?? "Sentinel-2 MSI (Copernicus) & Sentinel-1 C-SAR (ESA)"}
        />
      )}
      {/* Header */}
      <header className="bg-[#244324]/95 border-b border-emerald-800/40 backdrop-blur-md sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="h-9 w-9 rounded-xl overflow-hidden border border-emerald-500/40 bg-emerald-950/60 shadow-sm flex items-center justify-center group-hover:scale-105 transition">
              <img src="/logo.png" alt="Crop Health AI Logo" className="h-full w-full object-cover" />
            </div>
            <div>
              <span className="font-bold text-lg text-white tracking-tight">CropHealth</span>
              <span className="text-xs text-slate-400 ml-2">/ Analyze</span>
            </div>
          </Link>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/" className="text-slate-400 hover:text-white transition">
              Home
            </Link>
            <Link href="/farmer-dashboard" className="text-emerald-400 hover:text-emerald-300 font-semibold transition flex items-center gap-1">
              <span>🌾</span>
              <span>Farmer Dashboard</span>
            </Link>
            <Link href="/history" className="text-slate-400 hover:text-white transition flex items-center gap-1">
              <span>📈</span>
              <span>History</span>
            </Link>
            <Link href="/crop" className="text-emerald-300 hover:text-white font-semibold transition hidden sm:inline">
              🌱 Crop
            </Link>
            <Link href="/deforestation" className="text-teal-300 hover:text-white font-semibold transition hidden sm:inline">
              🌲 Deforestation
            </Link>
            <Link href="/heat" className="text-amber-300 hover:text-white font-semibold transition hidden sm:inline">
              🔥 Urban Heat
            </Link>
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-slate-800/80 border-slate-700">
              <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
              <span>Explainable AI & Uncertainty Active</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl xl:max-w-[1440px] 2xl:max-w-[1536px] mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8 relative z-10">
        {/* Page Title Header (Floating above the full-page background) */}
        <div className="text-center space-y-3 pt-2 pb-1">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950/85 text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-lg">
            <span>🌾</span>
            <span>Multimodal Satellite Agricultural Intelligence</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight drop-shadow-[0_4px_16px_rgba(0,0,0,0.95)]">
            Agricultural Field Analysis
          </h1>
          <p className="text-slate-200 text-sm sm:text-base max-w-2xl mx-auto drop-shadow-[0_2px_8px_rgba(0,0,0,0.95)] font-medium leading-relaxed">
            Locate any farm worldwide, acquire Sentinel-1 SAR &amp; Sentinel-2 optical observations, and run biophysical crop health analytics.
          </p>
        </div>

        {/* 1. Location Search Box & Quick Agricultural Presets */}
        <div className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-4 backdrop-blur-md">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <label htmlFor="location-input" className="block text-sm font-semibold text-slate-200">
              Field Location Search &amp; Geocoding
            </label>
            <span className="text-xs text-slate-400">
              Instant coordinates, gazetteer lookup &amp; auto-AOI centering
            </span>
          </div>
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
            <input
              id="location-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter place, city, village or coordinates (e.g., Tadepalligudem, Bhimavaram, Vijayawada, Guntur)..."
              className="flex-1 px-4 py-3 bg-slate-950 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition text-sm sm:text-base"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 text-slate-950 font-bold rounded-xl transition shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Locating Field...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>Search Place</span>
                </>
              )}
            </button>
          </form>

          {/* Quick Agricultural Hotspots */}
          <div className="pt-1">
            <span className="text-xs text-slate-400 block mb-2 font-medium">Quick Agricultural Presets:</span>
            <div className="flex flex-wrap gap-2">
              {[
                { label: "🌾 Tadepalligudem, AP", query: "Tadepalligudem, Andhra Pradesh, India" },
                { label: "🌾 Bhimavaram, AP", query: "Bhimavaram, Andhra Pradesh, India" },
                { label: "🌾 Vijayawada, AP", query: "Vijayawada, Andhra Pradesh, India" },
                { label: "🌶️ Guntur, AP", query: "Guntur, Andhra Pradesh, India" },
                { label: "🌾 Ludhiana, Punjab", query: "Ludhiana, Punjab, India" },
                { label: "🍇 Champagne, France", query: "Champagne, France" },
                { label: "🌽 Iowa Corn Belt, USA", query: "Des Moines, Iowa, USA" },
              ].map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => {
                    setQuery(preset.query);
                    handleSearch(undefined, preset.query);
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-emerald-950 hover:text-emerald-300 hover:border-emerald-500/50 text-slate-300 border border-slate-700/80 transition"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Location Active Status Card */}
          {currentLocation && (
            <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-xs text-emerald-200 flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-base text-emerald-400">📍</span>
                <div>
                  <strong className="text-emerald-300 font-semibold text-sm block">
                    {currentLocation.display_name}
                  </strong>
                  <span className="text-slate-400 font-mono text-[11px]">
                    Exact Coordinates: {currentLocation.latitude.toFixed(4)}°N, {currentLocation.longitude.toFixed(4)}°E • Field Radius: {radiusKm} km
                  </span>
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold text-[11px] flex items-center gap-1">
                ✓ Field AOI Locked &amp; Ready
              </span>
            </div>
          )}

          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Location Notice</strong>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}
        </div>

        {/* 2. Interactive Map Section with 5 Layer Selectors */}
        <div className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-4 backdrop-blur-md">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <span>🗺️</span>
                <span>Interactive Field Map</span>
              </h2>
              <p className="text-xs text-slate-400">
                Click map to reposition marker or switch overlays via layer buttons.
              </p>
            </div>

            {/* Quick Layer Switcher */}
            <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs">
              <span className="text-slate-500 px-2 font-medium">Layers:</span>
              <button
                onClick={() => setSelectedLayer("condition")}
                disabled={!conditionResult}
                className={`px-2.5 py-1 rounded-lg transition font-bold ${
                  selectedLayer === "condition"
                    ? "bg-gradient-to-r from-emerald-500 to-lime-500 text-slate-950 shadow"
                    : conditionResult
                    ? "text-emerald-400 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                🎯 Condition
              </button>
              <button
                onClick={() => setSelectedLayer("satellite")}
                disabled={!satelliteImage}
                className={`px-2.5 py-1 rounded-lg transition font-medium ${
                  selectedLayer === "satellite"
                    ? "bg-emerald-500 text-slate-950 shadow"
                    : satelliteImage
                    ? "text-slate-300 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                Satellite
              </button>
              <button
                onClick={() => setSelectedLayer("ndvi")}
                disabled={!indicesResult && !conditionResult}
                className={`px-2.5 py-1 rounded-lg transition font-medium ${
                  selectedLayer === "ndvi"
                    ? "bg-emerald-500 text-slate-950 shadow"
                    : (indicesResult || conditionResult)
                    ? "text-slate-300 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                NDVI
              </button>
              <button
                onClick={() => setSelectedLayer("ndmi")}
                disabled={!indicesResult && !conditionResult}
                className={`px-2.5 py-1 rounded-lg transition font-medium ${
                  selectedLayer === "ndmi"
                    ? "bg-cyan-500 text-slate-950 shadow"
                    : (indicesResult || conditionResult)
                    ? "text-slate-300 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                NDMI
              </button>
              <button
                onClick={() => setSelectedLayer("ndre")}
                disabled={!indicesResult && !conditionResult}
                className={`px-2.5 py-1 rounded-lg transition font-medium ${
                  selectedLayer === "ndre"
                    ? "bg-lime-500 text-slate-950 shadow"
                    : (indicesResult || conditionResult)
                    ? "text-slate-300 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                NDRE
              </button>
              <button
                onClick={() => setSelectedLayer("evi")}
                disabled={!indicesResult && !conditionResult}
                className={`px-2.5 py-1 rounded-lg transition font-medium ${
                  selectedLayer === "evi"
                    ? "bg-teal-500 text-slate-950 shadow"
                    : (indicesResult || conditionResult)
                    ? "text-slate-300 hover:bg-slate-800"
                    : "text-slate-600 cursor-not-allowed"
                }`}
              >
                EVI
              </button>
              {sentinel1Result && (
                <>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_vv")}
                    className={`px-2.5 py-1 rounded-lg transition font-medium ${
                      selectedLayer === "sentinel1_vv"
                        ? "bg-sky-400 text-slate-950 font-bold shadow"
                        : "text-sky-300 hover:bg-slate-800 border border-sky-500/30"
                    }`}
                  >
                    S1 VV
                  </button>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_vh")}
                    className={`px-2.5 py-1 rounded-lg transition font-medium ${
                      selectedLayer === "sentinel1_vh"
                        ? "bg-violet-400 text-slate-950 font-bold shadow"
                        : "text-violet-300 hover:bg-slate-800 border border-violet-500/30"
                    }`}
                  >
                    S1 VH
                  </button>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_ratio")}
                    className={`px-2.5 py-1 rounded-lg transition font-medium ${
                      selectedLayer === "sentinel1_ratio"
                        ? "bg-amber-400 text-slate-950 font-bold shadow"
                        : "text-amber-300 hover:bg-slate-800 border border-amber-500/30"
                    }`}
                  >
                    VV/VH
                  </button>
                </>
              )}
              {fusionResult && (
                <button
                  onClick={() => setSelectedLayer("quality_mask")}
                  className={`px-2.5 py-1 rounded-lg transition font-medium ${
                    selectedLayer === "quality_mask"
                      ? "bg-gradient-to-r from-emerald-400 to-teal-400 text-slate-950 font-bold shadow"
                      : "text-teal-300 hover:bg-slate-800 border border-teal-500/30"
                  }`}
                >
                  🛡️ Quality Mask
                </button>
              )}
              {stressResult?.spatial_stress_map?.stress_map_png && (
                <button
                  onClick={() => setSelectedLayer("stress")}
                  className={`px-2.5 py-1 rounded-lg transition font-medium ${
                    selectedLayer === "stress"
                      ? "bg-rose-500 text-white font-bold shadow ring-1 ring-rose-400"
                      : "text-rose-300 hover:bg-slate-800 border border-rose-500/30"
                  }`}
                >
                  ⚡ Stress Map
                </button>
              )}
            </div>
          </div>

          <FieldMap
            latitude={currentLocation.latitude}
            longitude={currentLocation.longitude}
            locationName={currentLocation.display_name}
            radiusKm={radiusKm}
            onCoordinatesChange={(lat, lon) => {
              setCurrentLocation((prev) => ({
                ...prev,
                latitude: lat,
                longitude: lon,
                display_name: `Custom Point (${lat.toFixed(5)}, ${lon.toFixed(5)})`,
              }));
              setConfirmedAOI(null);
            }}
            satelliteOverlayImage={satelliteImage ? `data:image/png;base64,${satelliteImage.image_base64}` : null}
            ndviOverlayImage={indicesResult?.ndvi.map || conditionResult?.indices.ndvi.map || null}
            ndmiOverlayImage={indicesResult?.ndmi.map || conditionResult?.indices.ndmi.map || null}
            ndreOverlayImage={indicesResult?.ndre.map || conditionResult?.indices.ndre.map || null}
            eviOverlayImage={indicesResult?.evi.map || conditionResult?.indices.evi.map || null}
            conditionOverlayImage={conditionResult?.condition_map || null}
            sentinel1VvOverlayImage={
              sentinel1Result?.map_layers?.vv_png ? `data:image/png;base64,${sentinel1Result.map_layers.vv_png}` : null
            }
            sentinel1VhOverlayImage={
              sentinel1Result?.map_layers?.vh_png ? `data:image/png;base64,${sentinel1Result.map_layers.vh_png}` : null
            }
            sentinel1RatioOverlayImage={
              sentinel1Result?.map_layers?.ratio_png ? `data:image/png;base64,${sentinel1Result.map_layers.ratio_png}` : null
            }
            qualityMaskOverlayImage={
              fusionResult?.previews?.quality_mask ? `data:image/png;base64,${fusionResult.previews.quality_mask}` : null
            }
            stressOverlayImage={
              stressResult?.spatial_stress_map?.stress_map_png
                ? `data:image/png;base64,${stressResult.spatial_stress_map.stress_map_png}`
                : null
            }
            selectedLayer={selectedLayer}
            onLayerChange={(layer) => setSelectedLayer(layer)}
            satelliteOverlayBounds={overlayBounds}
          />

          {/* Location Info Banner */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm">
            <div>
              <span className="text-slate-400 text-xs block">Active Location</span>
              <strong className="text-white text-base">{currentLocation.display_name}</strong>
              <div className="text-slate-400 text-xs font-mono mt-0.5">
                Latitude: {currentLocation.latitude.toFixed(6)}° | Longitude: {currentLocation.longitude.toFixed(6)}°
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
                Radius: {radiusKm} km (~{(Math.PI * radiusKm * radiusKm).toFixed(2)} km²)
              </span>
            </div>
          </div>
        </div>

        {/* 3. AOI & Acquisition Controls */}
        <div className="bg-slate-950/85 border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-6 backdrop-blur-md">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>⚙️</span>
            <span>Area of Interest & Acquisition Criteria</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Radius Slider */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800">
              <div className="flex justify-between items-center text-sm">
                <span className="font-semibold text-slate-300">AOI Radius:</span>
                <span className="font-mono text-emerald-400 font-bold text-base">{radiusKm} km</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="10.0"
                step="0.5"
                value={radiusKm}
                onChange={(e) => setRadiusKm(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500">
                <span>0.5 km (Small Field)</span>
                <span>5 km (Community)</span>
                <span>10 km (Regional)</span>
              </div>
              <button
                onClick={handleApplyRadius}
                className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition border border-slate-700"
              >
                Confirm AOI Geometry
              </button>
            </div>

            {/* Cloud Coverage Slider */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800">
              <div className="flex justify-between items-center text-sm">
                <span className="font-semibold text-slate-300">Max Cloud Coverage:</span>
                <span className="font-mono text-emerald-400 font-bold text-base">{maxCloudCoverage}%</span>
              </div>
              <input
                type="range"
                min="5"
                max="80"
                step="5"
                value={maxCloudCoverage}
                onChange={(e) => setMaxCloudCoverage(parseInt(e.target.value, 10))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-500">
                <span>5% (Clear Sky)</span>
                <span>30% (Standard)</span>
                <span>80% (Permissive)</span>
              </div>
              <div className="text-[11px] text-slate-400 pt-1">
                Sentinel-2 Level-2A imagery with cloud coverage above this limit is automatically filtered out.
              </div>
            </div>
          </div>

          {/* Date Range Selection */}
          <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800">
            <span className="text-sm font-semibold text-slate-300 block">Acquisition Date Range</span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-2">
              <span className="text-xs text-slate-400">Presets:</span>
              {[
                { label: "Last 7 Days", days: 7 },
                { label: "Last 15 Days", days: 15 },
                { label: "Last 30 Days", days: 30 },
                { label: "Last 60 Days", days: 60 },
              ].map((p) => (
                <button
                  key={p.days}
                  onClick={() => setPresetRange(p.days)}
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-lg transition"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Phase 12: Multimodal Fusion Configuration */}
          <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-teal-500/30">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-teal-300 flex items-center gap-2">
                <span>🛰️</span>
                <span>Multimodal Fusion Criteria (Sentinel-1 + Sentinel-2)</span>
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                Multimodal Fusion
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="text-slate-400">Temporal Matching Window:</span>
                  <span className="font-mono text-teal-400 font-bold">±{temporalWindowDays} days</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="7"
                  step="1"
                  value={temporalWindowDays}
                  onChange={(e) => setTemporalWindowDays(parseInt(e.target.value, 10))}
                  className="w-full accent-teal-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                  <span>±1d (Strict)</span>
                  <span>±3d (Optimal)</span>
                  <span>±7d (Permissive)</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Default temporal matching window: ±3 days. This setting is configurable.
                </p>
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Feature Normalization Strategy</label>
                <select
                  value={normalizationMethod}
                  onChange={(e) => setNormalizationMethod(e.target.value as any)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs font-mono focus:outline-none focus:ring-1 focus:ring-teal-500"
                >
                  <option value="raw">Raw Physical Units (Reflectance / dB)</option>
                  <option value="min-max">Min-Max Normalized [0, 1]</option>
                  <option value="z-score">Z-Score Standardized (μ=0, σ=1)</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Select raw for physical interpretability or min-max / z-score for neural network training.
                </span>
              </div>
            </div>
          </div>

          {/* Phase 14: AI Growth Prediction & Trend Forecasting Configuration */}
          <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-indigo-500/30">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-indigo-300 flex items-center gap-2">
                <span>📈</span>
                <span>Crop Growth Forecasting & Trend Configuration</span>
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Growth Forecast
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Forecast Horizon</label>
                <select
                  value={forecastHorizon}
                  onChange={(e) => setForecastHorizon(e.target.value as any)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="short">Short-Term (2 revisit cycles ~10-14 days)</option>
                  <option value="medium">Medium-Term (4 revisit cycles ~24-30 days)</option>
                  <option value="long">Longer-Term (6 revisit cycles ~36-45 days)</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Revisit cadences are derived directly from actual satellite observation frequencies.
                </span>
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Forecasting Model Algorithm</label>
                <select
                  value={forecastModelType}
                  onChange={(e) => setForecastModelType(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="linear_trend">Linear Trend (OLS with Slope Damping)</option>
                  <option value="moving_average">Moving Average (3-Observation Rolling Mean)</option>
                  <option value="last_observation">Last Observation (Persistence Naive)</option>
                  <option value="seasonal_naive">Seasonal Naive (Day-of-Year Matching)</option>
                  <option value="best_validated">Auto-Select Best Walk-Forward Model</option>
                </select>
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Every algorithm is benchmarked via chronological walk-forward validation against the last-observation baseline.
                </span>
              </div>
            </div>
          </div>

          {/* Action Trigger Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-9 gap-3 pt-2">
            <button
              onClick={handleFetchSatelliteImage}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 text-white font-bold rounded-xl transition border border-slate-700 flex items-center justify-center gap-2 text-xs sm:text-sm shadow-md"
            >
              {satelliteLoadingStep ? (
                <>
                  <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                  <span>
                    {satelliteLoadingStep === "searching" && "Searching catalog..."}
                    {satelliteLoadingStep === "selecting" && "Selecting scene..."}
                    {satelliteLoadingStep === "requesting" && "Requesting RGB..."}
                    {satelliteLoadingStep === "loading" && "Rendering preview..."}
                  </span>
                </>
              ) : (
                <>
                  <span>🛰️</span>
                  <span>Satellite Preview</span>
                </>
              )}
            </button>

            <button
              onClick={handleCalculateAllIndices}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 text-emerald-400 font-bold rounded-xl transition border border-emerald-500/40 flex items-center justify-center gap-2 text-xs sm:text-sm shadow-md"
            >
              {indicesLoadingStep ? (
                <>
                  <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                  <span>
                    {indicesLoadingStep === "requesting_bands" && "Acquiring bands..."}
                    {indicesLoadingStep === "aligning_resampling" && "Aligning 20m bands..."}
                    {indicesLoadingStep === "applying_cloud_mask" && "Filtering clouds..."}
                    {indicesLoadingStep === "calculating_indices" && "Computing indices..."}
                    {indicesLoadingStep === "generating_colormaps" && "Rendering..."}
                    {indicesLoadingStep === "complete" && "Indices ready."}
                  </span>
                </>
              ) : (
                <>
                  <span>📊</span>
                  <span>Vegetation Indices</span>
                </>
              )}
            </button>

            <button
              onClick={handleCalculateCropCondition}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-gradient-to-r from-emerald-500 via-lime-500 to-emerald-400 hover:from-emerald-400 hover:to-lime-400 disabled:bg-slate-800 text-slate-950 font-black rounded-xl transition shadow-xl shadow-emerald-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {conditionLoadingStep ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Evaluating condition...</span>
                </>
              ) : (
                <>
                  <span>🎯</span>
                  <span>Crop Condition</span>
                </>
              )}
            </button>

            <button
              onClick={handleFetchSentinel1Analysis}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 disabled:bg-slate-800 text-slate-950 font-black rounded-xl transition shadow-xl shadow-sky-500/20 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {sentinel1Loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Acquiring SAR GRD...</span>
                </>
              ) : (
                <>
                  <span>📡</span>
                  <span>Sentinel-1 SAR</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunMultimodalFusion}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-gradient-to-r from-emerald-500 via-teal-500 to-sky-500 hover:from-emerald-400 hover:to-sky-400 disabled:bg-slate-800 text-slate-950 font-black rounded-xl transition shadow-xl shadow-teal-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {fusionLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Fusing S1 + S2...</span>
                </>
              ) : (
                <>
                  <span>🛰️</span>
                  <span>S1+S2 Fusion</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunEarlyStressDetection}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading)}
              className="py-3.5 bg-gradient-to-r from-rose-500 via-amber-500 to-rose-400 hover:from-rose-400 hover:to-amber-400 disabled:bg-slate-800 text-slate-950 font-black rounded-xl transition shadow-xl shadow-rose-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {stressLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Inferring Vigour Anomaly...</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Vigour Anomaly</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunCropGrowthPrediction}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading || explainabilityLoading)}
              className="py-3.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-600 hover:from-indigo-400 hover:to-purple-400 disabled:bg-slate-800 text-white font-black rounded-xl transition shadow-xl shadow-indigo-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {growthLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Forecasting Trend...</span>
                </>
              ) : (
                <>
                  <span>📈</span>
                  <span>Growth Forecast</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunExplainabilityAnalysis}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading || explainabilityLoading)}
              className="py-3.5 bg-gradient-to-r from-amber-500 via-emerald-500 to-teal-500 hover:from-amber-400 hover:to-teal-400 disabled:bg-slate-800 text-slate-950 font-black rounded-xl transition shadow-xl shadow-emerald-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {explainabilityLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Explaining AI...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>AI Explainability</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunFieldRiskMapping}
              disabled={Boolean(satelliteLoadingStep || indicesLoadingStep || conditionLoadingStep || sentinel1Loading || fusionLoading || stressLoading || growthLoading || explainabilityLoading || fieldRiskLoading)}
              className="py-3.5 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 disabled:bg-slate-800 text-white font-black rounded-xl transition shadow-xl shadow-teal-500/25 flex items-center justify-center gap-2 text-xs sm:text-sm"
            >
              {fieldRiskLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Mapping Risk...</span>
                </>
              ) : (
                <>
                  <span>🛡️</span>
                  <span>Field Risk Map</span>
                </>
              )}
            </button>
          </div>

          {satelliteError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Satellite Acquisition Notice</strong>
                <p>{satelliteError}</p>
              </div>
            </div>
          )}

          {indicesError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Multispectral Analysis Notice</strong>
                <p>{indicesError}</p>
              </div>
            </div>
          )}

          {conditionError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Crop Condition Analysis Notice</strong>
                <p>{conditionError}</p>
              </div>
            </div>
          )}

          {sentinel1Error && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Sentinel-1 SAR Acquisition Notice</strong>
                <p>{sentinel1Error}</p>
              </div>
            </div>
          )}

          {fusionError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Multimodal Data Fusion Notice</strong>
                <p>{fusionError}</p>
              </div>
            </div>
          )}

          {stressError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Vigour Anomaly Detection Notice</strong>
                <p>{stressError}</p>
              </div>
            </div>
          )}

          {growthError && (
            <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
              <span>⚠️</span>
              <div>
                <strong className="block font-semibold">Growth Prediction &amp; Trend Notice</strong>
                <p>{growthError}</p>
              </div>
            </div>
          )}
        </div>

        {/* SATELLITE SOURCES SECTION (Multimodal Sentinel-2 + Sentinel-1) */}
        <section className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl">🛰️</span>
                <h2 className="text-xl font-bold text-white tracking-tight">
                  Multimodal Satellite Platform: Satellite Sources
                </h2>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  Multispectral + SAR
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Combining high-resolution multispectral reflectance with synthetic aperture radar backscatter for agricultural intelligence.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Sentinel-2 Card */}
            <div className="p-5 rounded-xl bg-slate-950/80 border border-emerald-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">🌿</span>
                  <h3 className="font-bold text-white text-base">Sentinel-2</h3>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300">
                    Optical Spectral
                  </span>
                </div>
                <span className="text-xs text-slate-400">10m - 20m GSD</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Sentinel-2 provides optical spectral information across visible, red-edge, near-infrared, and shortwave infrared bands (B02, B03, B04, B05, B08, B11). Measures biochemical chlorophyll concentration, leaf internal structure, and canopy moisture.
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {["B02", "B03", "B04", "B05", "B08", "B11", "NDVI", "NDMI", "NDRE", "EVI"].map((b) => (
                  <span key={b} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-700 text-slate-300">
                    {b}
                  </span>
                ))}
              </div>
              <div className="text-[11px] text-amber-300/90 flex items-center gap-1.5 pt-1">
                <span>⚠️</span>
                <span>Optical sensors cannot penetrate clouds, persistent fog, or severe atmospheric aerosol scattering.</span>
              </div>
            </div>

            {/* Sentinel-1 Card */}
            <div className="p-5 rounded-xl bg-slate-950/80 border border-sky-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">📡</span>
                  <h3 className="font-bold text-white text-base">Sentinel-1</h3>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-500/20 text-sky-300">
                    C-Band SAR
                  </span>
                </div>
                <span className="text-xs text-slate-400">10m GRD (IW)</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Sentinel-1 provides active microwave radar backscatter information (5.405 GHz) and complements optical observations. Penetrates atmospheric haze and standard cloud cover; sensitive to crop structural geometry, stalk density, surface roughness, and dielectric soil moisture.
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {["VV Linear", "VH Linear", "VV (dB)", "VH (dB)", "VV/VH Cross-Ratio", "VV-VH Diff (dB)"].map((b) => (
                  <span key={b} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-700 text-sky-300">
                    {b}
                  </span>
                ))}
              </div>
              <div className="text-[11px] text-sky-300/90 flex items-center gap-1.5 pt-1">
                <span>ℹ️</span>
                <span>Complementary radar backscatter. Sentinel-1 does NOT automatically solve cloud-cover problems in every circumstance (e.g., surface standing water saturation or severe convective precipitation reflection).</span>
              </div>
            </div>
          </div>
        </section>

        {/* SAR INDICATORS SECTION */}
        {sentinel1Result && (
          <section className="bg-slate-900 border-2 border-sky-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            
      {/* Header */}
            <div className="text-center space-y-2 border-b border-slate-800 pb-6">
              <div className="flex items-center justify-center gap-2">
                <span className="text-xs uppercase tracking-widest text-sky-400 font-extrabold px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20">
                  Sentinel-1 C-Band SAR Pipeline
                </span>
                {sentinel1Result.is_demo && (
                  <span className="text-xs uppercase tracking-widest text-amber-400 font-bold px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30">
                    DEMO SAR DATA
                  </span>
                )}
              </div>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                SAR INDICATORS
              </h2>
              <p className="text-slate-400 text-sm max-w-2xl mx-auto">
                Calibrated Sentinel-1 Synthetic Aperture Radar backscatter observables. Physical dielectric and structural indicators for multimodal fusion.
              </p>
            </div>

            {/* Metadata Bar */}
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
              <div className="flex items-center gap-2 text-slate-300">
                <span className="text-sky-400 font-semibold">Scene:</span>
                <span>{sentinel1Result.scene_id}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <span className="text-sky-400 font-semibold">Acquisition:</span>
                <span>📅 {sentinel1Result.acquisition_date}</span>
              </div>
              {sentinel1Result.matched_sentinel2_date && (
                <div className="flex items-center gap-2 text-slate-300">
                  <span className="text-emerald-400 font-semibold">Matched S2:</span>
                  <span>{sentinel1Result.matched_sentinel2_date} (Δ {sentinel1Result.date_difference_days}d)</span>
                </div>
              )}
              <div className="flex items-center gap-2 text-slate-300">
                <span className="text-sky-400 font-semibold">Orbit:</span>
                <span>{sentinel1Result.orbit_direction || "Ascending"}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <span className="text-sky-400 font-semibold">Product:</span>
                <span>{sentinel1Result.product_type}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <span className="text-sky-400 font-semibold">Valid Pixels:</span>
                <span>{sentinel1Result.valid_pixel_percentage.toFixed(1)}%</span>
              </div>
            </div>

            {/* 3 Core Indicator Cards: VV, VH, VV/VH Ratio */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* VV Card */}
              <div className="bg-slate-950/80 border border-sky-500/40 rounded-2xl p-6 space-y-4 hover:border-sky-400 transition shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold text-white text-lg flex items-center gap-2">
                      <span>📡 VV Backscatter</span>
                    </h3>
                    <span className="text-[11px] text-sky-400 font-medium">Co-Polarization (Vertical-Vertical)</span>
                  </div>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_vv")}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition ${
                      selectedLayer === "sentinel1_vv"
                        ? "bg-sky-400 text-slate-950 font-bold"
                        : "bg-slate-900 text-sky-300 border border-sky-500/30 hover:bg-slate-800"
                    }`}
                  >
                    View Map
                  </button>
                </div>

                <div className="border-t border-b border-slate-800 py-3 space-y-1">
                  <div className="flex items-baseline justify-between">
                    <span className="text-slate-400 text-xs">Mean Backscatter:</span>
                    <span className="text-2xl font-black text-sky-400 font-mono">
                      {sentinel1Result.vv.mean.toFixed(2)} dB
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                    <span>Linear: {Math.pow(10, sentinel1Result.vv.mean / 10).toFixed(4)}</span>
                    <span>Median: {sentinel1Result.vv.median.toFixed(2)} dB</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                  <div>
                    <span className="text-[10px] text-slate-400 block">Min</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vv.min.toFixed(2)} dB</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Max</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vv.max.toFixed(2)} dB</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Std Dev</span>
                    <span className="text-slate-200 font-semibold">±{sentinel1Result.vv.std.toFixed(2)} dB</span>
                  </div>
                </div>

                <div className="space-y-1.5 text-xs text-slate-300">
                  <strong className="text-sky-300 block font-semibold">Physical Interpretation:</strong>
                  <p className="leading-relaxed text-[11px] text-slate-400">
                    {sentinel1Result.physical_interpretation?.vv ||
                      "Sensitive to surface roughness, soil moisture content, and vertical crop stalk density."}
                  </p>
                </div>

                <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80">
                  <span>Quality: <strong className="text-slate-200">{sentinel1Result.valid_pixel_percentage >= 90 ? "Excellent" : "Good"}</strong></span>
                  <span>Date: <strong className="text-slate-200">{sentinel1Result.acquisition_date}</strong></span>
                </div>
              </div>

              {/* VH Card */}
              <div className="bg-slate-950/80 border border-violet-500/40 rounded-2xl p-6 space-y-4 hover:border-violet-400 transition shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold text-white text-lg flex items-center gap-2">
                      <span>📡 VH Backscatter</span>
                    </h3>
                    <span className="text-[11px] text-violet-400 font-medium">Cross-Polarization (Vertical-Horizontal)</span>
                  </div>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_vh")}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition ${
                      selectedLayer === "sentinel1_vh"
                        ? "bg-violet-400 text-slate-950 font-bold"
                        : "bg-slate-900 text-violet-300 border border-violet-500/30 hover:bg-slate-800"
                    }`}
                  >
                    View Map
                  </button>
                </div>

                <div className="border-t border-b border-slate-800 py-3 space-y-1">
                  <div className="flex items-baseline justify-between">
                    <span className="text-slate-400 text-xs">Mean Backscatter:</span>
                    <span className="text-2xl font-black text-violet-400 font-mono">
                      {sentinel1Result.vh.mean.toFixed(2)} dB
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                    <span>Linear: {Math.pow(10, sentinel1Result.vh.mean / 10).toFixed(4)}</span>
                    <span>Median: {sentinel1Result.vh.median.toFixed(2)} dB</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                  <div>
                    <span className="text-[10px] text-slate-400 block">Min</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vh.min.toFixed(2)} dB</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Max</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vh.max.toFixed(2)} dB</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Std Dev</span>
                    <span className="text-slate-200 font-semibold">±{sentinel1Result.vh.std.toFixed(2)} dB</span>
                  </div>
                </div>

                <div className="space-y-1.5 text-xs text-slate-300">
                  <strong className="text-violet-300 block font-semibold">Physical Interpretation:</strong>
                  <p className="leading-relaxed text-[11px] text-slate-400">
                    {sentinel1Result.physical_interpretation?.vh ||
                      "Dominated by volume scattering within leafy crop canopy foliage, biomass volume, and canopy structural complexity."}
                  </p>
                </div>

                <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80">
                  <span>Quality: <strong className="text-slate-200">{sentinel1Result.valid_pixel_percentage >= 90 ? "Excellent" : "Good"}</strong></span>
                  <span>Date: <strong className="text-slate-200">{sentinel1Result.acquisition_date}</strong></span>
                </div>
              </div>

              {/* VV/VH Ratio Card */}
              <div className="bg-slate-950/80 border border-amber-500/40 rounded-2xl p-6 space-y-4 hover:border-amber-400 transition shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold text-white text-lg flex items-center gap-2">
                      <span>📐 VV/VH Cross-Ratio</span>
                    </h3>
                    <span className="text-[11px] text-amber-400 font-medium">Polarimetric Structural Indicator</span>
                  </div>
                  <button
                    onClick={() => setSelectedLayer("sentinel1_ratio")}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition ${
                      selectedLayer === "sentinel1_ratio"
                        ? "bg-amber-400 text-slate-950 font-bold"
                        : "bg-slate-900 text-amber-300 border border-amber-500/30 hover:bg-slate-800"
                    }`}
                  >
                    View Map
                  </button>
                </div>

                <div className="border-t border-b border-slate-800 py-3 space-y-1">
                  <div className="flex items-baseline justify-between">
                    <span className="text-slate-400 text-xs">Mean Difference:</span>
                    <span className="text-2xl font-black text-amber-400 font-mono">
                      {sentinel1Result.vv_vh_diff_db.mean.toFixed(2)} dB
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                    <span>Linear Ratio: {sentinel1Result.vv_vh_ratio.mean.toFixed(2)}</span>
                    <span>Median Diff: {sentinel1Result.vv_vh_diff_db.median.toFixed(2)} dB</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                  <div>
                    <span className="text-[10px] text-slate-400 block">Linear Ratio Mean</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vv_vh_ratio.mean.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Linear Ratio Median</span>
                    <span className="text-slate-200 font-semibold">{sentinel1Result.vv_vh_ratio.median.toFixed(2)}</span>
                  </div>
                </div>

                <div className="space-y-1.5 text-xs text-slate-300">
                  <strong className="text-amber-300 block font-semibold">Physical Interpretation:</strong>
                  <p className="leading-relaxed text-[11px] text-slate-400">
                    {sentinel1Result.physical_interpretation?.vv_vh_ratio ||
                      "Ratio of surface to volume scattering. Higher values indicate surface dominance; lower values indicate dense vegetative canopy."}
                  </p>
                </div>

                <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80">
                  <span>Valid Pixels: <strong className="text-slate-200">{sentinel1Result.valid_pixel_percentage.toFixed(1)}%</strong></span>
                  <span>Date: <strong className="text-slate-200">{sentinel1Result.acquisition_date}</strong></span>
                </div>
              </div>
            </div>

            {/* Scientific Remote Sensing Disclaimer */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-start gap-3">
              <span className="text-base">ℹ️</span>
              <div className="space-y-1">
                <strong className="text-slate-200 block font-semibold">
                  Scientific Protocol Disclaimer
                </strong>
                <p className="leading-relaxed">
                  Sentinel-1 C-band SAR observations measure physical structural backscatter and dielectric constant. Radar backscatter is NOT called directly &ldquo;crop health&rdquo;—it provides structural and moisture evidence that will be ingested alongside optical indices in downstream multimodal fusion models.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* PHASE 12: MULTIMODAL SATELLITE FEATURE-FUSION SECTION */}
        {fusionResult && (
          <section id="multimodal-fusion-results" className="bg-slate-900 border-2 border-teal-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            
      {/* Header */}
            <div className="text-center space-y-2 border-b border-slate-800 pb-6">
              <div className="flex items-center justify-center gap-2 flex-wrap">
                <span className="text-xs uppercase tracking-widest text-teal-400 font-extrabold px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/20">
                  Multimodal Feature-Fusion Pipeline
                </span>
                <span
                  className={`text-xs uppercase tracking-widest font-extrabold px-3 py-1 rounded-full border ${
                    fusionResult.fusion_readiness === "READY_FOR_AI"
                      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                      : fusionResult.fusion_readiness === "PARTIAL_OPTICAL_ONLY"
                      ? "bg-amber-500/15 text-amber-300 border-amber-500/40"
                      : "bg-rose-500/15 text-rose-300 border-rose-500/40"
                  }`}
                >
                  {fusionResult.fusion_readiness === "READY_FOR_AI"
                    ? "✓ READY FOR AI TRAINING"
                    : fusionResult.fusion_readiness === "PARTIAL_OPTICAL_ONLY"
                    ? "⚠ PARTIAL (OPTICAL ONLY)"
                    : "✕ INSUFFICIENT QUALITY"}
                </span>
                {fusionResult.is_demo && (
                  <span className="text-xs uppercase tracking-widest text-amber-400 font-bold px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30">
                    DEMO MULTIMODAL DATA
                  </span>
                )}
              </div>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                MULTIMODAL SATELLITE FUSION
              </h2>
              <p className="text-slate-400 text-sm max-w-3xl mx-auto">
                Scientifically aligned Sentinel-2 optical spectral features (B02–B11, NDVI, NDMI, NDRE, EVI) combined with Sentinel-1 SAR backscatter (VV, VH, VV/VH ratio) on a canonical 10m UTM grid with unified quality masking.
              </p>
            </div>

            {/* 1. Multimodal Data Readiness Summary Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Optical Card */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-emerald-500/30 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-emerald-400 flex items-center gap-1.5">
                    <span>🌿</span> Optical Data
                  </span>
                  <span>Sentinel-2 L2A</span>
                </div>
                <div className="text-sm font-bold text-white truncate" title={fusionResult.temporal_matching.optical_scene_id}>
                  {fusionResult.temporal_matching.optical_scene_id}
                </div>
                <div className="text-xs text-slate-400 space-y-1 font-mono pt-1">
                  <div>Date: <span className="text-slate-200">{fusionResult.temporal_matching.optical_date}</span></div>
                  <div>Features: <span className="text-emerald-400 font-bold">{fusionResult.features.optical_features.length} channels</span></div>
                </div>
              </div>

              {/* SAR Card */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-sky-500/30 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-sky-400 flex items-center gap-1.5">
                    <span>📡</span> SAR Data
                  </span>
                  <span>Sentinel-1 GRD</span>
                </div>
                <div className="text-sm font-bold text-white truncate" title={fusionResult.temporal_matching.sar_scene_id || "None"}>
                  {fusionResult.temporal_matching.sar_scene_id || "No SAR Match in Window"}
                </div>
                <div className="text-xs text-slate-400 space-y-1 font-mono pt-1">
                  <div>Date: <span className="text-slate-200">{fusionResult.temporal_matching.sar_date || "N/A"}</span></div>
                  <div>Features: <span className="text-sky-400 font-bold">{fusionResult.features.sar_features.length} channels</span></div>
                </div>
              </div>

              {/* Temporal Matching Card */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-teal-500/30 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-teal-400 flex items-center gap-1.5">
                    <span>⏱️</span> Temporal Match
                  </span>
                  <span>±{fusionResult.temporal_matching.max_temporal_window_days}d Window</span>
                </div>
                <div className="text-xl font-black text-white">
                  {fusionResult.temporal_matching.matched
                    ? `Δ ${fusionResult.temporal_matching.date_difference_days} days`
                    : "Unmatched"}
                </div>
                <div className="text-xs text-slate-400 pt-1">
                  Strategy: <span className="text-slate-200 font-mono text-[11px]">{fusionResult.temporal_matching.strategy}</span>
                </div>
              </div>

              {/* Spatial Alignment Card */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-indigo-500/30 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-indigo-400 flex items-center gap-1.5">
                    <span>📐</span> Spatial Canonical Grid
                  </span>
                  <span>10m Resampled</span>
                </div>
                <div className="text-lg font-bold text-white font-mono">
                  {fusionResult.spatial_alignment.canonical_crs}
                </div>
                <div className="text-xs text-slate-400 space-y-1 font-mono pt-1">
                  <div>Grid Shape: <span className="text-slate-200">{fusionResult.spatial_alignment.grid_shape[0]} × {fusionResult.spatial_alignment.grid_shape[1]} px</span></div>
                  <div>Area: <span className="text-slate-200">{fusionResult.spatial_alignment.aoi_area_sq_km.toFixed(2)} km²</span></div>
                </div>
              </div>
            </div>

            {/* 2. Quality Mask Breakdown & Raster Visualization */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Quality Mask Raster Image */}
              <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-bold text-white text-base flex items-center gap-2">
                      <span>🛡️ Unified Multimodal Quality Mask</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Pixel-wise validation combining SCL cloud/shadow filtering with SAR physical bounds.
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setSelectedLayer("quality_mask");
                      window.scrollTo({ top: 350, behavior: "smooth" });
                    }}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-500/20 text-teal-300 border border-teal-500/30 hover:bg-teal-500/30 transition flex items-center gap-1"
                  >
                    <span>View on Field</span>
                  </button>
                </div>

                {fusionResult.previews?.quality_mask ? (
                  <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-900 aspect-square max-w-[380px] mx-auto shadow-inner flex items-center justify-center">
                    <img
                      src={`data:image/png;base64,${fusionResult.previews.quality_mask}`}
                      alt="Multimodal Quality Mask"
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="h-64 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-center text-xs text-slate-500">
                    Quality mask preview not generated
                  </div>
                )}

                {/* 3-Tier Legend */}
                <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
                  <span className="text-xs font-semibold text-slate-300 block">3-Tier Quality Hierarchy:</span>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-300">
                      <span className="font-bold block">VALID (1)</span>
                      <span className="text-[10px] text-slate-400">Clean Optical + SAR</span>
                      <span className="font-mono font-bold block mt-1">{fusionResult.quality.valid_percentage.toFixed(1)}%</span>
                    </div>
                    <div className="p-2 rounded-lg bg-amber-950/60 border border-amber-500/40 text-amber-300">
                      <span className="font-bold block">PARTIAL (2)</span>
                      <span className="text-[10px] text-slate-400">Valid Optical Only</span>
                      <span className="font-mono font-bold block mt-1">{fusionResult.quality.partial_percentage.toFixed(1)}%</span>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-400">
                      <span className="font-bold block">INVALID (0)</span>
                      <span className="text-[10px] text-slate-500">Cloud / Water / NoData</span>
                      <span className="font-mono font-bold block mt-1">{fusionResult.quality.invalid_percentage.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Spatial Alignment & Canonical Grid Specification */}
              <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-5 flex flex-col justify-between">
                <div>
                  <h3 className="font-bold text-white text-base flex items-center gap-2 mb-1">
                    <span>⚙️ Canonical Geospatial Grid</span>
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Ensures strict spatial correspondence across disparate sensor geometry without interpolation artefacts.
                  </p>

                  <div className="space-y-3 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Projected Coordinate System:</span>
                      <span className="text-white font-bold">{fusionResult.spatial_alignment.canonical_crs} (Zone {fusionResult.spatial_alignment.utm_zone})</span>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Ground Sampling Distance (GSD):</span>
                      <span className="text-emerald-400 font-bold">{fusionResult.spatial_alignment.resolution_meters} meters / pixel</span>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Grid Dimensions:</span>
                      <span className="text-white font-bold">{fusionResult.spatial_alignment.grid_shape[0]} rows × {fusionResult.spatial_alignment.grid_shape[1]} cols</span>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Total Pixel Footprint:</span>
                      <span className="text-white font-bold">{fusionResult.quality.total_pixels.toLocaleString()} pixels (~{fusionResult.spatial_alignment.aoi_area_sq_km.toFixed(2)} km²)</span>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Normalization Applied:</span>
                      <span className="text-teal-400 font-bold uppercase">{fusionResult.features.normalization_applied}</span>
                    </div>
                  </div>
                </div>

                {/* AI Readiness Banner */}
                <div className={`p-4 rounded-xl border text-xs leading-relaxed space-y-1 ${
                  fusionResult.fusion_readiness === "READY_FOR_AI"
                    ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-200"
                    : fusionResult.fusion_readiness === "PARTIAL_OPTICAL_ONLY"
                    ? "bg-amber-950/40 border-amber-500/40 text-amber-200"
                    : "bg-rose-950/40 border-rose-500/40 text-rose-200"
                }`}>
                  <strong className="block font-semibold">
                    {fusionResult.fusion_readiness === "READY_FOR_AI" && "✓ AI Training Dataset Status: Full Multimodal Ready"}
                    {fusionResult.fusion_readiness === "PARTIAL_OPTICAL_ONLY" && "⚠ AI Training Dataset Status: Partial Multimodal (Optical Only)"}
                    {fusionResult.fusion_readiness === "INSUFFICIENT_QUALITY" && "✕ AI Training Dataset Status: Insufficient Valid Area"}
                  </strong>
                  <p>
                    {fusionResult.fusion_readiness === "READY_FOR_AI" && "Optical and SAR features are temporally matched and co-registered on the canonical UTM grid. All 13 feature channels are verified valid for model ingestion."}
                    {fusionResult.fusion_readiness === "PARTIAL_OPTICAL_ONLY" && "No SAR observation was acquired within the temporal window. Optical spectral features remain valid; SAR channels are populated with NoData (NaN) and masked as Partial."}
                    {fusionResult.fusion_readiness === "INSUFFICIENT_QUALITY" && "Over 70% of the AOI pixels are obscured by clouds, shadows, or sensor artifacts. Acquisition quality does not meet the minimum threshold for model training."}
                  </p>
                </div>
              </div>
            </div>

            {/* 3. Multimodal Feature Stack Breakdown (Optical vs SAR) */}
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <span>📊 Multimodal Feature Stack</span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-teal-300 font-mono">
                      {fusionResult.features.feature_count} Aligned Channels
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Summary statistics calculated across valid canonical pixels ({fusionResult.quality.valid_pixels.toLocaleString()} pixels).
                  </p>
                </div>
              </div>

              {/* Table of all 13 features */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Feature</th>
                      <th className="py-2.5 px-3 font-sans">Sensor / Modality</th>
                      <th className="py-2.5 px-3 font-sans">Description</th>
                      <th className="py-2.5 px-3 text-right">Mean</th>
                      <th className="py-2.5 px-3 text-right">Median</th>
                      <th className="py-2.5 px-3 text-right">Min</th>
                      <th className="py-2.5 px-3 text-right">Max</th>
                      <th className="py-2.5 px-3 text-right">Std</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {fusionResult.features.feature_names.map((name) => {
                      const stats = fusionResult.features.statistics[name];
                      const isSar = name.startsWith("sar_");
                      return (
                        <tr key={name} className="hover:bg-slate-900/50">
                          <td className="py-2 px-3 font-bold text-white flex items-center gap-1.5">
                            <span className={`w-2 h-2 rounded-full ${isSar ? "bg-sky-400" : "bg-emerald-400"}`} />
                            <span>{name}</span>
                          </td>
                          <td className="py-2 px-3 font-sans">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              isSar ? "bg-sky-500/20 text-sky-300" : "bg-emerald-500/20 text-emerald-300"
                            }`}>
                              {isSar ? "Sentinel-1 SAR" : "Sentinel-2 Optical"}
                            </span>
                          </td>
                          <td className="py-2 px-3 font-sans text-slate-300 text-[11px]">
                            {stats?.description || name}
                          </td>
                          <td className="py-2 px-3 text-right text-teal-300 font-bold">
                            {stats?.mean !== null && stats?.mean !== undefined ? stats.mean.toFixed(3) : "NaN"}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-200">
                            {stats?.median !== null && stats?.median !== undefined ? stats.median.toFixed(3) : "NaN"}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-400">
                            {stats?.min !== null && stats?.min !== undefined ? stats.min.toFixed(3) : "NaN"}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-400">
                            {stats?.max !== null && stats?.max !== undefined ? stats.max.toFixed(3) : "NaN"}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-400">
                            {stats?.std !== null && stats?.std !== undefined ? stats.std.toFixed(3) : "NaN"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 4. Dataset Export Center */}
            <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 border border-teal-500/40 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h3 className="font-bold text-white text-lg flex items-center gap-2">
                    <span>💾 Standardized Multimodal Dataset Export</span>
                  </h3>
                  <p className="text-xs text-slate-300 mt-1 max-w-2xl">
                    Export co-registered multimodal tensors ready for direct ingestion by PyTorch, TensorFlow, Scikit-Learn, or QGIS. Includes 13 feature layers, 3-tier quality mask, and unassigned label placeholder (<code className="text-teal-300">labels_placeholder = None</code>).
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => handleExportFusionDataset("npz")}
                    className="px-4 py-2.5 bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-slate-950 font-black rounded-xl text-xs sm:text-sm shadow-lg shadow-teal-500/20 transition flex items-center gap-2"
                  >
                    <span>📥</span>
                    <span>Download NumPy (.npz)</span>
                  </button>
                  <button
                    onClick={() => handleExportFusionDataset("json")}
                    className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl text-xs sm:text-sm border border-slate-700 transition flex items-center gap-2"
                  >
                    <span>📄</span>
                    <span>Download JSON</span>
                  </button>
                  <button
                    onClick={() => handleExportFusionDataset("geotiff")}
                    className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl text-xs sm:text-sm border border-slate-700 transition flex items-center gap-2"
                  >
                    <span>🗺️</span>
                    <span>Download GeoTIFF (.tif)</span>
                  </button>
                </div>
              </div>
            </div>

            {/* 5. Scientific Guardrail & Non-Fabrication Protocol */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-start gap-3">
              <span className="text-base">🛡️</span>
              <div className="space-y-1">
                <strong className="text-slate-200 block font-semibold">
                  Multimodal Pipeline Protocol & Machine Learning Boundary
                </strong>
                <p className="leading-relaxed">
                  This pipeline completes the mathematical and geospatial fusion pipeline (temporal matching, canonical UTM reprojection, and quality masking). The exported arrays represent true physical observations. Vigour Anomaly Detection, Crop Growth Prediction, and Field Risk models will be trained on these validated multimodal records in downstream model evaluation; no synthetic predictions or fabricated accuracy benchmarks are asserted here.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* PHASE 13: AI-POWERED VIGOUR ANOMALY DETECTION SECTION */}
        {stressResult && (
          <section id="early-crop-stress-detection" className="bg-slate-900 border-2 border-rose-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            
      {/* Header */}
            <div className="text-center space-y-2 border-b border-slate-800 pb-6">
              <div className="flex items-center justify-center gap-2 flex-wrap">
                <span className="text-xs uppercase tracking-widest text-rose-400 font-extrabold px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/20">
                  AI Vigour Anomaly Intelligence
                </span>
                <span
                  className={`text-xs uppercase tracking-widest font-extrabold px-3 py-1 rounded-full border ${
                    stressResult.predicted_class === "NO_SIGNIFICANT_STRESS"
                      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                      : stressResult.predicted_class === "POTENTIAL_STRESS"
                      ? "bg-amber-500/15 text-amber-300 border-amber-500/40"
                      : stressResult.predicted_class === "ELEVATED_STRESS"
                      ? "bg-rose-500/15 text-rose-300 border-rose-500/40"
                      : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  {stressResult.predicted_class === "NO_SIGNIFICANT_STRESS" && "✓ NO SIGNIFICANT STRESS DETECTED"}
                  {stressResult.predicted_class === "POTENTIAL_STRESS" && "⚠️ POTENTIAL STRESS SIGNAL DETECTED"}
                  {stressResult.predicted_class === "ELEVATED_STRESS" && "🚨 ELEVATED POTENTIAL STRESS SIGNAL"}
                  {stressResult.predicted_class === "NO_RELIABLE_ASSESSMENT" && "✕ NO RELIABLE ASSESSMENT"}
                </span>
                {stressResult.is_demo && (
                  <span className="text-xs uppercase tracking-widest text-amber-400 font-bold px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30">
                    DEMO MODEL &amp; DATA — SYNTHETIC DEMONSTRATOR
                  </span>
                )}
              </div>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                VIGOUR ANOMALY DETECTION
              </h2>
              <p className="text-slate-400 text-sm max-w-3xl mx-auto">
                Supervised production stress detection is currently unavailable pending independently verified agronomic ground-truth data. Aligns available Sentinel-1 SAR and Sentinel-2 optical observations to detect rule-based canopy vigor and moisture anomalies.
              </p>
            </div>

            {/* Assessment Status Bar / Alert Banner */}
            {stressResult.assessment_status !== "ASSESSMENT_AVAILABLE" && (
              <div className={`p-4 rounded-2xl border flex items-start gap-3 text-sm ${
                stressResult.assessment_status === "POOR_DATA_QUALITY"
                  ? "bg-amber-950/40 border-amber-500/40 text-amber-200"
                  : stressResult.assessment_status === "MODEL_TRAINING_BLOCKED_PENDING_GROUND_TRUTH_DATA"
                  ? "bg-blue-950/40 border-blue-500/40 text-blue-200"
                  : "bg-slate-950 border-slate-800 text-slate-300"
              }`}>
                <span className="text-xl">
                  {stressResult.assessment_status === "POOR_DATA_QUALITY" ? "⚠️" : "ℹ️"}
                </span>
                <div className="space-y-1">
                  <strong className="block font-semibold">
                    Assessment Status: {stressResult.assessment_status.replace(/_/g, " ")}
                  </strong>
                  <p className="text-xs opacity-90">
                    {stressResult.assessment_status === "POOR_DATA_QUALITY" &&
                      "Scene cloud coverage exceeds threshold (>60%) or valid pixel coverage is insufficient (<25%). Stress probability cannot be reliably estimated without ground verification."}
                    {stressResult.assessment_status === "MODEL_TRAINING_BLOCKED_PENDING_GROUND_TRUTH_DATA" &&
                      "A verified production model trained on documented agricultural ground-truth labels is pending data ingestion. Synthetic demo mode can be used to preview pipeline features."}
                    {stressResult.assessment_status === "INSUFFICIENT_OBSERVATIONS" &&
                      "Insufficient prior satellite observations available to compute historical rate-of-change temporal features."}
                  </p>
                </div>
              </div>
            )}

            {/* 1. Core Assessment Metrics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Card 1: Calibrated Stress Probability */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-rose-500/30 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-rose-400 flex items-center gap-1.5">
                    <span>📈</span> Stress Probability
                  </span>
                  <span className="font-mono text-[11px]">Calibrated</span>
                </div>
                <div>
                  <div className="text-3xl font-black font-mono text-white">
                    {stressResult.stress_probability !== null
                      ? `${(stressResult.stress_probability * 100).toFixed(1)}%`
                      : "N/A"}
                  </div>
                  <span className="text-xs text-slate-400 block mt-0.5">
                    {stressResult.stress_probability !== null
                      ? stressResult.stress_probability < 0.35
                        ? "Nominal canopy response"
                        : stressResult.stress_probability < 0.65
                        ? "Moderate vegetative stress signal"
                        : "Elevated risk of canopy degradation"
                      : "Unassessed quality"}
                  </span>
                </div>
                {/* Progress bar */}
                {stressResult.stress_probability !== null && (
                  <div className="space-y-1.5">
                    <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                      <div
                        className={`h-2.5 rounded-full transition-all duration-700 ${
                          stressResult.stress_probability < 0.35
                            ? "bg-emerald-500"
                            : stressResult.stress_probability < 0.65
                            ? "bg-amber-400"
                            : "bg-rose-500"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, stressResult.stress_probability * 100))}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>0% Nominal</span>
                      <span>35% Potential</span>
                      <span>65% Elevated</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Card 2: Model & Version */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <span>🤖</span> Active Model
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                    {stressResult.model_version}
                  </span>
                </div>
                <div>
                  <div className="text-base font-bold text-white truncate">
                    {stressResult.model_name}
                  </div>
                  <span className="text-xs text-slate-400 block mt-0.5">
                    Features Used: <strong className="text-teal-300">{stressResult.features_used_count} channels</strong>
                  </span>
                </div>
                <div className="pt-1 text-[11px] text-slate-500">
                  Random Forest ensemble with isotonic probability calibration and class balancing.
                </div>
              </div>

              {/* Card 3: Observation Dates */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <span>📅</span> Observation Timeline
                  </span>
                  <span className="text-emerald-400 font-mono text-xs">
                    {stressResult.observation_dates.prior_observations_count} Priors
                  </span>
                </div>
                <div className="space-y-1 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Optical (S2):</span>
                    <span className="text-white font-semibold">
                      {stressResult.observation_dates.current_optical_date || "None"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">SAR (S1):</span>
                    <span className="text-white font-semibold">
                      {stressResult.observation_dates.current_sar_date || "Missing"}
                    </span>
                  </div>
                  <div className="flex justify-between border-t border-slate-800/80 pt-1">
                    <span className="text-slate-400">Prediction Date:</span>
                    <span className="text-emerald-300 font-bold">
                      {stressResult.observation_dates.prediction_date}
                    </span>
                  </div>
                </div>
                <div className="text-[10px] text-slate-500">
                  Strict leakage prevention: all temporal rates derived strictly prior to prediction date.
                </div>
              </div>

              {/* Card 4: Data Quality */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <span>🛡️</span> Data Quality
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                    stressResult.data_quality.quality_verdict === "HIGH_QUALITY"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : stressResult.data_quality.quality_verdict === "ACCEPTABLE"
                      ? "bg-teal-500/20 text-teal-300"
                      : "bg-amber-500/20 text-amber-300"
                  }`}>
                    {stressResult.data_quality.quality_verdict}
                  </span>
                </div>
                <div className="space-y-1 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Valid Pixels:</span>
                    <span className="text-white font-semibold">{stressResult.data_quality.valid_pixel_pct.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Cloud Cover:</span>
                    <span className="text-white font-semibold">{stressResult.data_quality.cloud_coverage.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">S1-S2 Gap:</span>
                    <span className="text-white font-semibold">
                      {stressResult.data_quality.s1_s2_temporal_gap_days !== null
                        ? `${stressResult.data_quality.s1_s2_temporal_gap_days.toFixed(1)} days`
                        : "N/A"}
                    </span>
                  </div>
                </div>
                <div className="text-[10px] text-slate-500">
                  {stressResult.data_quality.missing_sar
                    ? "SAR data missing outside temporal window; optical-only inference active."
                    : "Multimodal co-registration within temporal window verified."}
                </div>
              </div>
            </div>

            {/* 2. Spatial Stress Raster Map & Field Inspection */}
            {stressResult.spatial_stress_map?.stress_map_png ? (
              <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h3 className="font-bold text-white text-lg flex items-center gap-2">
                      <span>🗺️ Georeferenced Spatial Stress Map</span>
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-rose-300 font-mono">
                        {stressResult.spatial_stress_map.resolution_meters}m UTM Grid
                      </span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Pixel-level stress probability inferred across valid canonical pixels. Clouds and invalid water masks are strictly preserved as transparent NoData.
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedLayer("stress")}
                    className="px-4 py-2 bg-gradient-to-r from-rose-500 to-amber-500 hover:from-rose-400 hover:to-amber-400 text-slate-950 font-bold rounded-xl text-xs transition shadow-lg shadow-rose-500/20 flex items-center gap-1.5 self-start sm:self-auto"
                  >
                    <span>🔍</span>
                    <span>Overlay on Interactive Field Map</span>
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                  <div className="md:col-span-2 h-72 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center overflow-hidden p-2">
                    <img
                      src={`data:image/png;base64,${stressResult.spatial_stress_map.stress_map_png}`}
                      alt="Vigour Anomaly Probability Map"
                      className="w-full h-full object-contain"
                    />
                  </div>

                  <div className="space-y-4 text-xs">
                    <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                      <span className="font-semibold text-slate-200 block">Spatial Stress Legend:</span>
                      <div className="space-y-1.5 font-mono">
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-emerald-600 inline-block" />
                          <span className="text-slate-300">Nominal / Healthy (&lt; 0.35)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-amber-500 inline-block" />
                          <span className="text-slate-300">Potential Stress (0.35 – 0.65)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-rose-600 inline-block" />
                          <span className="text-slate-300">Elevated Stress (&gt; 0.65)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded bg-slate-800 border border-slate-700 inline-block" />
                          <span className="text-slate-400">Masked NoData / Cloud</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1 font-mono text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Assessable Field Area:</span>
                        <span className="text-emerald-400 font-bold">
                          {stressResult.spatial_stress_map.assessable_area_pct.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">NoData Extrapolation:</span>
                        <span className="text-slate-300 font-bold">Strictly Disabled</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {/* 3. Explainability Section (Local Drivers & Global Feature Importance) */}
            {stressResult.explainability && (
              <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-6">
                <div>
                  <h3 className="font-bold text-white text-lg flex items-center gap-2">
                    <span>💡 Model Explainability &amp; Feature Sensitivity</span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-amber-300 font-mono">
                      {stressResult.explainability.method}
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Identification of the mathematical indicators exerting greatest leverage on the prediction. Distinguishes local observation drivers from overall global model architecture.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Local Contributions */}
                  <div className="space-y-3">
                    <span className="text-xs font-semibold text-rose-300 flex items-center gap-1.5">
                      <span>🎯</span> Local Observation Sensitivity (This Field)
                    </span>
                    <div className="space-y-2">
                      {stressResult.explainability.local_contributions?.map((c, idx) => (
                        <div
                          key={idx}
                          className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs"
                        >
                          <div>
                            <span className="font-mono font-bold text-white block">{c.feature}</span>
                            <span className="text-[11px] text-slate-400 font-mono">
                              Value: {c.value !== null && c.value !== undefined ? c.value.toFixed(3) : "N/A"}
                            </span>
                          </div>
                          <div className="text-right">
                            <span
                              className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                                c.direction === "increasing_stress"
                                  ? "bg-rose-500/20 text-rose-300"
                                  : "bg-emerald-500/20 text-emerald-300"
                              }`}
                            >
                              {c.direction === "increasing_stress" ? "↑ Increases Stress" : "↓ Moderates Stress"}
                            </span>
                            <span className="block font-mono text-[11px] text-slate-400 mt-0.5">
                              Leverage: {c.contribution.toFixed(3)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Global Feature Importance */}
                  <div className="space-y-3">
                    <span className="text-xs font-semibold text-teal-300 flex items-center gap-1.5">
                      <span>🌐</span> Global Model Importance (Random Forest Architecture)
                    </span>
                    <div className="space-y-2">
                      {stressResult.explainability.global_top_features?.map((f, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-slate-300 font-semibold">{f.feature}</span>
                            <span className="text-teal-400 font-bold">{(f.importance * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                            <div
                              className="bg-teal-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${Math.min(100, f.importance * 300)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Non-Causation Callout */}
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 space-y-1.5 leading-relaxed">
                  <span className="font-semibold text-amber-300 flex items-center gap-1.5">
                    <span>⚠️</span>
                    <span>Scientific Explainability Protocol &amp; Non-Causation Boundary:</span>
                  </span>
                  <p>
                    {stressResult.explainability.disclaimer}
                  </p>
                </div>
              </div>
            )}

            {/* 4. Actionable Field Guidance & Scientific Guardrails */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Recommended Field Scouting Actions */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                <span className="font-bold text-white text-sm flex items-center gap-2">
                  <span>🧑‍🌾</span> Recommended Field Verification Actions
                </span>
                <ul className="space-y-2 text-xs text-slate-300">
                  {stressResult.recommended_actions.map((act, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-emerald-400 font-bold mt-0.5">•</span>
                      <span>{act}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Model Limitations & Warnings */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                <span className="font-bold text-white text-sm flex items-center gap-2">
                  <span>ℹ️</span> Operational Limitations &amp; Quality Notes
                </span>
                <ul className="space-y-2 text-xs text-slate-400">
                  {stressResult.prediction_limitations.map((lim, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-amber-400 font-bold mt-0.5">•</span>
                      <span>{lim}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        )}

        {/* Phase 14: AI-POWERED CROP GROWTH & TREND PREDICTION Section */}
        {growthResult && (
          <section className="bg-slate-900 border-2 border-indigo-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            {/* Section Header */}
            <div className="text-center space-y-3 border-b border-slate-800 pb-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30">
                <span className="text-xs uppercase tracking-widest text-indigo-400 font-extrabold">
                  Multimodal AI Growth Engine
                </span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                CROP GROWTH &amp; TREND PREDICTION
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto">
                Provides baseline projections from historical satellite observations, subject to observation availability and validation limitations.
              </p>

              {/* Status Badges */}
              <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                {growthResult.is_demo ? (
                  <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/50 text-amber-300 font-bold text-xs">
                    <span>⚠️</span>
                    <span>DEMO DATA — SYNTHETIC FORECAST</span>
                  </div>
                ) : (
                  <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/15 border border-emerald-500/50 text-emerald-300 font-bold text-xs">
                    <span>🛰️</span>
                    <span>VERIFIED MULTIMODAL OBSERVATIONS</span>
                  </div>
                )}

                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-xs font-mono">
                  <span>🤖</span>
                  <span>{growthResult.model_name}</span>
                  <span className="text-indigo-400 font-bold">{growthResult.model_version}</span>
                </div>

                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-xs font-mono">
                  <span>📅</span>
                  <span>Horizon: {growthResult.forecast_horizon_label}</span>
                </div>

                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-xs font-mono">
                  <span>📡</span>
                  <span>{growthResult.observation_count} Historical Acquisitions</span>
                </div>
              </div>
            </div>

            {/* Growth Trend Chart Component */}
            <div className="space-y-2">
              <GrowthTrendChart
                historicalObservations={growthResult.historical_observations}
                forecastDates={growthResult.forecast_dates}
                forecastValues={growthResult.forecast_values}
                predictionIntervals={growthResult.prediction_intervals}
                selectedIndex={activeGrowthIndex}
                onSelectIndex={(idx) => setActiveGrowthIndex(idx)}
                isDemo={growthResult.is_demo}
              />
            </div>

            {/* Metric Cards for Active Selected Index */}
            {(() => {
              const stat = growthResult.trend_statistics[activeGrowthIndex];
              const metrics =
                growthResult.validation_metrics[activeGrowthIndex]?.all_models_evaluated?.[
                  growthResult.validation_metrics[activeGrowthIndex]?.selected_model
                ] || growthResult.validation_metrics[activeGrowthIndex]?.all_models_evaluated?.["LinearTrend"];
              const fcVals = growthResult.forecast_values[activeGrowthIndex] || [];
              const intervals = growthResult.prediction_intervals[activeGrowthIndex] || [];

              const directionBadgeStyle =
                stat?.trend_direction === "Increasing"
                  ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                  : stat?.trend_direction === "Decreasing"
                  ? "bg-rose-500/20 text-rose-400 border-rose-500/40"
                  : stat?.trend_direction === "Relatively Stable"
                  ? "bg-sky-500/20 text-sky-400 border-sky-500/40"
                  : "bg-slate-700/40 text-slate-400 border-slate-600";

              return (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Card 1: Trend Direction & Momentum */}
                  <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                        <span>📊</span> Trend Direction
                      </span>
                      <span
                        className={`text-[11px] font-bold px-2 py-0.5 rounded-full border ${directionBadgeStyle}`}
                      >
                        {stat?.trend_direction || "Insufficient"}
                      </span>
                    </div>
                    <div>
                      <div className="text-2xl font-black text-white">
                        {stat?.trend_slope_per_day !== null && stat?.trend_slope_per_day !== undefined
                          ? `${stat.trend_slope_per_day > 0 ? "+" : ""}${(stat.trend_slope_per_day * 30).toFixed(3)}`
                          : "N/A"}
                        <span className="text-xs text-slate-400 font-normal ml-1">
                          / month
                        </span>
                      </div>
                      <span className="text-xs text-slate-400 block mt-1">
                        Net Change:{" "}
                        <strong className="text-white font-mono">
                          {stat?.absolute_change !== null && stat?.absolute_change !== undefined
                            ? `${stat.absolute_change > 0 ? "+" : ""}${stat.absolute_change.toFixed(3)}`
                            : "N/A"}
                        </strong>
                        {stat?.relative_change_pct !== null && stat?.relative_change_pct !== undefined && (
                          <span className="text-slate-400 ml-1">
                            ({stat.relative_change_pct > 0 ? "+" : ""}
                            {stat.relative_change_pct.toFixed(1)}%)
                          </span>
                        )}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                      {stat?.description || "Slope calculated over validated observation period."}
                    </div>
                  </div>

                  {/* Card 2: Measured vs Projected Targets */}
                  <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                        <span>🎯</span> Target Values
                      </span>
                      <span className="text-[11px] font-mono text-indigo-400 font-semibold">
                        {activeGrowthIndex}
                      </span>
                    </div>
                    <div className="space-y-1.5 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Latest Observed:</span>
                        <span className="text-white font-bold">
                          {stat?.current_value !== null && stat?.current_value !== undefined
                            ? stat.current_value.toFixed(3)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Initial Observed:</span>
                        <span className="text-slate-300">
                          {stat?.first_value !== null && stat?.first_value !== undefined
                            ? stat.first_value.toFixed(3)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between border-t border-slate-800 pt-1">
                        <span className="text-slate-400">Projected Endpoint:</span>
                        <span className="text-indigo-300 font-black">
                          {fcVals.length > 0 ? fcVals[fcVals.length - 1].toFixed(3) : "N/A"}
                        </span>
                      </div>
                      {intervals.length > 0 && (
                        <div className="flex justify-between text-[11px] text-sky-400">
                          <span className="text-slate-400">80% Interval:</span>
                          <span>
                            [{intervals[intervals.length - 1].lower.toFixed(2)},{" "}
                            {intervals[intervals.length - 1].upper.toFixed(2)}]
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-800">
                      Projection bounded to physical radiometric limits.
                    </div>
                  </div>

                  {/* Card 3: Model & Walk-Forward Validation */}
                  <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                        <span>🧪</span> Walk-Forward Accuracy
                      </span>
                      <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {growthResult.validation_metrics[activeGrowthIndex]?.selected_model || "LinearTrend"}
                      </span>
                    </div>
                    <div className="space-y-1.5 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">MAE:</span>
                        <span className="text-emerald-400 font-bold">
                          {metrics?.mae !== null && metrics?.mae !== undefined
                            ? metrics.mae.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">RMSE:</span>
                        <span className="text-teal-400 font-bold">
                          {metrics?.rmse !== null && metrics?.rmse !== undefined
                            ? metrics.rmse.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">R² Coefficient:</span>
                        <span className="text-slate-300">
                          {metrics?.r2 !== null && metrics?.r2 !== undefined
                            ? metrics.r2.toFixed(3)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between border-t border-slate-800 pt-1">
                        <span className="text-slate-400">Validation Windows:</span>
                        <span className="text-slate-300 font-bold">
                          {metrics?.validation_windows_count ?? 0} folds
                        </span>
                      </div>
                    </div>
                    <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-800">
                      Strict expanding-window walk-forward; no temporal leakage.
                    </div>
                  </div>

                  {/* Card 4: Satellite Observation Cadence & Revisit */}
                  <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                        <span>🛰️</span> Satellite Timeline
                      </span>
                      <span className="text-[11px] font-mono text-emerald-400">
                        {growthResult.observation_count} Acquisitions
                      </span>
                    </div>
                    <div className="space-y-1.5 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Historical Span:</span>
                        <span className="text-white font-bold">
                          {stat?.time_span_days || 0} days
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Forecast Steps:</span>
                        <span className="text-indigo-400 font-bold">
                          {growthResult.forecast_dates.length} revisit dates
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Quality Status:</span>
                        <span className="text-emerald-400">
                          {growthResult.data_quality?.quality_status || "Standard"}
                        </span>
                      </div>
                      <div className="flex justify-between border-t border-slate-800 pt-1">
                        <span className="text-slate-400">Cloud Median:</span>
                        <span className="text-slate-300">
                          {growthResult.data_quality?.median_cloud_coverage ?? 0}%
                        </span>
                      </div>
                    </div>
                    <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-800">
                      Forecast intervals align with physical orbital return schedules.
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Historical Multi-Period Comparison (Section 12) */}
            {growthResult.historical_comparison && (
              <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <span>🔄</span>
                      <span>Historical Multi-Period Trajectory Comparison</span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Direct empirical comparison between earlier and recent observation periods, followed by forecasted continuation.
                    </p>
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                    Section 12 Standard
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Prior Window */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                      {growthResult.historical_comparison.prior_period?.period_label}
                    </span>
                    <div className="text-xs text-slate-300 font-mono">
                      📅 {growthResult.historical_comparison.prior_period?.date_range?.[0]} →{" "}
                      {growthResult.historical_comparison.prior_period?.date_range?.[1]}
                    </div>
                    <div className="text-xs text-slate-400">
                      Acquisitions:{" "}
                      <strong className="text-white font-mono">
                        {growthResult.historical_comparison.prior_period?.observation_count}
                      </strong>
                    </div>
                    <div className="pt-2 border-t border-slate-800 space-y-1 text-xs font-mono">
                      {["NDVI", "NDMI", "NDRE", "EVI"].map((idx) => {
                        const val = growthResult.historical_comparison.prior_period?.mean_values?.[idx];
                        return (
                          <div key={idx} className="flex justify-between">
                            <span className="text-slate-400">{idx} Mean:</span>
                            <span className="text-white">{val !== null && val !== undefined ? val.toFixed(3) : "N/A"}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Recent Window */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider block">
                      {growthResult.historical_comparison.current_period?.period_label}
                    </span>
                    <div className="text-xs text-slate-300 font-mono">
                      📅 {growthResult.historical_comparison.current_period?.date_range?.[0]} →{" "}
                      {growthResult.historical_comparison.current_period?.date_range?.[1]}
                    </div>
                    <div className="text-xs text-slate-400">
                      Acquisitions:{" "}
                      <strong className="text-white font-mono">
                        {growthResult.historical_comparison.current_period?.observation_count}
                      </strong>
                    </div>
                    <div className="pt-2 border-t border-slate-800 space-y-1 text-xs font-mono">
                      {["NDVI", "NDMI", "NDRE", "EVI"].map((idx) => {
                        const val = growthResult.historical_comparison.current_period?.mean_values?.[idx];
                        const delta = growthResult.historical_comparison.period_to_period_change?.[idx];
                        return (
                          <div key={idx} className="flex justify-between">
                            <span className="text-slate-400">{idx} Mean:</span>
                            <span className="text-white">
                              {val !== null && val !== undefined ? val.toFixed(3) : "N/A"}{" "}
                              {delta !== null && delta !== undefined && (
                                <span className={delta >= 0 ? "text-emerald-400 text-[10px]" : "text-rose-400 text-[10px]"}>
                                  ({delta >= 0 ? "+" : ""}{delta.toFixed(3)})
                                </span>
                              )}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Forecast Continuation */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <span className="text-xs font-semibold text-teal-400 uppercase tracking-wider block">
                      Forecast Continuation Window
                    </span>
                    <div className="text-xs text-slate-300 font-mono">
                      📅 {growthResult.forecast_dates?.[0]} →{" "}
                      {growthResult.forecast_dates?.[growthResult.forecast_dates.length - 1]}
                    </div>
                    <div className="text-xs text-slate-400">
                      Projected Cycles:{" "}
                      <strong className="text-white font-mono">
                        {growthResult.forecast_dates.length} steps
                      </strong>
                    </div>
                    <div className="pt-2 border-t border-slate-800 space-y-1 text-xs font-mono">
                      {["NDVI", "NDMI", "NDRE", "EVI"].map((idx) => {
                        const fc = growthResult.historical_comparison.forecast_continuation?.[idx];
                        return (
                          <div key={idx} className="flex justify-between">
                            <span className="text-slate-400">{idx} Target:</span>
                            <span className="text-teal-300 font-bold">
                              {fc?.first_forecast !== null && fc?.first_forecast !== undefined
                                ? `${fc.first_forecast.toFixed(3)} → ${fc.terminal_forecast?.toFixed(3)}`
                                : "N/A"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex items-start gap-2">
                  <span className="text-amber-400 font-bold">ℹ️</span>
                  <span>
                    <strong>Empirical Measurement Standard:</strong> Reported period variations represent mathematical changes in satellite-measured reflectance and radar backscatter. They are not automatically attributed to drought, disease, pests, or nutrient deficiencies without in-field ground verification.
                  </span>
                </div>
              </div>
            )}

            {/* Model Benchmark Table across all candidate baselines */}
            {(() => {
              const evaluatedModels = growthResult.validation_metrics[activeGrowthIndex]?.all_models_evaluated || {};
              const selectedModel = growthResult.validation_metrics[activeGrowthIndex]?.selected_model;

              return (
                <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <span>📊</span>
                        <span>Walk-Forward Baseline Model Benchmarks ({activeGrowthIndex})</span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Chronological out-of-sample expanding-window evaluation (1-step-ahead walk-forward).
                      </p>
                    </div>
                    <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                      Active: {selectedModel || "LinearTrend"}
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300 font-mono">
                      <thead className="bg-slate-900 text-slate-400 text-[11px] uppercase border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">Candidate Model</th>
                          <th className="py-2.5 px-3">MAE</th>
                          <th className="py-2.5 px-3">RMSE</th>
                          <th className="py-2.5 px-3">R² Score</th>
                          <th className="py-2.5 px-3">MAPE</th>
                          <th className="py-2.5 px-3">Validation Folds</th>
                          <th className="py-2.5 px-3">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {Object.entries(evaluatedModels).map(([mName, mMetrics]: [string, any]) => {
                          const isCurrent = mName === selectedModel;
                          return (
                            <tr
                              key={mName}
                              className={`transition ${isCurrent ? "bg-indigo-500/10 font-bold text-white" : "hover:bg-slate-900/50"}`}
                            >
                              <td className="py-2 px-3 flex items-center gap-2">
                                {isCurrent && <span className="text-indigo-400">✓</span>}
                                <span>{mName}</span>
                              </td>
                              <td className="py-2 px-3 text-emerald-400">
                                {mMetrics.mae !== null ? mMetrics.mae.toFixed(4) : "N/A"}
                              </td>
                              <td className="py-2 px-3 text-teal-400">
                                {mMetrics.rmse !== null ? mMetrics.rmse.toFixed(4) : "N/A"}
                              </td>
                              <td className="py-2 px-3 text-slate-300">
                                {mMetrics.r2 !== null ? mMetrics.r2.toFixed(3) : "N/A"}
                              </td>
                              <td className="py-2 px-3 text-slate-400">
                                {mMetrics.mape !== null ? `${mMetrics.mape.toFixed(1)}%` : "N/A"}
                              </td>
                              <td className="py-2 px-3 text-slate-400">
                                {mMetrics.validation_windows_count ?? 0}
                              </td>
                              <td className="py-2 px-3">
                                {isCurrent ? (
                                  <span className="px-2 py-0.5 rounded bg-indigo-500/30 text-indigo-300 text-[10px]">
                                    Selected
                                  </span>
                                ) : (
                                  <span className="text-slate-500 text-[10px]">Benchmark</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()}

            {/* Warnings, Operational Limitations, & Scientific Disclaimer */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Warnings & Data Quality */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                <span className="font-bold text-white text-sm flex items-center gap-2">
                  <span>🛡️</span> Data Quality &amp; Pipeline Notes
                </span>
                <ul className="space-y-2 text-xs text-slate-300">
                  {growthResult.warnings.map((w, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-amber-400 font-bold mt-0.5">•</span>
                      <span>{w}</span>
                    </li>
                  ))}
                  {growthResult.warnings.length === 0 && (
                    <li className="text-slate-500">No data quality anomalies flagged during historical window.</li>
                  )}
                </ul>
              </div>

              {/* Operational Limitations */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                <span className="font-bold text-white text-sm flex items-center gap-2">
                  <span>ℹ️</span> Operational Limitations &amp; Boundaries
                </span>
                <ul className="space-y-2 text-xs text-slate-400">
                  {growthResult.limitations.map((lim, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-indigo-400 font-bold mt-0.5">•</span>
                      <span>{lim}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Scientific Disclaimer Banner */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-1.5 leading-relaxed">
              <span className="font-semibold text-amber-300 flex items-center gap-1.5">
                <span>⚠️</span>
                <span>Scientific Forecast Disclaimer &amp; Agronomic Protocol:</span>
              </span>
              <p>{growthResult.scientific_disclaimer}</p>
            </div>
          </section>
        )}

        {/* Phase 15: EXPLAINABLE AI, CONFIDENCE & UNCERTAINTY INTELLIGENCE Section */}
        {explainabilityLoading && (
          <section className="bg-slate-900/60 border border-amber-500/30 rounded-3xl p-10 flex flex-col items-center justify-center gap-4 text-center animate-pulse">
            <div className="w-10 h-10 border-4 border-amber-500 border-t-transparent rounded-full animate-spin" />
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-white">Generating Explainable AI Diagnostics...</h3>
              <p className="text-xs text-slate-400 max-w-md">
                Displays feature attribution and sensitivity information where supported by available models and input data.
              </p>
            </div>
          </section>
        )}

        {explainabilityError && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
            <span>⚠️</span>
            <div>
              <strong className="block font-semibold">Explainability Assessment Notice</strong>
              <p>{explainabilityError}</p>
            </div>
          </div>
        )}

        {explainabilityResult && (
          <section className="bg-slate-900 border-2 border-amber-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            <ExplainabilityPanel
              data={explainabilityResult}
              selectedIndex={activeGrowthIndex}
              onSelectIndex={(idx) => setActiveGrowthIndex(idx)}
            />
          </section>
        )}

        {/* Phase 16: FIELD-LEVEL AI RISK MAPPING & SPATIAL INTELLIGENCE Section */}
        {fieldRiskLoading && (
          <section className="bg-slate-900/60 border border-emerald-500/30 rounded-3xl p-10 flex flex-col items-center justify-center gap-4 text-center animate-pulse">
            <div className="w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-white">Generating Field-Level AI Risk Mapping...</h3>
              <p className="text-xs text-slate-400 max-w-md">
                Resampling multimodal Sentinel-1 and Sentinel-2 rasters to canonical 10m UTM grid, running calibrated spatial stress inference, and synthesizing multi-criteria risk evidence.
              </p>
            </div>
          </section>
        )}

        {fieldRiskError && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-600/80 text-rose-200 text-sm flex items-start gap-3">
            <span>⚠️</span>
            <div>
              <strong className="block font-semibold">Field Risk Mapping Notice</strong>
              <p>{fieldRiskError}</p>
            </div>
          </div>
        )}

        {fieldRiskResult && (
          <section
            id="field-risk-map-section"
            className="bg-slate-900 border-2 border-emerald-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500"
          >
            {/* Section Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-xs font-bold uppercase tracking-wider mb-2">
                  <span>🛰️ Field-Level Spatial Intelligence Layer</span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center gap-2.5">
                  <span>Field-Level AI Risk Mapping</span>
                </h2>
                <p className="text-sm text-slate-400 mt-1 max-w-2xl">
                  Visualizes spatial indicators and heuristic risk categories. These categories are not validated agronomic diagnoses. High-resolution pixel-level intelligence on a canonical 10m grid.
                </p>
              </div>

              <div className="flex items-center gap-2 self-start sm:self-center">
                <span className="text-xs font-mono px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300">
                  ID: {fieldRiskResult.analysis_id?.slice(0, 8)}...
                </span>
                {fieldRiskResult.is_demo && (
                  <span className="text-xs font-bold px-3 py-1.5 rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    DEMO DATA
                  </span>
                )}
              </div>
            </div>

            {/* Spatial Layer Selector */}
            <RiskLayerSelector
              selectedLayer={selectedRiskLayer}
              onSelectLayer={(key) => setSelectedRiskLayer(key)}
              availableLayerKeys={fieldRiskResult.available_layer_keys}
            />

            {/* Main Interactive Map */}
            <FieldRiskMap
              analysisId={fieldRiskResult.analysis_id}
              layers={fieldRiskResult.layers || {}}
              selectedLayer={selectedRiskLayer}
              geometry={fieldRiskResult.aoi_metadata?.geometry}
              latitude={fieldRiskResult.aoi_metadata?.latitude}
              longitude={fieldRiskResult.aoi_metadata?.longitude}
              radiusKm={fieldRiskResult.aoi_metadata?.radius_km}
              isDemo={fieldRiskResult.is_demo}
              observationDate={fieldRiskResult.statistics?.observation_date}
              onSelectLayer={(key) => setSelectedRiskLayer(key)}
              onPixelInspect={(data) => setInspectedPixel(data)}
              onInspectionLoading={(loading) => setPixelLoading(loading)}
            />

            {/* Active Layer Legend */}
            <FieldRiskLegend
              layerKey={selectedRiskLayer}
              legendData={fieldRiskResult.layers?.[selectedRiskLayer]?.legend}
            />

            {/* Two-Column Diagnostic Grid: Pixel Inspection + Field Statistics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PixelInspectionPanel
                inspection={inspectedPixel}
                isLoading={pixelLoading}
                onClear={() => setInspectedPixel(null)}
              />

              {fieldRiskResult.statistics && (
                <FieldRiskStatistics
                  statistics={fieldRiskResult.statistics}
                  isDemo={fieldRiskResult.is_demo}
                />
              )}
            </div>

            {/* Classification Evidence & Methodology */}
            <RiskEvidencePanel
              evidence={{
                classification_methodology: fieldRiskResult.classification_methodology,
                data_quality: fieldRiskResult.data_quality,
                model_provenance: fieldRiskResult.model_provenance,
                warnings: fieldRiskResult.warnings || [],
                scientific_limitations: fieldRiskResult.scientific_limitations || [],
              }}
            />
          </section>
        )}

        {/* 4. PHASE 5: Dedicated VEGETATION INDICES Section */}
        {indicesResult && (
          <section className="bg-slate-900 border-2 border-emerald-500/60 rounded-3xl p-6 sm:p-10 space-y-8 shadow-2xl animate-in fade-in duration-500">
            {/* Section Header */}
            <div className="text-center space-y-2 border-b border-slate-800 pb-6">
              <span className="text-xs uppercase tracking-widest text-emerald-400 font-extrabold px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                Sentinel-2 Multispectral Pipeline
              </span>
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                VEGETATION INDICES
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto">
                Comprehensive spectral measurements derived from Sentinel-2 Level-2A bands (B02, B04, B05, B08, B11) with spatial resampling and cloud filtering.
              </p>

              {indicesResult.scene.is_demo && (
                <div className="mt-3 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/50 text-amber-300 font-bold text-xs">
                  <span>⚠️</span>
                  <span>DEMO DATA — NOT LIVE SATELLITE ANALYSIS</span>
                </div>
              )}
            </div>

            {/* 4 Index Cards Summary Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Card 1: NDVI */}
              <div
                onClick={() => {
                  setActiveInspectionIndex("ndvi");
                  setSelectedLayer("ndvi");
                }}
                className={`p-6 rounded-2xl bg-slate-950 border transition cursor-pointer space-y-4 hover:border-emerald-500/60 ${
                  activeInspectionIndex === "ndvi"
                    ? "border-emerald-500 shadow-lg shadow-emerald-500/10 ring-1 ring-emerald-500/40"
                    : "border-slate-800"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🌿</span>
                      <h3 className="text-xl font-bold text-white">NDVI</h3>
                    </div>
                    <span className="text-xs text-emerald-400 font-medium">Vegetation vigor indicator</span>
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    B08, B04
                  </span>
                </div>

                <div className="space-y-1.5 text-sm pt-2 border-t border-slate-800/80">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Mean:</span>
                    <span className="font-mono font-bold text-emerald-300">
                      {indicesResult.ndvi.statistics.mean !== null ? indicesResult.ndvi.statistics.mean : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Median:</span>
                    <span className="font-mono font-bold text-slate-200">
                      {indicesResult.ndvi.statistics.median !== null ? indicesResult.ndvi.statistics.median : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Range:</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {indicesResult.ndvi.statistics.min !== null ? indicesResult.ndvi.statistics.min : "N/A"} to{" "}
                      {indicesResult.ndvi.statistics.max !== null ? indicesResult.ndvi.statistics.max : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs pt-1">
                    <span className="text-slate-500">Valid Pixels:</span>
                    <span className="font-mono text-slate-400">
                      {indicesResult.ndvi.statistics.valid_pixel_percentage}%
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 pt-1 flex items-center justify-between">
                  <span>Formula: (B08 - B04) / (B08 + B04)</span>
                  <span className="text-emerald-400 font-semibold text-xs">View Map →</span>
                </div>
              </div>

              {/* Card 2: NDMI */}
              <div
                onClick={() => {
                  setActiveInspectionIndex("ndmi");
                  setSelectedLayer("ndmi");
                }}
                className={`p-6 rounded-2xl bg-slate-950 border transition cursor-pointer space-y-4 hover:border-cyan-500/60 ${
                  activeInspectionIndex === "ndmi"
                    ? "border-cyan-500 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/40"
                    : "border-slate-800"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">💧</span>
                      <h3 className="text-xl font-bold text-white">NDMI</h3>
                    </div>
                    <span className="text-xs text-cyan-400 font-medium">Vegetation moisture-related indicator</span>
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    B08, B11 (20m resampled)
                  </span>
                </div>

                <div className="space-y-1.5 text-sm pt-2 border-t border-slate-800/80">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Mean:</span>
                    <span className="font-mono font-bold text-cyan-300">
                      {indicesResult.ndmi.statistics.mean !== null ? indicesResult.ndmi.statistics.mean : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Median:</span>
                    <span className="font-mono font-bold text-slate-200">
                      {indicesResult.ndmi.statistics.median !== null ? indicesResult.ndmi.statistics.median : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Range:</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {indicesResult.ndmi.statistics.min !== null ? indicesResult.ndmi.statistics.min : "N/A"} to{" "}
                      {indicesResult.ndmi.statistics.max !== null ? indicesResult.ndmi.statistics.max : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs pt-1">
                    <span className="text-slate-500">Valid Pixels:</span>
                    <span className="font-mono text-slate-400">
                      {indicesResult.ndmi.statistics.valid_pixel_percentage}%
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 pt-1 flex items-center justify-between">
                  <span>Formula: (B08 - B11) / (B08 + B11)</span>
                  <span className="text-cyan-400 font-semibold text-xs">View Map →</span>
                </div>
              </div>

              {/* Card 3: NDRE */}
              <div
                onClick={() => {
                  setActiveInspectionIndex("ndre");
                  setSelectedLayer("ndre");
                }}
                className={`p-6 rounded-2xl bg-slate-950 border transition cursor-pointer space-y-4 hover:border-lime-500/60 ${
                  activeInspectionIndex === "ndre"
                    ? "border-lime-500 shadow-lg shadow-lime-500/10 ring-1 ring-lime-500/40"
                    : "border-slate-800"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🌾</span>
                      <h3 className="text-xl font-bold text-white">NDRE</h3>
                    </div>
                    <span className="text-xs text-lime-400 font-medium">Red-edge vegetation indicator</span>
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    B08, B05 (20m resampled)
                  </span>
                </div>

                <div className="space-y-1.5 text-sm pt-2 border-t border-slate-800/80">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Mean:</span>
                    <span className="font-mono font-bold text-lime-300">
                      {indicesResult.ndre.statistics.mean !== null ? indicesResult.ndre.statistics.mean : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Median:</span>
                    <span className="font-mono font-bold text-slate-200">
                      {indicesResult.ndre.statistics.median !== null ? indicesResult.ndre.statistics.median : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Range:</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {indicesResult.ndre.statistics.min !== null ? indicesResult.ndre.statistics.min : "N/A"} to{" "}
                      {indicesResult.ndre.statistics.max !== null ? indicesResult.ndre.statistics.max : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs pt-1">
                    <span className="text-slate-500">Valid Pixels:</span>
                    <span className="font-mono text-slate-400">
                      {indicesResult.ndre.statistics.valid_pixel_percentage}%
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 pt-1 flex items-center justify-between">
                  <span>Formula: (B08 - B05) / (B08 + B05)</span>
                  <span className="text-lime-400 font-semibold text-xs">View Map →</span>
                </div>
              </div>

              {/* Card 4: EVI */}
              <div
                onClick={() => {
                  setActiveInspectionIndex("evi");
                  setSelectedLayer("evi");
                }}
                className={`p-6 rounded-2xl bg-slate-950 border transition cursor-pointer space-y-4 hover:border-teal-500/60 ${
                  activeInspectionIndex === "evi"
                    ? "border-teal-500 shadow-lg shadow-teal-500/10 ring-1 ring-teal-500/40"
                    : "border-slate-800"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🍃</span>
                      <h3 className="text-xl font-bold text-white">EVI</h3>
                    </div>
                    <span className="text-xs text-teal-400 font-medium">Enhanced vegetation index</span>
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    B08, B04, B02
                  </span>
                </div>

                <div className="space-y-1.5 text-sm pt-2 border-t border-slate-800/80">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Mean:</span>
                    <span className="font-mono font-bold text-teal-300">
                      {indicesResult.evi.statistics.mean !== null ? indicesResult.evi.statistics.mean : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Median:</span>
                    <span className="font-mono font-bold text-slate-200">
                      {indicesResult.evi.statistics.median !== null ? indicesResult.evi.statistics.median : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Range:</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {indicesResult.evi.statistics.min !== null ? indicesResult.evi.statistics.min : "N/A"} to{" "}
                      {indicesResult.evi.statistics.max !== null ? indicesResult.evi.statistics.max : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs pt-1">
                    <span className="text-slate-500">Valid Pixels:</span>
                    <span className="font-mono text-slate-400">
                      {indicesResult.evi.statistics.valid_pixel_percentage}%
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 pt-1 flex items-center justify-between">
                  <span>Formula: 2.5×(NIR-RED)/(NIR+6RED-7.5BLUE+1)</span>
                  <span className="text-teal-400 font-semibold text-xs">View Map →</span>
                </div>
              </div>
            </div>

            {/* Detailed Inspection Panel with Colormap and Scientific Legend */}
            <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>
                      {activeInspectionIndex === "ndvi" && "🌿"}
                      {activeInspectionIndex === "ndmi" && "💧"}
                      {activeInspectionIndex === "ndre" && "🌾"}
                      {activeInspectionIndex === "evi" && "🍃"}
                    </span>
                    <span>
                      {indicesResult[activeInspectionIndex].name} — {indicesResult[activeInspectionIndex].description}
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Bands used: {indicesResult[activeInspectionIndex].bands_used.join(", ")}
                  </p>
                </div>

                <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
                  {(["ndvi", "ndmi", "ndre", "evi"] as const).map((key) => (
                    <button
                      key={key}
                      onClick={() => {
                        setActiveInspectionIndex(key);
                        setSelectedLayer(key);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition uppercase ${
                        activeInspectionIndex === key
                          ? "bg-slate-800 text-white shadow"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {key}
                    </button>
                  ))}
                </div>
              </div>

              {/* Render Selected Index Colormap */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center shadow-inner">
                <img
                  src={indicesResult[activeInspectionIndex].map}
                  alt={`${indicesResult[activeInspectionIndex].name} map for ${currentLocation.display_name}`}
                  className="w-full max-h-[500px] object-contain"
                />

                <div className="absolute top-4 right-4 bg-slate-950/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 text-xs text-slate-300 shadow-md">
                  <span>Spectral Grid: Sentinel-2 10 m Resolution</span>
                </div>
              </div>

              {/* Continuous Scientific Scale Legend with user-specified non-misleading labels */}
              <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300">
                    {activeInspectionIndex === "ndvi" && "NDVI Scientific Continuous Scale"}
                    {activeInspectionIndex === "ndmi" && "NDMI Moisture-Related Continuous Scale"}
                    {activeInspectionIndex === "ndre" && "NDRE Red-Edge Continuous Scale"}
                    {activeInspectionIndex === "evi" && "EVI Enhanced Continuous Scale"}
                  </span>
                  <span className="text-slate-400">
                    {activeInspectionIndex === "ndvi" && "Low vegetation → High vegetation"}
                    {activeInspectionIndex === "ndmi" && "Lower moisture-related signal → Higher moisture-related signal"}
                    {activeInspectionIndex === "ndre" && "Lower red-edge vegetation signal → Higher red-edge vegetation signal"}
                    {activeInspectionIndex === "evi" && "Lower vegetation signal → Higher vegetation signal"}
                  </span>
                </div>

                {/* Color Gradients */}
                {activeInspectionIndex === "ndvi" && (
                  <div className="h-4 w-full rounded-md bg-gradient-to-r from-red-600 via-amber-400 via-yellow-300 via-lime-400 to-emerald-700 shadow-inner" />
                )}
                {activeInspectionIndex === "ndmi" && (
                  <div className="h-4 w-full rounded-md bg-gradient-to-r from-[#a6611a] via-[#dfc27d] via-[#80cdc1] via-[#35978f] to-[#01665e] shadow-inner" />
                )}
                {activeInspectionIndex === "ndre" && (
                  <div className="h-4 w-full rounded-md bg-gradient-to-r from-[#b2182b] via-[#ef8a62] via-[#f7f7f7] via-[#67a9cf] to-[#2166ac] shadow-inner" />
                )}
                {activeInspectionIndex === "evi" && (
                  <div className="h-4 w-full rounded-md bg-gradient-to-r from-[#be9664] via-[#ebcd6e] via-[#b6d957] via-[#52b144] to-[#0d5926] shadow-inner" />
                )}

                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>
                    {activeInspectionIndex === "ndvi" && "Low Vegetation (< 0.1)"}
                    {activeInspectionIndex === "ndmi" && "Lower Moisture Signal (< -0.2)"}
                    {activeInspectionIndex === "ndre" && "Lower Red-Edge Signal (< 0.1)"}
                    {activeInspectionIndex === "evi" && "Lower Vegetation (< 0.1)"}
                  </span>
                  <span>
                    {activeInspectionIndex === "ndvi" && "Moderate (0.3 – 0.5)"}
                    {activeInspectionIndex === "ndmi" && "Moderate Moisture (0.0 – 0.2)"}
                    {activeInspectionIndex === "ndre" && "Moderate Red-Edge (0.2 – 0.35)"}
                    {activeInspectionIndex === "evi" && "Moderate Vegetation (0.2 – 0.4)"}
                  </span>
                  <span>
                    {activeInspectionIndex === "ndvi" && "High Vegetation (≥ 0.7)"}
                    {activeInspectionIndex === "ndmi" && "Higher Moisture Signal (≥ 0.4)"}
                    {activeInspectionIndex === "ndre" && "Higher Red-Edge Signal (≥ 0.5)"}
                    {activeInspectionIndex === "evi" && "Higher Vegetation (≥ 0.6)"}
                  </span>
                </div>
              </div>

              {/* Statistics Metrics for Selected Index */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-0.5">Minimum</span>
                  <span className="text-white font-mono font-bold text-base">
                    {indicesResult[activeInspectionIndex].statistics.min !== null
                      ? indicesResult[activeInspectionIndex].statistics.min
                      : "N/A"}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-0.5">Maximum</span>
                  <span className="text-white font-mono font-bold text-base">
                    {indicesResult[activeInspectionIndex].statistics.max !== null
                      ? indicesResult[activeInspectionIndex].statistics.max
                      : "N/A"}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-emerald-500/40 bg-emerald-950/20">
                  <span className="text-xs text-emerald-400 font-semibold block mb-0.5">Mean</span>
                  <span className="text-emerald-300 font-mono font-black text-base">
                    {indicesResult[activeInspectionIndex].statistics.mean !== null
                      ? indicesResult[activeInspectionIndex].statistics.mean
                      : "N/A"}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-0.5">Median</span>
                  <span className="text-white font-mono font-bold text-base">
                    {indicesResult[activeInspectionIndex].statistics.median !== null
                      ? indicesResult[activeInspectionIndex].statistics.median
                      : "N/A"}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-0.5">Std Dev</span>
                  <span className="text-white font-mono font-bold text-base">
                    {indicesResult[activeInspectionIndex].statistics.std_dev !== null
                      ? indicesResult[activeInspectionIndex].statistics.std_dev
                      : "N/A"}
                  </span>
                </div>
              </div>

              {/* Download Active Index Map */}
              <div className="flex justify-end pt-2">
                <a
                  href={indicesResult[activeInspectionIndex].map}
                  download={`${activeInspectionIndex}_${currentLocation.query.replace(/\s+/g, "_")}_${indicesResult.scene.acquisition_date.split("T")[0]}.png`}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition flex items-center gap-2"
                >
                  <span>⬇️</span>
                  <span>Download {indicesResult[activeInspectionIndex].name} Map (PNG)</span>
                </a>
              </div>
            </div>

            {/* Scientific Notice / Guardrail */}
            <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-2">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5 text-sm">
                <span>ℹ️</span>
                <span>Scientific Terminology & Interpretation Guidelines</span>
              </span>
              <p className="leading-relaxed">
                Vegetation indices are physical surface reflectance indicators and signals. They do not constitute universal crop-health scores, automated disease diagnoses, or definitive proof of nitrogen deficiency or drought.
              </p>
              <ul className="list-disc pl-5 space-y-1 text-slate-400">
                <li><strong className="text-slate-300">NDVI:</strong> Quantifies canopy chlorophyll absorption vigor and biomass density.</li>
                <li><strong className="text-slate-300">NDMI:</strong> Reflects canopy moisture-related signal based on shortwave infrared water absorption.</li>
                <li><strong className="text-slate-300">NDRE:</strong> Evaluates red-edge spectral transition sensitivity, useful in dense canopies.</li>
                <li><strong className="text-slate-300">EVI:</strong> Optimizes vegetation signal with soil background decoupling and aerosol resistance.</li>
              </ul>
              <p className="text-slate-500 pt-1">
                Field decisions should corroborate satellite indicator signals with soil moisture readings, local weather history, and ground truth agronomist inspections.
              </p>
            </div>
          </section>
        )}

        {/* 5. PHASE 6: CROP CONDITION INTELLIGENCE SECTION */}
        {conditionResult && (
          <section id="analytical-results" className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl space-y-8">
            
      {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-xs font-bold uppercase tracking-wider">
                    Crop Condition Intelligence Layer
                  </span>
                  {conditionResult.scene.is_demo && (
                    <span className="px-2.5 py-1 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40 text-xs font-bold uppercase tracking-wider animate-pulse">
                      DEMO DATA
                    </span>
                  )}
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-white flex items-center gap-2">
                  <span>🎯</span>
                  <span>Crop Condition Intelligence</span>
                </h2>
                <p className="text-xs sm:text-sm text-slate-400 mt-1">
                  Heuristic spectral condition scoring combining NDVI (Vigor) + NDMI (Moisture) + NDRE (Red-Edge) + EVI (Canopy Biomass).
                </p>
              </div>

              {/* View Overlay on Map CTA */}
              <button
                onClick={() => {
                  setSelectedLayer("condition");
                  window.scrollTo({ top: 350, behavior: "smooth" });
                }}
                className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-lime-500 text-slate-950 font-bold text-xs sm:text-sm shadow-lg shadow-emerald-500/20 transition flex items-center justify-center gap-2 self-start sm:self-auto"
              >
                <span>🗺️</span>
                <span>View Condition Map on Field</span>
              </button>
            </div>

            {/* Top KPI Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Overall Condition Score */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border border-emerald-500/30 shadow-lg relative overflow-hidden">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Vegetation Condition
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl sm:text-4xl font-black text-emerald-400">
                    {conditionResult.overall_condition}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs text-slate-400">Score:</span>
                  <span className="font-mono font-bold text-white text-base">
                    {conditionResult.condition_score} / 100
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 mt-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-emerald-500 to-lime-400 h-1.5 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, conditionResult.condition_score))}%` }}
                  />
                </div>
              </div>

              {/* Dominant Condition Zone */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 shadow-md">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Dominant Condition Zone
                </span>
                <span className="text-xl sm:text-2xl font-black text-white block mt-1">
                  {conditionResult.dominant_zone}
                </span>
                <span className="text-xs text-slate-400 block mt-2">
                  Calculated across {conditionResult.quality.valid_pixels.toLocaleString()} valid pixels
                </span>
              </div>

              {/* Data Quality & Confidence */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 shadow-md">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Analysis Data Quality
                </span>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className={`px-2.5 py-0.5 rounded text-xs font-black uppercase ${
                      conditionResult.quality.quality_level === "Excellent"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        : conditionResult.quality.quality_level === "Good"
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                        : conditionResult.quality.quality_level === "Fair"
                        ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                        : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                    }`}
                  >
                    {conditionResult.quality.quality_level}
                  </span>
                  <span className="text-xl sm:text-2xl font-mono font-black text-white">
                    {conditionResult.quality.valid_percentage}%
                  </span>
                </div>
                <span className="text-xs text-slate-400 block mt-2">
                  Valid Observations: {conditionResult.quality.valid_percentage}% of AOI
                </span>
              </div>

              {/* Stress Zones Detected */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 shadow-md">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Spectral Stress Patterns
                </span>
                <span className="text-2xl sm:text-3xl font-black text-amber-400 block mt-1">
                  {conditionResult.patterns.length}
                </span>
                <span className="text-xs text-slate-400 block mt-2">
                  {conditionResult.patterns.length === 0
                    ? "No co-occurring stress patterns identified"
                    : `${conditionResult.patterns.length} possible stress zone(s) flagged`}
                </span>
              </div>
            </div>

            {/* Quality Warnings Alert (if any) */}
            {conditionResult.warnings && conditionResult.warnings.length > 0 && (
              <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-600/60 text-amber-200 text-xs sm:text-sm space-y-1">
                <div className="flex items-center gap-2 font-bold text-amber-300">
                  <span>⚠️</span>
                  <span>Data Quality & Confidence Disclaimers:</span>
                </div>
                <ul className="list-disc pl-6 space-y-1 text-slate-300">
                  {conditionResult.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Spatial Condition Zones & Map Preview */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Raster Image Display */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-4 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-bold text-white flex items-center gap-2">
                      <span>🗺️</span>
                      <span>Discrete Spatial Condition Raster</span>
                    </span>
                    <span className="text-xs text-slate-400">Classified 0–5</span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">
                    Color-coded spatial classification isolating low-vigor and stress-prone micro-zones across the field.
                  </p>

                  <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-900 aspect-square max-w-[380px] mx-auto shadow-inner">
                    <img
                      src={conditionResult.condition_map}
                      alt="Crop Condition Zone Map"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>

                {/* Discrete Legend */}
                <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                    <span>Heuristic Vegetation Condition Scale:</span>
                    <span className="text-[10px] text-slate-400">(Non-diagnostic)</span>
                  </div>
                  <div className="grid grid-cols-5 gap-1 text-[11px] text-center font-medium">
                    <div className="py-1 rounded bg-[#b91c1c] text-white">Very Low</div>
                    <div className="py-1 rounded bg-[#ea580c] text-white">Low</div>
                    <div className="py-1 rounded bg-[#ca8a04] text-slate-950">Moderate</div>
                    <div className="py-1 rounded bg-[#65a30d] text-slate-950">Good</div>
                    <div className="py-1 rounded bg-[#15803d] text-white">Very Good</div>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-400 px-1">
                    <span>0–20</span>
                    <span>20–40</span>
                    <span>40–60</span>
                    <span>60–80</span>
                    <span>80–100</span>
                  </div>
                </div>
              </div>

              {/* Condition Distribution Bars */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-5">
                <span className="text-sm font-bold text-white block">
                  Condition Zone Distribution (% of Valid Field Area)
                </span>

                <div className="space-y-4 text-xs">
                  {/* Very Good */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#15803d]"></span>
                        <span>Very Good Condition (80–100)</span>
                      </span>
                      <span className="font-mono text-emerald-400 font-bold">{conditionResult.zones.very_good}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-[#15803d] h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${conditionResult.zones.very_good}%` }}
                      />
                    </div>
                  </div>

                  {/* Good */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#65a30d]"></span>
                        <span>Good Condition (60–80)</span>
                      </span>
                      <span className="font-mono text-lime-400 font-bold">{conditionResult.zones.good}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-[#65a30d] h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${conditionResult.zones.good}%` }}
                      />
                    </div>
                  </div>

                  {/* Moderate */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#ca8a04]"></span>
                        <span>Moderate Condition (40–60)</span>
                      </span>
                      <span className="font-mono text-yellow-400 font-bold">{conditionResult.zones.moderate}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-[#ca8a04] h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${conditionResult.zones.moderate}%` }}
                      />
                    </div>
                  </div>

                  {/* Low */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#ea580c]"></span>
                        <span>Low Condition (20–40)</span>
                      </span>
                      <span className="font-mono text-orange-400 font-bold">{conditionResult.zones.low}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-[#ea580c] h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${conditionResult.zones.low}%` }}
                      />
                    </div>
                  </div>

                  {/* Very Low */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#b91c1c]"></span>
                        <span>Very Low Condition (0–20)</span>
                      </span>
                      <span className="font-mono text-rose-400 font-bold">{conditionResult.zones.very_low}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-[#b91c1c] h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${conditionResult.zones.very_low}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-900 flex justify-between items-center text-xs text-slate-400">
                  <span>Analyzed AOI Field Coverage:</span>
                  <span className="font-mono text-slate-200 font-semibold">
                    ~{conditionResult.area_represented_km2} km²
                  </span>
                </div>
              </div>
            </div>

            {/* Possible Spectral Stress Patterns */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>⚠️</span>
                    <span>Possible Spectral Stress Patterns</span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Rule-based spectral pattern recognition across vigor (NDVI/EVI), moisture (NDMI), and red-edge (NDRE) channels.
                  </p>
                </div>
              </div>

              {conditionResult.patterns.length === 0 ? (
                <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 text-slate-400 text-xs sm:text-sm text-center">
                  ✅ No significant co-occurring stress patterns detected in the valid canopy observations.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {conditionResult.patterns.map((p, idx) => (
                    <div
                      key={idx}
                      className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-2 hover:border-slate-700 transition"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white text-sm flex items-center gap-1.5">
                          <span>🔍</span>
                          <span>{p.label}</span>
                        </span>
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                              p.severity === "high"
                                ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                                : p.severity === "moderate"
                                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                                : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                            }`}
                          >
                            {p.severity} Severity
                          </span>
                          <span className="font-mono font-bold text-xs text-slate-300">
                            {p.percentage}% Area
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {p.explanation}
                      </p>
                      <div className="pt-2 text-[11px] text-slate-500 italic">
                        * Note: Spectral patterns represent radiometric signals and require agronomist ground verification before taking corrective action.
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 6. INDEX COMPARISON CHART & TABLE */}
            <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800 space-y-6">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <span>📊</span>
                  <span>Multispectral Index Comparison</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Comparative summary of NDVI, NDMI, NDRE, and EVI statistical distributions across the valid AOI pixels.
                </p>
              </div>

              {/* Comparative Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-900 text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-4">Index</th>
                      <th className="py-3 px-4">Physical Indicator</th>
                      <th className="py-3 px-4">Bands</th>
                      <th className="py-3 px-4 text-right">Mean</th>
                      <th className="py-3 px-4 text-right">Median</th>
                      <th className="py-3 px-4 text-right">Min</th>
                      <th className="py-3 px-4 text-right">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {/* NDVI */}
                    <tr className="hover:bg-slate-900/50">
                      <td className="py-3 px-4 font-bold text-emerald-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                        <span>NDVI</span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-300">Vegetation vigor & chlorophyll absorption</td>
                      <td className="py-3 px-4 text-slate-400">B04 (Red), B08 (NIR)</td>
                      <td className="py-3 px-4 text-right text-emerald-300 font-bold">{conditionResult.indices.ndvi.statistics.mean ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-200">{conditionResult.indices.ndvi.statistics.median ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndvi.statistics.min ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndvi.statistics.max ?? "N/A"}</td>
                    </tr>
                    {/* NDMI */}
                    <tr className="hover:bg-slate-900/50">
                      <td className="py-3 px-4 font-bold text-cyan-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                        <span>NDMI</span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-300">Moisture-related vegetation water signal</td>
                      <td className="py-3 px-4 text-slate-400">B08 (NIR), B11 (SWIR)</td>
                      <td className="py-3 px-4 text-right text-cyan-300 font-bold">{conditionResult.indices.ndmi.statistics.mean ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-200">{conditionResult.indices.ndmi.statistics.median ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndmi.statistics.min ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndmi.statistics.max ?? "N/A"}</td>
                    </tr>
                    {/* NDRE */}
                    <tr className="hover:bg-slate-900/50">
                      <td className="py-3 px-4 font-bold text-lime-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-lime-400"></span>
                        <span>NDRE</span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-300">Red-edge vegetation transition response</td>
                      <td className="py-3 px-4 text-slate-400">B08 (NIR), B05 (RedEdge)</td>
                      <td className="py-3 px-4 text-right text-lime-300 font-bold">{conditionResult.indices.ndre.statistics.mean ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-200">{conditionResult.indices.ndre.statistics.median ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndre.statistics.min ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.ndre.statistics.max ?? "N/A"}</td>
                    </tr>
                    {/* EVI */}
                    <tr className="hover:bg-slate-900/50">
                      <td className="py-3 px-4 font-bold text-teal-400 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-teal-400"></span>
                        <span>EVI</span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-300">Enhanced canopy biomass (soil-decoupled)</td>
                      <td className="py-3 px-4 text-slate-400">B02 (Blue), B04 (Red), B08 (NIR)</td>
                      <td className="py-3 px-4 text-right text-teal-300 font-bold">{conditionResult.indices.evi.statistics.mean ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-200">{conditionResult.indices.evi.statistics.median ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.evi.statistics.min ?? "N/A"}</td>
                      <td className="py-3 px-4 text-right text-slate-400">{conditionResult.indices.evi.statistics.max ?? "N/A"}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Explanatory Callout */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 space-y-1.5 leading-relaxed">
                <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                  <span>🔬</span>
                  <span>Physical Indicator Distinctions:</span>
                </span>
                <p>
                  These four indices reflect distinct physical radiative transfer mechanisms. <strong>NDVI</strong> is saturated in dense canopies; <strong>NDMI</strong> measures shortwave infrared liquid absorption inside cellular tissue; <strong>NDRE</strong> senses subtle chlorophyll content variations via the steep 705nm red-edge inflection; and <strong>EVI</strong> reduces atmospheric aerosol interference while minimizing soil brightness distortions.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* 6. PHASE 9: FIELD INTELLIGENCE & EXPLAINABLE SATELLITE ANALYSIS */}
        <AIFieldReport
          report={aiReport}
          loading={aiLoading}
          loadingStage={aiLoadingStage}
          error={aiError}
          onGenerate={handleGenerateAIReport}
          onRegenerate={handleGenerateAIReport}
          hasAnalyticalData={Boolean(conditionResult || indicesResult)}
          isDemoMode={Boolean(
            conditionResult?.scene?.is_demo ||
              indicesResult?.scene?.is_demo ||
              satelliteImage?.is_demo
          )}
        />
      </main>
    </div>
  );
}
