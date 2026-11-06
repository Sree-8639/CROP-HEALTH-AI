/**
 * Centralized API configuration for the Crop Health AI frontend.
 * Resolves API base URL dynamically from environment variables,
 * defaulting to http://127.0.0.1:8000 for local development.
 */
export const API_BASE_URL: string =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://127.0.0.1:8000";

export const API_BASE: string = API_BASE_URL;

export const API_ROUTES = {
  HEALTH: `${API_BASE_URL}/api/health`,
  DOCS: `${API_BASE_URL}/docs`,
  AOI_REGISTER: `${API_BASE_URL}/api/v1/aoi`,
  AOI_CALCULATE: `${API_BASE_URL}/api/v1/spatial/calculate-aoi`,
  GEOCODE: `${API_BASE_URL}/api/v1/geocode`,
  SATELLITE_PREVIEW: `${API_BASE_URL}/api/v1/satellite/preview`,
  INDICES: `${API_BASE_URL}/api/v1/analysis/indices`,
  CROP_CONDITION: `${API_BASE_URL}/api/v1/analysis/crop-condition`,
  SENTINEL1: `${API_BASE_URL}/api/v1/analysis/sentinel1`,
  FUSION: `${API_BASE_URL}/api/v1/analysis/fusion`,
  FUSION_EXPORT: `${API_BASE_URL}/api/v1/analysis/fusion/export`,
  STRESS_DETECTION: `${API_BASE_URL}/api/v1/analysis/stress-detection`,
  GROWTH_PREDICTION: `${API_BASE_URL}/api/v1/analysis/growth-prediction`,
  EXPLAINABILITY: `${API_BASE_URL}/api/v1/analysis/explainability`,
  FIELD_RISK_MAP: `${API_BASE_URL}/api/v1/analysis/field-risk-map`,
  FIELD_RISK_PIXEL: `${API_BASE_URL}/api/v1/analysis/field-risk-map/pixel-inspect`,
  AI_REPORT: `${API_BASE_URL}/api/v1/analysis/ai-report`,
  HISTORY: `${API_BASE_URL}/api/v1/analysis/history`,
  HISTORY_COMPARE: `${API_BASE_URL}/api/v1/analysis/history/compare`,
  FARMER_DASHBOARD_OVERVIEW: `${API_BASE_URL}/api/v1/farmer-dashboard/overview`,
  FARMER_DASHBOARD_FIELD: (fieldId: string) => `${API_BASE_URL}/api/v1/farmer-dashboard/field/${encodeURIComponent(fieldId)}`,
  FARMER_DASHBOARD_RECOMMENDATIONS: (fieldId: string) => `${API_BASE_URL}/api/v1/farmer-dashboard/field/${encodeURIComponent(fieldId)}/recommendations`,
  FARMER_DASHBOARD_COMPARE: `${API_BASE_URL}/api/v1/farmer-dashboard/compare`,
  FIELD_VALIDATION: `${API_BASE_URL}/api/v1/field-validation/observations`,
  CROP_METRICS: `${API_BASE_URL}/api/v1/ml/crop-metrics`,
  FOREST_METRICS: `${API_BASE_URL}/api/v1/forest/metrics`,
  FOREST_INFERENCE: `${API_BASE_URL}/api/v1/forest/inference`,
  FOREST_PRESETS: `${API_BASE_URL}/api/v1/forest/presets`,
  HEAT_METRICS: `${API_BASE_URL}/api/v1/heat/metrics`,
  HEAT_DOWNSCALE: `${API_BASE_URL}/api/v1/heat/downscale`,
  HEAT_TREE_PRIORITIES: `${API_BASE_URL}/api/v1/heat/tree-planting-priorities`,
};
