"""
Evidence-Linked Agricultural Recommendation Service (Phase 17).

Generates evidence-based observations and suggested investigative next steps
grounded directly in satellite reflectance, SAR backscatter, risk distributions,
and multi-temporal growth trends.

CRITICAL SCIENTIFIC & AGRONOMIC GUARDRAILS:
- Do NOT prescribe pesticides, chemical treatments, fertilizers, or specific irrigation quantities.
- Recommendations represent evidence-linked field observations and suggested investigative steps.
- Always identify data quality limitations and explicitly indicate whether physical ground verification is required.
- If observations are insufficient or cloud-degraded, explicitly declare insufficient evidence.
"""

import datetime
from typing import Dict, Any, List, Optional
from ml.farmer_dashboard.schema import AgriculturalRecommendation


class AgriculturalRecommendationService:
    """
    Agricultural Recommendation Service.
    
    Transforms multi-source satellite monitoring, risk analysis, stress inference,
    and temporal trend metrics into transparent, non-prescriptive, evidence-linked
    scouting and verification recommendations.
    """

    def generate_recommendations(
        self,
        field_id: str,
        observation_date: str,
        vegetation_metrics: Optional[Dict[str, Any]] = None,
        crop_condition: Optional[Dict[str, Any]] = None,
        stress_intelligence: Optional[Dict[str, Any]] = None,
        growth_forecast: Optional[Dict[str, Any]] = None,
        field_risk: Optional[Dict[str, Any]] = None,
        historical_monitoring: Optional[Dict[str, Any]] = None,
        data_reliability: Optional[Dict[str, Any]] = None,
    ) -> List[AgriculturalRecommendation]:
        """
        Generates structured, evidence-linked recommendations.
        """
        recommendations: List[AgriculturalRecommendation] = []
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        obs_date = observation_date or datetime.date.today().isoformat()

        # Extract context safely
        v_metrics = vegetation_metrics or {}
        c_condition = crop_condition or {}
        s_intel = stress_intelligence or {}
        g_forecast = growth_forecast or {}
        f_risk = field_risk or {}
        h_monitor = historical_monitoring or {}
        d_rel = data_reliability or {}

        # ---------------------------------------------------------------------
        # 0. Check for Insufficient Evidence Baseline
        # ---------------------------------------------------------------------
        valid_pixel_pct = d_rel.get("valid_pixel_percentage", 100.0)
        cloud_cov_pct = d_rel.get("cloud_coverage_percentage", 0.0)
        obs_count = h_monitor.get("observation_count", 0)

        if obs_count < 2 or valid_pixel_pct < 30.0:
            recommendations.append(
                AgriculturalRecommendation(
                    id=f"rec_insufficient_evidence_{field_id}",
                    category="Satellite Recheck",
                    title="Insufficient Observational Evidence for Definitive Assessment",
                    action=(
                        "Await subsequent cloud-free Sentinel-2 or Sentinel-1 acquisition passes before "
                        "undertaking significant field management alterations."
                    ),
                    reason=(
                        f"Current historical monitoring contains only {obs_count} observation(s) with "
                        f"{valid_pixel_pct:.1f}% valid canopy coverage. A minimum of two low-cloud "
                        "observations is required for establishing baseline trends and temporal change rates."
                    ),
                    evidence_source="Copernicus Data Space Sentinel-2 Quality Mask",
                    evidence_date=obs_date,
                    data_quality_limitations=[
                        f"Valid canopy pixel coverage is restricted to {valid_pixel_pct:.1f}%.",
                        f"Cloud and shadow obscuration is estimated at {cloud_cov_pct:.1f}%.",
                        "Time-series temporal baseline cannot be statistically confirmed.",
                    ],
                    field_verification_required=True,
                    priority="high",
                    created_at=now_iso,
                )
            )
            # If evidence is critically low, return early with caution
            if valid_pixel_pct < 20.0:
                return recommendations

        # ---------------------------------------------------------------------
        # 1. Field Scouting: Spatial Risk or Potential Stress Anomalies
        # ---------------------------------------------------------------------
        stress_coverage = f_risk.get("stress_signal_coverage_pct", 0.0)
        elevated_risk_pct = 0.0
        risk_dist = f_risk.get("risk_distribution", {})
        if "elevated_risk" in risk_dist:
            er_item = risk_dist["elevated_risk"]
            if hasattr(er_item, "percentage_of_field"):
                elevated_risk_pct = er_item.percentage_of_field
            elif isinstance(er_item, dict):
                elevated_risk_pct = er_item.get("percentage_of_field", 0.0)

        stress_prob = s_intel.get("stress_probability")
        stress_class = s_intel.get("stress_class") or ""

        if elevated_risk_pct >= 5.0 or stress_coverage >= 10.0 or stress_class in ("Elevated Stress", "Potential Stress"):
            severity = "high" if (elevated_risk_pct >= 15.0 or stress_coverage >= 20.0) else "medium"
            rec_id = f"rec_scouting_{field_id}"
            
            reasons = []
            if elevated_risk_pct >= 5.0:
                reasons.append(f"{elevated_risk_pct:.1f}% of the field area demonstrates multi-index elevated stress signals")
            if stress_coverage >= 10.0:
                reasons.append(f"{stress_coverage:.1f}% potential stress signal coverage across the canonical UTM grid")
            if stress_prob is not None and stress_prob >= 0.50:
                reasons.append(f"calibrated early stress probability is estimated at {stress_prob * 100:.1f}%")

            reason_str = "; ".join(reasons) if reasons else "Multi-sensor anomalies detected"

            recommendations.append(
                AgriculturalRecommendation(
                    id=rec_id,
                    category="Field Scouting",
                    title="Targeted Agronomic Scouting in Elevated Risk Sectors",
                    action=(
                        "Inspect zones flagged as 'Elevated Risk' in the spatial risk map. Perform visual "
                        "examination of canopy density, leaf discoloration, weed competition, and drainage patterns."
                    ),
                    reason=(
                        f"Field-level spatial risk analysis indicates {reason_str}. Physical scouting helps "
                        "differentiate biotic factors (pests, fungal infections) from abiotic stressors (soil compaction, waterlogging) "
                        "without making unverified remote diagnoses."
                    ),
                    evidence_source="Field-Level Spatial Risk Mapping & Multi-Index Overlay",
                    evidence_date=obs_date,
                    data_quality_limitations=[
                        "Remote sensing observes aggregate canopy reflectance, not microscopic pathogens or root pests.",
                        "Spatial resolution is 10m x 10m per pixel; sub-meter micro-variations require in-situ inspection.",
                    ],
                    field_verification_required=True,
                    priority=severity,
                    created_at=now_iso,
                )
            )

        # ---------------------------------------------------------------------
        # 2. Moisture Verification: NDMI Depletion or Canopy Water Loss
        # ---------------------------------------------------------------------
        ndmi_info = v_metrics.get("NDMI")
        if ndmi_info:
            latest_ndmi = getattr(ndmi_info, "latest_value", None) if not isinstance(ndmi_info, dict) else ndmi_info.get("latest_value")
            delta_ndmi = getattr(ndmi_info, "delta_absolute", None) if not isinstance(ndmi_info, dict) else ndmi_info.get("delta_absolute")
            
            if (latest_ndmi is not None and latest_ndmi < 0.05) or (delta_ndmi is not None and delta_ndmi < -0.06):
                priority = "high" if (latest_ndmi is not None and latest_ndmi < -0.05) or (delta_ndmi is not None and delta_ndmi < -0.12) else "medium"
                recommendations.append(
                    AgriculturalRecommendation(
                        id=f"rec_moisture_{field_id}",
                        category="Moisture Verification",
                        title="Canopy Water Content & Soil Moisture Verification",
                        action=(
                            "Assess soil moisture at root-zone depths using in-situ tensiometers, soil moisture "
                            "sensors, or manual core sampling in lagging sectors."
                        ),
                        reason=(
                            f"Normalized Difference Moisture Index (NDMI) is currently {latest_ndmi:.3f}"
                            f"{f' (shifted by {delta_ndmi:+.3f} from previous observation)' if delta_ndmi is not None else ''}. "
                            "A low or declining NDMI reflects diminished liquid water absorption in spongy mesophyll tissues, "
                            "suggesting potential moisture deficit or elevated crop evapotranspiration."
                        ),
                        evidence_source="Sentinel-2 MSI NIR (B8A) / SWIR (B11) Radiometry",
                        evidence_date=obs_date,
                        data_quality_limitations=[
                            "Optical NDMI reflects moisture within the upper canopy layer, not deep subsoil groundwater reservoirs.",
                            "Dew or recent surface precipitation immediately prior to acquisition may temporarily elevate values.",
                        ],
                        field_verification_required=True,
                        priority=priority,
                        created_at=now_iso,
                    )
                )

        # ---------------------------------------------------------------------
        # 3. Chlorophyll / Canopy Nitrogen Assessment: NDRE Signal
        # ---------------------------------------------------------------------
        ndre_info = v_metrics.get("NDRE")
        if ndre_info:
            latest_ndre = getattr(ndre_info, "latest_value", None) if not isinstance(ndre_info, dict) else ndre_info.get("latest_value")
            delta_ndre = getattr(ndre_info, "delta_absolute", None) if not isinstance(ndre_info, dict) else ndre_info.get("delta_absolute")

            if (latest_ndre is not None and latest_ndre < 0.20) or (delta_ndre is not None and delta_ndre < -0.05):
                recommendations.append(
                    AgriculturalRecommendation(
                        id=f"rec_chlorophyll_{field_id}",
                        category="Nutrient Assessment",
                        title="Canopy Chlorophyll & Foliar Nitrogen Evaluation",
                        action=(
                            "Perform chlorophyll meter (SPAD) measurements or collect representative leaf tissue "
                            "samples for standard laboratory nutrient profiling in lagging zones."
                        ),
                        reason=(
                            f"Normalized Difference Red Edge (NDRE) registered at {latest_ndre:.3f}"
                            f"{f' with a {delta_ndre:+.3f} delta' if delta_ndre is not None else ''}. "
                            "The red-edge spectral transition band (B5/B8) is highly sensitive to total leaf chlorophyll content. "
                            "Declines suggest potential vegetative chlorosis or nitrogen assimilation constraints."
                        ),
                        evidence_source="Sentinel-2 MSI Red Edge (B5) / NIR (B8) Spectral Indices",
                        evidence_date=obs_date,
                        data_quality_limitations=[
                            "NDRE variations can arise from chlorosis, salinity stress, root compaction, or natural crop ripening.",
                            "Do not apply fertilizers without laboratory confirmation of soil and leaf chemistry.",
                        ],
                        field_verification_required=True,
                        priority="medium",
                        created_at=now_iso,
                    )
                )

        # ---------------------------------------------------------------------
        # 4. Satellite Observation Recheck / Cloud Mitigation
        # ---------------------------------------------------------------------
        if cloud_cov_pct >= 20.0 or valid_pixel_pct < 80.0:
            recommendations.append(
                AgriculturalRecommendation(
                    id=f"rec_recheck_{field_id}",
                    category="Satellite Recheck",
                    title="Schedule Multi-Spectral Satellite Recheck Upon Next Orbit",
                    action=(
                        "Monitor upcoming Sentinel-2 Copernicus revisit windows (typically 3–5 day return cycle) "
                        "to obtain clear sky imagery for obscured field sectors."
                    ),
                    reason=(
                        f"Current scene exhibits {cloud_cov_pct:.1f}% atmospheric cloud/shadow cover with "
                        f"{valid_pixel_pct:.1f}% valid canopy pixels. Atmospheric filtering has masked portions of "
                        "the parcel, preventing continuous spatial field intelligence."
                    ),
                    evidence_source="Copernicus Sentinel-2 Level-2A Scene Classification Mask (SCL)",
                    evidence_date=obs_date,
                    data_quality_limitations=[
                        "Cirrus clouds and cloud shadow penumbra can subtly distort surface reflectance before full cloud flagging.",
                    ],
                    field_verification_required=False,
                    priority="medium" if cloud_cov_pct >= 35.0 else "low",
                    created_at=now_iso,
                )
            )

        # ---------------------------------------------------------------------
        # 5. Radar Co-Registration & Sensor Complementarity
        # ---------------------------------------------------------------------
        sar_status = d_rel.get("sar_sensor_status", "")
        missing_sensors = s_intel.get("missing_sensors", [])
        if "Sentinel-1 SAR" in missing_sensors or "Missing" in sar_status:
            recommendations.append(
                AgriculturalRecommendation(
                    id=f"rec_sar_sync_{field_id}",
                    category="Sensor Synchronization",
                    title="Review Sentinel-1 SAR Orbital Overpass Co-Registration",
                    action=(
                        "Review forthcoming Sentinel-1 C-band synthetic aperture radar acquisitions to supplement "
                        "optical multispectral indices with structural backscatter (VV and VH polarizations)."
                    ),
                    reason=(
                        "Current analysis relies exclusively on optical spectral bands. Synthetic Aperture Radar (SAR) "
                        "backscatter penetrates clouds and measures physical canopy geometry and soil surface roughness, "
                        "substantially reducing false alarms caused by surface optical discoloration."
                    ),
                    evidence_source="Multimodal Sensor Alignment & Temporal Pairing Ledger",
                    evidence_date=obs_date,
                    data_quality_limitations=[
                        "SAR data was not acquired within the configured ±3-day temporal fusion window.",
                    ],
                    field_verification_required=False,
                    priority="low",
                    created_at=now_iso,
                )
            )

        # ---------------------------------------------------------------------
        # 6. Temporal Growth Trajectory & Phenology Review
        # ---------------------------------------------------------------------
        slope = g_forecast.get("historical_slope_per_day")
        forecast_values = g_forecast.get("projected_values", [])
        if slope is not None and slope < -0.0015:
            recommendations.append(
                AgriculturalRecommendation(
                    id=f"rec_trend_review_{field_id}",
                    category="Temporal Review",
                    title="Review Crop Phenology Calendar & Historical Growth Trajectory",
                    action=(
                        "Cross-reference the observed negative growth slope with the sowing date and anticipated "
                        "maturity schedule. Verify whether the vegetative decline represents expected senescence or "
                        "premature crop stress."
                    ),
                    reason=(
                        f"Time-series trend analysis indicates a downward NDVI trajectory with an estimated slope "
                        f"of {slope * 1000:.2f} mNDVI/day. If the crop is mid-vegetative, this rate of decline warrants "
                        "investigation into root health, pest infestations, or unseasonal water stress."
                    ),
                    evidence_source="Time-Series Trend & Walk-Forward Forecaster",
                    evidence_date=obs_date,
                    data_quality_limitations=[
                        "Trend projection assumes stationary environmental dynamics without sudden meteorological disruptions.",
                        "Phenological phase calendars must be calibrated to the specific cultivar planted.",
                    ],
                    field_verification_required=True,
                    priority="medium",
                    created_at=now_iso,
                )
            )

        # Default fallback if field is healthy and stable
        if len(recommendations) == 0:
            recommendations.append(
                AgriculturalRecommendation(
                    id=f"rec_routine_monitoring_{field_id}",
                    category="Routine Monitoring",
                    title="Maintain Standard Crop Monitoring Schedule",
                    action=(
                        "Continue scheduled visual inspections and routine agronomic practices. Review the next "
                        "scheduled satellite observation."
                    ),
                    reason=(
                        "Vegetation vigor, moisture indices, and radar signals across the parcel remain within "
                        "expected normative ranges with no significant anomalous stress signals identified."
                    ),
                    evidence_source="Multispectral & Multimodal Feature Fusion Engine",
                    evidence_date=obs_date,
                    data_quality_limitations=[],
                    field_verification_required=False,
                    priority="info",
                    created_at=now_iso,
                )
            )

        return recommendations


# Global singleton instance
agricultural_recommendation_service = AgriculturalRecommendationService()
