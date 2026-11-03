import os
import json
import datetime
from typing import Dict, Any, Optional
from pydantic import ValidationError

from models.ai_models import (
    FieldEvidenceInput,
    AIFieldReport,
    ObservationItem,
    PatternItem,
    HistoricalInterpretation,
    AreaToWatch,
    DataQualityAssessment,
)
from services.ai_provider import AIProvider, get_ai_provider, AIProviderError, MockDeterministicProvider

SYSTEM_PROMPT = """You are an expert Remote Sensing and Agronomic Intelligence Interpretation Assistant.
Your task is to analyze pre-calculated optical satellite evidence (Sentinel-2 L2A) and produce an explainable, evidence-backed Field Intelligence Report.

CRITICAL SCIENTIFIC GUARDRAILS AND CONSTRAINTS:
1. You are an interpretation assistant, NOT a diagnostic medical lab or clinical agronomist.
2. DO NOT make definitive disease diagnoses (e.g., DO NOT say "Your crop has fungal blight" or "Crop has mosaic virus").
3. DO NOT assert definitive causes (e.g., DO NOT say "The crop definitely has nitrogen deficiency" or "Drought has ruined the crop").
4. DO NOT provide prescriptive chemical, fertilizer, or pesticide spraying advice (e.g., DO NOT say "Spray chlorpyrifos" or "Apply 50kg urea").
5. DO NOT predict crop yields in tonnes or monetary figures.
6. DO NOT invent or fabricate measurements. Only reference metrics explicitly present in the provided evidence.
7. Use cautious, evidence-grounded scientific language:
   - "The observed spectral pattern may be consistent with..."
   - "This area exhibits a lower near-infrared vegetation index signal."
   - "This variation can arise from multiple factors including phenology, moisture availability, soil background, or canopy density."
   - "Ground-truth field verification is recommended."
8. QUALITY AWARENESS:
   - If cloud coverage is high (>20%) or valid pixel percentage is low (<80%), you MUST downgrade confidence and explicitly cite atmospheric/pixel limitations in the report.

OUTPUT FORMAT:
You MUST respond with a valid JSON object matching the requested schema strictly. Do not include markdown codeblocks or commentary outside the JSON."""

class AIAnalysisService:
    """
    Service responsible for constructing structured field evidence prompts,
    enforcing agronomic scientific guardrails, invoking the AI provider,
    and validating structured field reports.
    """

    def __init__(self, provider: Optional[AIProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        if self._provider is not None:
            return self._provider
        return get_ai_provider()

    def build_evidence_prompt(self, evidence: FieldEvidenceInput) -> str:
        """
        Transforms verified field evidence into a clean, structured JSON context prompt.
        Omits unobserved or null attributes to avoid hallucination.
        """
        raw_dict = evidence.model_dump(exclude_none=True)
        return json.dumps(raw_dict, indent=2)

    async def generate_field_intelligence(
        self,
        evidence: FieldEvidenceInput,
        max_retries: int = 1,
    ) -> AIFieldReport:
        """
        Generates an explainable field intelligence report from structured evidence.
        Validates against AIFieldReport schema with automated single retry on schema failure.
        """
        prompt = self.build_evidence_prompt(evidence)
        provider = self.provider

        raw_response = ""
        try:
            raw_response = await provider.generate_analysis(prompt, SYSTEM_PROMPT)
            cleaned_json = self._clean_json_string(raw_response)
            parsed = json.loads(cleaned_json)
            # Stamp metadata
            parsed["is_demo"] = evidence.is_demo or isinstance(provider, MockDeterministicProvider)
            parsed["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return AIFieldReport.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as parse_err:
            if max_retries > 0:
                # Retry once with explicit schema correction
                correction_prompt = (
                    f"Previous response was not valid JSON or failed schema validation: {str(parse_err)}.\n"
                    f"Evidence:\n{prompt}\n\nPlease regenerate ONLY valid JSON strictly following the required fields."
                )
                try:
                    retry_raw = await provider.generate_analysis(correction_prompt, SYSTEM_PROMPT)
                    parsed_retry = json.loads(self._clean_json_string(retry_raw))
                    parsed_retry["is_demo"] = evidence.is_demo or isinstance(provider, MockDeterministicProvider)
                    parsed_retry["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    return AIFieldReport.model_validate(parsed_retry)
                except Exception:
                    pass

            # Safe fallback if parsing or validation fails
            return self._generate_safe_fallback_report(evidence)
        except AIProviderError as prov_err:
            # Safe fallback on provider timeout or API connection failure
            return self._generate_safe_fallback_report(evidence, error_note=str(prov_err))
        except Exception as general_err:
            return self._generate_safe_fallback_report(evidence, error_note=str(general_err))

    def _clean_json_string(self, raw_str: str) -> str:
        """Strips markdown code fences and extraneous leading/trailing whitespace."""
        cleaned = raw_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _generate_safe_fallback_report(
        self,
        evidence: FieldEvidenceInput,
        error_note: Optional[str] = None,
    ) -> AIFieldReport:
        """
        Constructs a deterministic, scientifically safe fallback report directly
        from the mathematical indices when external AI generation is unavailable.
        """
        ndvi_mean = evidence.indices.ndvi.mean if evidence.indices and evidence.indices.ndvi else None
        ndmi_mean = evidence.indices.ndmi.mean if evidence.indices and evidence.indices.ndmi else None
        valid_pct = evidence.quality.valid_percentage if evidence.quality else 95.0
        cloud_pct = evidence.observation.cloud_coverage if evidence.observation else 5.0

        headline = "Automated Field Evidence Synthesis (Rule-Based Engine)"
        if ndvi_mean is not None:
            if ndvi_mean > 0.6:
                headline = f"Canopy Vigor Summary: High Optical Greenness (NDVI ~{ndvi_mean:.2f})"
            elif ndvi_mean > 0.35:
                headline = f"Canopy Vigor Summary: Moderate Vegetative Activity (NDVI ~{ndvi_mean:.2f})"
            else:
                headline = f"Canopy Vigor Summary: Low-to-Sparse Foliage Signal (NDVI ~{ndvi_mean:.2f})"

        obs_items = []
        if ndvi_mean is not None:
            obs_items.append(
                ObservationItem(
                    title="Optical Vegetation Index Signal",
                    description="Red and Near-Infrared surface reflectance measurements indicate current chlorophyll absorption.",
                    evidence=[f"NDVI Mean: {ndvi_mean:.3f}"]
                )
            )
        if ndmi_mean is not None:
            obs_items.append(
                ObservationItem(
                    title="Canopy Moisture Reflectance Indicator",
                    description="Normalized Difference Moisture Index reflects relative water absorption bands.",
                    evidence=[f"NDMI Mean: {ndmi_mean:.3f}"]
                )
            )

        patterns = [
            PatternItem(
                pattern="Baseline Optical Reflectance Distribution",
                evidence=f"Valid pixel coverage: {valid_pct:.1f}%, Cloud coverage: {cloud_pct:.1f}%",
                severity="low",
                explanation=(
                    "Reflectance values represent surface optical canopy signals. Variations may reflect planting density, "
                    "phenological stage, or soil moisture differences. Cause cannot be determined from optical indices alone."
                )
            )
        ]

        hist_interp = None
        if evidence.historical and evidence.historical.trends:
            ndvi_tr = evidence.historical.trends.get("ndvi", {})
            hist_interp = HistoricalInterpretation(
                summary=evidence.historical.summary or "Multi-temporal observations show stable optical trajectory.",
                trend=ndvi_tr.get("trend", "stable").lower().replace(" ", "_")
                if ndvi_tr.get("trend", "stable").lower() in ["increasing", "stable", "decreasing", "insufficient_data"]
                else "stable"
            )

        limitations = [
            "Analytical interpretation generated from rule-based fallback engine.",
            "Optical satellite data captures top-of-canopy reflectance and does not directly inspect soil pathogens or root health."
        ]
        if cloud_pct > 20.0:
            limitations.append(f"Elevated cloud coverage ({cloud_pct:.1f}%) may attenuate surface spectral accuracy.")

        return AIFieldReport(
            headline=headline,
            summary=(
                "Field analysis compiled from Sentinel-2 surface reflectance observations. "
                "The canopy presents measurable spectral indicators across the visible, red-edge, and near-infrared bands. "
                "Physical inspection and ground verification are recommended to validate local agricultural conditions."
            ),
            observations=obs_items,
            possible_patterns=patterns,
            historical_interpretation=hist_interp,
            areas_to_watch=[
                AreaToWatch(
                    description="Peripheral and low-reflectance field zones",
                    reason="Variability in spatial vegetation index response."
                )
            ],
            data_quality=DataQualityAssessment(
                assessment="Quality metrics evaluated from Sentinel-2 scene metadata and valid pixel masks.",
                limitations=limitations
            ),
            recommended_next_steps=[
                "Confirm observed patterns against in-situ field scouting logbooks.",
                "Review irrigation timing, fertilization dates, and crop growth stage.",
                "Inspect areas with persistent low vegetation signals for physical soil compaction or drainage variations."
            ],
            confidence="high" if valid_pct >= 90.0 and cloud_pct <= 10.0 else "moderate",
            is_demo=evidence.is_demo,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            provider_used="rule-based-fallback" if not error_note else f"fallback ({error_note[:40]})"
        )

# Global singleton
ai_analysis_service = AIAnalysisService()
