import os
import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from models.ai_models import (
    FieldEvidenceInput,
    LocationEvidence,
    ObservationEvidence,
    IndicesEvidence,
    IndexStatsEvidence,
    ConditionEvidence,
    HistoricalEvidence,
    ComparisonEvidence,
    QualityEvidence,
    AIFieldReport,
)
from services.ai_provider import (
    AIProvider,
    MockDeterministicProvider,
    GenericHTTPLLMProvider,
    get_ai_provider,
    AIProviderError,
)
from services.ai_analysis_service import AIAnalysisService, ai_analysis_service
from main import app

client = TestClient(app)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_complete_evidence():
    return FieldEvidenceInput(
        location=LocationEvidence(
            latitude=16.81,
            longitude=81.52,
            name="Tadepalligudem Farmland",
            radius_km=2.0,
            area_sq_km=12.56,
        ),
        observation=ObservationEvidence(
            date="2026-10-25",
            cloud_coverage=4.2,
            valid_pixel_percentage=98.5,
            scene_id="S2A_MSIL2A_20261025T050721_N0511_R019_T44QPF_20261025T084512",
            data_source="Copernicus Sentinel-2",
        ),
        indices=IndicesEvidence(
            ndvi=IndexStatsEvidence(mean=0.62, median=0.64, min=0.15, max=0.88),
            ndmi=IndexStatsEvidence(mean=0.18, median=0.19, min=-0.12, max=0.45),
            ndre=IndexStatsEvidence(mean=0.41, median=0.42, min=0.08, max=0.67),
            evi=IndexStatsEvidence(mean=0.55, median=0.57, min=0.12, max=0.82),
        ),
        condition=ConditionEvidence(
            score=74.5,
            category="Good",
            dominant_zone="Good Vigor",
            zones={"very_good": 35.0, "good": 45.0, "moderate": 15.0, "low": 5.0, "very_low": 0.0},
        ),
        historical=HistoricalEvidence(
            observation_count=5,
            start_date="2026-08-01",
            latest_date="2026-10-25",
            trends={
                "ndvi": {"trend": "Increasing", "delta": 0.08},
                "ndmi": {"trend": "Stable", "delta": 0.01},
            },
            summary="Multi-temporal trajectory shows steady vegetation increase over the 60-day window.",
        ),
        comparison=ComparisonEvidence(
            delta_ndvi=0.07,
            delta_ndmi=0.02,
            delta_ndre=0.05,
            delta_evi=0.06,
        ),
        quality=QualityEvidence(
            valid_percentage=98.5,
            quality_level="Excellent",
            limitations=[],
        ),
        is_demo=False,
    )

# -------------------------------------------------------------------------
# Unit Tests: Input Construction & Missing Fields
# -------------------------------------------------------------------------

def test_evidence_input_construction(sample_complete_evidence):
    prompt_str = ai_analysis_service.build_evidence_prompt(sample_complete_evidence)
    parsed = json.loads(prompt_str)

    assert parsed["location"]["latitude"] == 16.81
    assert parsed["indices"]["ndvi"]["mean"] == 0.62
    assert parsed["condition"]["score"] == 74.5
    assert parsed["quality"]["valid_percentage"] == 98.5

def test_missing_index_handling():
    # Only NDVI provided, NDMI/NDRE/EVI omitted
    partial_evidence = FieldEvidenceInput(
        location=LocationEvidence(latitude=16.81, longitude=81.52),
        observation=ObservationEvidence(date="2026-10-25", cloud_coverage=2.0, valid_pixel_percentage=100.0),
        indices=IndicesEvidence(ndvi=IndexStatsEvidence(mean=0.45)),
        quality=QualityEvidence(valid_percentage=100.0, quality_level="Excellent"),
    )
    prompt_str = ai_analysis_service.build_evidence_prompt(partial_evidence)
    parsed = json.loads(prompt_str)

    assert "ndvi" in parsed["indices"]
    # Unprovided indices are not fabricated
    assert "ndmi" not in parsed["indices"] or parsed["indices"]["ndmi"] is None
    assert "condition" not in parsed

# -------------------------------------------------------------------------
# Unit Tests: Quality-Aware Reasoning & Degradation
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quality_aware_confidence_downgrade():
    # High cloud coverage (35%) and low valid pixel percentage (60%)
    low_quality_evidence = FieldEvidenceInput(
        location=LocationEvidence(latitude=16.81, longitude=81.52),
        observation=ObservationEvidence(date="2026-10-25", cloud_coverage=35.0, valid_pixel_percentage=60.0),
        indices=IndicesEvidence(ndvi=IndexStatsEvidence(mean=0.55)),
        quality=QualityEvidence(valid_percentage=60.0, quality_level="Poor", limitations=["Severe cloud obstruction"]),
    )

    mock_provider = MockDeterministicProvider()
    service = AIAnalysisService(provider=mock_provider)
    report = await service.generate_field_intelligence(low_quality_evidence)

    # In low quality conditions, confidence must be downgraded (not high)
    assert report.confidence in ["low", "moderate"]
    assert any("cloud" in lim.lower() or "coverage" in lim.lower() for lim in report.data_quality.limitations)

# -------------------------------------------------------------------------
# Unit Tests: Deterministic Demo AI Report
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deterministic_demo_ai_consistency(sample_complete_evidence):
    sample_complete_evidence.is_demo = True
    provider = MockDeterministicProvider()
    service = AIAnalysisService(provider=provider)

    report_1 = await service.generate_field_intelligence(sample_complete_evidence)
    report_2 = await service.generate_field_intelligence(sample_complete_evidence)

    assert report_1.headline == report_2.headline
    assert report_1.summary == report_2.summary
    assert report_1.is_demo is True
    assert len(report_1.observations) > 0
    assert len(report_1.possible_patterns) > 0

# -------------------------------------------------------------------------
# Unit Tests: Pydantic Validation & Single Retry on Invalid JSON
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_invalid_json_recovers():
    class FlakyProvider(AIProvider):
        def __init__(self):
            self.call_count = 0

        async def generate_analysis(self, prompt: str, system_prompt: str = "") -> str:
            self.call_count += 1
            if self.call_count == 1:
                return "NOT VALID JSON at all {broken"
            # Valid on retry
            return json.dumps({
                "headline": "Recovered Field Intelligence Analysis",
                "summary": "Recovered valid response after schema retry.",
                "observations": [
                    {"title": "Valid Signal", "description": "NDVI indicates healthy canopy absorption.", "evidence": ["NDVI: 0.62"]}
                ],
                "possible_patterns": [
                    {"pattern": "High Greenness", "evidence": "NDVI 0.62", "severity": "low", "explanation": "Adequate canopy development."}
                ],
                "areas_to_watch": [
                    {"description": "Perimeter boundaries", "reason": "Edge effects."}
                ],
                "data_quality": {
                    "assessment": "Valid observational quality.",
                    "limitations": []
                },
                "recommended_next_steps": ["Inspect border rows."],
                "confidence": "high"
            })

        async def generate_summary(self, text: str) -> str:
            return "Summary"

    flaky_prov = FlakyProvider()
    service = AIAnalysisService(provider=flaky_prov)
    evidence = FieldEvidenceInput(
        indices=IndicesEvidence(ndvi=IndexStatsEvidence(mean=0.62)),
        quality=QualityEvidence(valid_percentage=98.0, quality_level="Excellent"),
    )

    report = await service.generate_field_intelligence(evidence, max_retries=1)
    assert flaky_prov.call_count == 2
    assert report.headline == "Recovered Field Intelligence Analysis"
    assert report.confidence == "high"

@pytest.mark.asyncio
async def test_fallback_on_unrecoverable_failure():
    class AlwaysFailsProvider(AIProvider):
        async def generate_analysis(self, prompt: str, system_prompt: str = "") -> str:
            raise AIProviderError("Connection refused by remote host")

        async def generate_summary(self, text: str) -> str:
            return "Summary"

    service = AIAnalysisService(provider=AlwaysFailsProvider())
    evidence = FieldEvidenceInput(
        indices=IndicesEvidence(ndvi=IndexStatsEvidence(mean=0.58)),
        quality=QualityEvidence(valid_percentage=95.0, quality_level="Good"),
    )

    report = await service.generate_field_intelligence(evidence)
    assert isinstance(report, AIFieldReport)
    assert "fallback" in (report.provider_used or "").lower() or "synthesis" in report.headline.lower()
    assert report.confidence in ["moderate", "high"]

# -------------------------------------------------------------------------
# Scientific Guardrails Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scientific_guardrails_compliance(sample_complete_evidence):
    provider = MockDeterministicProvider()
    service = AIAnalysisService(provider=provider)
    report = await service.generate_field_intelligence(sample_complete_evidence)

    combined_text = f"{report.headline} {report.summary} {' '.join(o.description for o in report.observations)}"

    # Guardrails: Must NOT diagnose disease or prescribe chemicals
    prohibited_phrases = [
        "your crop has disease",
        "crop definitely has",
        "nitrogen deficiency confirmed",
        "spray pesticide",
        "apply fertilizer immediately",
        "yield will be",
        "tonnes per hectare",
    ]

    for phrase in prohibited_phrases:
        assert phrase not in combined_text.lower(), f"Guardrail violation: found '{phrase}' in AI response."

# -------------------------------------------------------------------------
# Security & Privacy Tests
# -------------------------------------------------------------------------

def test_api_key_redacted_in_exceptions():
    provider = GenericHTTPLLMProvider(
        provider_name="openai",
        api_key="sk-SECRET_API_KEY_12345",
        model="gpt-4o",
    )

    assert provider._api_key == "sk-SECRET_API_KEY_12345"

    # Redaction utility test
    err_msg = f"Failed to call OpenAI with key {provider._api_key} because of 401"
    redacted = provider._redact_secrets(err_msg)
    assert "sk-SECRET_API_KEY_12345" not in redacted
    assert "[REDACTED]" in redacted

def test_endpoint_no_credentials_leaked(sample_complete_evidence):
    response = client.post(
        "/api/v1/analysis/ai-report",
        json=sample_complete_evidence.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    res_text = response.text

    # Verify no secret patterns exist in response
    assert "sk-" not in res_text
    assert "AI_API_KEY" not in res_text
    assert "COPERNICUS_CLIENT_SECRET" not in res_text

# -------------------------------------------------------------------------
# API Endpoint Integration Test
# -------------------------------------------------------------------------

def test_api_ai_report_endpoint(sample_complete_evidence):
    response = client.post(
        "/api/v1/analysis/ai-report",
        json=sample_complete_evidence.model_dump(exclude_none=True),
    )
    assert response.status_code == 200
    data = response.json()

    assert "headline" in data
    assert "summary" in data
    assert "observations" in data
    assert "possible_patterns" in data
    assert "data_quality" in data
    assert "recommended_next_steps" in data
    assert data["confidence"] in ["low", "moderate", "high"]

def test_groq_provider_resolution(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test123456789")
    monkeypatch.delenv("AI_MODEL", raising=False)

    provider = get_ai_provider()
    assert isinstance(provider, GenericHTTPLLMProvider)
    assert provider.provider_name == "groq"
    assert provider.model == "llama-3.3-70b-versatile"
    assert provider._api_key == "gsk_test123456789"
