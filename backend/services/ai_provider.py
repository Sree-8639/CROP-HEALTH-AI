import os
import json
import abc
import hashlib
from typing import Optional, Dict, Any
import httpx

class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    pass

class AIProvider(abc.ABC):
    """
    Abstract interface for AI model providers to ensure loose coupling
    and prevent vendor lock-in.
    """

    @abc.abstractmethod
    async def generate_analysis(self, prompt: str, system_prompt: str) -> str:
        """Generates structured analysis text from prompt."""
        pass

    @abc.abstractmethod
    async def generate_summary(self, prompt: str, system_prompt: str) -> str:
        """Generates concise text summary from prompt."""
        pass

class MockDeterministicProvider(AIProvider):
    """
    Deterministic mock provider for DEMO_MODE, automated unit testing,
    and offline operation without external API keys.
    """

    async def generate_analysis(self, prompt: str, system_prompt: str) -> str:
        # Generate a deterministic response based on hash of input prompt
        return self._build_deterministic_report(prompt)

    async def generate_summary(self, prompt: str, system_prompt: str) -> str:
        return (
            "Field displays moderate vegetative vigor across the analyzed canopy. "
            "Recent optical indices indicate stable chlorophyll activity and moisture presence."
        )

    def _build_deterministic_report(self, prompt: str) -> str:
        # Extract numerical metrics if present in prompt to ground the mock response
        ndvi_val = 0.58
        ndmi_val = 0.16
        cloud_val = 4.5
        valid_val = 98.2

        try:
            parsed_data = json.loads(prompt)
            if "indices" in parsed_data and parsed_data["indices"]:
                if "ndvi" in parsed_data["indices"] and parsed_data["indices"]["ndvi"]:
                    ndvi_val = parsed_data["indices"]["ndvi"].get("mean", ndvi_val)
                if "ndmi" in parsed_data["indices"] and parsed_data["indices"]["ndmi"]:
                    ndmi_val = parsed_data["indices"]["ndmi"].get("mean", ndmi_val)
            if "observation" in parsed_data and parsed_data["observation"]:
                cloud_val = parsed_data["observation"].get("cloud_coverage", cloud_val)
                valid_val = parsed_data["observation"].get("valid_pixel_percentage", valid_val)
            if "quality" in parsed_data and parsed_data["quality"]:
                valid_val = parsed_data["quality"].get("valid_percentage", valid_val)
        except Exception:
            pass

        # Calculate confidence based on quality criteria
        confidence = "high"
        if cloud_val > 20.0 or valid_val < 80.0:
            confidence = "low"
        elif cloud_val > 10.0 or valid_val < 90.0:
            confidence = "moderate"

        limitations = [
            "Optical sensors capture canopy surface reflectance and cannot directly measure soil subsurface conditions.",
            "Cloud shadows and atmospheric aerosols may induce localized spectral variance.",
        ]
        if cloud_val > 20.0:
            limitations.append(f"Elevated cloud coverage ({cloud_val:.1f}%) reduces observational confidence.")
        if valid_val < 80.0:
            limitations.append(f"Limited valid pixel coverage ({valid_val:.1f}%) warrants cautious field interpretation.")

        # Build scientifically grounded structured output
        report = {
            "headline": f"Canopy Vigor Analysis: Moderate-to-High Vegetative Activity (NDVI ~{ndvi_val:.2f})",
            "summary": (
                f"Sentinel-2 multispectral evidence demonstrates an active vegetative canopy with an average "
                f"NDVI of {ndvi_val:.2f} and an NDMI of {ndmi_val:.2f}. Spatial vigor distribution indicates "
                "consistent chlorophyll absorption across the central field zones, while peripheral margins "
                "show moderate variability. Optical data quality is robust with minimal cloud contamination."
            ),
            "observations": [
                {
                    "title": "Photosynthetic Activity & Canopy Density",
                    "description": "Optical reflectance reflects active chlorophyll absorption in the red band paired with strong near-infrared scatter.",
                    "evidence": [f"NDVI Mean: {ndvi_val:.3f}", f"NDRE Mean: {(ndvi_val * 0.55):.3f}"]
                },
                {
                    "title": "Canopy Moisture Status",
                    "description": "Normalized Difference Moisture Index indicates moderate water content within upper foliage.",
                    "evidence": [f"NDMI Mean: {ndmi_val:.3f}"]
                },
                {
                    "title": "Data Quality & Atmospheric Clearance",
                    "description": "Satellite acquisition exhibits clear surface visibility across the specified Area of Interest.",
                    "evidence": [f"Cloud Cover: {cloud_val:.1f}%", f"Valid Pixels: {valid_val:.1f}%"]
                }
            ],
            "possible_patterns": [
                {
                    "pattern": "Uniform Central Foliage with Marginal Variance",
                    "evidence": f"Zonal distribution indicates >65% of the AOI in moderate-to-good vigor ranges.",
                    "severity": "low",
                    "explanation": (
                        "Field boundary variations are common due to road proximity, border furrows, or differing soil moisture retention. "
                        "This spatial pattern is consistent with standard field management and micro-topography rather than acute distress."
                    )
                }
            ],
            "historical_interpretation": {
                "summary": (
                    "Available multi-temporal observations indicate that vegetation index signals have remained relatively consistent "
                    "across the observed acquisition window, with minor seasonal fluctuations consistent with phenological development."
                ),
                "trend": "stable" if ndvi_val >= 0.45 else "decreasing"
            },
            "areas_to_watch": [
                {
                    "description": "Low-vigor clusters along perimeter sections",
                    "reason": "Lower NDVI and NDMI signals observed in marginal pixel groupings."
                }
            ],
            "data_quality": {
                "assessment": "Observational assessment derived from cloud-screened optical Sentinel-2 Level-2A imagery.",
                "limitations": limitations
            },
            "recommended_next_steps": [
                "Conduct ground-truth spot inspections in perimeter areas identified with lower spectral response.",
                "Compare current observation with local planting dates and recent irrigation cycles.",
                "Cross-reference with soil moisture probe data or local agricultural logbooks."
            ],
            "confidence": confidence,
            "is_demo": True,
            "provider_used": "mock-deterministic"
        }
        return json.dumps(report)

class GenericHTTPLLMProvider(AIProvider):
    """
    Standard HTTP-based AI Provider supporting Gemini, OpenAI, or custom LLM endpoints.
    Uses strict environment variable isolation and never exposes credentials.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        base_url: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url or os.getenv("AI_BASE_URL")
        self.provider_name = (provider_name or os.getenv("AI_PROVIDER", "gemini")).lower()

    async def generate_analysis(self, prompt: str, system_prompt: str) -> str:
        return await self._call_model(prompt, system_prompt, response_format="json")

    async def generate_summary(self, prompt: str, system_prompt: str) -> str:
        return await self._call_model(prompt, system_prompt, response_format="text")

    async def _call_model(self, prompt: str, system_prompt: str, response_format: str = "json") -> str:
        if not self._api_key:
            raise AIProviderError("AI API key is missing. Set GROQ_API_KEY or AI_API_KEY in environment.")

        # Determine endpoint style
        provider_name = self.provider_name or os.getenv("AI_PROVIDER", "groq").lower()

        if "gemini" in provider_name and not self._api_key.startswith("gsk_"):
            # Google Gemini REST API format
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self._api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nEvidence for interpretation:\n{prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json" if response_format == "json" else "text/plain"
                }
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code != 200:
                        raise AIProviderError(f"AI API request failed with status {resp.status_code}.")
                    res_data = resp.json()
                    candidates = res_data.get("candidates", [])
                    if not candidates:
                        raise AIProviderError("No response candidate returned by AI model.")
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return text
            except httpx.TimeoutException:
                raise AIProviderError("AI API request timed out. Model service busy.")
            except Exception as e:
                # Never include self._api_key in exception string
                err_msg = self._redact_secrets(str(e))
                raise AIProviderError(f"AI generation error: {err_msg}")
        else:
            # Groq or OpenAI-compatible Chat Completions format
            is_groq = "groq" in provider_name or self._api_key.startswith("gsk_")
            default_endpoint = "https://api.groq.com/openai/v1/chat/completions" if is_groq else "https://api.openai.com/v1/chat/completions"
            endpoint = self.base_url or default_endpoint
            
            # Default model for Groq if set to gemini default
            active_model = self.model
            if is_groq and ("gemini" in active_model.lower() or not active_model):
                active_model = "llama-3.3-70b-versatile"

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": active_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            if response_format == "json":
                body["response_format"] = {"type": "json_object"}

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(endpoint, json=body, headers=headers)
                    if resp.status_code != 200:
                        err_text = self._redact_secrets(resp.text)
                        raise AIProviderError(f"AI API request failed ({resp.status_code}): {err_text}")
                    res_data = resp.json()
                    choices = res_data.get("choices", [])
                    if not choices:
                        raise AIProviderError("No response choice returned by AI model.")
                    text = choices[0].get("message", {}).get("content", "")
                    return text
            except Exception as e:
                err_msg = self._redact_secrets(str(e))
                raise AIProviderError(f"AI generation error: {err_msg}")

    def _redact_secrets(self, text: str) -> str:
        """Sanitizes sensitive keys and credentials from logs and error strings."""
        if not text:
            return ""
        if self._api_key:
            return text.replace(self._api_key, "[REDACTED]")
        return text

def get_ai_provider() -> AIProvider:
    """
    Factory function resolving configured AI provider.
    Supports Groq (https://console.groq.com/keys), Gemini, OpenAI, or MockDeterministicProvider.
    Falls back gracefully to MockDeterministicProvider when DEMO_MODE=true or API keys are absent.
    """
    demo_mode = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
    provider_name = os.getenv("AI_PROVIDER", "groq").lower()
    
    # Read either GROQ_API_KEY or AI_API_KEY
    api_key = (os.getenv("GROQ_API_KEY") or os.getenv("AI_API_KEY") or "").strip()

    # Determine default model according to provider
    if "groq" in provider_name or api_key.startswith("gsk_"):
        provider_name = "groq"
        default_model = "llama-3.3-70b-versatile"
    elif "gemini" in provider_name:
        default_model = "gemini-1.5-flash"
    else:
        default_model = "gpt-4o-mini"

    model = os.getenv("AI_MODEL") or default_model

    if demo_mode or not api_key:
        return MockDeterministicProvider()

    return GenericHTTPLLMProvider(api_key=api_key, model=model, provider_name=provider_name)
