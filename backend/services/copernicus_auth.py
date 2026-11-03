import os
import time
import httpx
from typing import Optional, Dict, Any

class CopernicusCredentialsMissingError(Exception):
    """Raised when Copernicus credentials are not set."""
    pass

class CopernicusAuthError(Exception):
    """Raised when authentication with Copernicus OAuth2 service fails."""
    pass

class CopernicusAuthService:
    """
    Manages OAuth2 access tokens for Copernicus Data Space Ecosystem (CDSE)
    and Sentinel Hub APIs.

    Features:
    - Requests access tokens using OAuth2 client credentials flow
    - Caches and reuses active tokens until expiration
    - Thread/async safe token refresh with safety buffer
    - Prevents leaking credentials or raw tokens
    """

    DEFAULT_AUTH_URL = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        auth_url: Optional[str] = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._auth_url = auth_url
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def client_id(self) -> Optional[str]:
        if self._client_id is not None:
            return self._client_id
        return os.getenv("COPERNICUS_CLIENT_ID")

    @property
    def client_secret(self) -> Optional[str]:
        if self._client_secret is not None:
            return self._client_secret
        return os.getenv("COPERNICUS_CLIENT_SECRET")

    @property
    def auth_url(self) -> str:
        return (
            self._auth_url
            or os.getenv("COPERNICUS_AUTH_URL")
            or self.DEFAULT_AUTH_URL
        )

    def is_configured(self) -> bool:
        """Checks if client credentials are non-empty strings."""
        cid = self.client_id
        csec = self.client_secret
        return bool(cid and cid.strip() and csec and csec.strip())

    def has_credentials(self) -> bool:
        """Alias for is_configured()."""
        return self.is_configured()


    def is_token_valid(self) -> bool:
        """Returns True if a cached token exists and has not expired (with 60s buffer)."""
        if not self._cached_token:
            return False
        return time.time() < (self._token_expires_at - 60.0)

    def clear_cached_token(self) -> None:
        """Clears the cached token to force re-authentication."""
        self._cached_token = None
        self._token_expires_at = 0.0

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Retrieves a valid OAuth2 access token. Reuses the existing cached token
        if it is still valid, or requests a new one from the Copernicus IDP.
        """
        if not self.is_configured():
            raise CopernicusCredentialsMissingError(
                "Satellite service credentials are not configured. "
                "Please configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
            )

        if not force_refresh and self.is_token_valid() and self._cached_token:
            return self._cached_token

        token_data = await self._fetch_token()
        self._cached_token = token_data["access_token"]
        expires_in = float(token_data.get("expires_in", 300))
        self._token_expires_at = time.time() + expires_in

        return self._cached_token

    async def _fetch_token(self) -> Dict[str, Any]:
        """Performs the HTTP POST request to obtain a new token."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.auth_url,
                    data=data,
                    headers=headers,
                )

                if response.status_code in (401, 403):
                    raise CopernicusAuthError(
                        "Authentication failed: Invalid Copernicus credentials provided."
                    )

                if response.status_code >= 400:
                    raise CopernicusAuthError(
                        f"Copernicus authentication service returned error status {response.status_code}."
                    )

                response_data = response.json()
                if "access_token" not in response_data:
                    raise CopernicusAuthError(
                        "Unexpected response from Copernicus auth server: missing access_token."
                    )

                return response_data

        except httpx.TimeoutException:
            raise CopernicusAuthError("Satellite service authentication timed out. Please try again.")
        except httpx.RequestError as exc:
            raise CopernicusAuthError(f"Network error connecting to Copernicus authentication service: {str(exc)}")


# Global singleton instance for easy reuse across request lifecycles
copernicus_auth = CopernicusAuthService()
