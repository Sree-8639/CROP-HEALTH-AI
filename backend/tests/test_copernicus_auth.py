import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from services.copernicus_auth import (
    CopernicusAuthService,
    CopernicusAuthError,
    CopernicusCredentialsMissingError,
)

@pytest.mark.asyncio
async def test_auth_missing_credentials():
    service = CopernicusAuthService(client_id="", client_secret="")
    assert not service.is_configured()
    with pytest.raises(CopernicusCredentialsMissingError) as exc_info:
        await service.get_access_token()
    assert "credentials are not configured" in str(exc_info.value)

@pytest.mark.asyncio
async def test_auth_token_fetch_and_caching():
    service = CopernicusAuthService(
        client_id="test-client-id",
        client_secret="test-client-secret",
        auth_url="https://mock-auth.dataspace.copernicus.eu/token",
    )
    assert service.is_configured()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "mock-access-token-12345",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        # First call fetches from IDP
        token1 = await service.get_access_token()
        assert token1 == "mock-access-token-12345"
        assert service.is_token_valid()
        assert mock_post.call_count == 1

        # Second call reuses cached token without calling network
        token2 = await service.get_access_token()
        assert token2 == "mock-access-token-12345"
        assert mock_post.call_count == 1

@pytest.mark.asyncio
async def test_auth_token_renewal_when_expired():
    service = CopernicusAuthService(
        client_id="test-client-id",
        client_secret="test-client-secret",
    )

    # Set an expired token
    service._cached_token = "old-token"
    service._token_expires_at = time.time() - 100.0  # Already expired
    assert not service.is_token_valid()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "fresh-token-67890",
        "expires_in": 1800,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        token = await service.get_access_token()
        assert token == "fresh-token-67890"
        assert mock_post.call_count == 1

@pytest.mark.asyncio
async def test_auth_invalid_credentials_error():
    service = CopernicusAuthService(
        client_id="wrong-id",
        client_secret="wrong-secret",
    )

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(CopernicusAuthError) as exc_info:
            await service.get_access_token()
        assert "Invalid Copernicus credentials" in str(exc_info.value)
