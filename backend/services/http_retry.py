"""
backend/services/http_retry.py:
Robust asynchronous HTTP retry & timeout handler for Copernicus CDSE and Sentinel Hub APIs.
Features:
- Exponential backoff with random jitter
- Automated 401 token refresh and header injection
- Transparent handling of rate limits (HTTP 429) and gateway timeouts (HTTP 504/502/503/500)
- Configurable timeout and retry bounds
"""

import asyncio
import random
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
import httpx

logger = logging.getLogger(__name__)


async def request_with_retry(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    data_payload: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    timeout_seconds: float = 50.0,
    token_refresh_callback: Optional[Callable[[], Awaitable[str]]] = None,
) -> httpx.Response:
    """
    Executes an HTTP request with automated retries, token refresh on 401,
    and exponential backoff on transient errors.
    """
    current_headers = dict(headers or {})
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                if method.upper() == "POST":
                    response = await client.post(
                        url,
                        headers=current_headers,
                        json=json_payload,
                        data=data_payload,
                    )
                elif method.upper() == "GET":
                    response = await client.get(
                        url,
                        headers=current_headers,
                        params=data_payload,
                    )
                else:
                    response = await client.request(
                        method,
                        url,
                        headers=current_headers,
                        json=json_payload,
                        data=data_payload,
                    )

                # 1. Handle token expiry (401 Unauthorized)
                if response.status_code == 401 and token_refresh_callback is not None and attempt < max_retries:
                    logger.warning("CDSE returned 401 Unauthorized. Refreshing OAuth2 token (attempt %d/%d)...", attempt, max_retries)
                    new_token = await token_refresh_callback()
                    current_headers["Authorization"] = f"Bearer {new_token}"
                    await asyncio.sleep(0.5)
                    continue

                # 2. Handle rate limits (429) and transient server errors (500, 502, 503, 504)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)

                    logger.warning(
                        "CDSE request to %s returned status %d. Backing off for %.2fs (attempt %d/%d)...",
                        url, response.status_code, delay, attempt, max_retries
                    )
                    await asyncio.sleep(delay)
                    continue

                return response

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                logger.warning(
                    "Network error %s requesting %s. Retrying in %.2fs (attempt %d/%d)...",
                    type(exc).__name__, url, delay, attempt, max_retries
                )
                await asyncio.sleep(delay)
                continue
            break

    if last_exception:
        raise last_exception
    raise httpx.RequestError(f"HTTP request to {url} failed after {max_retries} attempts.")

