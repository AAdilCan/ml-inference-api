"""API-key authentication middleware.

When api_keys is set in settings, every request to non-exempt paths must
include the key either in the X-API-Key header or ?api_key query param.
"""
from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.core.config import settings

log = logging.getLogger(__name__)

_EXEMPT_PATHS = frozenset({"/health", "/metrics", "/docs", "/redoc", "/openapi.json"})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        valid_keys = settings.valid_api_keys
        if not valid_keys or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not key or key not in valid_keys:
            log.warning("unauthorized request from %s", request.client)
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)

        return await call_next(request)
