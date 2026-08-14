import time
from loguru import logger
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        logger.info(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 2)
        logger.info(f"← {request.method} {request.url.path} | {response.status_code} | {duration}ms")
        return response
