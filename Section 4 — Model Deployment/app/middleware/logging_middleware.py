import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger("app.middleware.logging")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = getattr(request.state, "request_id", "unknown")
        start_time = time.perf_counter()
        
        # We can't log the body here easily without consuming it, so we log basic info
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path
        )
        
        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(process_time, 2)
            )
            return response
        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(process_time, 2),
                error=str(e)
            )
            raise
