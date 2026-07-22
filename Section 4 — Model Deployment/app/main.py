from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api.v1.router import router as v1_router
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.utils.logger import setup_logger

setup_logger()
logger = structlog.get_logger("app.main")

app = FastAPI(
    title="Section 4 - Model Deployment",
    version="1.0.0"
)

# Middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(v1_router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("unhandled_exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal Server Error",
            "code": "INTERNAL_ERROR"
        }
    )
