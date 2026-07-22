import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from app.schemas.generate import GenerateRequest, GenerateResponse, GenerationMetrics
from app.schemas.health import HealthResponse
from app.services.ollama_client import OllamaClient
from app.config import get_settings

router = APIRouter()
ollama_client = OllamaClient()

@router.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
            res.raise_for_status()
        return HealthResponse(
            status="ok",
            ollama="reachable",
            model=settings.model_name,
            version=settings.app_version
        )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Ollama is unreachable")

@router.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest):
    request_id = getattr(request.state, "request_id", "unknown")
    try:
        result = await ollama_client.generate(
            prompt=body.prompt,
            model=body.model,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            system=body.system
        )
        
        metrics = GenerationMetrics(
            prompt_tokens=result.get("prompt_eval_count"),
            completion_tokens=result.get("eval_count"),
            total_duration_ms=result.get("total_duration", 0) / 1e6, # convert ns to ms
            load_duration_ms=result.get("load_duration", 0) / 1e6,
            eval_rate_tps=None # We could calculate this if we had eval_duration
        )
        
        if result.get("eval_duration") and result.get("eval_count"):
            metrics.eval_rate_tps = (result["eval_count"] / (result["eval_duration"] / 1e9))
            
        return GenerateResponse(
            request_id=request_id,
            text=result.get("response", ""),
            model=result.get("model", ""),
            metrics=metrics
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="OLLAMA_UNAVAILABLE")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="OLLAMA_TIMEOUT")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v1/stream")
async def stream_generate(request: Request, body: GenerateRequest):
    async def event_generator():
        try:
            async for chunk in ollama_client.stream_generate(
                prompt=body.prompt,
                model=body.model,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                system=body.system
            ):
                yield chunk
        except asyncio.CancelledError:
            pass # Client disconnected
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
