from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    model: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    system: Optional[str] = None

class GenerationMetrics(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_duration_ms: Optional[float] = None
    load_duration_ms: Optional[float] = None
    eval_rate_tps: Optional[float] = None

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    model: str
    metrics: GenerationMetrics

class ErrorResponse(BaseModel):
    request_id: str
    error: str
    code: str
    detail: Optional[str] = None
