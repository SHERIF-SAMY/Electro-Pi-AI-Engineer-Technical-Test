from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    ollama: str
    model: str
    version: str
