import httpx
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional

from app.config import get_settings

logger = logging.getLogger("app.services.ollama_client")

class OllamaClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self.timeout = self.settings.request_timeout

    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Blocking generation."""
        target_model = model or self.settings.model_name
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError as e:
                logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama returned HTTP error: {e.response.status_code} - {e.response.text}")
                raise

    async def stream_generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Streaming generation."""
        target_model = model or self.settings.model_name
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=self.settings.stream_timeout) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            yield f"data: {json.dumps(data)}\n\n"
                            
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed NDJSON from Ollama: {line}")
                            continue
                            
            except httpx.ConnectError as e:
                logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
                yield f"data: {{\"error\": \"connection failed\"}}\n\n"
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama returned HTTP error: {e.response.status_code}")
                yield f"data: {{\"error\": \"HTTP {e.response.status_code}\"}}\n\n"
