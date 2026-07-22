import pytest
import respx
import httpx
from app.services.ollama_client import OllamaClient
from app.config import get_settings

@pytest.fixture
def ollama_client():
    return OllamaClient()

@pytest.mark.asyncio
@respx.mock
async def test_generate_success(ollama_client):
    settings = get_settings()
    url = f"{settings.ollama_host}/api/generate"
    
    mock_response = {
        "model": "qwen2.5:0.5b",
        "created_at": "2026-07-22T20:00:00.000Z",
        "response": "Hello world",
        "done": True,
    }
    
    respx.post(url).mock(return_value=httpx.Response(200, json=mock_response))
    
    result = await ollama_client.generate("Say hello")
    assert result["response"] == "Hello world"
    assert result["done"] is True

@pytest.mark.asyncio
@respx.mock
async def test_generate_connection_error(ollama_client):
    settings = get_settings()
    url = f"{settings.ollama_host}/api/generate"
    
    respx.post(url).mock(side_effect=httpx.ConnectError("Connection failed"))
    
    with pytest.raises(httpx.ConnectError):
        await ollama_client.generate("Say hello")
