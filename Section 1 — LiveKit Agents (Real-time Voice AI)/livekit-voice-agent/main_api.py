import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from livekit.api import AccessToken, VideoGrants
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

app = FastAPI()

# Mount static files (CSS, JS) from frontend dir
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class TokenRequest(BaseModel):
    participant_name: str
    room_name: str

@app.post("/api/token")
async def get_token(request: TokenRequest):
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    grant = VideoGrants(
        room_join=True,
        room=request.room_name,
    )
    
    access_token = AccessToken(
        api_key, 
        api_secret
    )
    
    access_token.with_identity(request.participant_name)
    access_token.with_name(request.participant_name)
    access_token.with_grants(grant)

    return {"token": access_token.to_jwt()}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
