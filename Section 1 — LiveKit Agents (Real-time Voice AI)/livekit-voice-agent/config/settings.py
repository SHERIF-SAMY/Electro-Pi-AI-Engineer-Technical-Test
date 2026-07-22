from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    groq_api_key: str = ""

    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    use_mock_stt: bool = True
    use_mock_tts: bool = True

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
