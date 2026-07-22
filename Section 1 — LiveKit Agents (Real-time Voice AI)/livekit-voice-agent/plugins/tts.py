import logging

from livekit.agents.tts import TTS

from config.settings import Settings
from mocks.mock_tts import MockTTS

logger = logging.getLogger("plugins.tts")


def create_tts(settings: Settings) -> TTS:
    if settings.use_mock_tts:
        logger.info("Using MockTTS")
        return MockTTS()

    try:
        from livekit.plugins.cartesia import TTS as CartesiaTTS
        logger.info("Using Cartesia TTS")
        return CartesiaTTS()
    except ImportError:
        logger.warning("Cartesia plugin not available, falling back to MockTTS")
        return MockTTS()
