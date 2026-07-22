import logging

from livekit.agents.stt import STT

from config.settings import Settings
from mocks.mock_stt import MockSTT

logger = logging.getLogger("plugins.stt")


def create_stt(settings: Settings) -> STT:
    if settings.use_mock_stt:
        logger.info("Using MockSTT")
        return MockSTT()

    try:
        from livekit.plugins.deepgram import STT as DeepgramSTT
        logger.info("Using Deepgram STT")
        return DeepgramSTT()
    except ImportError:
        logger.warning("Deepgram plugin not available, falling back to MockSTT")
        return MockSTT()
