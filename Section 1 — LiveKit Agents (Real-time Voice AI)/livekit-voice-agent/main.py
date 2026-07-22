import logging
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import AgentServer, cli

from agent.session import start_agent_session
from config.settings import Settings

logger = logging.getLogger("main")

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log"),
    ],
)

logger.info("Starting LiveKit Voice AI Agent")
logger.info("Mock STT: %s, Mock TTS: %s", settings.use_mock_stt, settings.use_mock_tts)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx):
    await start_agent_session(ctx, settings)


if __name__ == "__main__":
    cli.run_app(server)
