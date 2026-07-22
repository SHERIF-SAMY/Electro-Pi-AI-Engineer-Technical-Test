import asyncio
import logging

from livekit.agents import utils
from livekit.agents.tts import (
    TTS,
    ChunkedStream,
    SynthesizeStream,
    TTSCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("mock.tts")


class MockTTS(TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=24000,
            num_channels=1,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return MockChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return MockSynthesizeStream(tts=self, conn_options=conn_options)


class MockChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter) -> None:
        logger.info("MockTTS response: \"%s\"", self._input_text)
        print(f"\n--- Agent: {self._input_text} ---\n")

        output_emitter.initialize(
            request_id=utils.shortuuid("mock_tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        output_emitter.push(b"\x00\x00" * 240)
        output_emitter.flush()


class MockSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)

    async def _run(self, output_emitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid("mock_tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )

        input_text = ""
        async for data in self._input_ch:
            if isinstance(data, str):
                input_text += data
                continue
            elif isinstance(data, SynthesizeStream._FlushSentinel):
                if input_text:
                    logger.info("MockTTS response: \"%s\"", input_text)
                    print(f"\n--- Agent: {input_text} ---\n")
                    output_emitter.push(b"\x00\x00" * 240)
                    output_emitter.flush()
                    input_text = ""
