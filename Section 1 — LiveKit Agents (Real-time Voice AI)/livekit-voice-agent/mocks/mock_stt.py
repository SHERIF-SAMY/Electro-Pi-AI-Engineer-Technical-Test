import asyncio
import logging

from livekit.agents import NOT_GIVEN, NotGivenOr, utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.audio import AudioBuffer

logger = logging.getLogger("mock.stt")


class MockSTT(STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        text = ""
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text=text, language="en")],
        )

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return MockRecognizeStream(stt=self, conn_options=conn_options, language=language)


class MockRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._language = language
        self._pending_text: str | None = None

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info("MockSTT: reading from stdin (type your input and press Enter)")

        try:
            while True:
                text = await loop.run_in_executor(None, input)
                text = text.strip()
                if text:
                    logger.info("MockSTT transcript: \"%s\"", text)
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                SpeechData(
                                    text=text,
                                    language=self._language or "en",
                                )
                            ],
                        )
                    )

                self._input_ch.recv_nowait()

        except asyncio.CancelledError:
            logger.debug("MockSTT stream cancelled")
        except EOFError:
            logger.info("MockSTT: EOF received, ending stream")
        except utils.aio.ChanClosed:
            logger.debug("MockSTT: input channel closed")
