import logging

from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, inference
from livekit.plugins import langchain

from agent.persona import SYSTEM_PROMPT
from agent.graph import create_graph
from config.settings import Settings
from plugins.stt import create_stt
from plugins.tts import create_tts

logger = logging.getLogger("agent.session")


import json
import asyncio

async def start_agent_session(ctx: JobContext, settings: Settings) -> None:
    stt = create_stt(settings)
    tts = create_tts(settings)

    llm = langchain.LLMAdapter(graph=create_graph())

    vad = inference.VAD()

    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        if getattr(ev, "is_final", False):
            payload = {"type": "user", "text": getattr(ev, "transcript", "")}
            asyncio.create_task(ctx.room.local_participant.publish_data(json.dumps(payload).encode("utf-8"), topic="pipeline_logs"))

    @session.on("conversation_item_added")
    def on_item_added(ev):
        item = getattr(ev, "item", None)
        if not item: return
        
        # Determine the role or type safely (might be a dict or object depending on LangChain adapter internals)
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        
        if role == "assistant" and content:
            # We don't want to log empty assistant messages (like when they just do tool calls)
            payload = {"type": "agent", "text": str(content)}
            asyncio.create_task(ctx.room.local_participant.publish_data(json.dumps(payload).encode("utf-8"), topic="pipeline_logs"))
            
        tool_calls = getattr(item, "tool_calls", None)
        if tool_calls:
            try:
                tool_names = [tc.get("name", "tool") if isinstance(tc, dict) else getattr(tc, "name", "tool") for tc in tool_calls]
                payload = {"type": "tool", "text": f"Agent is calling tools: {', '.join(tool_names)}"}
                asyncio.create_task(ctx.room.local_participant.publish_data(json.dumps(payload).encode("utf-8"), topic="pipeline_logs"))
            except Exception as e:
                logger.error(f"Error parsing tool calls: {e}")

    agent = Agent(
        instructions=SYSTEM_PROMPT,
    )

    logger.info(
        "Starting AgentSession. Room: %s. Participant identity will be available after connect.",
        ctx.room.name,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    logger.info("AgentSession started successfully")
