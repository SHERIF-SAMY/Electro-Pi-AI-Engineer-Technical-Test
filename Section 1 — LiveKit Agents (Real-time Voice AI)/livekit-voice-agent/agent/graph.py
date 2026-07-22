from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import get_order_status
from config.settings import Settings

settings = Settings()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_graph():
    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
    )
    
    # Define our tools
    tools = [get_order_status]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Define the tool node
    tool_node = ToolNode(tools)
    
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def route_tools(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, route to tools
        if last_message.tool_calls:
            return "tools"
        
        # Otherwise, end the conversation
        return END

    # Build the graph
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "chatbot")
    
    return builder.compile()
