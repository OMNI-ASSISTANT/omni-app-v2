"""
J.A.R.V.I.S – LiveKit Agents with Gemini Live API
--------------------------------------------------
This version uses LiveKit Agents framework for seamless
Gemini Live API integration with automatic audio streaming.

Features:
* LiveKit Agents with built-in Gemini Live API
* Automatic audio streaming and room management
* All original tool functions preserved
* Production-ready for Railway deployment
"""
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio
import logging
import os
import subprocess
import io
import json
import pathlib
import sys
from typing import Annotated

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, function_tool, RunContext, Agent, AgentSession, RoomInputOptions
from livekit.plugins import google
from eleven import call_user
from google.genai import types

# ───────── ENV & LOG ─────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────── Tool Function Definitions ─────────
search = DuckDuckGoSearchRun()

# Global variable to store current user info
current_user_info = None

@function_tool
async def get_user_info(
    context: "RunContext",
) -> str:
    """Get information about the current user in the room."""
    global current_user_info
    
    if current_user_info:
        logger.info(f"User info retrieved: {current_user_info}")
        from upstash_redis import Redis
        redis = Redis.from_env()
        memory = redis.get(current_user_info["identity"])
        return f"User information: {str(current_user_info) + "Memories: " + str(memory)}"
    else:
        return "No user information available. Please wait for a user to join."

@function_tool
async def add_memory(
    context: "RunContext",
    memory: str,
) -> str:
    """Add a memory to the current user."""
    from upstash_redis import Redis
    redis = Redis.from_env()
    try:
        current_memory = redis.get(current_user_info["identity"])
    except Exception as e:
        current_memory = ""
    if current_memory:
        current_memory += "\n" + memory
    else:
        current_memory = memory
    redis.set(current_user_info["identity"], current_memory)
    return "Memory added successfully."

@function_tool
async def web_search(
    context: "RunContext",
    query: Annotated[str, "The search query to run"],
) -> str:
    """Perform a web search using DuckDuckGo and return results as text."""
    logger.info(f"Running web search for: {query}")
    result = search.run(query)
    return result

@function_tool()
async def send_call_agent(
    context: RunContext,
    goal: Annotated[str, "Goal of the call"],
    user: Annotated[str, "User to call"],
    to_number: Annotated[str, "Phone number to call"],
) -> str:
    """Initiate a phone call using ElevenLabs conversational AI agent."""
    logger.info(f"Initiating call to {to_number} with agent {goal} {user}")
    return call_user(goal=goal, user=user, phone_number=to_number)


@function_tool()
async def search_videos(
    context: RunContext,
    query: Annotated[str, "Search query for finding videos (e.g., 'person holding water bottle', 'window', 'car driving')"],
    question: Annotated[str, "What needs to be found out about the video?"],
    top_k: Annotated[int, "Number of top results to return (default: 3)"] = 3
) -> str:
    """Search for videos using AI-powered vector database and extract key frames as images for Gemini Live."""
    logger.info(f"Searching videos for: {query}")
    
    try:
        import requests
        import cv2
        import tempfile
        import base64
        import numpy as np
        from livekit.agents.llm import ImageContent
        from livekit.agents import get_job_context
    except ImportError as e:
        return f"Error: Required library not available: {str(e)}"
    
    api_base_url = "https://macbook-pro.tail1dc532.ts.net"
    
    # Check API health and perform search
    health_response = requests.get(f"{api_base_url}/", timeout=10)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if not health_data.get("model_loaded", False):
            return "Video search unavailable - AI model not loaded."
    
    # Search for videos
    search_data = {"query": query, "top_k": min(top_k, 10)}
    response = requests.post(
        f"{api_base_url}/search",
        headers={"Content-Type": "application/json"},
        json=search_data,
        timeout=30
    )
    import time
    if response.status_code != 200:
        return f"Video search failed: {response.status_code}"
        
    results = response.json().get("results", [])
    if not results:
        return f"No videos found matching '{query}'. Try different terms like 'person', 'car', 'window'."
    
    # Get best match and extract frames
    best_match = results[0]
    filename = best_match.get("filename", "Unknown")
    similarity_percent = round(best_match.get("similarity", 0) * 100, 1)
    video_url = f"{api_base_url}/vids/{filename}"

    # Download the video file and save it to disk
    video_response = requests.get(video_url, timeout=60)
    video_path = filename
    with open(video_path, "wb") as f:
        f.write(video_response.content)

    if video_response.status_code != 200:
        print(f"Found video '{filename}' but couldn't download it.")

    #ask gemini what the video is about/what it shows thru question
    from google import genai
    from google.genai import types
    import dotenv
    dotenv.load_dotenv()
    #get from .env
    client = genai.Client(api_key=dotenv.get_key(".env", "GOOGLE_API_KEY"))
    video_temp = client.files.upload(file=video_path)
    # wait for file to be ACTIVE
    while True:
        try:
            response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=['Answer the following question based on the video: '+question, video_temp]
        )
            break
        except Exception as e:
            #file has not been loaded yet
            time.sleep(1)
    video_summary = response.text
    print(video_summary)
    #delete video file
    os.remove(video_path)
    #delete video temp file
    client.files.delete(video_temp.id)
    return video_summary
    #| GET | `/vids/{filename}` | Stream individual video for playback | None |



# ───────── Main Agent Entry Point ─────────

class OmniAgent(Agent):
    """Custom Agent class for Omni."""
    def __init__(self):
        tool_list = [get_user_info, send_call_agent, search_videos, web_search, add_memory]
        
        instructions = (
            "You are Omni, a helpful AI assistant. "
            "When a user joins the room, use the get_user_info tool to find out their name and other details. "
            "Always greet users by their name when you know it. Be helpful, concise, and friendly. "
            "Start conversations by calling get_user_info to learn about the user."
            " Use the add_memory tool to add memories to the user, even when not directly prompted."
            " Always speak in British English (en-GB) with a natural UK accent and use UK spelling."
        )

        super().__init__(
            instructions=instructions,
            tools=tool_list,
            llm=google.beta.realtime.RealtimeModel(
                # Default to a British-accent voice; override with OMNI_VOICE env var
                voice=os.getenv("OMNI_VOICE", "Puck"),
                # If supported by the plugin, set language/locale for TTS
                language_code=os.getenv("OMNI_LANGUAGE", "en-GB"),
            ),
        )

async def entrypoint(ctx: JobContext):
    """Main agent entry point."""
    logger.info("Starting Omni Agent...")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found.")
        raise ValueError("GOOGLE_API_KEY is required.")

    await ctx.connect()
    
    agent = OmniAgent()
    session = AgentSession()

    def on_participant_connected(participant):
        global current_user_info
        print(f"🎉 Participant connected!")
        print(f"   Identity: {participant.identity}")
        print(f"   Name: {participant.name}")
        print(f"   SID: {participant.sid}")
        print(f"   Attributes: {getattr(participant, 'attributes', 'No attributes')}")
        print(f"   Metadata: {participant.metadata}")
        
        # Store user info globally for the agent to access
        current_user_info = {
            "name": participant.name or participant.identity,
            "identity": participant.identity,
            "sid": participant.sid,
            "state": getattr(participant, 'state', 'Unknown'),
            "kind": getattr(participant, 'kind', 'Unknown')
        }
        print(f"✅ Stored user info: {current_user_info}")

    def on_participant_disconnected(participant):
        global current_user_info
        print(f"👋 Participant disconnected: {participant.identity}")
        current_user_info = None
        print("🔄 Cleared user info")

    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)

    print(f"🏠 Agent is in the room, waiting for users...")

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(close_on_disconnect=False, video_enabled=True)
    )
    logger.info("Agent session ended.")

# ───────── Entry Point ─────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )