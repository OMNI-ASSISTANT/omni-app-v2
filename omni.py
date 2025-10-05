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
    """Custom Agent class for Omni with video search capabilities."""
    
    def __init__(self, participant_info=None):
        # Create tool list
        tool_list = [
            send_call_agent,
            search_videos,
            web_search, 
        ]

        # Build dynamic system prompt
        if participant_info and participant_info.get('name'):
            user_name = participant_info['name']
            instructions = (f"Your name is Omni, a helpful assistant. You are speaking with {user_name}. "
                            f"Be polite and helpful, and use their name when appropriate. Keep your responses to 2 sentences or less.")
        else:
            instructions = ("Your name is Omni, a helpful assistant. "
                            "You are waiting for a user to join. Be polite and helpful. "
                            "Keep your responses to 2 sentences or less.")

        super().__init__(
            instructions=instructions,
            tools=tool_list,
            llm=google.beta.realtime.RealtimeModel(),
        )

def get_participant_kind_name(kind):
    """Convert participant kind number to readable name."""
    kind_map = { 0: "Standard User", 1: "Ingress", 2: "Egress", 3: "SIP", 4: "Agent" }
    return kind_map.get(kind, f"Unknown ({kind})")

def print_participant_info(participant, is_local=False):
    """Print detailed participant information."""
    prefix = "Local" if is_local else "Remote"
    print(f"\n🔍 {prefix} Participant Details:")
    print(f"  - SID: {participant.sid}")
    print(f"  - Identity: {participant.identity}")
    print(f"  - Name: {participant.name or 'No name set'}")
    print(f"  - Kind: {get_participant_kind_name(getattr(participant, 'kind', 0))}")
    print(f"  - Tracks: {len(participant.track_publications)}")
    print("=" * 50)

async def entrypoint(ctx: JobContext):
    """Main agent entry point called by LiveKit Agents framework."""
    logger.info("Starting Omni Agent with Gemini Live API")
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY is required.")

    await ctx.connect()

    agent = OmniAgent()
    session = AgentSession()

    def on_participant_connected(participant):
        print("\n🎉 NEW PARTICIPANT JOINED!")
        print_participant_info(participant)
        
        participant_info = {'name': participant.name or participant.identity}
        new_agent = OmniAgent(participant_info=participant_info)
        print(f"🤖 Updating agent context for {participant_info['name']}")
        session.update_agent(new_agent)

    def on_participant_disconnected(participant):
        print("\n👋 PARTICIPANT LEFT!")
        print_participant_info(participant)
        
        print("🤖 Resetting agent to default state.")
        session.update_agent(OmniAgent())

    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)

    print("🏠 Room connected! Waiting for participants...")
    print_participant_info(ctx.room.local_participant, is_local=True)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(video_enabled=True, close_on_disconnect=False)
    )
    logger.info("Agent session ended.")

# ───────── Entry Point ─────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )