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
        # Determine if running on Railway
        RUNNING_ON_RAILWAY = any(
            env in os.environ for env in (
                "RAILWAY_ENVIRONMENT",
                "RAILWAY_PROJECT_ID", 
                "RAILWAY_SERVICE_NAME",
            )
        )

        # Create tool list
        tool_list = [
            send_call_agent,
            search_videos,
            web_search, 
        ]

        # Build dynamic system prompt with participant info
        base_instructions = """You are **Omni**, a hyper-proactive assistant with external tools.
                • Think briefly before acting; be concise and helpful.
                • Reply to the user in ≤ 2 sentences and suggest useful next steps when appropriate.
                • Ask one clarifying question only if a critical detail is missing.
                • Use tools proactively to help users accomplish their goals.
                • For phone calls, ask for the number and confirm before calling.
                • For web searches, use precise queries and suggest related topics.
                • For video searches, use descriptive queries about visual content (e.g., "person holding bottle", "car driving", "window").
                • Video searches extract key frames as images that I can see and analyze - perfect for understanding visual content!
                • When users mention videos, proactively search to extract and analyze the visual frames.
                • Always confirm destructive actions before proceeding."""
        
        # Add participant context if available
        if participant_info:
            participant_context = f"""
                
                **CURRENT USER CONTEXT:**
                • User Name: {participant_info.get('name', 'Not provided')}
                • User Identity: {participant_info.get('identity', 'Unknown')}
                • Connection State: {participant_info.get('state', 'Unknown')}
                • User Type: {participant_info.get('kind_name', 'Standard User')}
                • Has Audio: {participant_info.get('has_audio', False)}
                • Has Video: {participant_info.get('has_video', False)}
                
                Use this context to personalize your responses and be more helpful!"""
            base_instructions += participant_context

        super().__init__(
            instructions=base_instructions,
            tools=tool_list,
            llm=google.beta.realtime.RealtimeModel(),
        )


def get_participant_kind_name(kind):
    """Convert participant kind number to readable name."""
    kind_map = {
        0: "Standard User",
        1: "Ingress",
        2: "Egress", 
        3: "SIP",
        4: "Agent"
    }
    return kind_map.get(kind, f"Unknown ({kind})")

def update_agent_context(agent, participant_info):
    """Update the agent's system prompt with current participant context."""
    base_instructions = """You are **Omni**, a hyper-proactive assistant.
*   Be concise and helpful.
*   Reply in 2 sentences or less.
*   Use your tools to help the user."""
    
    # Add participant context if available
    if participant_info and participant_info.get('name'):
        user_name = participant_info['name']
        participant_context = f"""

You are currently speaking with '{user_name}'. Address them by their name when it makes sense.
Here is some info about them:
- Identity: {participant_info.get('identity', 'Unknown')}
- Has Audio: {participant_info.get('has_audio', False)}
- Has Video: {participant_info.get('has_video', False)}"""
        base_instructions = f"Hello! Your name is Omni and you are a helpful assistant. The user you are talking to is named {user_name}. Please be polite and helpful, and use their name when you can."
        print(f"✅ Agent context updated for user: {user_name}")
    else:
        print("✅ Agent context reset to default. No user name available.")
    
    # Update the agent's instructions directly
    try:
        agent._instructions = base_instructions
        print(f"🔄 System prompt updated successfully. New prompt: '{base_instructions[:100]}...'")
    except Exception as e:
        print(f"⚠️ Failed to update agent context: {e}")


def print_participant_info(participant, is_local=False):
    """Print detailed participant information."""
    prefix = "Local" if is_local else "Remote"
    print(f"\n🔍 {prefix} Participant Details:")
    print(f"  - SID: {participant.sid}")
    print(f"  - Identity: {participant.identity}")
    print(f"  - Name: {participant.name or 'No name set'}")
    print(f"  - State: {getattr(participant, 'state', 'Unknown')}")
    print(f"  - Kind: {get_participant_kind_name(getattr(participant, 'kind', 0))}")
    print(f"  - Attributes: {getattr(participant, 'attributes', {}) or 'No attributes'}")
    print(f"  - Metadata: {participant.metadata or 'No metadata'}")
    print(f"  - Permissions: {getattr(participant, 'permissions', 'Unknown')}")
    print(f"  - Track Publications: {len(participant.track_publications)}")
    
    # Print track information if available
    if participant.track_publications:
        print(f"  - Active Tracks:")
        for track_sid, track_pub in participant.track_publications.items():
            print(f"    * {track_pub.kind}: {track_pub.name or 'Unnamed'} (muted: {track_pub.muted})")
    print("=" * 50)

async def entrypoint(ctx: JobContext):
    """Main agent entry point called by LiveKit Agents framework."""
    logger.info("Starting Omni Agent with Gemini Live API")

    # Connect to the room first
    await ctx.connect()
    
    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file")
        logger.error("Get your Google API key from: https://makersuite.google.com/app/apikey")
        logger.error("Then add this line to your .env file: GOOGLE_API_KEY=your-api-key-here")
        raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini Live API")
        
    # Create the initial agent
    agent = OmniAgent()
    
    # Store agent reference in job context for tool access
    ctx._agent = agent

    # Set up event listeners for participant events
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"\n🎉 NEW PARTICIPANT JOINED THE ROOM!")
        print_participant_info(participant, is_local=False)
        
        # Extract participant info for context
        participant_info = {
            'name': participant.name or participant.identity,
            'identity': participant.identity,
            'state': getattr(participant, 'state', 'JOINED'),
            'kind_name': get_participant_kind_name(getattr(participant, 'kind', 0)),
            'has_audio': any(pub.kind == 'audio' for pub in participant.track_publications.values()),
            'has_video': any(pub.kind == 'video' for pub in participant.track_publications.values())
        }
        
        # Update agent's system prompt with participant context
        print(f"🤖 Updating Omni with user context: {participant_info['name']}")
        update_agent_context(agent, participant_info)
        
    @ctx.room.on("participant_disconnected") 
    def on_participant_disconnected(participant):
        print(f"\n👋 PARTICIPANT LEFT THE ROOM!")
        print_participant_info(participant, is_local=False)
        
        # Reset agent context when user leaves
        print(f"🤖 Resetting Omni to default state")
        update_agent_context(agent, None)
        
    @ctx.room.on("participant_attributes_changed")
    def on_participant_attributes_changed(changed_attributes, participant):
        # The local participant is the agent itself. We must ignore its own attribute changes
        # to avoid wiping the user's context from the system prompt.
        if participant.identity == ctx.room.local_participant.identity:
            print(f"ℹ️ Agent's own attributes changed: {changed_attributes}. Ignoring for context update.")
            return

        print(f"\n📝 Remote participant attributes changed!")
        print(f"  - Participant: {participant.identity}")
        print(f"  - Changed attributes: {changed_attributes}")
        
        # Update agent context with new attributes for the remote participant
        participant_info = {
            'name': participant.name or participant.identity,
            'identity': participant.identity,
            'state': getattr(participant, 'state', 'JOINED'),
            'kind_name': get_participant_kind_name(getattr(participant, 'kind', 0)),
            'has_audio': any(pub.kind == 'audio' for pub in participant.track_publications.values()),
            'has_video': any(pub.kind == 'video' for pub in participant.track_publications.values()),
            'attributes': getattr(participant, 'attributes', {})
        }
        print(f"🤖 Re-updating Omni with new context for user: {participant_info['name']}")
        update_agent_context(agent, participant_info)
        
    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        print(f"\n📡 NEW TRACK PUBLISHED!")
        print(f"  - Participant: {participant.identity}")
        print(f"  - Track: {publication.kind} - {publication.name or 'Unnamed'}")
        print(f"  - Muted: {publication.muted}")
        
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"\n🔊 SUBSCRIBED TO TRACK!")
        print(f"  - Participant: {participant.identity}")
        print(f"  - Track: {publication.kind} - {publication.name or 'Unnamed'}")
        print(f"  - Track SID: {track.sid}")
    
    # Print initial room state
    logger.info("=== INITIAL ROOM STATE ===")
    print(f"\n🏠 Room connected! Waiting for participants...")
    if ctx.room.local_participant:
        print_participant_info(ctx.room.local_participant, is_local=True)
    
    # Create session with Gemini Live API
    session = AgentSession()

    await session.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(video_enabled=True, close_on_disconnect=False), 
        agent=agent
    )
    logger.info("Agent session started successfully")

# ───────── Entry Point ─────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )