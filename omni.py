"""
J.A.R.V.I.S – LiveKit Agents with Gemini Live API
--------------------------------------------------
This version uses LiveKit Agents framework for seamless
Gemini Live API integration with automatic audio streaming.

Features:
* LiveKit Agents with built-in Gemini Live API
* Automatic audio streaming and room management
* All original tool functions preserved
* Web search capability via DuckDuckGo API
* Production-ready for Railway deployment
"""
import asyncio
import logging
import os
import subprocess
import io
import json
import pathlib
import sys
from typing import Annotated
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, function_tool, RunContext, Agent, AgentSession, RoomInputOptions
from livekit.plugins import google
from eleven import call_user, get_call_summaries, get_recent_call_summaries
from google.genai import types
import time
# ───────── ENV & LOG ─────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose websocket debug logs
logging.getLogger('websockets.client').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)

# Global variable to store current user info
current_user_info = None

@function_tool
async def get_user_info(
    context: "RunContext",
) -> str:
    """Get information about the current user in the room."""
    global current_user_info
    #wait a second

    print(f"Getting user info for: {current_user_info}")
    while True:
        if current_user_info:
            logger.info(f"User info retrieved: {current_user_info}")
            from upstash_redis import Redis
            redis = Redis.from_env()
            memory = redis.get(current_user_info["identity"])
            return f"User information: {str(current_user_info) + "Memories: " + str(memory)}"
        else:
            time.sleep(1)

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
    query: Annotated[str, "The search query to look up on the web"],
    topic: Annotated[str, "The topic of the search - general, or if recent info needed, use news. Nothing else or will error. Must be 'general' or 'news.' "],
) -> str:
    """Search the web for current information and return results."""
    logger.info(f"Searching web for: {query}")
    print(topic)
    from tavily import TavilyClient
    client = TavilyClient("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")
    response = client.search(
        query=query,
        topic=topic
    )
    return response

@function_tool
async def get_current_datetime(
    context: "RunContext",
) -> str:
    """Get the current date and time."""
    now = datetime.now()
    formatted_datetime = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    logger.info(f"Current date and time: {formatted_datetime}")
    return f"The current date and time is {formatted_datetime}"

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
async def get_call_history(
    context: RunContext,
    user_identity: Annotated[str, "User identity to retrieve call history for (optional - uses current user if not provided)"] = None,
    limit: Annotated[int, "Maximum number of calls to retrieve (default: 5)"] = 5,
) -> str:
    """Retrieve call history and summaries from previous ElevenLabs calls for a specific user."""
    global current_user_info
    
    # Use current user if no user_identity provided
    if not user_identity and current_user_info:
        user_identity = current_user_info["identity"]
    elif not user_identity:
        return "No user identity provided and no current user found."
    
    logger.info(f"Retrieving call history for: {user_identity}")
    
    summaries = get_call_summaries(user_identity, limit)
    
    if not summaries:
        return f"No call history found for {user_identity}."
    
    # Format the summaries for display
    result = f"Call history for {user_identity} ({len(summaries)} calls):\n\n"
    for idx, call in enumerate(summaries, 1):
        result += f"Call {idx}:\n"
        result += f"  Goal: {call.get('goal', 'N/A')}\n"
        result += f"  Status: {call.get('status', 'N/A')}\n"
        result += f"  Duration: {call.get('duration', 0)} seconds\n"
        result += f"  Summary: {call.get('summary', 'No summary available')}\n"
        result += f"  Completed: {call.get('completed_at', 'N/A')}\n"
        if call.get('transcript'):
            result += f"  Transcript: {call.get('transcript')[:200]}...\n"
        result += "\n"
    
    return result


@function_tool()
async def search_videos(
    context: RunContext,
    query: Annotated[str, "Search query for finding videos (e.g., 'person holding water bottle', 'window', 'car driving')"],
    question: Annotated[str, "What needs to be found out about the video?"],
    top_k: Annotated[int, "Number of top results to return (default: 3)"] = 3
) -> str:
    """Search for videos using AI-powered vector database and extract key frames as images for Gemini Live."""
    logger.info(f"Searching videos for: {query}")
    
    global current_user_info
    
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
    
    # Get user email from current user info
    if not current_user_info or not current_user_info.get("identity"):
        return "Error: User identity not available. Cannot perform video search."
    
    user_email = current_user_info["identity"]
    api_base_url = "http://localhost:9999"
    
    # Check API health with email parameter
    health_response = requests.get(f"{api_base_url}/?email={user_email}", timeout=10)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if not health_data.get("model_loaded", False):
            # Try to warmup the model for this user
            logger.info(f"Model not loaded for {user_email}, calling warmup...")
            warmup_response = requests.post(
                f"{api_base_url}/warmup",
                headers={"Content-Type": "application/json"},
                json={"email": user_email},
                timeout=10
            )
            if warmup_response.status_code == 200:
                return "Video search is initializing for your account. This takes 1-2 minutes. Please try again shortly."
            else:
                return "Video search unavailable - AI model not loaded and warmup failed."
    
    # Search for videos with email
    search_data = {
        "email": user_email,
        "query": query,
        "top_k": min(top_k, 10)
    }
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
    # Prefer env var, fallback to .env key if present
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = dotenv.get_key(".env", "GOOGLE_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        return "Error: GOOGLE_API_KEY not configured. Set env var or .env."

    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=video_path)

    # Wait for the uploaded file to become ACTIVE
    max_wait_seconds = 60
    start_time = time.time()
    active_file = None
    while time.time() - start_time < max_wait_seconds:
        try:
            f = client.files.get(name=getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None))
            state = getattr(f, "state", None)
            # state may be literal "ACTIVE" or object with name
            state_name = getattr(state, "name", state)
            if state_name == "ACTIVE":
                active_file = f
                break
        except Exception:
            pass
        time.sleep(1)

    if not active_file:
        return "Error: Uploaded file did not become ACTIVE in time. Try again."

    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=['Answer the following question based on the video: ' + question, active_file]
        )
    except Exception as e:
        # Common transient errors include quota or file state races
        return f"Error generating analysis: {e}"

    video_summary = getattr(response, "text", None) or ""
    print(video_summary)
    #delete video file
    os.remove(video_path)
    #delete video temp file
    try:
        client.files.delete(name=getattr(active_file, "name", None) or getattr(active_file, "id", None))
    except Exception:
        pass
    return video_summary
    #| GET | `/vids/{filename}` | Stream individual video for playback | None |



# ───────── Main Agent Entry Point ─────────

class OmniAgent(Agent):
    """Custom Agent class for Omni."""
    def __init__(self):
        tool_list = [get_user_info, send_call_agent, get_call_history, search_videos, add_memory, web_search, get_current_datetime]
        instructions = (
            "You are Omni, a helpful AI assistant. "
            "When a user joins the room, use the get_user_info tool to find out their name and other details. "
            "Never wait, always call get_user_info immediately, as the first thing you do."
            "Always greet users by their name when you know it. Be helpful, concise, and friendly. "
            "Use the add_memory tool to add memories to the user, even when not directly prompted. "
            "You have access to web search for real-time information. Use the web_search tool when you need current information. "
            "Always speak in British English (en-GB) with a natural UK accent and use UK spelling."
            "If the user asks for recent information, use the web_search tool with the topic 'news', otherwise use the topic 'general'."
            "If you use any other topic, the tool will error."
            "Be extremly proactive - never ask for confirmation, and proactively call web search and add_memory tools when appropriate."
            "Often when the user is talking about something that happened in the past, they want you to use search_videos to find out more information."
        )

        super().__init__(
            instructions=instructions,
            tools=tool_list,
            llm=google.beta.realtime.RealtimeModel(
                # Default to a British-accent voice; override with OMNI_VOICE env var
                voice=os.getenv("OMNI_VOICE", "Puck"),
                # Set BCP-47 language tag (e.g., en-GB) for STT/TTS
                language=os.getenv("OMNI_LANGUAGE", "en-GB"),
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

    # Register event handlers BEFORE starting the session
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)

    print(f"🏠 Agent is in the room, waiting for users...")
    
    # Check for participants already in the room
    print(f"🔍 Checking for existing participants...")
    print(f"   Room name: {ctx.room.name}")
    print(f"   Room SID: {ctx.room.sid if hasattr(ctx.room, 'sid') else 'N/A'}")
    
    # Try to access remote participants
    if hasattr(ctx.room, 'remote_participants'):
        print(f"   Remote participants: {len(ctx.room.remote_participants)}")
        for participant in ctx.room.remote_participants.values():
            print(f"   Found existing participant: {participant.identity}")
            on_participant_connected(participant)
    else:
        print(f"   No remote_participants attribute found")
        
    # Also check all participants
    if hasattr(ctx.room, 'participants'):
        print(f"   All participants: {ctx.room.participants}")

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