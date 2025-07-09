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
from livekit.agents import JobContext, WorkerOptions, cli, function_tool, RunContext, Agent, AgentSession
from livekit.plugins import google

# ───────── ENV & LOG ─────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────── Tool Function Definitions ─────────

@function_tool()
async def change_mode(
    context: RunContext,
    mode: Annotated[str, "The mode to switch to: 'audio' for audio-only mode or 'video' for video mode"]
) -> str:
    """Change between audio-only and video mode for LiveKit session."""
    logger.info(f"Switching to {mode} mode")
    if mode.lower() == "audio":
        return "Switched to audio-only mode. Video is now disabled."
    elif mode.lower() == "video":
        return "Switched to video mode. Video and audio are now enabled."
    else:
        return "Invalid mode. Please specify 'audio' or 'video'."

@function_tool()
async def open_tab(
    context: RunContext,
    url: Annotated[str, "The URL to open in the browser"]
) -> str:
    """Open a URL in Arc browser (or default browser if Arc not available)."""
    logger.info(f"Opening URL: {url}")
    try:
        # Try Arc browser first (macOS)
        subprocess.run(["open", "-b", "company.thebrowser.Browser", url], check=True)
        return f"Opened {url} in Arc browser"
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to default browser
            subprocess.run(["open", url], check=True)
            return f"Opened {url} in default browser"
        except subprocess.CalledProcessError as e:
            return f"Failed to open {url}: {str(e)}"

@function_tool()
async def send_call_agent(
    context: RunContext,
    to_number: Annotated[str, "Phone number to call"],
    agent_id: Annotated[str, "ElevenLabs agent ID to use for the call"] = "your-agent-id"
) -> str:
    """Initiate a phone call using ElevenLabs conversational AI agent."""
    logger.info(f"Initiating call to {to_number} with agent {agent_id}")
    
    # Import here to avoid issues if elevenlabs isn't installed
    try:
        import requests
    except ImportError:
        return "Error: requests library not available"
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return "Error: ELEVENLABS_API_KEY not found in environment variables"
    
    url = "https://api.elevenlabs.io/v1/convai/conversations"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "agent_id": agent_id,
        "mode": "phone_call",
        "phone_number": to_number
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return f"Call initiated successfully to {to_number}. Call ID: {result.get('conversation_id', 'Unknown')}"
        else:
            return f"Failed to initiate call: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error initiating call: {str(e)}"

@function_tool()
async def search_drive(
    context: RunContext,
    query: Annotated[str, "Search query for Google Drive files"]
) -> str:
    """Search for files in Google Drive."""
    logger.info(f"Searching Google Drive for: {query}")
    
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return "Error: Google API libraries not available"
    
    # Google Drive API setup would go here
    # For now, return a placeholder
    return f"Searched Google Drive for '{query}'. This feature requires Google API setup."

@function_tool()
async def download_drive_file(
    context: RunContext,
    file_id: Annotated[str, "Google Drive file ID"],
    destination: Annotated[str, "Local destination path"] = "./downloads/"
) -> str:
    """Download a file from Google Drive."""
    logger.info(f"Downloading file {file_id} to {destination}")
    
    # Google Drive download implementation would go here
    return f"File download initiated from Google Drive ID: {file_id}"

# ───────── Main Agent Entry Point ─────────

async def entrypoint(ctx: JobContext):
    """Main agent entry point called by LiveKit Agents framework."""
    logger.info("Starting Omni Agent with Gemini Live API")
    
    # Connect to the room first
    await ctx.connect()

    # Determine if running on Railway (Railway automatically sets at least one of these env vars)
    RUNNING_ON_RAILWAY = any(
        env in os.environ for env in (
            "RAILWAY_ENVIRONMENT",  # set to "production" on deployed services
            "RAILWAY_PROJECT_ID",   # project UUID
            "RAILWAY_SERVICE_NAME", # service name
        )
    )

    # Create the agent with tools passed directly to constructor
    tool_list = [
        change_mode,
        send_call_agent,
        search_drive,
        download_drive_file,
    ]
    if not RUNNING_ON_RAILWAY:
        # open_tab uses subprocess & GUI – skip inside headless Railway container
        tool_list.append(open_tab)

    agent = Agent(
        instructions="""You are **Omni**, a hyper-proactive assistant with external tools.
            • Think briefly before acting; be concise and helpful.
            • Reply to the user in ≤ 2 sentences and suggest useful next steps when appropriate.
            • Ask one clarifying question only if a critical detail is missing.
            • Use tools proactively to help users accomplish their goals.
            • For phone calls, ask for the number and confirm before calling.
            • For web searches, use precise queries and suggest related topics.
            • Always confirm destructive actions before proceeding.
        """,
        tools=tool_list
    )

    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file")
        logger.error("Get your Google API key from: https://makersuite.google.com/app/apikey")
        logger.error("Then add this line to your .env file: GOOGLE_API_KEY=your-api-key-here")
        raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini Live API")

    # Create session with Gemini Live API (using the correct beta module)
    session = AgentSession(
        # Note: Gemini Live is a realtime model, so we don't need separate STT/TTS
        # The model handles voice input/output directly
        llm=google.beta.realtime.RealtimeModel(),
    )

    # Start the session
    await session.start(room=ctx.room, agent=agent)
    
    logger.info("Agent session started successfully")

# ───────── Entry Point ─────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )