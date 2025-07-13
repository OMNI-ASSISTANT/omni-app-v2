#!/usr/bin/env python3
"""
Simple test script to create a LiveKit session that will trigger your Omni agent
and show up on your LiveKit Cloud dashboard.
"""

import asyncio
import os
from livekit import api, rtc
from dotenv import load_dotenv

load_dotenv()

async def test_agent_session():
    """Create a test session to trigger the agent"""
    
    # Your LiveKit credentials
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY") 
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    print(f"🚀 Connecting to LiveKit: {url}")
    
    # Generate a token for the test user
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("test-user")
        .with_name("Test User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="test-room",
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    
    # Connect to the room
    room = rtc.Room()
    
    print("🔌 Connecting to room: test-room")
    await room.connect(url, token)
    
    print("✅ Connected! Your agent should now be triggered.")
    print("📊 Check your LiveKit Cloud dashboard - you should see a new session!")
    print("🎤 Enabling microphone to test voice interaction...")
    
    # Enable microphone (this will trigger your agent)
    try:
        await room.local_participant.set_microphone_enabled(True)
        print("🎙️ Microphone enabled - agent should respond!")
    except Exception as e:
        print(f"⚠️ Microphone setup failed (normal in headless mode): {e}")
    
    # Keep the session alive for 30 seconds
    print("⏱️ Keeping session alive for 30 seconds...")
    await asyncio.sleep(30)
    
    print("🏁 Test session complete!")
    await room.disconnect()

if __name__ == "__main__":
    asyncio.run(test_agent_session()) 