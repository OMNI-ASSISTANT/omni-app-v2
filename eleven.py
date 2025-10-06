from elevenlabs import ElevenLabs
from elevenlabs.types.conversation_initiation_client_data_request_input import ConversationInitiationClientDataRequestInput
import os
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

def call_user(goal: str, user: str, phone_number: str):
    """
    Initiate a phone call using ElevenLabs conversational AI.
    
    To receive call summaries:
    1. Set ELEVENLABS_WEBHOOK_URL in your .env file (e.g., https://your-domain.com/webhook/elevenlabs/call-completed)
    2. Configure this webhook URL in your ElevenLabs dashboard under Agent Settings
    3. The webhook server (webhook_server.py) will automatically store summaries in Redis
    
    Args:
        goal: The objective of the call
        user: User identity for tracking
        phone_number: Phone number to call
    
    Returns:
        Success message with call initiation details
    """
    client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY", "sk_cf430628198af45d71bb18b6e6b9fa8e0a5b3b2c790467ee"),
    )

    # Note: Webhook URL should be configured in ElevenLabs dashboard
    # Set ELEVENLABS_WEBHOOK_URL in .env for reference
    webhook_url = os.getenv("ELEVENLABS_WEBHOOK_URL")
    if webhook_url:
        print(f"📞 Call will send completion data to webhook: {webhook_url}")
    else:
        print("⚠️  No webhook URL configured. Set ELEVENLABS_WEBHOOK_URL in .env")

    call_response = client.conversational_ai.twilio.outbound_call(
        agent_id="LJ7aDuf9TaSPch9MsBic",
        agent_phone_number_id="BkXlFb0SSzu1FRwIzIWs",
        to_number=phone_number,
        conversation_initiation_client_data=ConversationInitiationClientDataRequestInput(
            dynamic_variables={"goal": goal, "user": user}
        ),
    )
    
    return f"Call initiated to {phone_number}. Goal: {goal}. User: {user}. Summaries will be sent to webhook when call completes."


def get_call_summaries(user_identity: str, limit: int = 5):
    """
    Retrieve call summaries for a specific user from Redis.
    
    Args:
        user_identity: The user's identity to look up
        limit: Maximum number of call summaries to retrieve
    
    Returns:
        List of call summaries or empty list if none found
    """
    try:
        redis = Redis.from_env()
        calls = redis.lrange(f"elevenlabs:user:{user_identity}:calls", 0, limit - 1)
        if calls:
            return [eval(call) for call in calls]
        return []
    except Exception as e:
        print(f"Error retrieving call summaries: {e}")
        return []


def get_recent_call_summaries(limit: int = 10):
    """
    Retrieve recent call summaries across all users.
    
    Args:
        limit: Maximum number of call summaries to retrieve
    
    Returns:
        List of recent call summaries
    """
    try:
        redis = Redis.from_env()
        calls = redis.lrange("elevenlabs:recent_calls", 0, limit - 1)
        if calls:
            return [eval(call) for call in calls]
        return []
    except Exception as e:
        print(f"Error retrieving recent calls: {e}")
        return []