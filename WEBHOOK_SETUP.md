# ElevenLabs Webhook Setup Guide

This guide explains how to set up webhooks to receive call summaries from ElevenLabs back to Omni.

## Overview

When an ElevenLabs call completes, the webhook system:
1. Receives the call summary, transcript, and metadata
2. Stores it in Redis for easy retrieval
3. Automatically adds summaries to user memories
4. Makes call history available via the `get_call_history` tool in Omni

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Add the following to your `.env` file:

```bash
# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_WEBHOOK_URL=https://your-domain.com/webhook/elevenlabs/call-completed

# Webhook Server Configuration (optional)
WEBHOOK_PORT=5000

# Upstash Redis (already configured)
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token
```

### 3. Expose Your Webhook Server

You need to make your webhook server accessible to ElevenLabs. Choose one option:

#### Option A: Deploy to Production (Recommended)

Deploy `webhook_server.py` to your production server (Railway, Render, Heroku, etc.) and use that URL.

#### Option B: Use Ngrok for Testing

For local testing:

```bash
# In terminal 1: Start the webhook server
python webhook_server.py

# In terminal 2: Expose it with ngrok
ngrok http 5000

# Copy the ngrok URL (e.g., https://abc123.ngrok.io)
# Add it to your .env:
# ELEVENLABS_WEBHOOK_URL=https://abc123.ngrok.io/webhook/elevenlabs/call-completed
```

### 4. Configure ElevenLabs Dashboard

1. Go to [ElevenLabs Dashboard](https://elevenlabs.io/conversational-ai)
2. Navigate to your Conversational AI Agent settings
3. Find the **Webhooks** or **Event Callbacks** section
4. Add your webhook URL: `https://your-domain.com/webhook/elevenlabs/call-completed`
5. Enable the following events:
   - `conversation.ended`
   - `call.completed`
   - Any other relevant call completion events

### 5. Run the Services

#### Start Webhook Server:
```bash
python webhook_server.py
```

#### Start Omni Agent:
```bash
python omni.py start
```

## How It Works

### Call Flow

1. **User asks Omni to make a call**: "Call John at +1234567890 to schedule a meeting"
2. **Omni initiates call**: Uses `send_call_agent` tool
3. **ElevenLabs makes the call**: Agent converses with the recipient
4. **Call completes**: ElevenLabs sends webhook to your server
5. **Webhook receives data**: Call summary, transcript, duration, etc.
6. **Data stored in Redis**:
   - By call ID: `elevenlabs:call:{call_id}`
   - By user: `elevenlabs:user:{user_identity}:calls`
   - Recent calls: `elevenlabs:recent_calls`
   - User memory: Added to user's memory in Redis

### Retrieving Summaries

#### In Omni Conversation:
Users can ask Omni:
- "What calls have I made?"
- "Show me my call history"
- "What was the summary of my last call?"

Omni will use the `get_call_history` tool to retrieve and display summaries.

#### Via API:
```bash
# Get user's call history
curl http://localhost:5000/api/calls/user_identity

# Get recent calls (all users)
curl http://localhost:5000/api/calls/recent?limit=10
```

#### Programmatically:
```python
from eleven import get_call_summaries, get_recent_call_summaries

# Get specific user's calls
user_calls = get_call_summaries("user_identity", limit=5)

# Get all recent calls
recent = get_recent_call_summaries(limit=10)
```

## Data Structure

Each call summary stored in Redis contains:

```python
{
    'call_id': 'unique_call_id',
    'conversation_id': 'conversation_id',
    'status': 'completed',
    'transcript': 'Full conversation transcript...',
    'summary': 'AI-generated summary of the call',
    'duration': 120,  # seconds
    'goal': 'Schedule a meeting',
    'user': 'user_identity',
    'timestamp': '2025-10-06T12:00:00Z',
    'completed_at': '2025-10-06T12:02:00Z'
}
```

## Testing the Webhook

### Test with curl:

```bash
curl -X POST http://localhost:5000/webhook/elevenlabs/call-completed \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "test_123",
    "conversation_id": "conv_456",
    "status": "completed",
    "transcript": "Test transcript",
    "summary": "This was a test call",
    "duration": 60,
    "custom_data": {
      "goal": "Test goal",
      "user": "test_user"
    }
  }'
```

### Check health:
```bash
curl http://localhost:5000/webhook/health
```

## Troubleshooting

### Webhook not receiving data:
1. Check that webhook URL is correct in ElevenLabs dashboard
2. Verify webhook server is running and accessible
3. Check server logs: `tail -f webhook_server.log`
4. Test webhook endpoint manually with curl

### Can't retrieve summaries:
1. Verify Redis connection is working
2. Check that calls are being stored: `redis.lrange("elevenlabs:recent_calls", 0, 0)`
3. Verify user identity matches between call initiation and retrieval

### Webhook server crashes:
1. Check Python version (3.7+)
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check Redis environment variables are set correctly

## Production Deployment

### Railway Deployment

1. Add webhook_server.py to your Railway service
2. Add a `Procfile`:
   ```
   web: python webhook_server.py
   agent: python omni.py start
   ```
3. Configure environment variables in Railway dashboard
4. Use the Railway-provided URL as your webhook URL

### Docker Deployment

Update your `Dockerfile` to include:
```dockerfile
# Add Flask
RUN pip install flask

# Copy webhook server
COPY webhook_server.py .

# You may want to run both services
CMD python webhook_server.py & python omni.py start
```

## Security Considerations

1. **Validate Webhook Signatures**: Consider adding webhook signature verification
2. **Use HTTPS**: Always use HTTPS for production webhook URLs
3. **Rate Limiting**: Add rate limiting to webhook endpoint
4. **API Key Security**: Store ElevenLabs API key securely (env vars, not in code)
5. **Redis Security**: Use authentication for Redis in production

## Support

For issues or questions:
- Check ElevenLabs documentation: https://elevenlabs.io/docs
- Check webhook server logs
- Verify environment variables are set correctly

