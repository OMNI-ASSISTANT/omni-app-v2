"""
ElevenLabs Webhook Server
--------------------------
Receives call completion callbacks from ElevenLabs and stores summaries in Redis.
Run this server alongside your main omni.py application.
"""

from flask import Flask, request, jsonify
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Redis connection
redis = Redis.from_env()

@app.route('/webhook/elevenlabs/call-completed', methods=['POST'])
def elevenlabs_webhook():
    """Receive call completion webhook from ElevenLabs."""
    try:
        # Get the webhook payload
        data = request.get_json()
        logger.info(f"Received ElevenLabs webhook: {data}")
        
        # Extract relevant information
        call_id = data.get('call_id', 'unknown')
        conversation_id = data.get('conversation_id', 'unknown')
        status = data.get('status', 'unknown')
        transcript = data.get('transcript', '')
        summary = data.get('summary', '')
        duration = data.get('duration', 0)
        
        # Get custom variables (goal and user) from the webhook
        custom_data = data.get('custom_data', {})
        goal = custom_data.get('goal', 'Unknown goal')
        user_identity = custom_data.get('user', 'unknown_user')
        
        # Create a structured summary object
        call_summary = {
            'call_id': call_id,
            'conversation_id': conversation_id,
            'status': status,
            'transcript': transcript,
            'summary': summary,
            'duration': duration,
            'goal': goal,
            'user': user_identity,
            'timestamp': datetime.utcnow().isoformat(),
            'completed_at': data.get('completed_at', datetime.utcnow().isoformat())
        }
        
        # Store in Redis with multiple keys for easy retrieval
        # 1. Store by call_id
        redis.set(f"elevenlabs:call:{call_id}", str(call_summary))
        
        # 2. Add to user's call history
        redis.lpush(f"elevenlabs:user:{user_identity}:calls", str(call_summary))
        
        # 3. Store in a global list of recent calls
        redis.lpush("elevenlabs:recent_calls", str(call_summary))
        redis.ltrim("elevenlabs:recent_calls", 0, 99)  # Keep only last 100 calls
        
        # 4. If there's a summary, add it to user's memories
        if summary:
            try:
                current_memory = redis.get(user_identity)
                memory_entry = f"\n[Call Summary - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}] Goal: {goal}. Summary: {summary}"
                if current_memory:
                    redis.set(user_identity, current_memory + memory_entry)
                else:
                    redis.set(user_identity, memory_entry)
                logger.info(f"Added call summary to {user_identity}'s memory")
            except Exception as e:
                logger.error(f"Error adding to memory: {e}")
        
        logger.info(f"Successfully stored call summary for {user_identity} (Call ID: {call_id})")
        
        return jsonify({
            'status': 'success',
            'message': 'Call summary received and stored',
            'call_id': call_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/webhook/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'elevenlabs-webhook'}), 200

@app.route('/api/calls/<user_identity>', methods=['GET'])
def get_user_calls(user_identity):
    """Retrieve call history for a specific user."""
    try:
        calls = redis.lrange(f"elevenlabs:user:{user_identity}:calls", 0, -1)
        return jsonify({
            'user': user_identity,
            'calls': [eval(call) for call in calls] if calls else []
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving calls: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/calls/recent', methods=['GET'])
def get_recent_calls():
    """Retrieve recent calls across all users."""
    try:
        limit = int(request.args.get('limit', 10))
        calls = redis.lrange("elevenlabs:recent_calls", 0, limit - 1)
        return jsonify({
            'calls': [eval(call) for call in calls] if calls else []
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving recent calls: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('WEBHOOK_PORT', 5000))
    logger.info(f"Starting webhook server on port {port}...")
    logger.info(f"Webhook endpoint: http://localhost:{port}/webhook/elevenlabs/call-completed")
    app.run(host='0.0.0.0', port=port, debug=False)

