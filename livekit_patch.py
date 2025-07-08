"""LiveKit Audio Integration Patch
---------------------------------
Drop‑in mixin that turns Omni’s local PyAudio loops into true
cloud‑native WebRTC tracks. Import and call `enable_livekit()`
right after creating your `AudioLoop` to attach:

from livekit_patch import enable_livekit
loop = AudioLoop()
enable_livekit(loop)

Requires env vars:
  LK_URL, LK_API_KEY, LK_API_SECRET
"""

import os, asyncio
from livekit import rtc, tokens
from livekit.rtc.audio_frame import AudioFrame

# ───────── Token helper ─────────

def _build_token(identity: str = "omni-core", room: str = "omni") -> str:
    api_key = os.getenv("LK_API_KEY")
    api_secret = os.getenv("LK_API_SECRET")
    grant = tokens.VideoGrant(room=room)
    at = tokens.AccessToken(api_key, api_secret, identity=identity, grants=grant)
    at.ttl = 3600  # 1 hour
    return at.to_jwt()

# ───────── Patch entrypoint ─────

def enable_livekit(loop_obj):
    """Monkey‑patch AudioLoop instance with LiveKit I/O."""

    async def _lk_connect(self):
        url = os.getenv("LK_URL")
        token = _build_token()
        room = await rtc.connect(url, token, auto_subscribe=True)

        # Track for publishing TTS back
        self._lk_tts = room.add_track(rtc.LocalAudioTrack())

        @room.on("track_subscribed")
        async def _on_track(track, pub, participant):
            async for frame in track:
                pcm = frame.data.tobytes()
                await self.out_q.put({"mime_type": "audio/pcm", "data": pcm})

        self._lk_room = room
        print("[LiveKit] connected →", room.name)

    async def _lk_play(self):
        while True:
            pcm = await self.audio_in.get()
            frame = AudioFrame(data=pcm,
                               sample_rate=self.SAMPLE_RATE,
                               num_channels=1,
                               samples_per_channel=len(pcm)//2)
            await self._lk_tts.write_audio_frame(frame)

    # inject tasks at runtime
    loop_obj._livekit_tasks = [
        asyncio.create_task(_lk_connect(loop_obj)),
        asyncio.create_task(_lk_play(loop_obj)),
    ]

    print("LiveKit patch applied – PyAudio loops skipped in cloud mode")
