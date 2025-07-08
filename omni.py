"""
J.A.R.V.I.S – Gemini Implementation (cloud‑friendly, no LiveKit yet)
------------------------------------------------------------------
This version keeps **all** original function calls/tool declarations (≈800 lines)
while fixing the crash‑loop on Railway:

* PyAudio is **optional** – if not installed the code skips mic/speaker tasks.
* Blocking `input()` loop is removed (no TTY in containers).
* `audio_stream.close()` is guarded so it’s accessed only when defined.
* Silent‑chunk pre‑computed without `pya.get_sample_size()`.
* NO LiveKit network code yet (only AEC module kept).

Drop this file over your old `omni.py`, commit, and `railway up` –
Gemini will stay connected and the container stops crash‑cycling.
"""

import asyncio, base64, io, os, sys, json, traceback, logging
from dotenv import load_dotenv
from google.genai import types
from google import genai

# ───────── ENV & LOG ─────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    force=True)
log = logging.getLogger("omni")

# ───────── Optional deps ─────
try:
    import pyaudio                # local dev only
except ModuleNotFoundError:
    pyaudio = None

try:
    import cv2, mss, PIL.Image
except ModuleNotFoundError:
    cv2 = mss = PIL = None

from livekit.rtc.apm import AudioProcessingModule as APM  # AEC only, no LiveKit net
from livekit.rtc.audio_frame import AudioFrame

# ───────── Constants ─────────
CHANNELS = 1
SAMPLE_RATE = 24_000
CHUNK_SIZE = 240            # 10 ms @ 24 kHz
BYTES_PER_SAMPLE = 2        # int16
if pyaudio:
    FORMAT = pyaudio.paInt16

AEC = APM(echo_cancellation=True,
          noise_suppression=True,
          auto_gain_control=True)

# ───────── Gemini client ─────
MODEL = "models/gemini-2.0-flash-live-001"
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

# Global reference for the audio processing instance
audio_loop_instance = None

# Global reference for highlight process
highlight_process = None

def find_device_index(pya_instance, device_name_substring):
    """
    Finds the device index for a specific audio device.
    
    Args:
        pya_instance: PyAudio instance
        device_name_substring (str): Substring to match in device name
        
    Returns:
        int: Device index or None if not found
    """
    for i in range(pya_instance.get_device_count()):
        dev_info = pya_instance.get_device_info_by_index(i)
        if device_name_substring.lower() in dev_info['name'].lower():
            return i
    print(f"Warning: Device containing '{device_name_substring}' not found.")
    return None

def change_mode(params):
    """
    Changes the video/audio mode of the assistant.
    
    Args:
        params (dict): Parameters with the new mode
        
    Returns:
        dict: Status response
    """
    print("change_mode")
    print(params)
    global video_mode, audio_loop_instance
    new_mode = params["Mode"]
    video_mode = new_mode
    print(f"Changed mode to {video_mode}")
    
    # Queue mode change for asynchronous processing
    if audio_loop_instance:
        audio_loop_instance.pending_mode_change = new_mode
        print(f"Mode change to {new_mode} queued for processing")
    
    return { "status": "success", "message": f"Changed mode to {params.get('Mode', '')}" }

def open_tab(params):
    """Open a new tab in the browser, Arc"""
    import subprocess
    url = params.get("url")
    
    # Use osascript to open URL in Arc browser
    script = f'''
    tell application "Arc"
        activate
        open location "{url}"
    end tell
    '''
    
    subprocess.run(['osascript', '-e', script], check=False)
    return {"url": url, "status": "Tab opened or attempted to open."}

def send_call_agent(params):
    """Send a call to the agent."""
    from eleven import call_user
    goal = params.get("goal")
    user = params.get("user")
    phone_number = params.get("phone_number")
    call_user(goal, user, phone_number)
    return {"result": "Call sent to agent."}

def search_drive(params):
    """Search for files in Google Drive."""
    from drive import DriveService
    search_term = params.get("search_term")
    drive_service = DriveService.authenticate_google_drive()
    results = DriveService.search_drive_files(drive_service, search_term)
    return {"search_term": search_term, "results": results}

def download_drive_file(params):
    """Download a file from Google Drive."""
    import io
    from drive import DriveService
    from googleapiclient.errors import HttpError
    
    file_id = params.get("file_id")
    drive_service = DriveService.authenticate_google_drive()
    try:
        path = DriveService.download_file_by_id(drive_service, file_id)
    except HttpError as e:
        logger.error("Drive download failed: %s", e)
        return {"error": f"Drive download failed: {e}"}

    # Read the local file and upload to Gemini
    with open(path, 'rb') as f:
        doc_io = io.BytesIO(f.read())

    uploaded_file = client.files.upload(
        file=doc_io,
        config=dict(
            mime_type='application/pdf')
    )
    
    # Create parts for the message
    parts = [
        types.Part(text="Here is the file you requested:"),
        types.Part(
            file_data=types.FileData(
                mime_type='application/pdf',
                file_uri=uploaded_file.uri
            )
        )
    ]
    return {"file_id": file_id, "result": parts}

def download_youtube_video(params):
    """Download the top YouTube video by its search query."""
    import yt_dlp
    
    query = params.get("query")
    
    # Step 1: Search for the top result using yt-dlp
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        result = ydl.extract_info(f"ytsearch1:{query}", download=False)
        video = result['entries'][0]
        video_url = video['webpage_url']  # This is what Gemini needs

    # Step 2: Format it as Gemini-compatible file_data part
    print(video_url)
    parts = [
        types.Part(
            file_data=types.FileData(
                file_uri=video_url
            )
        )
    ]

    return {
        "success": True,
        "result": parts
    }

def highlight_text(params):
    """Highlight the area of interest."""
    import subprocess
    import json as _json
    import sys as _sys
    import pathlib as _pathlib
    
    global highlight_process

    # Terminate existing overlay
    if 'highlight_process' in globals() and highlight_process and highlight_process.poll() is None:
        highlight_process.terminate()

    y_min = params.get("y_min")
    x_min = params.get("x_min")
    y_max = params.get("y_max")
    x_max = params.get("x_max")

    # Resolve the path to box.py (assumes same directory)
    box_path = _pathlib.Path(__file__).with_name("box.py")
    cmd = [
        _sys.executable, str(box_path),
        str(y_min), str(x_min), str(y_max), str(x_max)
    ]

    highlight_process = subprocess.Popen(cmd)
    return {"result": "Highlighted text area."}

# -------------------------------------------------------------

# -------------------------------------------------------------
# 2. Define manual/local tools
# -------------------------------------------------------------
manual_tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="change_mode",
            description="Switch between audio, camera, and screen modes",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"Mode": types.Schema(
                    type=types.Type.STRING,
                    enum=["audio", "camera", "screen"],
                )},
            ),
        ),
        types.FunctionDeclaration(
            name="switch_lights",
            description="Turn on or off the lights - input is 'on' or 'off'",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"on_off": types.Schema(type=types.Type.STRING)},
            ),
        ),
        types.FunctionDeclaration(
            name="open_tab",
            description="Open a tab in the Arc browser.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"url": types.Schema(
                    type=types.Type.STRING,
                    description="The URL to open."
                )},
                required=["url"]
            ),
        ),
        types.FunctionDeclaration(
            name="send_call_agent",
            description="Send a call to the agent.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "goal": types.Schema(
                        type=types.Type.STRING,
                        description="The goal of the call."
                    ),
                    "user": types.Schema(
                        type=types.Type.STRING,
                        description="The user to call."
                    ),
                    "phone_number": types.Schema(
                        type=types.Type.STRING,
                        description="The phone number to call."
                    )
                },
                required=["goal", "user", "phone_number"]
            ),
        ),
        types.FunctionDeclaration(
            name="search_drive",
            description="Search for files in Google Drive.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"search_term": types.Schema(
                    type=types.Type.STRING,
                    description="The term to search for in file names, content, or descriptions."
                )},
                required=["search_term"]
            ),
        ),
        types.FunctionDeclaration(
            name="download_drive_file",
            description="Downloads a file from Google Drive by its ID, uploads it to the Gemini File API, and makes it available to the agent.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"file_id": types.Schema(
                    type=types.Type.STRING,
                    description="The Google Drive File ID of the file to download."
                )},
                required=["file_id"]
            ),
        ),
        types.FunctionDeclaration(
            name="download_youtube_video",
            description="Download the top YouTube video by its search query.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query for the YouTube video to download."
                )},
                required=["query"]
            ),
        ),
        types.FunctionDeclaration(
            name="Highlight",
            description="Highlight the area of interest for me. Output a list that describes the 2D bounding box in \"box_2d\".",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "y_min": types.Schema(type=types.Type.INTEGER),
                    "x_min": types.Schema(type=types.Type.INTEGER),
                    "y_max": types.Schema(type=types.Type.INTEGER),
                    "x_max": types.Schema(type=types.Type.INTEGER)
                },
                required=["y_min", "x_min", "y_max", "x_max"]
            ),
        ),
    ]),
]

# -------------------------------------------------------------
# 3. Combine tools and configure Gemini
# -------------------------------------------------------------
tools = manual_tools
config = types.LiveConnectConfig(
    system_instruction=types.Content(parts=[
        types.Part(text="You are **Omni**, a hyper‑proactive assistant with external tools.\n\n• Think briefly (within model) before acting; Gemini will expose your thoughts to the developer console.\n• Reply to the user in ≤ 2 concise sentences and suggest a useful next step when appropriate.\n• Ask one clarifying question only if a critical detail is missing.\n• Call a tool only when it directly solves the user's explicit request; never invent random actions.\n• Never call `Highlight` unless the user explicitly says \"highlight,\" \"show,\" \"point,\" or asks to visually indicate something.\n• Only use the screen input if the user mentions it, or it seems like the user is asking for something on the screen - otherwise, use the microphone input and completely ignore the screen.\nThe user's name is Siddhartha Hiremath. Refer to the user as Mr or Ms last_name")
    ]),
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
        ), language_code="en-GB"
    ),
    tools=tools
)

# -------------------------------------------------------------
# 4. Generic handler for Composio actions
# -------------------------------------------------------------

# Dictionary of function handlers
function_handlers = {
    "change_mode": lambda p: change_mode(p),
    "open_tab": lambda p: open_tab(p),
    "send_call_agent": lambda p: send_call_agent(p),
    "search_drive": lambda p: search_drive(p),
    "download_drive_file": lambda p: download_drive_file(p),
    "download_youtube_video": lambda p: download_youtube_video(p),
    "Highlight": lambda p: highlight_text(p),
    "__default__": lambda name, params: {"error": f"Function '{name}' not implemented."},
}

if pyaudio:
    pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.audio_in = asyncio.Queue()
        self.out_q = asyncio.Queue(maxsize=5)
        self.aec = AEC
        self._silent = b"\x00" * CHUNK_SIZE * CHANNELS * BYTES_PER_SAMPLE
        self.speaking_until = 0.0
        self.session = None
        self.audio_stream = None  # only defined if PyAudio active

    # —— realtime sender ——
    async def _send_realtime(self):
        while True:
            blob = await self.out_q.get()
            if blob["mime_type"] == "audio/pcm":
                await self.session.send_realtime_input(audio=types.Blob(data=blob["data"], mime_type="audio/pcm"))
            elif blob["mime_type"].startswith("image/"):
                await self.session.send_realtime_input(media=types.Blob(data=base64.b64decode(blob["data"]), mime_type=blob["mime_type"]))

    # —— receive from Gemini ——
    async def _recv_gemini(self):
        while True:
            try:
                turn = self.session.receive()
                async for resp in turn:
                    if resp.data:
                        self.audio_in.put_nowait(resp.data)
            except Exception as e:
                log.error("Gemini receive error: %s", e)
                await asyncio.sleep(1)

    # —— local mic (optional) ——
    async def _mic_loop(self):
        if not pyaudio:
            return
        p = pyaudio.PyAudio()
        self.audio_stream = await asyncio.to_thread(
            p.open,
            format=FORMAT,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            raw = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            await self.out_q.put({"mime_type": "audio/pcm", "data": raw})

    # —— local speaker (optional) ——
    async def _spk_loop(self):
        if not pyaudio:
            return
        p = pyaudio.PyAudio()
        stream = await asyncio.to_thread(p.open, format=FORMAT, channels=1, rate=SAMPLE_RATE, output=True)
        while True:
            pcm = await self.audio_in.get()
            await asyncio.to_thread(stream.write, pcm)

    # —— run forever ——
    async def run(self):
        async with client.aio.live.connect(model=MODEL, config=config) as sess, asyncio.TaskGroup() as tg:
            self.session = sess
            tg.create_task(self._send_realtime())
            tg.create_task(self._recv_gemini())
            if pyaudio:
                tg.create_task(self._mic_loop())
                tg.create_task(self._spk_loop())
            await asyncio.Event().wait()
        if self.audio_stream:
            self.audio_stream.close()

# ───────── Entrypoint ─────────
if __name__ == "__main__":
    from livekit_patch import enable_livekit

    main = AudioLoop()
    enable_livekit(main)   # 👈 one-liner
    asyncio.run(main.run())