"""
J.A.R.V.I.S - Gemini Implementation
-----------------------------------
This module provides the Gemini-based implementation of the J.A.R.V.I.S assistant.
It leverages Google's Gemini API for natural language processing and response generation,
along with audio processing capabilities for voice input/output and video modes for visual interaction.
The system can perform various tasks through tools and integrations with external services.
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import json
from google.genai import types
import logging  # For debugging and logging
from dotenv import load_dotenv

#echo canceler

# Load environment variables
load_dotenv()

# Configure logging for better visibility
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Ensure stdout is not buffered for real-time feedback
sys.stdout.reconfigure(line_buffering=True)  # For Python 3.7+

# Import required libraries
import cv2
import pyaudio
import numpy as np
 # switched to LiveKit's cross‑platform AEC that ships pre‑built wheels 🚀
from livekit.rtc.apm import AudioProcessingModule as APM
from livekit.rtc.audio_frame import AudioFrame
import PIL.Image
import mss

import argparse

from google.genai import types
from google import genai


# Polyfill for asyncio TaskGroup on Python versions < 3.11
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 24000 # Matched with RECEIVE_SAMPLE_RATE
RECEIVE_SAMPLE_RATE = 24000
 # 10 ms of mono‑16‑bit samples at 24 kHz → 240 samples
CHUNK_SIZE = 240

# --- LiveKit APM: echo cancellation + NS + AGC ---
AEC_PROCESSOR = APM(
    echo_cancellation=True,
    noise_suppression=True,
    auto_gain_control=True,
)

# Gemini model selection
MODEL = "models/gemini-2.0-flash-live-001"  # Back to working model

# Default video mode
video_mode = "none"

# Initialize Gemini client with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
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

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.task_group = None  # Store reference to task group
        
        # Reference to global AEC for this instance
        self.aec = AEC_PROCESSOR

        self.far_end_buffer = asyncio.Queue(maxsize=10) # Buffer for far-end audio
        self._silent_chunk = b'\x00' * CHUNK_SIZE * CHANNELS * pya.get_sample_size(FORMAT) # Precompute silent chunk

        # Store tasks so they can be canceled
        self.video_task = None
        self.current_mode = video_mode
        self.pending_mode_change = None  # Flag for pending mode changes

        # Timestamp (monotonic) until which Omni is considered "speaking"
        # Initialized to time in the past → not speaking
        self.speaking_until = 0.0

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        print("Starting camera mode")
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        try:
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
        finally:
            # Release the VideoCapture object
            cap.release()
            print("Camera mode stopped")

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        print("Starting screen mode")
        try:
            while True:
                frame = await asyncio.to_thread(self._get_screen)
                if frame is None:
                    break

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
        finally:
            print("Screen mode stopped")

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if "data" in msg and "mime_type" in msg:
                if msg["mime_type"] == "audio/pcm":
                    # Send audio data
                    await self.session.send_realtime_input(
                        audio=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                    )
                elif msg["mime_type"].startswith("image/"):
                    # Send image data
                    await self.session.send_realtime_input(
                        media=types.Blob(data=base64.b64decode(msg["data"]), mime_type=msg["mime_type"])
                    )
                else:
                    print(f"Unsupported mime type: {msg['mime_type']}")

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        #search for the device name "MacBook Pro Microphone"
        for i in range(pya.get_device_count()):
            if "MacBook Pro Microphone" in pya.get_device_info_by_index(i)['name']:
                input_device_index = i
                break
        
        print(f"Using input device: {pya.get_device_info_by_index(input_device_index)['name']}")

        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=input_device_index, # Use MacBook Pro Microphone
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            # If we're inside the speaking window, skip mic capture
            if asyncio.get_event_loop().time() < self.speaking_until:
                await asyncio.sleep(CHUNK_SIZE / SEND_SAMPLE_RATE)  # wait 10 ms
                continue

            # ---- capture mic frame ----
            raw_data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

            # ---- get simultaneous far‑end playback (speaker) frame ----
            try:
                far_end_data = self.far_end_buffer.get_nowait()
                self.far_end_buffer.task_done()
            except asyncio.QueueEmpty:
                far_end_data = self._silent_chunk  # silence fallback

            # Ensure far‑end slice is exactly one 10 ms chunk
            bytes_per_chunk = CHUNK_SIZE * CHANNELS * 2  # int16 → 2 bytes
            if len(far_end_data) != bytes_per_chunk:
                if len(far_end_data) > bytes_per_chunk:
                    far_end_data = far_end_data[:bytes_per_chunk]
                else:
                    far_end_data = far_end_data.ljust(bytes_per_chunk, b'\x00')

            # Frames must be 10 ms long → samples_per_channel == CHUNK_SIZE
            samples = CHUNK_SIZE

            mic_frame = AudioFrame(
                data=memoryview(raw_data),
                sample_rate=SEND_SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=samples,
            )
            far_frame = AudioFrame(
                data=memoryview(far_end_data),
                sample_rate=SEND_SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=samples,
            )

            # Feed speaker audio (reverse stream) first, then process mic
            self.aec.process_reverse_stream(far_frame)
            self.aec.process_stream(mic_frame)

            # Retrieve processed bytes and enqueue
            processed_bytes = mic_frame.data.tobytes()
            await self.out_queue.put({"data": processed_bytes, "mime_type": "audio/pcm"})

    async def update_video_mode(self, new_mode):
        """Update the video mode and manage tasks dynamically."""
        print(f"Updating video mode from {self.current_mode} to {new_mode}")
        
        # Skip if no change or no task group
        if new_mode == self.current_mode or not self.task_group:
            return
        
        # Cancel existing video task if it exists
        if self.video_task and not self.video_task.done():
            print(f"Canceling existing {self.current_mode} task")
            self.video_task.cancel()
            try:
                await asyncio.wait_for(self.video_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Start new task based on mode
        self.current_mode = new_mode
        if new_mode == "camera":
            print("Creating camera task")
            self.video_task = self.task_group.create_task(self.get_frames())
        elif new_mode == "screen":
            print("Creating screen task")
            self.video_task = self.task_group.create_task(self.get_screen())
        else:
            # No video task for audio only
            self.video_task = None
        
        print(f"Mode updated to {new_mode}")

    async def check_mode_changes(self):
        """Periodically check for pending mode changes and apply them"""
        while True:
            if self.pending_mode_change and self.pending_mode_change != self.current_mode:
                new_mode = self.pending_mode_change
                print(f"Processing pending mode change to {new_mode}")
                await self.update_video_mode(new_mode)
                self.pending_mode_change = None
            await asyncio.sleep(0.5)  # Check every half second

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        logger.debug("Starting receive_audio method")
        while True:
            try:
                logger.debug("Waiting for session.receive()")
                turn = self.session.receive()
                logger.debug(f"Got turn object: {type(turn)}")
                
                # Add explicit counter to track iterations
                response_count = 0
                async for response in turn:
                    response_count += 1
                    logger.debug(f"Processing response #{response_count}")
                    
                    # Force flush stdout to ensure prints are visible
                    print(f"DEBUG: Received response: {response}", flush=True)
                    sys.stdout.flush()  # Extra flush for good measure
                    
                    # Handle tool calls
                    if response.tool_call:
                        print(f"Received tool call: {response.tool_call}", flush=True)
                        f_responses = []
                        for fc in response.tool_call.function_calls:
                            try:
                                params = fc.args if isinstance(fc.args, dict) else json.loads(fc.args or "{}")
                                print(f"Function call: {fc.name}, params: {params}", flush=True)
                            except json.JSONDecodeError:
                                params = {}
                                print(f"JSON decode error for function {fc.name}, using empty params", flush=True)

                            handler = function_handlers.get(fc.name)           # exact match
                            if not handler:                                    # fallback
                                handler = function_handlers["__default__"]
                                print(f"Using default handler for function: {fc.name}", flush=True)
                                sys.stdout.flush()  # Extra flush to ensure visibility
                            else:
                                print(f"Found handler for function: {fc.name}", flush=True)

                            print(f"Executing function: {fc.name}", flush=True)
                            result = handler(fc.name, params) if handler == function_handlers["__default__"] \
                                    else handler(params)
                            print(f"Function result: {result}", flush=True)

                            f_responses.append(types.FunctionResponse(
                                id=fc.id, name=fc.name, response=result
                            ))

                        print(f"Sending tool responses: {f_responses}", flush=True)
                        await self.session.send_tool_response(function_responses=f_responses)
                        continue
                    
                    if data := response.data:
                        logger.debug("Received audio data")
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="", flush=True)
                        sys.stdout.flush()  # Extra flush

                if response_count == 0:
                    logger.debug("No responses were received in this turn")
                    
            except Exception as e:
                logger.error(f"Error in receive_audio: {e}")
                logger.error(traceback.format_exc())
                # Continue the loop even after errors
                await asyncio.sleep(1)  # Small delay to avoid tight loop
            
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            logger.debug("Turn completed, emptying audio queue")
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        # Set to None to use the default output device
        output_device_index = None 

        # Print confirmation message
        print("Using default output device.")

        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            output_device_index=output_device_index, # Set to None for default
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            # Calculate chunk duration in seconds and extend speaking window
            duration_sec = len(bytestream) / (RECEIVE_SAMPLE_RATE * CHANNELS * 2)  # bytes → seconds
            # Add a 50 ms safety buffer on top of actual duration
            self.speaking_until = max(self.speaking_until, asyncio.get_event_loop().time() + duration_sec + 0.05)
            # Put audio into far-end buffer *before* playing
            await self.far_end_buffer.put(bytestream)
            # Far‑end audio is now supplied to the AEC in listen_audio
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.task_group = tg

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Store global reference to this instance
                global audio_loop_instance
                audio_loop_instance = self

                # Always create these core tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Add the mode change checker task
                tg.create_task(self.check_mode_changes())
                
                # Initialize with the current video mode
                await self.update_video_mode(video_mode)

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)
        

if __name__ == "__main__":
    # Enable debug mode for asyncio
    main = AudioLoop()
    asyncio.run(main.run(), debug=True)