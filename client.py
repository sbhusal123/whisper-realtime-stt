import asyncio
import sounddevice as sd
import numpy as np
import websockets

# Audio configuration
SAMPLE_RATE = 8000  # 8 kHz
CHANNELS = 1
DTYPE = 'int16'
CHUNK_DURATION = 0.2  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# WebSocket server URI
WS_URI = "ws://localhost:5050/ws"

async def send_audio(websocket):
    """
    Captures audio from the microphone and sends it over the WebSocket.
    """
    loop = asyncio.get_event_loop()

    def callback(indata, frames, time, status):
        if status:
            print(f"Recording status: {status}")
        # Schedule the sending of audio data in the event loop
        asyncio.run_coroutine_threadsafe(
            websocket.send(indata.tobytes()), loop
        )

    # Start the input stream with the callback
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, blocksize=CHUNK_SIZE, callback=callback):
        print("Started recording...")
        await asyncio.Future()  # Run indefinitely

async def receive_transcriptions(websocket):
    """
    Receives transcribed text from the WebSocket server and prints it.
    """
    try:
        async for message in websocket:
            print("Transcription:", message)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed.")

async def main():
    async with websockets.connect(WS_URI) as websocket:
        print("Connected to the WebSocket server.")
        # Create tasks for sending audio and receiving transcriptions
        send_task = asyncio.create_task(send_audio(websocket))
        receive_task = asyncio.create_task(receive_transcriptions(websocket))
        # Wait for both tasks to complete
        await asyncio.gather(send_task, receive_task)

if __name__ == "__main__":
    asyncio.run(main())
