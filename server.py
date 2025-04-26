from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import numpy as np

app = FastAPI()

# Determine the device to run the model on
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# Initialize the ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1,
)

# Resampler to convert 8kHz audio to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)


# sample rate of audio chunk expected
SAMPLE_RATE = 8000
FRAME_SIZE = 2 # no of seconds audio to take
CHUNK_SIZE = SAMPLE_RATE * FRAME_SIZE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            print("Recieved data...")

            
            if len(buffer) % CHUNK_SIZE == 0:
                pcm_chunk = buffer[:CHUNK_SIZE-100] if len(buffer) - CHUNK_SIZE > 100 else buffer[:CHUNK_SIZE]
                buffer = bytearray()

                # Convert bytes to float32 tensor
                audio_np = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                waveform = torch.from_numpy(audio_np).unsqueeze(0)

                waveform = resampler(waveform)

                # remove noise here

                # Transcribe the audio
                result = asr_pipeline(waveform.squeeze(0).numpy(), chunk_length_s=1)
                text = result["text"]
                
                await websocket.send_text(text)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, port=5050)
