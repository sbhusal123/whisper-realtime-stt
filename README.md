# Realtime Speech To Text With Whisper Models:

## Server:

> ``python server.py``

Server expects an audio with SAMPLE_RATE=8000, for every audio chunk passed through socket, it is collected till 2s audio can be constructed
then last 2s audio is transcribed.

```sh
# sample rate of audio chunk expected
SAMPLE_RATE=8000

# audio frame size (in secs) to take for transcription
FRAME_SIZE=2
CHUNK_SIZE=SAMPLE_RATE*FRAME_SIZE
```


Example Client Implementation: [CLient](./client.py)
