# transcriber.py

import whisper
import logging
import os

logger = logging.getLogger(__name__)

# Load the Whisper model once to avoid reloading each time
model = whisper.load_model("turbo")  # You can change "base" to other model sizes like "small", "medium", "large"

def transcribe_audio(audio_path):
    try:
        logger.info(f"Transcribing audio file {audio_path} using Whisper model.")

        # Transcribe the audio file
        result = model.transcribe(audio_path)
        full_text = result["text"]

        # Collect segments with timestamps if available
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        logger.info("Transcription successful.")
        return full_text, segments

    except Exception as e:
        logger.error(f"Error during transcription of file {audio_path}: {e}")
        raise  # Re-raise the exception to be handled by the calling function


