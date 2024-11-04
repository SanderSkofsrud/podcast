# transcriber.py
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

# Load the model with CPU inference and INT8 quantization for speed
model_size = "tiny"  # or "base" if needed
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    logger.info(f"Loading audio file {audio_path} for transcription.")

    segments_generator, info = model.transcribe(
        audio_path,
        language='en',  # Specify language
        beam_size=1     # Use greedy decoding for speed
    )

    segments = []
    full_text = ""
    for segment in segments_generator:
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })
        full_text += segment.text + " "

    logger.info("Transcription successful.")
    return full_text.strip(), segments
