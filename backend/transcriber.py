import whisper
import logging
import warnings
import torch

logger = logging.getLogger(__name__)

# Suppress the specific FutureWarning from torch.load
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`"
)

# Determine if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load the Whisper models once to avoid reloading each time
logger.info("Loading Whisper models...")
model_tiny = whisper.load_model("tiny", device=device)
model_small = whisper.load_model("small", device=device)
logger.info("Whisper models loaded.")

def transcribe_audio(audio_path, mode='fast'):
    try:
        if mode == 'fast':
            model = model_tiny
            logger.info(f"Using 'tiny' Whisper model for transcription.")
        elif mode == 'accurate':
            model = model_small
            logger.info(f"Using 'small' Whisper model for transcription.")
        else:
            logger.error(f"Invalid mode '{mode}' provided to transcribe_audio.")
            raise ValueError("Invalid mode. Choose 'fast' or 'accurate'.")

        logger.info(f"Transcribing audio file {audio_path} using Whisper model ({mode}).")

        # Transcribe the audio file with options to speed up transcription
        result = model.transcribe(
            audio_path,
            verbose=False,
            fp16=(device == "cuda"),  # Use FP16 if running on GPU
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=False,
            task='transcribe'  # Ensure transcription (not translation)
            # Removed 'suppress_silence' parameter
        )
        full_text = result["text"]

        # Collect segments with timestamps
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
        raise
