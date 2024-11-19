# podcast_processor/transcription/models.py

import logging
import whisper
from multiprocessing import current_process

logger = logging.getLogger(__name__)

def load_whisper_model(model_name: str, device: str) -> whisper.Whisper:
    """
    Load the specified Whisper model onto the designated device.
    """
    logger.info(f"Process {current_process().name}: Loading Whisper model '{model_name}' on device '{device}'.")
    try:
        model = whisper.load_model(model_name, device=device)
        logger.info(f"Process {current_process().name}: Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Process {current_process().name}: Failed to load model '{model_name}': {e}")
        return None
