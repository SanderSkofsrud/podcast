# transcriber.py

import requests
import logging
from config.settings import get_whisper_endpoint, get_whisper_api_key

logger = logging.getLogger(__name__)

# Get Whisper endpoint and API key from configuration
WHISPER_ENDPOINT = get_whisper_endpoint()
WHISPER_API_KEY = get_whisper_api_key()

def transcribe_audio(audio_path):
    try:
        logger.info(f"Transcribing audio file {audio_path} using Whisper model.")

        # Prepare headers
        headers = {
            'Authorization': f'Bearer {WHISPER_API_KEY}',
            # 'Content-Type': 'multipart/form-data'  # Not needed; requests will set this
        }

        # Prepare the files payload
        files = {
            'file': open(audio_path, 'rb'),
        }

        # Prepare data payload if needed
        data = {
            'language': 'en',  # Adjust based on your model's requirements
            'response_format': 'verbose_json',  # To get segments if supported
        }

        # Send POST request to the Whisper endpoint
        response = requests.post(
            WHISPER_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=600  # Adjust timeout as needed
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()

        # Extract full_text and segments
        full_text = result.get('text', '').strip()
        segments = result.get('segments', [])

        logger.info("Transcription successful.")
        return full_text, segments

    except Exception as e:
        logger.error(f"Error during transcription of file {audio_path}: {e}")
        raise  # Re-raise the exception to be handled by the calling function
