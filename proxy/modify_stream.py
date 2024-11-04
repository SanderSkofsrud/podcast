# modify_stream.py

from mitmproxy import http
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_SERVER_URL = 'http://localhost:5000/process_audio'

def response(flow: http.HTTPFlow) -> None:
    # Check if the response is an audio stream
    content_type = flow.response.headers.get("Content-Type", "")
    if "audio/mpeg" in content_type or "audio/mp3" in content_type:
        logger.info(f"Intercepted audio stream: {flow.request.url}")
        audio_data = flow.response.content

        # Send the audio data to the backend server
        files = {'audio': ('audio.mp3', audio_data)}
        try:
            response = requests.post(BACKEND_SERVER_URL, files=files)
            if response.status_code == 200:
                flow.response.content = response.content
                # Update Content-Length header
                flow.response.headers['Content-Length'] = str(len(flow.response.content))
                logger.info("Replaced audio stream with ad-free version.")
            else:
                logger.error(f"Backend server error: {response.status_code}")
                # Optionally handle the error, e.g., keep the original audio
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to backend server: {e}")
            # Optionally handle the exception
