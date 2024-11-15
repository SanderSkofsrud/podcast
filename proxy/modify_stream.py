import asyncio
import logging
import threading

from mitmproxy import http, ctx

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend server URL
BACKEND_SERVER_URL = 'http://localhost:5001/process_audio'


class AudioStreamModifier:
    def __init__(self):
        self.flow_cache = {}
        self.loop = None  # Placeholder for the event loop

    def load(self, loader):
        """
        Called when the addon is loaded.
        Stores the event loop for later use.
        """
        self.loop = asyncio.get_running_loop()
        logger.info("AudioStreamModifier addon loaded and event loop acquired.")

    def response(self, flow: http.HTTPFlow) -> None:
        """
        Intercept HTTP responses to modify audio streams.
        """
        content_type = flow.response.headers.get("Content-Type", "")
        if "audio/mpeg" in content_type or "audio/mp3" in content_type:
            logger.info(f"Intercepted audio stream: {flow.request.url}")

            # Replace the original response with an empty response
            original_response = flow.response
            flow.response = http.Response.make(
                200,  # (optional) status code
                b'',  # empty content
                original_response.headers  # preserve original headers
            )
            flow_id = flow.id

            # Store the flow for later use
            self.flow_cache[flow_id] = flow

            # Start processing the audio in a separate thread
            threading.Thread(target=self.process_audio_stream, args=(flow_id, original_response), daemon=True).start()

    def process_audio_stream(self, flow_id, original_response):
        """
        Send the intercepted audio to the backend for processing.
        """
        try:
            # Send the audio data to the backend
            headers = {'Content-Type': 'audio/mpeg'}
            logger.info(f"Sending audio data to backend for flow {flow_id}")

            response = requests.post(
                BACKEND_SERVER_URL,
                data=original_response.content,
                headers=headers
            )

            if response.status_code == 200:
                processed_content = response.content
                logger.info(f"Received processed audio for flow {flow_id}, size {len(processed_content)} bytes")

                # Schedule the update_flow_response to run in the main thread
                self.loop.call_soon_threadsafe(self.update_flow_response, flow_id, processed_content)
            else:
                logger.error(f"Backend server error: {response.status_code}")
                # Fallback to original response
                self.send_original_response(flow_id, original_response)
        except Exception as e:
            logger.error(f"Error processing flow {flow_id}: {e}", exc_info=True)
            # Fallback to original response
            self.send_original_response(flow_id, original_response)

    def update_flow_response(self, flow_id, content):
        """
        Update the intercepted flow with the processed audio content.
        """
        flow = self.flow_cache.pop(flow_id, None)
        if flow:
            flow.response.content = content
            flow.response.headers['Content-Length'] = str(len(content))
            logger.info(f"Sent processed audio to client for flow {flow_id}")

    def send_original_response(self, flow_id, original_response):
        """
        Restore the original audio response in case of an error.
        """
        flow = self.flow_cache.pop(flow_id, None)
        if flow:
            flow.response.content = original_response.content
            flow.response.headers = original_response.headers
            logger.info(f"Sent original audio to client for flow {flow_id} due to an error.")


addons = [
    AudioStreamModifier()
]
