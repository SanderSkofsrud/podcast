from mitmproxy import http
import requests
import logging
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_SERVER_URL = 'http://localhost:5001/process_audio'

class AudioStreamModifier:
    def __init__(self):
        self.chunk_queues = {}

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        content_type = flow.response.headers.get("Content-Type", "")
        if "audio/mpeg" in content_type or "audio/mp3" in content_type:
            logger.info(f"Intercepted audio stream: {flow.request.url}")
            flow.response.stream = True  # Enable streaming
            self.chunk_queues[flow.id] = queue.Queue()
            threading.Thread(target=self.send_chunks_to_backend, args=(flow,)).start()

    def http_response_chunk(self, flow: http.HTTPFlow, chunk: http.ResponseData) -> None:
        if flow.id in self.chunk_queues:
            # Put the chunk into the queue
            self.chunk_queues[flow.id].put(chunk.content)
            if chunk.last:
                # Signal that we're done
                self.chunk_queues[flow.id].put(None)

            # You can modify the chunk here if needed
            # chunk.content = modify_chunk(chunk.content)

    def send_chunks_to_backend(self, flow):
        def chunk_generator():
            while True:
                chunk = self.chunk_queues[flow.id].get()
                if chunk is None:
                    break
                yield chunk
            # Clean up
            del self.chunk_queues[flow.id]

        try:
            headers = {'Content-Type': 'audio/mpeg'}  # Set appropriate headers
            response = requests.post(
                BACKEND_SERVER_URL,
                data=chunk_generator(),
                headers=headers,
                stream=True  # Enable streaming
            )
            if response.status_code == 200:
                flow.response.content = response.content
                flow.response.headers['Content-Length'] = str(len(flow.response.content))
                logger.info("Replaced audio stream with ad-free version.")
            else:
                logger.error(f"Backend server error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to backend server: {e}")

addons = [
    AudioStreamModifier()
]
