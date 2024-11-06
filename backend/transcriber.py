# transcriber.py
import azure.cognitiveservices.speech as speechsdk
import logging
from config.settings import get_speech_key, get_speech_region
import threading  # Import threading module

logger = logging.getLogger(__name__)

# Azure Speech Service key and region
SPEECH_KEY = get_speech_key()
SERVICE_REGION = get_speech_region()

def transcribe_audio(audio_path):
    try:
        logger.info(f"Loading audio file {audio_path} for transcription.")
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
        speech_config.speech_recognition_language = "en-US"
        audio_input = speechsdk.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        full_text = ""
        segments = []

        # Synchronization event to wait until transcription is complete
        done = threading.Event()

        def handle_final_result(evt):
            nonlocal full_text, segments
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text
                offset = result.offset / 10_000_000  # Convert ticks to seconds
                duration = result.duration / 10_000_000  # Convert ticks to seconds
                segments.append({'start': offset, 'end': offset + duration, 'text': text})
                full_text += text + " "
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized.")
 
        def session_stopped(evt):
            logger.info("Transcription session stopped.")
            done.set()

        def canceled(evt):
            logger.error(f"Transcription canceled: {evt.reason}")
            done.set()

        # Connect event handlers
        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.session_stopped.connect(session_stopped)
        speech_recognizer.canceled.connect(canceled)

        # Start continuous transcription
        speech_recognizer.start_continuous_recognition()
        logger.info("Transcription in progress...")

        # Wait until the transcription is done
        done.wait()

        # Stop the recognizer
        speech_recognizer.stop_continuous_recognition()

        logger.info("Transcription successful.")
        return full_text.strip(), segments
    except Exception as e:
        logger.error(f"Error during transcription of file {audio_path}: {e}")
        # Re-raise the exception to be handled by the calling function
        raise e
