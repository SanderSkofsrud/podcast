# transcriber.py
import azure.cognitiveservices.speech as speechsdk
import logging
from config.settings import get_speech_key, get_speech_region
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

        def handle_final_result(evt):
            nonlocal full_text, segments
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text
                offset = result.offset / 10_000_000
                duration = result.duration / 10_000_000
                segments.append({'start': offset, 'end': offset + duration, 'text': text})
                full_text += text + " "

        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.start_continuous_recognition()
        logger.info("Transcription in progress...")

        result = speech_recognizer.recognize_once_async().get()
        speech_recognizer.stop_continuous_recognition()

        logger.info("Transcription successful.")
        return full_text.strip(), segments
    except Exception as e:
        logger.error(f"Error during transcription of file {audio_path}: {e}")

        return "", []  # Return empty text and segments in case of error