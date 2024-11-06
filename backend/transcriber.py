from pydub import AudioSegment
import openai
import logging
from config.settings import (
    get_whisper_api_key,
    get_whisper_api_version,
    get_whisper_deployment_id,
    get_whisper_endpoint,
)
import os
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=get_whisper_api_key(),
    azure_endpoint=get_whisper_endpoint(),
    api_version=get_whisper_api_version(),
)

def split_audio(audio_path, chunk_size_ms=240000):  # Adjust chunk size if needed
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]
    return chunks

def transcribe_audio_chunks(audio_path):
    try:
        logger.info(f"Transcribing audio file {audio_path} in chunks.")

        # Split audio into manageable chunks
        chunks = split_audio(audio_path)
        full_text = ""
        all_segments = []

        for i, chunk in enumerate(chunks):
            chunk_path = f"temp_chunk_{i}.mp3"
            chunk.export(chunk_path, format="mp3")

            with open(chunk_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model=get_whisper_deployment_id(),
                    response_format='verbose_json',
                    language='en'
                )

            os.remove(chunk_path)  # Clean up temporary chunk file

            # Append transcription results
            full_text += response.text.strip() + " "
            all_segments.extend(response.segments)

        logger.info("Transcription of all chunks successful.")
        return full_text.strip(), all_segments

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise


