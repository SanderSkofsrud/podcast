# podcast_processor/ad_detection/detect_ads.py

import json
import os
import time
import logging
import openai
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import RateLimitError, APIConnectionError, OpenAIError

from podcast_processor.ad_detection.rate_limiter import rate_limiter
from podcast_processor.ad_detection.retry import retry_request
from podcast_processor.ad_detection.token_utils import num_tokens_from_messages
from podcast_processor.config import (
    AD_DETECTIONS_DIR,
    LLM_MODELS,
    TRANSCRIPTIONS_DIR,
    LLM_MODEL_CONFIGS, WHISPER_MODELS
)
from podcast_processor.ad_detection.utils import (
    parse_ads_response,
    save_ad_detections,
    deduplicate_ads,
    validate_ads
)
from podcast_processor.evaluation.evaluate import ads_overlap
from podcast_processor.transcription.utils import normalize_text

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_transcription(file_path: str) -> str:
    """
    Read the transcription text from a file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading transcription from '{file_path}': {e}")
        return ""

OVERLAP_TOKENS = 100  # Number of tokens to overlap between chunks

@retry_request(
    max_retries=5,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=60.0,
    allowed_errors=(RateLimitError, APIConnectionError)
)
def detect_ads(model_name: str, transcription_segments: List[Dict]) -> List[Dict]:
    """
    Detect advertisements in a transcription using the specified LLM model.
    Implements chunking to handle long transcriptions.

    Args:
        model_name (str): The name of the LLM model to use.
        transcription_segments (List[Dict]): List of transcription segments with 'start', 'end', and 'text'.

    Returns:
        List[Dict]: List of detected ads with 'text', 'start', 'end', and 'models'.
    """
    ads = []
    transcription_lines = []
    for segment in transcription_segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        transcription_lines.append(f"[{start:.2f} - {end:.2f}] {text}")

    # Fetch model configurations
    model_config = LLM_MODEL_CONFIGS.get(model_name, {})
    context_window = model_config.get("context_window", 32_768)  # Corrected value
    max_tokens_per_minute = model_config.get("max_tokens_per_minute", 30_000)  # Default remains if unspecified

    # Define max_output_length based on model's capabilities
    max_output_length = model_config.get("max_output_length", 4096)
    max_response_tokens = max_output_length  # Maximum tokens the response can have

    # Initialize encoding for the model
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.error(f"Encoding not found for model '{model_name}'.")
        return ads

    # Calculate tokens in the prompt template (excluding the chunk)
    prompt_template = (
        "You will be provided with a podcast transcription that contains advertisements. "
        "Each line of the transcription includes the start and end times in seconds, followed by the text.\n\n"
        "Transcription:\n"
        # "{chunk}\n\n"  # We don't include the chunk here when calculating tokens
        "Identify and list the advertisements in the transcription.\n\n"
        "For each advertisement, provide the following details in JSON format exactly as shown below without any additional text:\n"
        "[\n"
        "  {\n"
        "    \"text\": \"Ad text\",\n"
        "    \"start\": Start time in seconds,\n"
        "    \"end\": End time in seconds\n"
        "  },\n"
        "  ...\n"
        "]\n\n"
        "Ensure the response is a valid JSON array without any code blocks, markdown, or additional text. Keep each ad entry as concise as possible."
    )
    prompt_template_tokens = len(encoding.encode(prompt_template))

    # Calculate maximum available tokens for the prompt (including the chunk)
    max_available_tokens_for_prompt = int((context_window - max_response_tokens - prompt_template_tokens) * 0.9) # 10% buffer
    logger.info(f"Max available tokens for prompt (including chunk): {max_available_tokens_for_prompt} for model '{model_name}'.")

    # Calculate max_chunk_tokens
    max_chunk_tokens = max_available_tokens_for_prompt - prompt_template_tokens
    if max_chunk_tokens <= 0:
        logger.error(f"Prompt template is too large for the model '{model_name}' context window.")
        return ads

    logger.info(f"Max chunk tokens: {max_chunk_tokens} for model '{model_name}'.")

    # Split the transcription into chunks with overlapping tokens
    chunks = []
    current_chunk = []
    current_tokens = 0
    for line in transcription_lines:
        line_tokens = len(encoding.encode(line + '\n'))  # Include newline
        if current_tokens + line_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            # Start new chunk with overlap
            overlap_tokens = 0
            overlap_chunk = []
            # Add overlapping tokens from the end of the current_chunk
            for prev_line in reversed(current_chunk):
                overlap_line_tokens = len(encoding.encode(prev_line + '\n'))
                if overlap_tokens + overlap_line_tokens >= OVERLAP_TOKENS:
                    break
                overlap_chunk.insert(0, prev_line)
                overlap_tokens += overlap_line_tokens
            current_chunk = overlap_chunk + [line]
            current_tokens = sum(len(encoding.encode(l + '\n')) for l in current_chunk)
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    # Initialize an empty list to collect ads from all chunks
    all_ads = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        prompt = (
            "You will be provided with a podcast transcription that contains advertisements. "
            "Each line of the transcription includes the start and end times in seconds, followed by the text.\n\n"
            f"Transcription:\n{chunk}\n\n"
            "Identify and list the advertisements in the transcription.\n\n"
            "For each advertisement, provide the following details in JSON format exactly as shown below without any additional text:\n"
            "[\n"
            "  {\n"
            "    \"text\": \"Ad text\",\n"
            "    \"start\": Start time in seconds,\n"
            "    \"end\": End time in seconds\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Ensure the response is a valid JSON array without any code blocks, markdown, or additional text. Keep each ad entry as concise as possible."
        )

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that helps identify advertisements in podcast transcriptions.",
            },
            {"role": "user", "content": prompt},
        ]

        # Count tokens in the prompt (including the chunk)
        prompt_tokens = num_tokens_from_messages(messages, model_name)

        # Total tokens for this request
        total_tokens = prompt_tokens + max_output_length

        # Check if total_tokens exceeds context_window
        if total_tokens > context_window:
            logger.error(
                f"Total tokens ({total_tokens}) exceed context window ({context_window}) for model '{model_name}'. Skipping chunk {chunk_index}."
            )
            continue

        # Acquire tokens from the rate limiter
        rate_limiter.acquire(model_name, total_tokens)

        logger.info(f"Processing chunk {chunk_index} with {prompt_tokens} prompt tokens and {max_response_tokens} response tokens for model '{model_name}'.")

        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that helps identify advertisements in podcast transcriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,  # For deterministic output
                max_tokens=max_response_tokens,
            )
            ads_content = response.choices[0].message.content.strip()

            # Parse the ads from the response
            chunk_ads = parse_ads_response(ads_content)

            # Validate and clean the ads
            chunk_ads = validate_ads(chunk_ads)
            logger.info(f"Detected {len(chunk_ads)} ads in chunk {chunk_index} using model '{model_name}'.")
            logger.info(f"Chunk {chunk_index} ads: {chunk_ads} model: {model_name}")

            # Associate ads with the model
            for ad in chunk_ads:
                ad.setdefault('models', []).append(model_name)
                all_ads.append(ad)

        except OpenAIError as e:
            logger.error(f"OpenAI API error while processing chunk {chunk_index} with model '{model_name}': {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error while processing chunk {chunk_index} with model '{model_name}': {e}")
            continue

    # Deduplicate ads
    deduped_ads = deduplicate_ads(all_ads, overlap_threshold=0.5)

    return deduped_ads

def detect_ads_for_model(llm_model_name: str, run_dir: str):
    """
    Detect advertisements in all transcriptions using a specific LLM model.
    Also, measure the processing time for each model.

    Args:
        llm_model_name (str): The name of the LLM model to use.
        run_dir (str): The directory for the current run.

    Returns:
        Tuple[Dict[str, Dict[str, List[Dict]]], float]: Detected ads per whisper_model, and total processing time.
    """
    results = {}  # Dict[str, Dict[str, List[Dict]]], maps whisper_model to audio files and their detected ads
    total_processing_time = 0.0  # Initialize total processing time
    transcription_base_dir = TRANSCRIPTIONS_DIR  # This is the 'data/transcriptions' directory

    # Iterate over active Whisper models
    for whisper_model in WHISPER_MODELS:
        whisper_model_dir = os.path.join(transcription_base_dir, whisper_model)
        if not os.path.isdir(whisper_model_dir):
            logger.warning(f"Transcription directory for Whisper model '{whisper_model}' does not exist.")
            continue

        # Iterate over each transcription file in the Whisper model directory
        for transcription_file in os.listdir(whisper_model_dir):
            if not transcription_file.endswith('_segments.json'):
                continue
            audio_file = transcription_file.replace('_segments.json', '.mp3')
            transcription_path = os.path.join(whisper_model_dir, transcription_file)
            transcription_segments = load_transcription_segments(transcription_path)
            if not transcription_segments:
                thread_name = threading.current_thread().name
                logger.warning(f"Thread {thread_name}: Empty transcription segments for '{audio_file}'. Skipping ad detection.")
                continue

            # Measure the start time
            start_time = time.time()
            # Process the transcription segments using the LLM model
            ads = detect_ads(llm_model_name, transcription_segments)
            # Measure the end time and calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_processing_time += elapsed_time
            logger.info(f"Model '{llm_model_name}' processed '{audio_file}' in {elapsed_time:.2f} seconds.")

            # Initialize results for this whisper_model if not already
            if whisper_model not in results:
                results[whisper_model] = {}
            results[whisper_model][audio_file] = ads

            # Save the detected ads, including the whisper_model
            save_ad_detections(ads, whisper_model, llm_model_name, audio_file, AD_DETECTIONS_DIR)

    return results, total_processing_time

def detect_ads_in_transcriptions(run_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, List[Dict]]]], Dict[str, float]]:
    """
    Detect advertisements in all transcriptions using all specified LLM models.
    Returns a tuple containing:
        - ad_detections: Dict mapping whisper_model to llm_model to audio files and their detected ads.
        - processing_times: Dict mapping LLM model names to their total processing time.

    Args:
        run_dir (str): The directory for the current run.

    Returns:
        Tuple[Dict[str, Dict[str, Dict[str, List[Dict]]]], Dict[str, float]]: Detected ads and processing times.
    """
    llm_models = LLM_MODELS
    ad_detections = {}
    processing_times = {}

    logger.info(f"Starting ad detection with {len(llm_models)} parallel threads.")

    with ThreadPoolExecutor(max_workers=len(llm_models)) as executor:
        future_to_model = {executor.submit(detect_ads_for_model, model, run_dir): model for model in llm_models}

        for future in as_completed(future_to_model):
            llm_model_name = future_to_model[future]
            try:
                detections, processing_time = future.result()
                processing_times[llm_model_name] = processing_time
                for whisper_model, audio_files in detections.items():
                    if whisper_model not in ad_detections:
                        ad_detections[whisper_model] = {}
                    if llm_model_name not in ad_detections[whisper_model]:
                        ad_detections[whisper_model][llm_model_name] = {}
                    ad_detections[whisper_model][llm_model_name].update(audio_files)
                logger.info(f"Model '{llm_model_name}' completed in {processing_time:.2f} seconds.")
            except ZeroDivisionError as e:
                logger.error(f"ZeroDivisionError during deduplication: {e}")
                continue
            except Exception as e:
                logger.error(f"Error detecting ads with model {llm_model_name}: {e}")
                continue

    return ad_detections, processing_times

def load_transcription_segments(file_path: str) -> List[Dict]:
    """
    Load the transcription segments from a JSON file.

    Args:
        file_path (str): Path to the transcription segments JSON file.

    Returns:
        List[Dict]: List of transcription segments.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
            return segments
    except Exception as e:
        logger.error(f"Error loading transcription segments from '{file_path}': {e}")
        return []
