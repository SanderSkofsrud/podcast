# ad_detector.py

import logging
import time
import openai
import json
from config.settings import client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def detect_ad_segments(transcription):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that detects advertisement segments in a podcast transcription. "
                "Given the transcription with timestamps, identify all advertisement segments and return them in JSON format "
                "with each advertisement containing 'text', 'start', and 'end' in seconds.\n"
                "Example format:\n"
                "[\n"
                "    {\n"
                "        \"text\": \"Example ad\",\n"
                "        \"start\": 10.5,\n"
                "        \"end\": 70.04\n"
                "    }\n"
                "]\n"
                "Only include advertisement segments. Do not include any additional text."
            )
        }
    ]

    user_content = (
        "Identify all advertisement segments in the following transcription and return them in the specified JSON format.\n\n"
        f"{transcription}"
    )

    messages.append({"role": "user", "content": user_content})

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )

            content = response.choices[0].message.content.strip()

            logger.info(f"Assistant's response:\n{content}")

            ad_segments = json.loads(content)

            if isinstance(ad_segments, list):
                for segment in ad_segments:
                    if not all(k in segment for k in ("text", "start", "end")):
                        raise ValueError("Each advertisement segment must contain 'text', 'start', and 'end' keys.")
                break
            else:
                raise ValueError("Response is not a list.")

        except json.JSONDecodeError as jde:
            logger.error(f"JSON decode error: {jde}. Response was: {content}")
        except openai.error.RateLimitError as e:
            logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            logger.error(f"Error detecting ad segments: {e}")
            ad_segments = []
            break
    else:
        logger.error("Max retries exceeded for ad segment detection.")
        ad_segments = []

    logger.info(f"Advertisement segments: {ad_segments}")
    return ad_segments
