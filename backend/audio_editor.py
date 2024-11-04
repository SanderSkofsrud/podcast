# audio_editor.py
from pydub import AudioSegment
import logging
from itertools import chain

logger = logging.getLogger(__name__)

def remove_ad_segments(audio_path, segments_to_remove):
    audio = AudioSegment.from_file(audio_path)
    keep_ranges = []
    last_end = 0

    for segment in segments_to_remove:
        start = int(segment['start'] * 1000)
        end = int(segment['end'] * 1000)
        if last_end < start:
            keep_ranges.append((last_end, start))
        last_end = end
    if last_end < len(audio):
        keep_ranges.append((last_end, len(audio)))

    # Extract all segments to keep
    segments_to_keep = (audio[start:end] for start, end in keep_ranges)

    # Efficiently concatenate segments
    final_audio = audio._spawn(data=b''.join(segment.raw_data for segment in segments_to_keep))

    edited_audio_path = "edited_" + audio_path
    final_audio.export(edited_audio_path, format="mp3")
    logger.info(f"Exported edited audio to {edited_audio_path}.")
    return edited_audio_path
