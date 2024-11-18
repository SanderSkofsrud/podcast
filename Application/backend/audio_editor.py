# audio_editor.py

from pydub import AudioSegment
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def remove_ad_segments(audio_path, ad_segments):

    try:
        audio = AudioSegment.from_file(audio_path)
        original_length_ms = len(audio)
        logger.info(f"Original audio length: {original_length_ms / 1000:.2f} seconds")

        ad_segments_sorted = sorted(ad_segments, key=lambda x: x['start'])

        keep_ranges = []
        last_end = 0

        for segment in ad_segments_sorted:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)

            if last_end < start_ms:
                keep_ranges.append((last_end, start_ms))
            last_end = end_ms

        if last_end < len(audio):
            keep_ranges.append((last_end, len(audio)))

        logger.info(f"Segments to keep: {keep_ranges}")
        logger.info(f"Segments to remove: {ad_segments_sorted}")

        segments_to_keep = [audio[start:end] for start, end in keep_ranges]
        final_audio = sum(segments_to_keep)

        edited_length_ms = len(final_audio)
        logger.info(f"Edited audio length: {edited_length_ms / 1000:.2f} seconds")

        if edited_length_ms >= original_length_ms:
            logger.warning("Edited audio is not shorter than the original. Check if ad segments are correctly identified.")

        edited_audio_filename = f"edited_{os.path.basename(audio_path)}"
        final_audio.export(edited_audio_filename, format="mp3")
        logger.info(f"Exported edited audio to {edited_audio_filename}.")
        return edited_audio_filename
    except Exception as e:
        logger.error(f"Error during audio editing: {e}")
        raise e
