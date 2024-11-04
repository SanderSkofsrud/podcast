# app.py
from flask import Flask, request, send_file
from transcriber import transcribe_audio
from classifier import classify_texts
from audio_editor import remove_ad_segments
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request.")

    # Save the uploaded audio file
    audio_file = request.files['audio']
    audio_path = 'temp_audio.mp3'
    audio_file.save(audio_path)
    logger.info(f"Saved audio file to {audio_path}.")

    # Transcribe audio
    logger.info("Starting transcription.")
    transcription, segments = transcribe_audio(audio_path)
    logger.info("Transcription completed.")

    # Classify segments in batch
    logger.info("Starting batch classification of segments.")
    texts = [segment['text'] for segment in segments]
    labels = classify_texts(texts)
    ad_segments = []
    for segment, label in zip(segments, labels):
        text = segment['text']
        logger.debug(f"Segment '{text[:30]}...' classified as '{label}'.")
        if label == 'advertisement':
            ad_segments.append({'start': segment['start'], 'end': segment['end']})
    logger.info("Batch classification completed.")

    # Remove ad segments
    logger.info("Starting audio editing to remove ads.")
    try:
        edited_audio_path = remove_ad_segments(audio_path, ad_segments)
        logger.info(f"Audio editing completed. Edited file saved to {edited_audio_path}.")
    except Exception as e:
        logger.error(f"Error during audio editing: {e}")
        return "Error during audio editing.", 500

    # Return the edited audio file
    try:
        response = send_file(edited_audio_path, mimetype='audio/mp3')
    except Exception as e:
        logger.error(f"Error sending edited audio file: {e}")
        return "Error sending edited audio file.", 500
    finally:
        # Clean up temporary files
        os.remove(audio_path)
        os.remove(edited_audio_path)
        logger.info(f"Removed temporary files {audio_path} and {edited_audio_path}.")

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
