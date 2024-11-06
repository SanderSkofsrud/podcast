# app.py

from flask import Flask, request, send_file, after_this_request
from transcriber import transcribe_audio
from classifier import classify_texts
from audio_editor import remove_ad_segments
import os
import logging
import uuid
import json
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request.")

    # Save the uploaded audio file with a unique filename
    audio_file = request.files['audio']
    unique_id = str(uuid.uuid4())
    audio_filename = f"temp_audio_{unique_id}.mp3"
    audio_file.save(audio_filename)
    logger.info(f"Saved audio file to {audio_filename}.")

    try:
        # Transcribe audio
        logger.info("Starting transcription.")
        transcription, segments = transcribe_audio(audio_filename)
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
        edited_audio_filename = remove_ad_segments(audio_filename, ad_segments)
        logger.info(f"Audio editing completed. Edited file saved to {edited_audio_filename}.")

        # Return the edited audio file
        @after_this_request
        def remove_files(response):
            try:
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logger.info(f"Removed temporary file {audio_filename}.")
                if os.path.exists(edited_audio_filename):
                    os.remove(edited_audio_filename)
                    logger.info(f"Removed temporary file {edited_audio_filename}.")
            except Exception as e:
                logger.error(f"Error removing temporary files: {e}")
            return response

        # Read the file into a BytesIO object
        with open(edited_audio_filename, 'rb') as f:
            file_data = BytesIO(f.read())

        # Delete the edited audio file immediately after reading
        if os.path.exists(edited_audio_filename):
            os.remove(edited_audio_filename)
            logger.info(f"Removed temporary file {edited_audio_filename}.")

        response = send_file(
            file_data,
            mimetype='audio/mp3',
            as_attachment=True,
            download_name='edited_audio.mp3'
        )
        return response

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return "Error processing audio.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)