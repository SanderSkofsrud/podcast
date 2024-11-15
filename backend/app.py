# app.py

from flask import Flask, request, send_file, after_this_request
from transcriber import transcribe_audio
from classifier import classify_texts
from audio_editor import remove_ad_segments
import os
import logging
import uuid
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request.")
    if 'audio/mpeg' not in request.headers.get('Content-Type', ''):
        logger.info(f"Unsupported media type: {request.headers.get('Content-Type', '')}")
        return "Unsupported media type. Please send audio data as 'audio/mpeg'.", 415

    # Generate a unique filename for the temporary audio file
    unique_id = str(uuid.uuid4())
    audio_filename = f"temp_audio_{unique_id}.mp3"

    # Open a file to write the incoming data
    with open(audio_filename, 'wb') as f:
        # Read the data in chunks from the request stream
        chunk_size = 4096  # Adjust the chunk size as needed
        while True:
            chunk = request.stream.read(chunk_size)
            logger.debug(f"Received chunk of size {len(chunk)} bytes.")
            if not chunk:
                break
            f.write(chunk)
            f.flush()
            # Optionally, you can start processing the data here if your functions support streaming

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

        # Read the edited audio file into a BytesIO object
        with open(edited_audio_filename, 'rb') as f:
            file_data = BytesIO(f.read())

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
