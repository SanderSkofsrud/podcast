# app.py

from flask import Flask, request, send_file, after_this_request, jsonify
from flask_cors import CORS
from transcriber import transcribe_audio
from ad_detector import detect_ad_segments
from audio_editor import remove_ad_segments
import os
import logging
import uuid
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#This CORS could be reduced to lesser access.
app = Flask(__name__)
CORS(app)
status_dict = {}

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request.")

    mode = request.form.get('mode', 'fast').lower()
    if mode not in ['fast', 'accurate']:
        return jsonify({"error": "Invalid mode. Choose 'fast' or 'accurate'."}), 400

    logger.info(f"Processing mode: {mode}")

    unique_id = str(uuid.uuid4())
    status_dict[unique_id] = "Received audio processing request"
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided."}), 400
    audio_filename = f"temp_audio_{unique_id}.mp3"
    audio_file.save(audio_filename)
    logger.info(f"Saved audio file to {audio_filename}.")

    try:
        status_dict[unique_id] = "Transcribing audio"
        transcription, segments = transcribe_audio(audio_filename, mode=mode)
        logger.info("Transcription completed.")

        status_dict[unique_id] = "Detecting advertisement segments"
        ad_segments = detect_ad_segments(transcription)
        logger.info(f"Identified {len(ad_segments)} advertisement segments.")
        logger.debug(f"Advertisement segments: {ad_segments}")  

        status_dict[unique_id] = "Removing advertisement segments"
        edited_audio_filename = remove_ad_segments(audio_filename, ad_segments)
        logger.info(f"Audio editing completed. Edited file saved to {edited_audio_filename}.")

        status_dict[unique_id] = "Completed"

        @after_this_request
        def remove_files(response):
            try:
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    logger.info(f"Removed temporary file {audio_filename}.")
                if os.path.exists(edited_audio_filename):
                    os.remove(edited_audio_filename)
                    logger.info(f"Removed temporary file {edited_audio_filename}.")
                del status_dict[unique_id]  
            except Exception as e:
                logger.error(f"Error removing temporary files: {e}")
            return response

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
        status_dict[unique_id] = "Error"
        return jsonify({"error": "Error processing audio."}), 500

@app.route('/status/<request_id>', methods=['GET'])
def get_status(request_id):
    status = status_dict.get(request_id, "Unknown request ID")
    return jsonify({"status": status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
